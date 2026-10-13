import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

spec = importlib.util.spec_from_file_location("fit_check", Path(__file__).parent / "fit-check.py")
fc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fc)


@pytest.fixture
def photo(tmp_path):
    """Left half red, right half blue."""
    p = tmp_path / "outfit.png"
    im = Image.new("RGB", (40, 40), (255, 0, 0))
    im.paste((0, 0, 255), (20, 0, 40, 40))
    im.save(p)
    return p


REPORT = {
    "overall_vibe": "bold color blocking", "score": 8,
    "strengths": ["strong contrast"], "improvements": ["add a neutral layer"],
    "occasion": "weekend outing",
    "palette": [
        {"hex": "#FF0000", "name": "red", "role": "top"},
        {"hex": "#0000ff", "name": "blue", "role": "pants"},
        {"hex": "#FFFFFF", "name": "white", "role": "shoes"},
    ],
    "palette_notes": "Primary colors pop; add cream to soften.",
}


def message(stop_reason, content):
    return SimpleNamespace(stop_reason=stop_reason, content=content)


def text(s):
    return SimpleNamespace(type="text", text=s)


def tool_use(id_, name, input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


class FakeClient:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_load_image_ok(photo):
    data, media_type = fc.load_image(str(photo))
    assert media_type == "image/png" and data


def test_load_image_rejects_bad_input(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("hi")
    with pytest.raises(ValueError, match="unsupported"):
        fc.load_image(str(txt))
    fake = tmp_path / "fake.jpg"
    fake.write_text("not an image")
    with pytest.raises(ValueError, match="not a readable image"):
        fc.load_image(str(fake))


def test_dominant_colors_finds_both_halves(photo):
    data, _ = fc.load_image(str(photo))
    colors = fc.tool_dominant_colors(data, 2)
    assert {c["hex"] for c in colors} == {"#FF0000", "#0000FF"}
    assert sum(c["share"] for c in colors) == pytest.approx(1.0, abs=0.01)


def test_palette_validation():
    ok = fc.StyleFeedback.model_validate(REPORT)
    assert [c.hex for c in ok.palette] == ["#FF0000", "#0000FF", "#FFFFFF"]
    with pytest.raises(ValueError, match="3-5"):
        fc.StyleFeedback.model_validate({**REPORT, "palette": REPORT["palette"][:2]})
    bad = [{**REPORT["palette"][0], "hex": "red"}] + REPORT["palette"][1:]
    with pytest.raises(ValueError, match="#RRGGBB"):
        fc.StyleFeedback.model_validate({**REPORT, "palette": bad})
    with pytest.raises(ValueError, match="1-10"):
        fc.StyleFeedback.model_validate({**REPORT, "score": 11})


def test_agent_loop_runs_tool_then_returns_report(photo):
    data, media_type = fc.load_image(str(photo))
    client = FakeClient(
        message("tool_use", [tool_use("t1", "dominant_colors", {"count": 2})]),
        message("end_turn", [text(json.dumps(REPORT))]),
    )
    report = fc.build_report(data, media_type, client=client)
    assert report.score == 8 and len(report.palette) == 3
    first = client.calls[0]
    assert first["thinking"] == {"type": "adaptive"}
    assert first["messages"][0]["content"][0]["type"] == "image"
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "t1" and "#FF0000" in result["content"]


def test_agent_loop_reports_tool_errors_and_refusals(photo):
    data, media_type = fc.load_image(str(photo))
    client = FakeClient(
        message("tool_use", [tool_use("t1", "nope", {})]),
        message("refusal", []),
    )
    with pytest.raises(RuntimeError, match="declined"):
        fc.build_report(data, media_type, client=client)
    assert client.calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_main_missing_file_returns_1(capsys):
    assert fc.main(["/no/such/photo.jpg"]) == 1
    assert "error" in capsys.readouterr().err
