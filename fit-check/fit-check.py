#!/usr/bin/env python3
"""Fit Check: send an outfit photo to Claude and get structured style feedback plus a color palette.

Claude (Sonnet, adaptive thinking, vision) looks at the photo and runs a tool-use loop that
measures real pixel colors with Pillow, then returns a schema-validated report as structured output.
"""
import argparse
import base64
import io
import json
import os
import re
import sys
from pathlib import Path

import anthropic
from anthropic import transform_schema
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, field_validator

MODEL = os.environ.get("FIT_CHECK_MODEL", "claude-sonnet-5-5")
MAX_TURNS = 6
MAX_IMAGE_BYTES = 5 * 1024 * 1024  # API limit for a base64 image
MEDIA_TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
               ".gif": "image/gif", ".webp": "image/webp"}
HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}$")


class PaletteColor(BaseModel):
    hex: str = Field(description="Color as #RRGGBB")
    name: str = Field(description="Short human name, e.g. 'dusty rose'")
    role: str = Field(description="Where it shows up in the outfit, e.g. 'jacket' or 'accent'")

    @field_validator("hex")
    @classmethod
    def valid_hex(cls, v: str) -> str:
        if not HEX_RE.match(v):
            raise ValueError(f"not a #RRGGBB hex code: {v!r}")
        return v.upper()


class StyleFeedback(BaseModel):
    overall_vibe: str = Field(description="One-line read on the style, e.g. 'relaxed smart-casual'")
    score: int = Field(description="Overall outfit score from 1 to 10")
    strengths: list[str] = Field(description="What works, specific to this photo")
    improvements: list[str] = Field(description="Concrete, actionable changes")
    occasion: str = Field(description="Occasions this outfit suits best")
    palette: list[PaletteColor] = Field(description="3 to 5 colors that define the outfit")
    palette_notes: str = Field(description="How the colors work together, and one color to add")

    @field_validator("palette")
    @classmethod
    def palette_size(cls, v: list[PaletteColor]) -> list[PaletteColor]:
        if not 3 <= len(v) <= 5:
            raise ValueError(f"palette must have 3-5 colors, got {len(v)}")
        return v

    @field_validator("score")
    @classmethod
    def score_range(cls, v: int) -> int:
        if not 1 <= v <= 10:
            raise ValueError(f"score must be 1-10, got {v}")
        return v


# --------------------------------------------------------------------------- image


def load_image(path: str) -> tuple[bytes, str]:
    """Read and validate an image file. Returns (raw bytes, media type)."""
    p = Path(path)
    media_type = MEDIA_TYPES.get(p.suffix.lower())
    if media_type is None:
        raise ValueError(f"unsupported image type {p.suffix or '(none)'}; use jpg, png, gif or webp")
    data = p.read_bytes()
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(f"{path} is {len(data) / 1e6:.1f} MB; the limit is 5 MB")
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"{path} is not a readable image: {exc}") from exc
    return data, media_type


# ------------------------------------------------------------------------- tools


def tool_image_info(data: bytes) -> dict:
    with Image.open(io.BytesIO(data)) as im:
        return {"width": im.width, "height": im.height, "format": im.format, "mode": im.mode}


def tool_dominant_colors(data: bytes, count: int = 5) -> list[dict]:
    """Quantize the image and return the most common colors by share of pixels."""
    count = max(1, min(int(count), 8))
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB")
        im.thumbnail((200, 200))
        quant = im.quantize(colors=count, method=Image.Quantize.MEDIANCUT)
    palette = quant.getpalette() or []
    total = quant.width * quant.height
    out = []
    for n, idx in sorted(quant.getcolors(), reverse=True):
        r, g, b = palette[idx * 3: idx * 3 + 3]
        out.append({"hex": f"#{r:02X}{g:02X}{b:02X}", "share": round(n / total, 3)})
    return out


TOOL_DEFS = [
    {
        "name": "image_info",
        "description": "Pixel dimensions and format of the photo.",
        "strict": True,
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "dominant_colors",
        "description": "Most common colors in the photo as hex codes with their share of pixels. "
                       "Includes background, so use it to check your own reading of the outfit.",
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"count": {"type": "integer", "description": "How many colors, 1-8"}},
            "required": ["count"],
            "additionalProperties": False,
        },
    },
]


def run_tool(data: bytes, name: str, args: dict):
    if name == "image_info":
        return tool_image_info(data)
    if name == "dominant_colors":
        return tool_dominant_colors(data, int(args.get("count", 5)))
    raise ValueError(f"unknown tool: {name}")


# ------------------------------------------------------------------------- agent

SYSTEM = (
    "You are Fit Check, a candid but kind personal stylist. Look at the photo and judge the "
    "outfit: fit, proportions, color, layering, accessories and occasion. Use the "
    "dominant_colors tool to ground the palette in real pixels, then keep only the colors that "
    "belong to the outfit (ignore walls, floors and skin). Return 3 to 5 palette colors as "
    "#RRGGBB hex codes. Be specific to this photo and do not invent items you cannot see. If "
    "the photo has no visible outfit, say so in overall_vibe and score it 1."
)


def parse_report(response) -> StyleFeedback:
    text = next(b.text for b in response.content if b.type == "text")
    return StyleFeedback.model_validate_json(text)


def build_report(data: bytes, media_type: str, client: anthropic.Anthropic | None = None,
                 model: str = MODEL) -> StyleFeedback:
    """Tool-use loop: Claude inspects the photo and the pixels, then returns a StyleFeedback."""
    client = client or anthropic.Anthropic()
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media_type,
                                         "data": base64.standard_b64encode(data).decode("ascii")}},
            {"type": "text", "text": "Give me style feedback on this outfit and a color palette."},
        ],
    }]
    for _ in range(MAX_TURNS):
        response = client.messages.create(
            model=model,
            max_tokens=16000,
            system=SYSTEM,
            thinking={"type": "adaptive"},
            tools=TOOL_DEFS,
            output_config={"format": {"type": "json_schema", "schema": transform_schema(StyleFeedback)}},
            messages=messages,
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("The model declined this request.")
        if response.stop_reason == "max_tokens":
            raise RuntimeError("Response was cut off (max_tokens); try a smaller image.")
        if response.stop_reason != "tool_use":
            return parse_report(response)

        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            try:
                out = {"type": "tool_result", "tool_use_id": block.id,
                       "content": json.dumps(run_tool(data, block.name, block.input))}
            except Exception as exc:  # report tool failures back to the model
                out = {"type": "tool_result", "tool_use_id": block.id,
                       "content": f"error: {exc}", "is_error": True}
            results.append(out)
        messages.append({"role": "user", "content": results})
    raise RuntimeError(f"No final answer after {MAX_TURNS} turns.")


def format_report(report: StyleFeedback) -> str:
    lines = [f"Fit Check: {report.overall_vibe}  ({report.score}/10)", f"Best for: {report.occasion}", "",
             "Strengths:"]
    lines += [f"  + {s}" for s in report.strengths]
    lines += ["", "Improvements:"]
    lines += [f"  - {s}" for s in report.improvements]
    lines += ["", "Palette:"]
    lines += [f"  {c.hex}  {c.name} ({c.role})" for c in report.palette]
    lines += ["", report.palette_notes]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Get style feedback and a color palette for an outfit photo.")
    parser.add_argument("image", help="path to a jpg, png, gif or webp photo (max 5 MB)")
    parser.add_argument("--json", action="store_true", help="print the raw JSON report")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args(argv)

    try:
        data, media_type = load_image(args.image)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        report = build_report(data, media_type, model=args.model)
    except anthropic.AuthenticationError:
        print("error: invalid or missing ANTHROPIC_API_KEY (see .env.example)", file=sys.stderr)
        return 1
    except (anthropic.APIError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(report.model_dump_json(indent=2) if args.json else format_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
