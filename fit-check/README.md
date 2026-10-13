# Fit Check

A CLI that sends an outfit photo to Claude with vision and returns structured style feedback plus a 3-5 color palette (hex codes).

## How it works

1. **Pillow** validates the image (jpg, png, gif or webp, max 5 MB).
2. **Claude** (`claude-sonnet-5-5`, vision, adaptive thinking) looks at the photo and runs an agentic tool-use loop. It calls `image_info` and `dominant_colors` so the palette is grounded in real pixel data instead of guesses.
3. The final answer is **structured output**: a JSON schema generated from the `StyleFeedback` pydantic model and validated before printing (score 1-10, palette of 3-5 `#RRGGBB` colors).

## Usage

```bash
pip install -r requirements.txt
cp .env.example .env   # add your ANTHROPIC_API_KEY, then export it
python fit-check.py outfit.jpg
python fit-check.py outfit.jpg --json
```

See [QUICKSTART.md](QUICKSTART.md) for a 2-minute walkthrough.

## Docker

```bash
docker build -t fit-check .
docker run --rm -e ANTHROPIC_API_KEY -v "$PWD":/data fit-check /data/outfit.jpg
```

## Tests

```bash
pytest
```

Tests cover image validation, the color tool and the agent loop, and mock the API, so no key or network is needed.

## License

MIT, see [LICENSE](LICENSE).
