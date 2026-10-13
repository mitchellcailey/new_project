# Quickstart

1. Python 3.10+ is required. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Set your key (never commit it):
   ```bash
   cp .env.example .env
   export ANTHROPIC_API_KEY=sk-ant-...
   ```
3. Run it on a photo of an outfit:
   ```bash
   python fit-check.py outfit.jpg
   ```
4. Supported formats: jpg, png, gif, webp, up to 5 MB. Add `--json` for machine-readable output.
5. Run the tests: `pytest`
