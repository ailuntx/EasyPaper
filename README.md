# EasyPaper

EasyPaper is a multi-agent academic paper generation service. It turns a small set of metadata
(title, idea, method, data, experiments, references) into a structured LaTeX paper and optionally
compiles it into a PDF through a typesetting agent.

## Features

- Multi-agent pipeline: planning, writing, review, typesetting, and optional VLM review
- FastAPI service with health and agent discovery endpoints
- CLI scripts for metadata-driven generation and paper assembly demos
- LaTeX output with citation validation, figure/table injection, and review loop

## Requirements

- Python 3.11+
- LaTeX toolchain (`pdflatex` + `bibtex`) for PDF compilation
- [Poppler](https://poppler.freedesktop.org/) — required by `pdf2image` for PDF-to-image conversion
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `apt install poppler-utils`
- Model API keys configured in YAML (see [Config](#config))

## Quickstart

1. Install core dependencies:

```bash
pip install -e .
```

To install development tools (pytest, ipython, etc.):

```bash
pip install -e ".[dev]"
```

To enable Claude VLM review:

```bash
pip install -e ".[vlm]"
```

2. Copy the example config and fill in your API keys:

```bash
cp configs/example.yaml configs/dev.yaml
# Edit configs/dev.yaml — replace YOUR_API_KEY with real keys
```

3. Set the config path (or create a `.env` file):

```bash
# Option A: environment variable
export AGENT_CONFIG_PATH=./configs/dev.yaml

# Option B: .env file (auto-loaded by python-dotenv)
echo 'AGENT_CONFIG_PATH=./configs/dev.yaml' > .env
```

4. Start the server:

```bash
uvicorn src.main:app --reload --port 8000
```

5. Verify health:

```bash
curl http://localhost:8000/healthz
```

## Generate a Paper

### API (JSON)

```bash
curl -X POST http://localhost:8000/metadata/generate \
  -H "Content-Type: application/json" \
  -d @economist_example/metadata.json
```

### CLI

```bash
python scripts/generate_paper.py --input economist_example/metadata.json
```

## Config

The application loads configuration from `AGENT_CONFIG_PATH` (defaults to `./configs/dev.yaml`).
You can also set this variable in a `.env` file at the project root.

See `configs/example.yaml` for a fully commented configuration template. Each agent entry defines
its model and optional agent-specific settings.

Key fields per agent:
- `model_name` — LLM model identifier
- `api_key` — API key for the model provider
- `base_url` — API endpoint URL

Additional top-level sections:
- `skills` — skills system toggle and active skill list
- `tools` — ReAct tool configuration (citation validation, paper search, etc.)
- `vlm_service` — shared VLM provider for visual review (supports OpenAI-compatible and Claude)

## Service Endpoints

- `GET /healthz` — health check
- `GET /config` — current app config
- `GET /list_agents` — list registered agents and endpoints
- Agent-specific routes are registered under `/agent/*` and `/metadata/*`

## Repository Layout

- `src/` — FastAPI app, agent implementations, shared utilities
- `configs/` — YAML configs for agents and models
- `scripts/` — CLI utilities and demos
- `economist_example/` — sample metadata input