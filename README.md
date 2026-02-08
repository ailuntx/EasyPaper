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
- LaTeX toolchain (pdflatex + bibtex) for PDF compilation
- Model API keys configured in YAML (see Config)

## Quickstart

1. Install dependencies:

```
pip install -e .
```

2. (Optional) set the config path:

```
export AGENT_CONFIG_PATH=./configs/dev.yaml
```

3. Start the server:

```
uvicorn src.main:app --reload --port 8000
```

4. Verify health:

```
curl http://localhost:8000/healthz
```

## Generate a Paper

### API (JSON)

```
curl -X POST http://localhost:8000/metadata/generate \
  -H "Content-Type: application/json" \
  -d @examples/transkg_metadata.json
```

### CLI

```
python scripts/generate_paper.py --input examples/transkg_metadata.json
```

### Assembly Demo

```
python scripts/assemble_paper_demo.py --config example_jsons/paper_chain_config.json
```

## Config

The application loads configuration from `AGENT_CONFIG_PATH` and defaults to `./configs/dev.yaml`.
Each agent entry defines its model and optional agent-specific settings.

Key fields:
- `model_name`
- `api_key`
- `base_url`

## Service Endpoints

- `GET /healthz` — health check
- `GET /config` — current app config
- `GET /list_agents` — list registered agents and endpoints
- Agent-specific routes are registered under `/agent/*` and `/metadata/*`

## Repository Layout

- `src/` — FastAPI app, agent implementations, shared utilities
- `configs/` — YAML configs for agents and models
- `scripts/` — CLI utilities and demos
- `examples/`, `example_jsons/` — sample inputs
- `results/` — generated outputs
- `tests/` — test suite