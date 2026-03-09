# EasyPaper

EasyPaper is a multi-agent academic paper generation system. It turns a small set of metadata
(title, idea, method, data, experiments, references) into a structured LaTeX paper and optionally
compiles it into a PDF through a typesetting agent.

## Features

- **Python SDK** — `pip install easypaper`, then `import easypaper` in your own project
- **Streaming generation** — async generator yields real-time progress events at each phase
- Multi-agent pipeline: planning, writing, review, typesetting, and optional VLM review
- Optional FastAPI server mode with health and agent discovery endpoints
- LaTeX output with citation validation, figure/table injection, and review loop

## Requirements

- Python 3.11+
- LaTeX toolchain (`pdflatex` + `bibtex`) for PDF compilation
- [Poppler](https://poppler.freedesktop.org/) — required by `pdf2image` for PDF-to-image conversion
  - macOS: `brew install poppler`
  - Ubuntu/Debian: `apt install poppler-utils`
- Model API keys configured in YAML (see [Config](#config))

## SDK Usage

Install from PyPI:

```bash
pip install easypaper
```

### One-shot generation

```python
import asyncio
from easypaper import EasyPaper, PaperMetaData

async def main():
    ep = EasyPaper(config_path="config.yaml")

    metadata = PaperMetaData(
        title="My Paper Title",
        idea_hypothesis="...",
        method="...",
        data="...",
        experiments="...",
    )

    result = await ep.generate(metadata)
    print(result.status, result.total_word_count)
    for sec in result.sections:
        print(f"  {sec.section_type}: {sec.word_count} words")

asyncio.run(main())
```

### Streaming generation

Use `generate_stream()` to receive real-time progress events via async generator:

```python
import asyncio
from easypaper import EasyPaper, PaperMetaData, EventType

async def main():
    ep = EasyPaper(config_path="config.yaml")
    metadata = PaperMetaData(
        title="My Paper Title",
        idea_hypothesis="...",
        method="...",
        data="...",
        experiments="...",
    )

    async for event in ep.generate_stream(metadata):
        if event.event_type == EventType.PHASE_START:
            print(f"▶ [{event.phase}] {event.message}")
        elif event.event_type == EventType.SECTION_COMPLETE:
            print(f"  ✎ {event.phase} done")
        elif event.event_type == EventType.COMPLETE:
            result = event.data["result"]
            print(f"Done! {result['total_word_count']} words")

asyncio.run(main())
```

`GenerationEvent` fields:

| Field | Type | Description |
|---|---|---|
| `event_type` | `EventType` | `PHASE_START`, `PHASE_COMPLETE`, `SECTION_COMPLETE`, `PROGRESS`, `WARNING`, `ERROR`, `COMPLETE` |
| `phase` | `str` | Logical phase name (e.g. `"planning"`, `"introduction"`, `"body"`) |
| `message` | `str` | Human-readable description |
| `data` | `dict \| None` | Structured payload (section content, final result, etc.) |
| `timestamp` | `datetime` | When the event was created |

A complete working example is available in [`user_case/`](user_case/).

## Server Mode

To run EasyPaper as a FastAPI service (requires the `server` extra):

```bash
pip install "easypaper[server]"
```

1. Copy the example config and fill in your API keys:

```bash
cp configs/example.yaml configs/dev.yaml
```

2. Start the server:

```bash
uvicorn easypaper.main:app --reload --port 8000
```

3. Generate via API:

```bash
curl -X POST http://localhost:8000/metadata/generate \
  -H "Content-Type: application/json" \
  -d @economist_example/metadata.json
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

## Repository Layout

- `easypaper/` — SDK core, agent implementations, event system, shared utilities
- `configs/` — YAML configs for agents and models
- `scripts/` — CLI utilities and demos
- `user_case/` — standalone usage example (independent environment)
- `economist_example/` — sample metadata input