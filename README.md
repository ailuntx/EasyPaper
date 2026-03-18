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

## Skills

EasyPaper includes a pluggable **Skills** system that injects writing constraints, venue-specific
formatting rules, and reviewer checkers into the generation pipeline. The repository ships
pre-built skills in [`skills/`](skills/):

| Category | Skills | Description |
|---|---|---|
| **Writing** | `anti-ai-style`, `academic-polish`, `latex-conventions` | Style constraints applied to all sections — eliminates AI-flavored phrasing, enforces academic tone, ensures LaTeX best practices |
| **Venues** | `neurips`, `icml`, `iclr`, `acl`, `aaai`, `colm`, `nature` | Conference/journal profiles with page limits, formatting rules, and venue-specific style requirements |
| **Reviewing** | `logic-check`, `style-check` | Reviewer checker prompts — detects logical contradictions, terminology inconsistencies, and style violations |

### Enabling skills

Add a `skills` section to your config YAML and point `skills_dir` to a directory containing
`.yaml` skill files:

```yaml
skills:
  enabled: true
  skills_dir: "./skills"    # path to skill YAML files
  active_skills:
    - "*"                   # "*" = load all skills; or list specific names
```

To use the built-in skills, copy the [`skills/`](skills/) directory into your project:

```bash
cp -r /path/to/easypaper/skills ./skills
```

If `skills_dir` does not exist or `skills.enabled` is `false`, skills are silently skipped —
no configuration is required for basic usage.

### Venue profiles

To apply venue-specific constraints (e.g. page limits, formatting), set `style_guide` in your
`PaperMetaData` to match a venue profile name:

```python
metadata = PaperMetaData(
    title="...",
    idea_hypothesis="...",
    method="...",
    data="...",
    experiments="...",
    style_guide="neurips",   # activates the neurips venue profile
)
```

### Custom skills

Each skill is a single YAML file with the following structure:

```yaml
name: my-custom-skill
description: "What this skill does"
type: writing_constraint   # writing_constraint | reviewer_checker | venue_profile
target_sections: ["*"]     # ["*"] = all sections, or specific ones
priority: 10               # lower = higher priority

system_prompt_append: |
  ## My Custom Rules
  - Rule 1: ...
  - Rule 2: ...

anti_patterns:
  - "word to avoid"
```

Drop the file into your `skills_dir` and it will be automatically loaded on the next run.
See the built-in skills in [`skills/`](skills/) for complete examples.

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
- `skills` — skills system toggle and active skill list (see [Skills](#skills))
- `tools` — ReAct tool configuration (citation validation, paper search, etc.)
- `vlm_service` — shared VLM provider for visual review (supports OpenAI-compatible and Claude)

## Repository Layout

- `easypaper/` — SDK core, agent implementations, event system, shared utilities
- `configs/` — YAML configs for agents and models
- `skills/` — backend YAML skills loaded by the Python service
- `plugins/easypaper/` — Claude/OpenCode plugin root (commands + SKILL.md prompts)
- `.claude-plugin/marketplace.json` — marketplace catalog
- `.opencode/opencode.json` — OpenCode/OpenClaw runtime configuration
- `AGENTS.md` — repository-level instructions for coding agents
- `scripts/` — CLI utilities and demos
- `user_case/` — standalone usage example (independent environment)
- `economist_example/` — sample metadata input

## Claude Code Plugin Market

This repository ships a Claude Code marketplace with one installable plugin located at `plugins/easypaper`.

### Installation

Add the marketplace:

```bash
/plugin marketplace add https://github.com/your-username/easypaper
```

Install the plugin from this marketplace:

```bash
/plugin install easypaper@easypaper
```

### Available Plugin

| Plugin | Source | Description |
|--------|--------|-------------|
| easypaper | `./plugins/easypaper` | Generate AI-powered academic papers from metadata interactively |

### Usage

After installation:

```bash
/easypaper
```

Related commands:

```bash
/paper-from-metadata
/paper-section
```

### Plugin Prerequisites

- Python 3.11+
- `easypaper` package installed (`pip install easypaper`)
- LaTeX toolchain (pdflatex + bibtex) for PDF compilation
- API key for LLM provider (configured via config file)

## OpenCode / OpenClaw Usage

This repository includes native OpenCode/OpenClaw configuration in `.opencode/opencode.json`.

### Run directly in this repository

```bash
opencode
```

The runtime loads:

- Plugin path: `./plugins/easypaper`
- Skills: `plugins/easypaper/skills/*/SKILL.md`
- Commands: `easypaper`, `paper-from-metadata`, `paper-section`

For both Claude Code and OpenCode/OpenClaw workflows, start the EasyPaper API service before generation:

```bash
uv run uvicorn easypaper.main:app --reload --port 8000
```
