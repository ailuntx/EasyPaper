# AGENTS

Repository-level instructions for coding agents working on EasyPaper.

## Project Scope

- EasyPaper is a metadata-to-paper generation system with Python SDK and FastAPI API.
- This repository also publishes a Claude Code plugin marketplace.
- The installable plugin root is `plugins/easypaper/`.

## Plugin/Marketplace Layout

- Marketplace manifest: `.claude-plugin/marketplace.json`
- Plugin manifest: `plugins/easypaper/.claude-plugin/plugin.json`
- Plugin commands: `plugins/easypaper/commands/`
- Plugin skills: `plugins/easypaper/skills/`
- OpenCode/OpenClaw config: `.opencode/opencode.json`

## EasyPaper API Workflow

For end-to-end paper generation, use the metadata agent API:

1. Ensure service is running:
   - `uv run uvicorn easypaper.main:app --reload --port 8000`
2. Submit request:
   - `POST /metadata/generate`
3. Optional section-only generation:
   - `POST /metadata/generate/section`

## Required Metadata Fields

- `title`
- `idea_hypothesis`
- `method`
- `data`
- `experiments`
- `references`

Optional fields include `style_guide`, `target_pages`, `template_path`, `compile_pdf`, and review options.

## Skills Source of Truth

- Backend YAML skills remain under `skills/` and are loaded by Python service config.
- Claude/OpenCode skill prompts live under `plugins/easypaper/skills/*/SKILL.md`.

## Validation Checklist

- Keep marketplace `source` pointing to `./plugins/easypaper`.
- Keep plugin version in `plugins/easypaper/.claude-plugin/plugin.json`.
- Keep README installation steps aligned with actual marketplace command syntax.
