---
description: Generate a full academic paper from metadata through the EasyPaper API workflow.
---

Use this skill when the user wants end-to-end paper drafting from structured metadata.

## Inputs to collect

- `title`
- `idea_hypothesis`
- `method`
- `data`
- `experiments`
- `references` (BibTeX entries or structured references)
- Optional: `style_guide`, `target_pages`, `template_path`, `compile_pdf`, `enable_review`, `max_review_iterations`

## Workflow

1. Validate that the EasyPaper API service is reachable.
2. Collect missing metadata fields from the user.
3. Build a `PaperGenerationRequest` payload.
4. Submit `POST /metadata/generate`.
5. Report generation progress and summarize outputs (`paper.tex`, `references.bib`, optional `paper.pdf`).

## API endpoint

- `POST /metadata/generate`
- Default base URL: `http://localhost:8000`

## Notes

- This workflow maps to the metadata-agent pipeline: planning, section drafting, synthesis, review loop, and optional PDF compilation.
- If the service is unavailable, instruct the user to start it with:
  - `uv run uvicorn easypaper.main:app --reload --port 8000`
