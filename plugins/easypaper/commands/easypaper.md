Run the EasyPaper end-to-end metadata workflow.

## Execution contract

1. Confirm API base URL (default `http://localhost:8000`).
2. Collect or validate metadata fields:
   - `title`
   - `idea_hypothesis`
   - `method`
   - `data`
   - `experiments`
   - `references`
3. Collect optional fields when provided:
   - `style_guide`
   - `target_pages`
   - `template_path`
   - `compile_pdf`
   - `enable_review`
   - `max_review_iterations`
4. Build a JSON payload that matches `PaperGenerationRequest`.
5. Call `POST /metadata/generate`.
6. Summarize output status, generated sections, and output artifacts.

## If service is not running

Tell the user to start the server:

```bash
uv run uvicorn easypaper.main:app --reload --port 8000
```

Then retry the request.

$ARGUMENTS
