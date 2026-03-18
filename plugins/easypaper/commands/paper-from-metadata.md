Generate a paper directly from metadata using the EasyPaper metadata agent API.

Use the `paper-from-metadata` skill and execute this flow:

1. Normalize user metadata into a `PaperGenerationRequest` JSON object.
2. Validate references and venue (`style_guide`) if provided.
3. Send request to `POST /metadata/generate` at the configured EasyPaper API URL.
4. Report generation status and key outputs.

Fallback behavior:

- If required fields are missing, ask targeted follow-up questions.
- If API returns validation errors, surface them and request corrections.

$ARGUMENTS
