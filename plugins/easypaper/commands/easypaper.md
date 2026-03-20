Run the EasyPaper end-to-end paper generation workflow with guided setup and metadata collection.

## Execution contract

### Phase 1: Environment Setup (First-time only)

1. **Check if environment is set up**:
   - Check if `.easypaper-env` directory exists
   - Check if `easypaper` package is importable: `python -c "import easypaper"`
   - Check if `pdflatex` command is available

2. **If environment is not ready**:
   - Use the `setup-environment` skill to automatically:
     - Create isolated virtual environment (prefer `uv`, fallback to `venv`)
     - Install easypaper package
     - Check and guide LaTeX installation
     - Verify all components are working

### Phase 2: Paper Generation

3. **Use the `paper-from-metadata` skill** which handles:
   - **Check for existing metadata**: Ask user if they have complete metadata file/JSON
   - **Collect metadata if needed**: If missing or incomplete, interactively collect all required fields:
     - Required: `title`, `idea_hypothesis`, `method`, `data`, `experiments`, `references`
     - Optional: `style_guide`, `target_pages`, `template_path` (absolute path), `compile_pdf`, `enable_review`, `max_review_iterations`
     - Advanced: `figures` (with absolute file_path), `tables`, `code_repository` (with absolute path if local_dir), `output_dir` (absolute path)
   - **Path handling**: Ensure all paths are absolute - convert relative paths to absolute using `pathlib.Path.resolve()`
   - **Review and confirm**: Display summary, allow edits, save to file, get confirmation
   - **Generate paper**: Use EasyPaper Python SDK directly. Prefer loading from a metadata file when the user has one (e.g. `metadata = PaperMetaData.model_validate_json_file("metadata.json")`); pass generation options from the same JSON (`output_dir`, `save_output`, `enable_vlm_review`, `max_review_iterations`) to `ep.generate(metadata, **options)`.
     ```python
     from easypaper import EasyPaper, PaperMetaData
     from pathlib import Path
     
     # Config path should be absolute
     config_path = Path("configs/openrouter.yaml").resolve()
     ep = EasyPaper(config_path=str(config_path))
     # If user has metadata.json (e.g. examples/meta.json): load and pass options
     metadata = PaperMetaData.model_validate_json_file("metadata.json")
     result = await ep.generate(metadata, **options)  # options from JSON if present
     ```
   - **Report results**: Show status, output files, absolute paths, summary

## Path Requirements

**IMPORTANT**: All paths in metadata must be absolute paths:
- `template_path`: Absolute path to LaTeX template file/directory
- `figures[].file_path`: Absolute paths to figure image files
- `code_repository.path`: Absolute path (if type is `local_dir`)
- `output_dir`: Absolute path to output directory
- `config_path`: Absolute path to EasyPaper config file

The skill will automatically convert relative paths to absolute paths, but users should be encouraged to provide absolute paths.

## User Experience Guidelines

- **First-time users**: Automatically trigger environment setup without asking
- **Clear progress**: Show what step you're on (e.g., "Step 1/2: Setting up environment...")
- **Error handling**: If any step fails, explain clearly and provide next steps
- **Flexibility**: Allow users to provide complete metadata or collect interactively
- **Path conversion**: Automatically convert relative paths to absolute and inform user
- **Reference**: When users ask about structure, reference `examples/meta.json` as the template (note: paths should be absolute)
- **Direct import**: Use EasyPaper as Python SDK - no API server needed

$ARGUMENTS
