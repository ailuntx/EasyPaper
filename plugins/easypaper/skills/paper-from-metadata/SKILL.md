---
description: Generate a full academic paper from metadata using EasyPaper Python SDK. Collects metadata interactively if not provided, then generates the paper directly.
---

Use this skill when the user wants to generate an academic paper from metadata. This skill handles both metadata collection and paper generation in one unified workflow.

**Recommended:** Have the user prepare a `metadata.json` (see `examples/meta.json` for the full schema), then load it with `PaperMetaData.model_validate_json_file("metadata.json")` and pass any generation options (`output_dir`, `save_output`, `enable_vlm_review`, `max_review_iterations`) to `ep.generate(metadata, **options)`. A file like `examples/meta.json` is fully supported.

## Workflow

### Phase 1: Check for Existing Metadata

1. **Ask user if they have complete metadata**:
   - "Do you already have a complete metadata file (JSON) or would you like me to collect it interactively?"
   - If user provides metadata (file path or JSON object), validate it against `examples/meta.json` structure
   - **IMPORTANT**: Convert all relative paths in metadata to absolute paths before validation
   - If metadata is provided but incomplete, proceed to collection phase for missing fields
   - If no metadata provided, proceed to collection phase

### Phase 2: Collect Metadata (if needed)

If metadata is missing or incomplete, collect all required fields interactively:

#### Required Fields (collect one by one with examples):

1. **Title**
   - Prompt: "Please provide the title of your paper."
   - Example: "Artificial intelligence tools expand scientists' impact but contract science's focus"
   - Validation: Non-empty string, 10-200 characters

2. **Idea/Hypothesis**
   - Prompt: "What is the core research question or hypothesis of your paper? Describe the main idea you want to explore."
   - Example: "The study hypothesizes a dual effect of AI adoption in science: while AI tools increase individual scientists' productivity, citations, and career advancement, they simultaneously narrow the collective scope of scientific exploration..."
   - Validation: Non-empty, at least 50 characters

3. **Method**
   - Prompt: "Describe the methodology used in your research. Include details about experimental design, data collection, analysis methods, and any tools or frameworks used."
   - Example: "The study analyzes 41,298,433 papers across biology, medicine, chemistry, physics, materials science, and geology (1980-2025), primarily from OpenAlex..."
   - Validation: Non-empty, should describe research approach

4. **Data**
   - Prompt: "What data sources, datasets, or materials did you use? Describe the data collection process and any preprocessing steps."
   - Example: "Primary bibliometric data come from OpenAlex, covering 41,298,433 papers in six natural science disciplines from 1980 to 2025..."
   - Validation: Non-empty, should describe data sources

5. **Experiments/Results**
   - Prompt: "Describe your experimental results, findings, or main outcomes. Include key metrics, comparisons, and interpretations."
   - Example: "Main findings show a clear individual-versus-collective divergence. Individual-level outcomes: Productivity: AI-using scientists publish 3.02x more papers..."
   - Validation: Non-empty, should describe results

6. **References**
   - Prompt: "Provide your references in BibTeX format or as a list of structured citations. You can provide them one by one or paste multiple at once."
   - Example: 
     ```json
     [
       "Wang, H. et al. Scientific discovery in the age of artificial intelligence. Nature 620, 47-60 (2023).",
       "LeCun, Y., Bengio, Y. & Hinton, G. Deep learning. Nature 521, 436-444 (2015)."
     ]
     ```
   - Validation: Non-empty array, at least 3 references recommended

#### Optional Fields (with smart defaults):

7. **Style Guide (Venue)**: "Which venue or style guide? Options: NeurIPS, ICML, ICLR, ACL, AAAI, COLM, Nature, or custom." (Default: "Nature")

8. **Target Pages**: "What is your target page count?" (Default: 20)

9. **Template Path**: 
   - Prompt: "Do you have a custom LaTeX template? If yes, provide the **absolute path** to the template file or directory."
   - **IMPORTANT**: Must be an absolute path. If user provides relative path, convert it to absolute path using `os.path.abspath()` or `pathlib.Path.resolve()`.
   - Example: `/Users/username/papers/templates/nature.zip` or `/home/user/templates/custom.tex`
   - Validation: Path must exist and be absolute
   - Default: null

10. **Compile PDF**: "Should the paper be compiled to PDF? (yes/no)" (Default: true)

11. **Enable Review**: "Enable VLM-based review and iterative improvement? (yes/no)" (Default: true)

12. **Max Review Iterations**: "Maximum number of review iterations (if review is enabled):" (Default: 3)

#### Advanced Options (optional):

13. **Figures**: 
   - Prompt: "Do you have figures to include? For each figure, provide: ID, **absolute file path**, caption, and description."
   - Format: Array of objects with `id`, `file_path` (must be absolute), `caption`, `description`
   - **IMPORTANT**: All `file_path` values must be absolute paths. Convert relative paths to absolute using `os.path.abspath()` or `pathlib.Path.resolve()`.
   - Example:
     ```json
     [
       {
         "id": "fig:architecture",
         "file_path": "/Users/username/papers/figures/architecture.png",
         "caption": "System architecture diagram",
         "description": "Shows the overall system design"
       }
     ]
     ```
   - Validation: Each `file_path` must be absolute and file must exist
   - Default: empty array

14. **Tables**: Array of table objects (Default: empty array)

15. **Code Repository**: 
   - Prompt: "Do you want to include code from a repository? Provide type (local_dir/git) and **absolute path** or URL."
   - For `local_dir` type: **absolute path** is required
   - For `git_repo` type: URL is required
   - **IMPORTANT**: If `local_dir` and user provides relative path, convert to absolute path using `os.path.abspath()` or `pathlib.Path.resolve()`.
   - Example for local_dir:
     ```json
     {
       "type": "local_dir",
       "path": "/Users/username/projects/my_code",
       "on_error": "fallback"
     }
     ```
   - Example for git_repo:
     ```json
     {
       "type": "git_repo",
       "url": "https://github.com/user/repo.git",
       "ref": "main"
     }
     ```
   - Validation: For local_dir, path must be absolute and directory must exist
   - Default: null

16. **Output Directory**: 
   - Prompt: "Where should the generated paper be saved? Provide an **absolute path** to the output directory."
   - **IMPORTANT**: Must be an absolute path. If user provides relative path, convert it to absolute path using `os.path.abspath()` or `pathlib.Path.resolve()`.
   - Example: `/Users/username/papers/output/my_paper` or `/home/user/output/output_20250120`
   - Validation: Path must be absolute (can create directory if doesn't exist)
   - Default: `{current_working_directory}/output_{timestamp}` (converted to absolute)

### Phase 3: Review and Confirm

Before generating:
1. Display summary of all collected metadata
2. **Verify all paths are absolute**: Check that `template_path`, `figures[].file_path`, `code_repository.path` (if local_dir), and `output_dir` are all absolute paths. Convert any relative paths found.
3. Ask "Would you like to modify any field? (yes/no)"
4. Optionally save metadata to `metadata.json` file (with all absolute paths)
5. Get final confirmation: "Ready to generate paper? (yes/no)"

### Phase 4: Generate Paper

1. **Check environment**:
   - Ensure easypaper package is installed (use `setup-environment` skill if needed)
   - Check if config file exists or ask user for config path
   - **IMPORTANT**: Config path should be absolute. If relative, convert to absolute.

2. **Import and initialize EasyPaper**:
   ```python
   from easypaper import EasyPaper, PaperMetaData
   from pathlib import Path
   
   # Convert config path to absolute if needed
   config_path = Path("configs/openrouter.yaml").resolve()  # or user-provided path
   
   # Initialize with config
   ep = EasyPaper(config_path=str(config_path))
   ```

3. **Obtain PaperMetaData (prefer loading from file)**:
   - **When user has a metadata file (recommended):** Load with `PaperMetaData.model_validate_json_file(path)`. Convert any relative paths in the file to absolute before or after load (e.g. resolve `template_path`, `figures[].file_path`, `code_repository.path`, `output_dir`).
   - **When metadata is from interactive collection or a dict:** Build `PaperMetaData` from the collected dict; ensure all path fields are absolute.
   - **examples/meta.json is fully supported:** All content fields (title, idea_hypothesis, method, data, experiments, references, figures, tables, template_path, style_guide, target_pages, code_repository, export_prompt_traces) are part of `PaperMetaData`. Fields `output_dir`, `save_output`, `enable_vlm_review`, `max_review_iterations` are generation options; pass them to `ep.generate(metadata, **options)` (see step 4).
   - Validate required fields are present.

4. **Generate paper**:
   ```python
   import json
   from pathlib import Path
   
   # If metadata was loaded from a full JSON (e.g. examples/meta.json), pass generation options
   with open("metadata.json", encoding="utf-8") as f:
       data = json.load(f)
   options = {}
   for key in ("output_dir", "save_output", "enable_vlm_review", "max_review_iterations"):
       if key in data:
           options[key] = data[key]
   # Resolve output_dir to absolute if present
   if "output_dir" in options:
       options["output_dir"] = str(Path(options["output_dir"]).resolve())
   
   result = await ep.generate(metadata=paper_metadata, **options)
   ```
   Or with explicit options:
   ```python
   result = await ep.generate(
       metadata=paper_metadata,
       output_dir=str(Path(metadata.get("output_dir", "output_default")).resolve()),
       compile_pdf=True,
       enable_review=True,
       max_review_iterations=3,
   )
   ```

   OR use streaming for progress updates:
   ```python
   async for event in ep.generate_stream(metadata, **options):
       print(f"{event.phase}: {event.message}")
   ```

5. **Report results**:
   - Show generation status
   - List output files: `paper.tex`, `references.bib`, `paper.pdf` (if compiled)
   - Provide absolute file paths
   - Show summary: word count, sections generated, etc.

## Path Handling Rules

**CRITICAL**: All paths in metadata must be absolute paths. Follow these rules:

1. **When collecting paths from user**:
   - Always ask for absolute paths explicitly
   - If user provides relative path, convert immediately using:
     ```python
     from pathlib import Path
     absolute_path = Path(relative_path).resolve()
     ```

2. **When reading metadata from file**:
   - After loading JSON, scan for all path fields and convert relative paths to absolute
   - Path fields to check: `template_path`, `figures[].file_path`, `code_repository.path` (if local_dir), `output_dir`

3. **When saving metadata**:
   - Save with all absolute paths (never save relative paths)

4. **Path fields that must be absolute**:
   - `template_path`: LaTeX template file/directory path
   - `figures[].file_path`: Figure image file paths
   - `code_repository.path`: Local code repository directory path (if type is local_dir)
   - `output_dir`: Output directory path
   - `config_path`: EasyPaper configuration file path

## Metadata Structure

The metadata should match the structure in `examples/meta.json`, but with **absolute paths**:

```json
{
  "title": "...",
  "idea_hypothesis": "...",
  "method": "...",
  "data": "...",
  "experiments": "...",
  "references": [...],
  "style_guide": "Nature",
  "target_pages": 20,
  "template_path": "/absolute/path/to/template.zip",
  "compile_pdf": true,
  "enable_vlm_review": true,
  "max_review_iterations": 3,
  "figures": [
    {
      "id": "fig:example",
      "file_path": "/absolute/path/to/figure.png",
      "caption": "...",
      "description": "..."
    }
  ],
  "tables": [],
  "code_repository": {
    "type": "local_dir",
    "path": "/absolute/path/to/code",
    "on_error": "fallback"
  },
  "output_dir": "/absolute/path/to/output"
}
```

## Best Practices

- **Progressive disclosure**: Start with required fields, then optional, then advanced
- **Examples**: Always provide examples when asking for input (use absolute paths in examples)
- **Validation**: Validate each field immediately and ask for correction if invalid
- **Path conversion**: Always convert relative paths to absolute paths immediately upon collection
- **Reference**: Always reference `examples/meta.json` when users ask about structure (but note paths should be absolute). That file is fully supported: content fields load into `PaperMetaData`; `output_dir`, `save_output`, `enable_vlm_review`, `max_review_iterations` are passed to `ep.generate(metadata, **options)`.
- **Flexibility**: Allow users to provide metadata in different formats (paste full JSON, or answer questions)
- **Direct import**: Use `from easypaper import EasyPaper, PaperMetaData` directly - no API calls needed
- **Config handling**: Ask user for config path if not found, or use default `configs/dev.yaml` (convert to absolute)

## Error Handling

- If easypaper package not installed, use `setup-environment` skill
- If config file missing, ask user for path or create default (ensure absolute path)
- If metadata validation fails, show specific errors and ask for corrections
- If path validation fails (file/directory doesn't exist), show clear error and ask for correct absolute path
- If generation fails, show error message and suggest fixes
- If relative path detected, automatically convert to absolute and inform user

