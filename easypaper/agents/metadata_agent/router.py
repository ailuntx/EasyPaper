"""
MetaData Agent Router
- **Description**:
    - FastAPI router for MetaData-based paper generation
    - Independent API - can be called directly without frontend
    - Endpoints:
        - POST /metadata/generate - Generate complete paper
        - POST /metadata/generate/section - Generate single section
        - GET /metadata/health - Health check
"""
from fastapi import APIRouter, HTTPException
from typing import TYPE_CHECKING

from .models import (
    PaperGenerationRequest,
    PaperGenerationResult,
    SectionGenerationRequest,
    SectionResult,
)

if TYPE_CHECKING:
    from .metadata_agent import MetaDataAgent


def create_metadata_router(agent: "MetaDataAgent") -> APIRouter:
    """
    Create FastAPI router for MetaData Agent
    
    Args:
        agent: MetaDataAgent instance
    
    Returns:
        FastAPI APIRouter with endpoints
    """
    router = APIRouter(prefix="/metadata", tags=["MetaData Paper Generation"])
    
    @router.post("/generate", response_model=PaperGenerationResult)
    async def generate_paper(request: PaperGenerationRequest) -> PaperGenerationResult:
        """
        Generate complete paper from MetaData
        
        **Independent API** - Can be called directly via curl/Postman
        
        ## Request Body
        
        ```json
        {
            "title": "Paper Title",
            "idea_hypothesis": "Research idea or hypothesis...",
            "method": "Method description...",
            "data": "Data or validation method...",
            "experiments": "Experiment design, results, findings...",
            "references": ["@article{key, title={...}, ...}", ...],
            "template_path": "path/to/template.zip",
            "style_guide": "ICML 2026",
            "compile_pdf": true,
            "figures_source_dir": null,
            "save_output": true,
            "output_dir": null
        }
        ```
        
        ## Response
        
        ```json
        {
            "status": "ok",
            "paper_title": "...",
            "sections": [...],
            "latex_content": "...",
            "output_path": "results/xxx/",
            "pdf_path": null,
            "total_word_count": 5000,
            "errors": []
        }
        ```
        
        ## Six-Phase Generation
        
        0. **Phase 0 (Planning)**: Create detailed paper plan (if enabled)
        1. **Phase 1 (Introduction)**: Leader section that sets tone and extracts contributions
        2. **Phase 2 (Body)**: Method, Experiment, Results, Related Work (can be parallel)
        3. **Phase 3 (Synthesis)**: Abstract and Conclusion based on prior sections
        3.5. **Review Loop**: Iterative feedback and revision (if enabled)
        4. **Phase 4 (PDF)**: Compile to PDF using template (if template_path provided)
        5. **Phase 5 (VLM Review)**: Check page overflow with VLM (if enable_vlm_review=true)
        """
        try:
            metadata = request.to_metadata()
            result = await agent.generate_paper(
                metadata=metadata,
                output_dir=request.output_dir,
                save_output=request.save_output,
                compile_pdf=request.compile_pdf,
                template_path=request.template_path,
                figures_source_dir=request.figures_source_dir,
                target_pages=request.target_pages,
                enable_review=request.enable_review,
                max_review_iterations=request.max_review_iterations,
                enable_planning=request.enable_planning,
                enable_vlm_review=request.enable_vlm_review,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/generate/section", response_model=SectionResult)
    async def generate_single_section(request: SectionGenerationRequest) -> SectionResult:
        """
        Generate a single section (for debugging or incremental generation)
        
        ## Request Body
        
        ```json
        {
            "section_type": "introduction",
            "metadata": {
                "title": "...",
                "idea_hypothesis": "...",
                "method": "...",
                "data": "...",
                "experiments": "...",
                "references": [...]
            },
            "intro_context": null,
            "prior_sections": null
        }
        ```
        
        ## Section Types
        
        - **Leader**: introduction
        - **Body**: method, experiment, result, related_work, discussion
        - **Synthesis**: abstract, conclusion (requires prior_sections)
        """
        try:
            result = await agent.generate_single_section(request)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/health")
    async def health_check():
        """
        Health check endpoint
        
        Returns agent status and configuration info.
        """
        return {
            "status": "ok",
            "agent": "metadata_agent",
            "model": agent.model_name,
            "description": "MetaData-based paper generation (Simple Mode)",
            "endpoints": [
                "POST /metadata/generate - Generate complete paper",
                "POST /metadata/generate/section - Generate single section",
                "GET /metadata/health - Health check",
            ],
        }
    
    @router.get("/schema")
    async def get_input_schema():
        """
        Get the input schema for paper generation
        
        Returns the expected format for PaperGenerationRequest.
        """
        return {
            "input_schema": {
                "title": {
                    "type": "string",
                    "description": "Paper title",
                    "required": False,
                    "default": "Untitled Paper",
                },
                "idea_hypothesis": {
                    "type": "string",
                    "description": "Research idea or hypothesis (natural language)",
                    "required": True,
                },
                "method": {
                    "type": "string",
                    "description": "Method/approach description (natural language)",
                    "required": True,
                },
                "data": {
                    "type": "string",
                    "description": "Data or validation method description",
                    "required": True,
                },
                "experiments": {
                    "type": "string",
                    "description": "Experiment design, execution, results, findings",
                    "required": True,
                },
                "references": {
                    "type": "array",
                    "items": "string (BibTeX entry)",
                    "description": "List of BibTeX reference entries",
                    "required": False,
                    "default": [],
                },
                "template_path": {
                    "type": "string",
                    "description": "Path to .zip template file for PDF compilation",
                    "required": False,
                },
                "style_guide": {
                    "type": "string",
                    "description": "Writing style guide (e.g., 'ICML', 'NeurIPS')",
                    "required": False,
                },
                "compile_pdf": {
                    "type": "boolean",
                    "description": "Whether to compile PDF (requires template_path)",
                    "required": False,
                    "default": True,
                },
                "figures_source_dir": {
                    "type": "string",
                    "description": "Directory containing figure files",
                    "required": False,
                },
                "save_output": {
                    "type": "boolean",
                    "description": "Whether to save output files to disk",
                    "required": False,
                    "default": True,
                },
                "output_dir": {
                    "type": "string",
                    "description": "Custom output directory path",
                    "required": False,
                },
                "export_prompt_traces": {
                    "type": "boolean",
                    "description": "Whether to export section-level prompt and evidence traces",
                    "required": False,
                    "default": False,
                },
                "code_repository": {
                    "type": "object",
                    "description": "Optional code/docs repository source for section-aware writing support",
                    "required": False,
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["local_dir", "git_repo"],
                            "description": "Source type: local folder or remote git repository",
                        },
                        "path": {
                            "type": "string",
                            "description": "Required when type=local_dir; local folder path",
                            "required": False,
                        },
                        "url": {
                            "type": "string",
                            "description": "Required when type=git_repo; repository URL",
                            "required": False,
                        },
                        "ref": {
                            "type": "string",
                            "description": "Optional git branch/tag/commit (default: main)",
                            "required": False,
                            "default": "main",
                        },
                        "subdir": {
                            "type": "string",
                            "description": "Optional sub-directory inside repository to scan",
                            "required": False,
                        },
                        "include_globs": {
                            "type": "array",
                            "items": "string",
                            "description": "Optional include glob patterns",
                            "required": False,
                            "default": [],
                        },
                        "exclude_globs": {
                            "type": "array",
                            "items": "string",
                            "description": "Optional exclude glob patterns",
                            "required": False,
                            "default": [],
                        },
                        "max_files": {
                            "type": "integer",
                            "description": "Maximum number of files to ingest",
                            "required": False,
                            "default": 5000,
                        },
                        "max_total_bytes": {
                            "type": "integer",
                            "description": "Maximum bytes to ingest across all files",
                            "required": False,
                            "default": 200000000,
                        },
                        "on_error": {
                            "type": "string",
                            "enum": ["fallback", "strict"],
                            "description": "Failure policy: fallback to normal flow or abort",
                            "required": False,
                            "default": "fallback",
                        },
                    },
                },
            },
            "example": {
                "title": "TransKG: Knowledge Graph Completion with Transformers",
                "idea_hypothesis": "We hypothesize that pre-trained Transformer models can better capture semantic relationships in knowledge graphs...",
                "method": "We propose TransKG, combining BERT with relation-aware attention...",
                "data": "We evaluate on FB15k-237, WN18RR, and YAGO3-10 datasets...",
                "experiments": "Compared against TransE, RotatE, achieving 0.391 MRR...",
                "references": [
                    "@inproceedings{bordes2013transE, title={Translating embeddings for modeling multi-relational data}, author={Bordes, Antoine}, year={2013}}"
                ],
                "template_path": "example_jsons/icml2026.zip",
                "style_guide": "ICML",
                "compile_pdf": True,
                "code_repository": {
                    "type": "local_dir",
                    "path": "examples/project_code",
                    "include_globs": ["**/*.py", "**/*.md"],
                    "exclude_globs": ["**/.git/**", "**/venv/**"],
                    "on_error": "fallback",
                },
                "export_prompt_traces": False,
            },
        }
    
    return router
