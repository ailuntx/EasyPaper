"""
Router for Typesetter Agent endpoints
"""
from fastapi import APIRouter, HTTPException, status
from typing import Optional, Any, Dict
import time
import logging
from .models import TypesetterPayload, TypesetterResult, CompilationResult, TemplateConfig


def create_typesetter_router(agent_instance):
    """
    Create router for typesetter agent endpoints
    - **Args**:
        - `agent_instance`: The TypesetterAgent instance

    - **Returns**:
        - `APIRouter`: FastAPI router with typesetter endpoints
    """
    router = APIRouter()
    logger = logging.getLogger("uvicorn.error")

    @router.post("/agent/typesetter/compile", response_model=TypesetterResult, status_code=status.HTTP_200_OK)
    async def compile_paper(payload: TypesetterPayload):
        """
        Compile LaTeX content into PDF
        - **Description**:
            - Fetches resources, generates BibTeX, injects template, compiles
            - Returns PDF path and source directory

        - **Args**:
            - `payload` (TypesetterPayload): Request payload with LaTeX content

        - **Returns**:
            - `TypesetterResult`: Compilation result or error
        """
        start = time.time()
        logger.info("typesetter.compile.request %s user=%s", payload.request_id, payload.user_id)

        try:
            # Extract parameters from payload
            latex_content = payload.payload.get("latex_content", "")
            sections = payload.payload.get("sections")  # Multi-file mode
            section_order = payload.payload.get("section_order")
            section_titles = payload.payload.get("section_titles")
            template_path = payload.payload.get("template_path")
            template_config_data = payload.payload.get("template_config")
            figure_ids = payload.payload.get("figure_ids", [])
            citation_ids = payload.payload.get("citation_ids", [])
            references = payload.payload.get("references", [])
            work_id = payload.payload.get("work_id")
            output_dir = payload.payload.get("output_dir")
            figures_source_dir = payload.payload.get("figures_source_dir")
            figure_paths = payload.payload.get("figure_paths", {})
            converted_tables = payload.payload.get("converted_tables", {})

            if not latex_content and not sections:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Either latex_content or sections must be provided"
                )

            # Parse template_config if provided
            template_config = None
            if template_config_data:
                if isinstance(template_config_data, TemplateConfig):
                    template_config = template_config_data
                elif isinstance(template_config_data, dict):
                    template_config = TemplateConfig(**template_config_data)

            # Run the agent
            agent_result = await agent_instance.run(
                latex_content=latex_content,
                sections=sections,
                section_order=section_order,
                section_titles=section_titles,
                template_path=template_path,
                template_config=template_config,
                figure_ids=figure_ids,
                citation_ids=citation_ids,
                references=references,
                work_id=work_id,
                output_dir=output_dir,
                figures_source_dir=figures_source_dir,
                figure_paths=figure_paths,
                converted_tables=converted_tables,
            )

            # Extract compilation result
            compilation_data = agent_result.get("compilation_result")
            
            if compilation_data:
                if isinstance(compilation_data, CompilationResult):
                    result = compilation_data
                else:
                    result = CompilationResult(**compilation_data)
            else:
                result = CompilationResult(
                    success=False,
                    errors=["No compilation result returned"]
                )

            latency = time.time() - start
            logger.info("typesetter.compile.complete %s latency=%.3f success=%s", 
                       payload.request_id, latency, result.success)

            return TypesetterResult(
                request_id=payload.request_id,
                status="ok" if result.success else "error",
                result=result,
                error=None if result.success else "; ".join(result.errors),
            )

        except Exception as e:
            latency = time.time() - start
            logger.error("typesetter.compile.error %s latency=%.3f error=%s", 
                        payload.request_id, latency, str(e))
            return TypesetterResult(
                request_id=payload.request_id,
                status="error",
                error=str(e)
            )

    return router
