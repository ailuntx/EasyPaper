"""
Router for Template Parser Agent endpoints
"""
from fastapi import APIRouter, HTTPException, status
from typing import Optional, Any, Dict
import time
import logging
from .models import TemplateParsePayload, TemplateParseResult, TemplateInfo


def create_template_router(agent_instance):
    """
    Create router for template parser agent endpoints
    - **Args**:
        - `agent_instance`: The TemplateParserAgent instance

    - **Returns**:
        - `APIRouter`: FastAPI router with template endpoints
    """
    router = APIRouter()
    logger = logging.getLogger("uvicorn.error")

    @router.post("/agent/template/parse", response_model=TemplateParseResult, status_code=status.HTTP_200_OK)
    async def parse_template(payload: TemplateParsePayload):
        """
        Parse a LaTeX template zip package
        - **Description**:
            - Extracts and analyzes LaTeX template zip files
            - Returns parsed format rules and structure

        - **Args**:
            - `payload` (TemplateParsePayload): Request payload with file path

        - **Returns**:
            - `TemplateParseResult`: Parsed template info or error
        """
        start = time.time()
        logger.info("template.parse.request %s user=%s", payload.request_id, payload.user_id)

        try:
            # Extract file information from payload
            file_path = payload.payload.get("file_path")
            template_id = payload.payload.get("template_id")

            if not file_path:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="file_path must be provided"
                )

            # Run the agent
            agent_result = await agent_instance.run(
                file_path=file_path,
                template_id=template_id
            )

            # Extract the template info
            template_info_dict = agent_result.get("template_info", {})
            
            # Convert to TemplateInfo model
            template_info = TemplateInfo(**template_info_dict)

            latency = time.time() - start
            logger.info("template.parse.complete %s latency=%.3f", payload.request_id, latency)

            return TemplateParseResult(
                request_id=payload.request_id,
                status="ok",
                result=template_info
            )

        except Exception as e:
            latency = time.time() - start
            logger.error("template.parse.error %s latency=%.3f error=%s", payload.request_id, latency, str(e))
            return TemplateParseResult(
                request_id=payload.request_id,
                status="error",
                error=str(e)
            )

    return router
