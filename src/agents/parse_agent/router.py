from fastapi import APIRouter, HTTPException, status
from typing import Optional, Any, Dict
import time
import logging
from .models import ParsePayload, ParseResult

def create_parse_router(agent_instance):
    """Create router for parse agent endpoints"""
    router = APIRouter()
    logger = logging.getLogger("uvicorn.error")

    @router.post("/agent/parse", response_model=ParseResult, status_code=status.HTTP_200_OK)
    async def parse_paper(payload: ParsePayload):
        """Parse and understand a research paper using the paper parser agent"""
        # basic metrics/logging
        start = time.time()
        logger.info("parse.request %s user=%s", payload.request_id, payload.user_id)

        try:
            # Extract file information from payload
            file_path = payload.payload.get("file_path")
            file_content = payload.payload.get("file_content")

            if not file_path and not file_content:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Either file_path or file_content must be provided"
                )

            # Run the agent
            agent_result = await agent_instance.run(file_path=file_path, file_content=file_content)

            # Extract the understanding result
            understand_result = agent_result.get("understand_result", {})

            latency = time.time() - start
            logger.info("parse.complete %s latency=%.3f", payload.request_id, latency)

            return ParseResult(
                request_id=payload.request_id,
                status="ok",
                result=understand_result
            )

        except Exception as e:
            latency = time.time() - start
            logger.error("parse.error %s latency=%.3f error=%s", payload.request_id, latency, str(e))
            return ParseResult(
                request_id=payload.request_id,
                status="error",
                error=str(e)
            )

    return router