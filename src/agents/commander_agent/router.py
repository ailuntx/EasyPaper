"""
Router for Commander Agent endpoints
- **Description**:
    - Commander converts FlowGram.ai data to unified SectionWritePayload
    - The output can be directly passed to Writer Agent
"""
from fastapi import APIRouter, HTTPException, status
from typing import Optional, Any, Dict
import time
import logging
from .models import CommanderPayload, CommanderResult
from ..writer_agent.section_models import SectionWritePayload


def create_commander_router(agent_instance):
    """
    Create router for commander agent endpoints
    - **Args**:
        - `agent_instance`: The CommanderAgent instance

    - **Returns**:
        - `APIRouter`: FastAPI router with commander endpoints
    """
    router = APIRouter()
    logger = logging.getLogger("uvicorn.error")

    @router.post("/agent/commander/prepare", response_model=CommanderResult, status_code=status.HTTP_200_OK)
    async def prepare_section(payload: CommanderPayload):
        """
        Prepare unified SectionWritePayload for Writer Agent
        - **Description**:
            - Collects context from FlowGram.ai research graph
            - Outputs SectionWritePayload that Writer can directly consume
            - This is the interface between FlowGram.ai and the Writer

        - **Args**:
            - `payload` (CommanderPayload): Request payload with work_id and section info

        - **Returns**:
            - `CommanderResult`: Contains section_write_payload for Writer
        """
        start = time.time()
        logger.info("commander.prepare.request %s user=%s", payload.request_id, payload.user_id)

        try:
            # Extract parameters from payload
            work_id = payload.payload.get("work_id")
            section_type = payload.payload.get("section_type", "introduction")
            section_title = payload.payload.get("section_title", "")
            user_prompt = payload.payload.get("user_prompt", "")
            template_id = payload.payload.get("template_id")
            explicit_node_ids = payload.payload.get("explicit_node_ids", [])
            word_count_limit = payload.payload.get("word_count_limit")

            if not work_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="work_id must be provided"
                )

            # Run the agent
            agent_result = await agent_instance.run(
                work_id=work_id,
                section_type=section_type,
                section_title=section_title,
                user_prompt=user_prompt,
                template_id=template_id,
                explicit_node_ids=explicit_node_ids,
                word_count_limit=word_count_limit,
            )

            # Extract the unified SectionWritePayload
            section_write_payload = agent_result.get("section_write_payload")
            
            latency = time.time() - start
            logger.info("commander.prepare.complete %s latency=%.3f section=%s", 
                       payload.request_id, latency, section_type)

            return CommanderResult(
                request_id=payload.request_id,
                status="ok",
                section_write_payload=section_write_payload,
            )

        except Exception as e:
            latency = time.time() - start
            logger.error("commander.prepare.error %s latency=%.3f error=%s", 
                        payload.request_id, latency, str(e))
            return CommanderResult(
                request_id=payload.request_id,
                status="error",
                error=str(e)
            )

    return router
