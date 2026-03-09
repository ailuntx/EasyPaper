"""
Models for Commander Agent
- **Description**:
    - Commander acts as adapter between FlowGram.ai and Writer Agent
    - Outputs unified SectionWritePayload format
"""
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
import uuid

# Import unified models from Writer Agent
from ..writer_agent.section_models import SectionWritePayload


class CommanderPayload(BaseModel):
    """
    Payload for Commander Agent request
    - **Description**:
        - Input parameters for paper section generation orchestration
        - Accepts FlowGram.ai specific parameters (work_id, node_ids)
    """
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    payload: Dict[str, Any]
    # Expected payload fields:
    # - work_id: str (required)
    # - section_type: str (abstract, introduction, method, etc.)
    # - section_title: str (optional, custom title)
    # - user_prompt: str
    # - template_id: Optional[str]
    # - explicit_node_ids: Optional[List[str]] (FlowGram variable refs)
    # - word_count_limit: Optional[int]


class CommanderResult(BaseModel):
    """
    Result from Commander Agent
    - **Description**:
        - Contains unified SectionWritePayload for Writer Agent
        - This is the interface between Commander and Writer
    """
    request_id: str
    status: str  # 'ok' or 'error'
    section_write_payload: Optional[SectionWritePayload] = None
    error: Optional[str] = None
