from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
import uuid

class ParsePayload(BaseModel):
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str]
    payload: Dict[str, Any]

class ParseResult(BaseModel):
    request_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None