"""
Models for Writer Agent
"""
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List, Set
import uuid


class ReviewResult(BaseModel):
    """
    Result from mini-review of generated content.
    """
    passed: bool = True
    issues: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    invalid_citations: List[str] = Field(default_factory=list)
    word_count: int = 0
    target_words: Optional[int] = None
    key_point_coverage: float = 1.0


class WriterPayload(BaseModel):
    """
    Payload for Writer Agent request
    - **Description**:
        - Contains the compiled prompt from Commander Agent
        - Supports iterative review with citation validation
    """
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    payload: Dict[str, Any]
    # Expected payload fields:
    # - system_prompt: str
    # - user_prompt: str
    # - section_type: str
    # - citation_format: str (default: "cite")
    # - constraints: List[str]
    
    # New fields for iterative review
    valid_citation_keys: List[str] = Field(default_factory=list)
    target_words: Optional[int] = None
    key_points: List[str] = Field(default_factory=list)
    max_iterations: int = 2
    enable_review: bool = True


class GeneratedContent(BaseModel):
    """
    Generated LaTeX content from Writer Agent
    - **Description**:
        - Contains the raw LaTeX content and metadata
        - Includes review results from iterative refinement
    """
    latex_content: str
    section_type: str
    word_count: int = 0
    citation_ids: List[str] = Field(default_factory=list)
    figure_ids: List[str] = Field(default_factory=list)
    table_ids: List[str] = Field(default_factory=list)
    # Review results
    iterations_used: int = 1
    review_passed: bool = True
    invalid_citations_removed: List[str] = Field(default_factory=list)


class WriterResult(BaseModel):
    """
    Result from Writer Agent
    - **Description**:
        - Contains generated LaTeX content or error
    """
    request_id: str
    status: str  # 'ok' or 'error'
    result: Optional[GeneratedContent] = None
    error: Optional[str] = None
    # Detailed review info
    review_history: List[ReviewResult] = Field(default_factory=list)
