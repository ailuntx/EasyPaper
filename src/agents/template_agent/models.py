"""
Models for Template Parser Agent
"""
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
import uuid


class TemplateInfo(BaseModel):
    """
    Parsed LaTeX template information
    - **Description**:
        - Contains all extracted format rules and constraints from a LaTeX template
    """
    template_id: str
    main_tex_path: str
    citation_style: str = "cite"  # \cite / \citep / \citet
    figure_placement: str = "htbp"  # [htbp] / [H] etc.
    section_commands: List[str] = Field(default_factory=lambda: ["section", "subsection", "subsubsection"])
    required_packages: List[str] = Field(default_factory=list)
    bib_style: Optional[str] = None
    document_class: str = "article"
    template_structure: Dict[str, Any] = Field(default_factory=dict)
    has_abstract: bool = True
    has_acknowledgment: bool = False
    column_format: str = "single"  # single / double
    raw_preamble: Optional[str] = None


class TemplateParsePayload(BaseModel):
    """
    Payload for template parsing request
    - **Description**:
        - Contains the file path to the uploaded template zip
    """
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    payload: Dict[str, Any]  # Should contain 'file_path' or 'file_content'


class TemplateParseResult(BaseModel):
    """
    Result of template parsing
    - **Description**:
        - Contains parsed template info or error message
    """
    request_id: str
    status: str  # 'ok' or 'error'
    result: Optional[TemplateInfo] = None
    error: Optional[str] = None
