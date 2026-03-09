"""
Models for Typesetter Agent
"""
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List
import uuid


class ResourceInfo(BaseModel):
    """
    Information about a fetched resource
    - **Description**:
        - Tracks downloaded assets for LaTeX compilation
    """
    resource_id: str
    resource_type: str  # 'figure', 'table', 'reference'
    original_path: Optional[str] = None
    local_path: Optional[str] = None
    status: str = "pending"  # pending, downloaded, failed


class BibEntry(BaseModel):
    """
    BibTeX entry for a reference
    - **Description**:
        - Represents a single BibTeX reference entry
        - When raw_bibtex is provided, other fields are optional
    """
    key: str
    entry_type: str = "article"
    title: Optional[str] = None  # Optional if raw_bibtex is provided
    authors: Optional[str] = None
    year: Optional[int] = None
    journal: Optional[str] = None
    booktitle: Optional[str] = None  # For inproceedings/conference papers
    venue: Optional[str] = None  # Generic venue field (maps to journal or booktitle)
    volume: Optional[str] = None
    pages: Optional[str] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    raw_bibtex: Optional[str] = None  # Raw BibTeX string - if provided, used directly


class CompilationResult(BaseModel):
    """
    Result of LaTeX compilation
    - **Description**:
        - Contains compilation status and output paths
        - When multi-file mode is used, section_files maps section types to file paths
          and section_errors maps section types to their specific compilation errors
    """
    success: bool
    pdf_path: Optional[str] = None
    source_path: Optional[str] = None
    log_content: Optional[str] = None
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    attempts: int = 0
    section_files: Dict[str, str] = Field(default_factory=dict)  # section_type -> file path
    section_errors: Dict[str, List[str]] = Field(default_factory=dict)  # section_type -> errors


class TypesetterPayload(BaseModel):
    """
    Payload for Typesetter Agent request
    - **Description**:
        - Contains LaTeX content and resources to compile
        - Supports two content modes (mutually exclusive):
          1. latex_content (str): Single concatenated LaTeX body (legacy)
          2. sections (Dict[str, str]): Per-section content for multi-file output
    """
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    payload: Dict[str, Any]
    # Expected payload fields:
    # - latex_content: str              (legacy single-string mode)
    # - sections: Dict[str, str]        (multi-file mode: section_type -> content)
    # - section_order: List[str]        (multi-file mode: order of body sections)
    # - section_titles: Dict[str, str]  (multi-file mode: section_type -> display title)
    # - template_path: str
    # - figure_ids: List[str]
    # - citation_ids: List[str]
    # - references: List[Dict] (reference metadata)
    # - work_id: str


class TypesetterResult(BaseModel):
    """
    Result from Typesetter Agent
    - **Description**:
        - Contains compiled PDF and source or error info
    """
    request_id: str
    status: str  # 'ok' or 'error'
    result: Optional[CompilationResult] = None
    error: Optional[str] = None


class TemplateConfig(BaseModel):
    """
    Template configuration for LaTeX compilation
    - **Description**:
        - Contains parsed template information and constraints
        - Used by Typesetter to build document preamble and apply formatting rules
        - Can be populated from TemplateInfo (parsed template) or manually specified

    - **Args**:
        - `template_id` (str, optional): ID of the source template
        - `document_class` (str): LaTeX document class (article, IEEEtran, etc.)
        - `document_class_options` (List[str]): Options for document class
        - `citation_style` (str): Citation command style (cite, citep, citet)
        - `column_format` (str): Column layout (single, double)
        - `raw_preamble` (str, optional): Full preamble from parsed template
        - `bib_style` (str): Bibliography style (plain, unsrt, ieee, etc.)
        - `required_packages` (List[str]): Additional packages to include
        - `figure_placement` (str): Default figure placement options
        - `paper_title` (str, optional): Paper title for title page
        - `paper_authors` (str, optional): Author names
    """
    template_id: Optional[str] = None
    document_class: str = "article"
    document_class_options: List[str] = Field(default_factory=lambda: ["11pt"])
    citation_style: str = "cite"  # cite / citep / citet
    column_format: str = "single"  # single / double
    raw_preamble: Optional[str] = None  # Full preamble from parsed template
    bib_style: str = "plain"
    required_packages: List[str] = Field(default_factory=list)
    figure_placement: str = "htbp"
    paper_title: Optional[str] = None
    paper_authors: Optional[str] = None
    has_abstract: bool = True
    has_acknowledgment: bool = False

    @classmethod
    def from_template_info(cls, template_info: Dict[str, Any]) -> "TemplateConfig":
        """
        Create TemplateConfig from parsed TemplateInfo dict
        - **Args**:
            - `template_info` (dict): Parsed template information

        - **Returns**:
            - `TemplateConfig`: Configuration object
        """
        return cls(
            template_id=template_info.get("template_id"),
            document_class=template_info.get("document_class", "article"),
            citation_style=template_info.get("citation_style", "cite"),
            column_format=template_info.get("column_format", "single"),
            raw_preamble=template_info.get("raw_preamble"),
            bib_style=template_info.get("bib_style", "plain"),
            required_packages=template_info.get("required_packages", []),
            figure_placement=template_info.get("figure_placement", "htbp"),
            has_abstract=template_info.get("has_abstract", True),
            has_acknowledgment=template_info.get("has_acknowledgment", False),
        )
