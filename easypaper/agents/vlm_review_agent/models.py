"""
VLM Review Agent Models
- **Description**:
    - Defines input/output models for VLM-based PDF review
    - Supports page overflow detection, underfill detection, and layout analysis
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class IssueSeverity(str, Enum):
    """Severity levels for layout issues"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class IssueType(str, Enum):
    """Types of layout issues that can be detected"""
    OVERFLOW = "overflow"           # Content exceeds page limit
    UNDERFILL = "underfill"         # Too much blank space
    WIDOW = "widow"                 # Single line at top of page
    ORPHAN = "orphan"               # Single line at bottom of page
    BAD_FIGURE = "bad_figure"       # Poorly placed figure
    BAD_TABLE = "bad_table"         # Poorly placed table
    EQUATION_OVERFLOW = "equation_overflow"  # Equation extends beyond margin
    MARGIN_VIOLATION = "margin_violation"    # Content beyond margins
    BAD_BREAK = "bad_break"         # Bad page/column break


class BlankSpace(BaseModel):
    """Represents a detected blank space in the page"""
    location: str                   # e.g., "bottom-left", "top-right", "center"
    size: str                       # e.g., "small", "medium", "large"
    estimated_lines: int = 0        # Estimated number of text lines that could fit
    page_number: int = 0


class LayoutIssue(BaseModel):
    """Represents a detected layout issue"""
    issue_type: IssueType
    severity: IssueSeverity
    description: str
    page_number: int
    location: Optional[str] = None  # Where on the page
    suggested_fix: Optional[str] = None


class SectionAdvice(BaseModel):
    """Advice for a specific section"""
    section_type: str               # e.g., "introduction", "method", etc.
    current_length: str             # e.g., "too_long", "appropriate", "too_short"
    recommended_action: str         # e.g., "trim", "keep", "expand"
    target_change: Optional[int] = None  # Target word change (+/- words)
    priority: int = 5               # 1-10, higher = more important to change
    specific_guidance: Optional[str] = None


class PageAnalysis(BaseModel):
    """Analysis result for a single page"""
    page_number: int
    fill_percentage: float          # 0-100
    is_overflow: bool = False       # Content extends beyond boundary
    has_significant_blank: bool = False
    blank_spaces: List[BlankSpace] = Field(default_factory=list)
    layout_issues: List[LayoutIssue] = Field(default_factory=list)
    is_last_content_page: bool = False  # Last page with main content (before references)
    is_references_page: bool = False    # Contains bibliography
    is_appendix_page: bool = False      # Contains appendix content
    body_content_percentage: float = 100.0  # 0-100, % of page that is main body content
    raw_vlm_response: Optional[str] = None  # Raw VLM response for debugging


class VLMReviewRequest(BaseModel):
    """
    Request for VLM-based PDF review
    
    - **Description**:
        - Specifies the PDF to review and check parameters
        
    - **Args**:
        - `pdf_path` (str): Path to the compiled PDF file
        - `page_limit` (int): Maximum allowed pages (e.g., 8 for ICML)
        - `template_type` (str): Template type for context (ICML, NeurIPS, etc.)
        - `check_overflow` (bool): Whether to check for page overflow
        - `check_underfill` (bool): Whether to check for excessive blank space
        - `check_layout` (bool): Whether to check for layout issues
        - `sections_info` (Dict): Optional info about sections for targeted advice
    """
    pdf_path: str
    page_limit: int = 8
    template_type: str = "ICML"
    check_overflow: bool = True
    check_underfill: bool = True
    check_layout: bool = True
    sections_info: Optional[Dict[str, Any]] = None  # Section word counts, etc.
    plan_context: Optional[Dict[str, Any]] = None  # Plan section targets from memory
    prior_vlm_issues: Optional[List[str]] = None  # Issues from prior VLM reviews
    
    # Thresholds
    min_fill_percentage: float = 0.85  # Last page should be at least 85% filled
    max_blank_area: float = 0.15       # Max blank area per page


class VLMReviewResult(BaseModel):
    """
    Result of VLM-based PDF review
    
    - **Description**:
        - Contains overall pass/fail status and detailed analysis
        
    - **Returns**:
        - `passed` (bool): Whether the PDF passes all checks
        - `total_pages` (int): Actual page count
        - `content_pages` (int): Pages with main content (excluding references)
        - `overflow_pages` (int): Number of pages exceeding limit
        - `issues` (List[LayoutIssue]): All detected issues
        - `page_analyses` (List[PageAnalysis]): Per-page analysis
        - `section_recommendations` (Dict): Recommendations for each section
        - `summary` (str): Human-readable summary
    """
    passed: bool
    total_pages: int
    content_pages: float = 0.0
    overflow_pages: float = 0.0
    underfill_detected: bool = False
    issues: List[LayoutIssue] = Field(default_factory=list)
    page_analyses: List[PageAnalysis] = Field(default_factory=list)
    section_recommendations: Dict[str, SectionAdvice] = Field(default_factory=dict)
    summary: str = ""
    
    # Actions needed
    needs_trim: bool = False
    needs_expand: bool = False
    trim_target_words: int = 0      # How many words to remove
    expand_target_words: int = 0    # How many words to add


class VLMResponse(BaseModel):
    """Response from VLM provider"""
    success: bool
    content: Optional[str] = None   # Parsed response
    raw_response: Optional[str] = None
    error: Optional[str] = None
    tokens_used: int = 0


# Trim/Expand priority configuration
SECTION_TRIM_PRIORITY = {
    # Higher number = trim first
    "discussion": 10,
    "related_work": 8,
    "experiment": 6,
    "result": 5,
    "method": 3,
    "introduction": 2,
    "abstract": 1,  # Never trim
    "conclusion": 4,
}

SECTION_EXPAND_PRIORITY = {
    # Higher number = expand first
    "experiment": 10,
    "result": 9,
    "discussion": 8,
    "related_work": 7,
    "method": 5,
    "introduction": 3,
    "conclusion": 2,
    "abstract": 1,  # Never expand
}

# Estimated words per page for different templates
WORDS_PER_PAGE = {
    "ICML": 850,
    "NeurIPS": 800,
    "ICLR": 820,
    "ACL": 780,
    "EMNLP": 780,
    "CVPR": 900,
    "ICCV": 900,
    "COLM": 800,
    "DEFAULT": 800,
}
