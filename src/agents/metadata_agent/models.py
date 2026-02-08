"""
MetaData Agent Models
- **Description**:
    - Defines input/output models for MetaData-based paper generation
    - PaperMetaData: User's simplified input (5 strings + references)
    - PaperGenerationResult: Complete generation result
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class OutputFormat(str, Enum):
    """Output format options"""
    LATEX = "latex"
    PDF = "pdf"


class FigureSpec(BaseModel):
    """
    Figure specification for paper generation
    - **Description**:
        - Defines a figure to be included in the paper
        - All fields except id and caption are optional
        
    - **Args**:
        - `id` (str): LaTeX label (e.g., "fig:architecture")
        - `caption` (str): Figure caption
        - `description` (str): Helps Writer understand/reference the figure
        - `section` (str): Suggested section placement
        - `file_path` (str, optional): Path to figure file (PDF, PNG, JPEG)
        - `wide` (bool): If True, use figure* for double-column spanning
        - `auto_generate` (bool): Mark for future auto-generation
        - `generation_prompt` (str, optional): Prompt for image generation (future)
    """
    id: str                              # LaTeX label: "fig:architecture"
    caption: str                         # Figure caption
    description: str = ""                # Helps Writer understand/reference
    section: str = ""                    # Suggested section placement
    
    # Source
    file_path: Optional[str] = None      # "figures/model.pdf"
    
    # Layout
    wide: bool = False                   # If True, use figure* for double-column spanning
    
    # Future: auto-generation
    auto_generate: bool = False          # Mark for future auto-generation
    generation_prompt: Optional[str] = None  # Prompt for image generation


class TableSpec(BaseModel):
    """
    Table specification for paper generation
    - **Description**:
        - Defines a table to be included in the paper
        - Content can be provided via file_path or inline content
        - Any readable format (CSV, Markdown, plain text) is converted to LaTeX
        
    - **Args**:
        - `id` (str): LaTeX label (e.g., "tab:results")
        - `caption` (str): Table caption
        - `description` (str): Helps Writer understand/reference the table
        - `section` (str): Suggested section placement
        - `file_path` (str, optional): Path to data file (CSV, MD, TXT)
        - `content` (str, optional): Inline data in any readable format
        - `wide` (bool): If True, use table* for double-column spanning
        - `auto_generate` (bool): Mark for future auto-generation
        - `data_source` (str, optional): Metadata field to extract from (future)
    """
    id: str                              # LaTeX label: "tab:results"
    caption: str                         # Table caption
    description: str = ""                # Helps Writer understand/reference
    section: str = ""                    # Suggested section placement
    
    # Source (choose one, or auto_generate)
    file_path: Optional[str] = None      # "data/results.csv" (CSV, MD, TXT)
    content: Optional[str] = None        # Inline data (any readable format)
    
    # Layout
    wide: bool = False                   # If True, use table* for double-column spanning
    
    # Future: auto-generation
    auto_generate: bool = False          # Mark for future auto-generation
    data_source: Optional[str] = None    # Metadata field to extract from


class PaperMetaData(BaseModel):
    """
    User's paper metadata - simplified input
    - **Description**:
        - Minimal input for paper generation
        - 5 natural language fields + BibTeX references
        - Optional figures and tables
        
    - **Args**:
        - `title` (str): Paper title
        - `idea_hypothesis` (str): Research idea or hypothesis
        - `method` (str): Method description
        - `data` (str): Data or validation method description
        - `experiments` (str): Experiment design, results, findings
        - `references` (List[str]): BibTeX entries
        - `template_path` (str, optional): Path to .zip template file
        - `style_guide` (str, optional): Writing style guide (e.g., "ICML", "NeurIPS")
        - `target_pages` (int, optional): Target page count (uses venue default if not set)
        - `figures` (List[FigureSpec], optional): Figure specifications
        - `tables` (List[TableSpec], optional): Table specifications
    """
    title: str = "Untitled Paper"
    idea_hypothesis: str
    method: str
    data: str
    experiments: str
    references: List[str] = Field(default_factory=list)
    template_path: Optional[str] = None  # Path to .zip template file
    style_guide: Optional[str] = None    # Writing style (can be extracted from template)
    target_pages: Optional[int] = None   # Target page count (overrides venue default)
    
    # Figures and tables (optional)
    figures: List[FigureSpec] = Field(default_factory=list)  # Optional figures
    tables: List[TableSpec] = Field(default_factory=list)    # Optional tables


class PaperGenerationRequest(BaseModel):
    """
    Request for paper generation
    - **Description**:
        - Wraps PaperMetaData with generation options
        
    - **Args**:
        - `title` (str): Paper title
        - `idea_hypothesis` (str): Research idea or hypothesis
        - `method` (str): Method description
        - `data` (str): Data description
        - `experiments` (str): Experiments description
        - `references` (List[str]): BibTeX entries
        - `template_path` (str, optional): Path to .zip template file
        - `style_guide` (str, optional): Writing style guide
        - `target_pages` (int, optional): Target page count
        - `compile_pdf` (bool): Whether to compile PDF (default: True if template provided)
        - `figures_source_dir` (str, optional): Directory containing figure files
        - `save_output` (bool): Whether to save output to disk
        - `output_dir` (str, optional): Directory for output files
        - `enable_review` (bool): Whether to enable review loop
        - `max_review_iterations` (int): Maximum review iterations
    """
    # Metadata fields (can pass directly)
    title: str = "Untitled Paper"
    idea_hypothesis: str = ""
    method: str = ""
    data: str = ""
    experiments: str = ""
    references: List[str] = Field(default_factory=list)
    
    # Figures and tables (optional)
    figures: List[FigureSpec] = Field(default_factory=list)
    tables: List[TableSpec] = Field(default_factory=list)
    
    # Template and style
    template_path: Optional[str] = None      # Path to .zip template file
    style_guide: Optional[str] = None        # Writing style (ICML, NeurIPS, etc.)
    target_pages: Optional[int] = None       # Target page count
    
    # Compilation options
    compile_pdf: bool = True                 # Compile to PDF if template provided
    figures_source_dir: Optional[str] = None # Directory with figure files (legacy)
    
    # Review options
    enable_review: bool = True               # Enable review loop
    max_review_iterations: int = 3           # Maximum review iterations
    
    # Planning options
    enable_planning: bool = True             # Enable planning phase
    
    # VLM Review options
    enable_vlm_review: bool = False          # Enable VLM-based PDF review (page overflow detection)
    
    # Output options
    save_output: bool = True
    output_dir: Optional[str] = None
    
    def to_metadata(self) -> PaperMetaData:
        """Convert request to PaperMetaData"""
        return PaperMetaData(
            title=self.title,
            idea_hypothesis=self.idea_hypothesis,
            method=self.method,
            data=self.data,
            experiments=self.experiments,
            references=self.references,
            template_path=self.template_path,
            style_guide=self.style_guide,
            target_pages=self.target_pages,
            figures=self.figures,
            tables=self.tables,
        )


class SectionResult(BaseModel):
    """Result for a single section"""
    section_type: str
    section_title: str = ""
    status: str  # 'ok', 'error'
    latex_content: str = ""
    word_count: int = 0
    error: Optional[str] = None


class PaperGenerationResult(BaseModel):
    """
    Result of paper generation
    - **Description**:
        - Contains generated paper content and metadata
        
    - **Returns**:
        - `status` (str): 'ok', 'partial', 'error'
        - `paper_title` (str): Paper title
        - `sections` (List[SectionResult]): Results for each section
        - `latex_content` (str): Complete assembled LaTeX
        - `output_path` (str, optional): Directory where files are saved
        - `pdf_path` (str, optional): Path to PDF if generated
        - `total_word_count` (int): Total word count
        - `target_word_count` (int, optional): Target word count
        - `review_iterations` (int): Number of review iterations performed
        - `errors` (List[str]): List of errors
    """
    status: str  # 'ok', 'partial', 'error'
    paper_title: str = ""
    sections: List[SectionResult] = Field(default_factory=list)
    latex_content: str = ""
    output_path: Optional[str] = None
    pdf_path: Optional[str] = None
    total_word_count: int = 0
    target_word_count: Optional[int] = None
    review_iterations: int = 0
    errors: List[str] = Field(default_factory=list)


class SectionGenerationRequest(BaseModel):
    """
    Request to generate a single section
    - **Description**:
        - For debugging or incremental generation
        
    - **Args**:
        - `section_type` (str): Type of section to generate
        - `metadata` (PaperMetaData): Paper metadata
        - `intro_context` (str, optional): Introduction content for context
        - `prior_sections` (Dict[str, str], optional): Already generated sections
    """
    section_type: str
    metadata: PaperMetaData
    intro_context: Optional[str] = None
    prior_sections: Optional[Dict[str, str]] = None


# Section source mapping
BODY_SECTION_SOURCES: Dict[str, List[str]] = {
    "related_work": ["references"],
    "method": ["method"],
    "experiment": ["data", "experiments"],
    "result": ["experiments"],
    "discussion": ["experiments"],
}

SYNTHESIS_SECTIONS = ["abstract", "conclusion"]

INTRODUCTION_SOURCES = ["idea_hypothesis", "method", "data", "experiments", "references"]

# Default section order for paper assembly
DEFAULT_SECTION_ORDER = [
    "abstract",
    "introduction",
    "related_work",
    "method",
    "experiment",
    "result",
    "conclusion",
]
