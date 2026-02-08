"""
Section Writing Models
- **Description**:
    - Argument-tree based JSON format for section-based paper writing
    - Uses Material/Claim structure to express reasoning relationships
    - Fully decoupled from FlowGram.ai - usable by any frontend
"""
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict, List, ForwardRef
from enum import Enum
import uuid


class SectionType(str, Enum):
    """
    Supported paper section types
    """
    ABSTRACT = "abstract"
    INTRODUCTION = "introduction"
    RELATED_WORK = "related_work"
    BACKGROUND = "background"
    METHOD = "method"
    EXPERIMENT = "experiment"
    RESULT = "result"
    ANALYSIS = "analysis"
    DISCUSSION = "discussion"
    CONCLUSION = "conclusion"
    ACKNOWLEDGMENT = "acknowledgment"
    APPENDIX = "appendix"
    CUSTOM = "custom"


class MaterialType(str, Enum):
    """
    Types of research materials
    """
    HYPOTHESIS = "hypothesis"
    QUESTION = "question"
    IDEA = "idea"
    METHOD = "method"
    EXPERIMENT = "experiment"
    DATA = "data"
    METRIC = "metric"
    RESULT = "result"
    FINDING = "finding"
    LITERATURE = "literature"
    CONCEPT = "concept"
    OBSERVATION = "observation"
    OTHER = "other"


class PointType(str, Enum):
    """
    Types of points in an argument
    - **Description**:
        - Renamed from ClaimType for more universal applicability
        - "Point" is more suitable for descriptive sections like Methodology
    """
    MAIN = "main"           # Primary point of the section
    SUB = "sub"             # Supporting sub-point
    BACKGROUND = "background"  # Background/context point


class PointRelation(str, Enum):
    """
    Relation types between points
    - **Description**:
        - How a sub-point relates to its parent point
    """
    ELABORATES = "elaborates"     # Provides more detail
    EXEMPLIFIES = "exemplifies"   # Provides an example
    CONTRASTS = "contrasts"       # Provides contrast/comparison
    SUPPORTS = "supports"         # Directly supports parent
    EXTENDS = "extends"           # Extends the argument


# Backward compatibility alias
ClaimType = PointType
ClaimRelation = PointRelation


# =============================================================================
# Core Material Model (replaces NodeContent)
# =============================================================================

class Material(BaseModel):
    """
    Research material - a piece of evidence or content
    - **Description**:
        - Represents any research content that supports a point
        - Replaces the old NodeContent model
        - Uses 'material' terminology instead of 'node' for decoupling
        - Supports explicit linking to resources via linked_* fields

    - **Args**:
        - `id` (str): Unique identifier for this material
        - `material_type` (str): Type of material (hypothesis, method, result, etc.)
        - `title` (str): Title of the material
        - `content` (str): Main text content
        - `linked_refs` (List[str]): IDs of references to cite when using this material
        - `linked_figures` (List[str]): IDs of figures to reference
        - `linked_tables` (List[str]): IDs of tables to reference
        - `linked_equations` (List[str]): IDs of equations to reference
        - `metadata` (Dict): Additional metadata
    """
    id: str
    material_type: str  # hypothesis, method, result, finding, literature, concept, etc.
    title: str = ""
    content: str = ""  # Main text content
    
    # Explicit resource linking (user can control what to cite when using this material)
    linked_refs: List[str] = Field(default_factory=list)        # Reference IDs to cite
    linked_figures: List[str] = Field(default_factory=list)     # Figure IDs to reference
    linked_tables: List[str] = Field(default_factory=list)      # Table IDs to reference
    linked_equations: List[str] = Field(default_factory=list)   # Equation IDs to reference
    
    metadata: Dict[str, Any] = Field(default_factory=dict)


# =============================================================================
# Point Model (argument tree node) - renamed from Claim
# =============================================================================

class Point(BaseModel):
    """
    A point/topic in the argument tree
    - **Description**:
        - Represents a key point being made in the section
        - "Point" is more universal than "Claim" for descriptive sections
        - Contains supporting and counter materials
        - Can have sub-points for hierarchical structure
        - Supports explicit resource linking for user control

    - **Args**:
        - `id` (str): Unique identifier for this point
        - `statement` (str): The main statement of this point
        - `point_type` (str): Type of point (main, sub, background)
        - `supporting_materials` (List[Material]): Evidence supporting this point
        - `counter_materials` (List[Material]): Counter-evidence or challenges
        - `sub_points` (List[Point]): Nested sub-points (recursive)
        - `linked_refs` (List[str]): IDs of references directly associated with this point
        - `relation_to_parent` (str, optional): How this relates to parent point
    """
    id: str
    statement: str  # The point statement
    point_type: str = "main"  # main, sub, background
    
    # Supporting evidence
    supporting_materials: List[Material] = Field(default_factory=list)
    
    # Counter-evidence or challenges (for balanced discussion)
    counter_materials: List[Material] = Field(default_factory=list)
    
    # Sub-points (recursive tree structure)
    sub_points: List["Point"] = Field(default_factory=list)
    
    # Explicit resource linking at point level
    linked_refs: List[str] = Field(default_factory=list)  # Reference IDs to cite for this point
    
    # Relation to parent point (if this is a sub-point)
    relation_to_parent: Optional[str] = None  # elaborates, exemplifies, contrasts, supports, extends


# Enable forward reference for recursive Point
Point.model_rebuild()

# Backward compatibility alias
Claim = Point


# =============================================================================
# Argument Structure (replaces explicit/implicit nodes)
# =============================================================================

class ArgumentStructure(BaseModel):
    """
    The argument structure for a section
    - **Description**:
        - Replaces flat explicit_nodes/implicit_nodes lists
        - Expresses the reasoning structure of the section
        - Contains thesis, points, and background context
        - Supports template syntax in user_prompt for referencing

    - **Template Syntax** (for user_prompt):
        - `{{point:id}}` - Reference a point by ID
        - `{{ref:id}}` - Reference a reference by ID
        - `{{fig:id}}` - Reference a figure by ID
        - `{{eq:id}}` - Reference an equation by ID
        Example: "Focus on {{point:p1}} and cite {{ref:bordes2013}}"
    """
    # Core thesis statement for this section (optional)
    thesis: Optional[str] = None
    
    # Main points with their supporting materials
    main_points: List[Point] = Field(default_factory=list)
    
    # Background context materials (not tied to specific points)
    background_context: List[Material] = Field(default_factory=list)
    
    # Backward compatibility property
    @property
    def main_claims(self) -> List[Point]:
        """Deprecated: use main_points instead"""
        return self.main_points


# =============================================================================
# Resource Models (for citations, figures, etc.)
# =============================================================================

class ReferenceInfo(BaseModel):
    """
    Reference/citation information
    - **Description**:
        - Represents a literature reference for citation
        - Supports two formats:
          1. BibTeX format: ref_id + bibtex (preferred)
          2. Structured format: ref_id + title + other fields
    """
    ref_id: str  # Citation key for \cite{} - required
    bibtex: Optional[str] = None  # Raw BibTeX string - preferred format
    title: Optional[str] = None  # Optional if bibtex is provided
    authors: Optional[str] = None
    year: Optional[int] = None
    venue: Optional[str] = None  # Conference/journal name
    doi: Optional[str] = None
    url: Optional[str] = None
    abstract: Optional[str] = None


class FigureInfo(BaseModel):
    """
    Figure resource information
    - **Description**:
        - Represents a figure for inclusion in the paper
    """
    figure_id: str  # ID for \includegraphics{}
    title: str = ""
    caption: str = ""
    file_path: Optional[str] = None
    file_type: Optional[str] = None  # png, jpg, pdf, etc.
    width: Optional[str] = None  # LaTeX width spec, e.g., "0.8\\textwidth"


class TableInfo(BaseModel):
    """
    Table resource information
    - **Description**:
        - Represents a table for inclusion in the paper
    """
    table_id: str  # Label for \ref{}
    title: str = ""
    caption: str = ""
    data: Optional[List[List[str]]] = None  # Table data as 2D array
    latex_content: Optional[str] = None  # Pre-formatted LaTeX table


class EquationInfo(BaseModel):
    """
    Equation/formula information
    - **Description**:
        - Represents a mathematical equation or formula
    """
    equation_id: str  # Label for \ref{}
    title: str = ""
    latex: str = ""  # LaTeX equation content
    description: Optional[str] = None


class TemplateRules(BaseModel):
    """
    LaTeX template formatting rules
    - **Description**:
        - Contains formatting rules extracted from template
    """
    document_class: Optional[str] = None
    citation_style: str = "cite"  # cite, citep, citet, etc.
    figure_placement: Optional[str] = None  # h, t, b, p, etc.
    column_format: Optional[str] = None  # single, double
    section_commands: List[str] = Field(default_factory=list)
    required_packages: List[str] = Field(default_factory=list)


class SectionResources(BaseModel):
    """
    External resources for the section
    - **Description**:
        - Contains references, figures, tables, equations
        - Separated from argument structure for clarity
    """
    references: List[ReferenceInfo] = Field(default_factory=list)
    figures: List[FigureInfo] = Field(default_factory=list)
    tables: List[TableInfo] = Field(default_factory=list)
    equations: List[EquationInfo] = Field(default_factory=list)
    template_rules: Optional[TemplateRules] = None


# =============================================================================
# Section Constraints Model
# =============================================================================

class SectionConstraints(BaseModel):
    """
    Writing constraints for section generation
    - **Description**:
        - Specifies constraints and preferences for content generation
    """
    word_count_limit: Optional[int] = None
    citation_format: str = "cite"  # cite, citep, citet
    language: str = "en"
    style_guide: Optional[str] = None  # ACL, IEEE, NeurIPS, etc.
    additional_instructions: List[str] = Field(default_factory=list)


# =============================================================================
# Section Write Payload (Main API Model)
# =============================================================================

class SectionWritePayload(BaseModel):
    """
    Universal payload for section writing
    - **Description**:
        - Main input model for Writer Agent section writing
        - Uses argument tree structure to express reasoning
        - Fully decoupled from any specific frontend
        - Supports explicit linking between materials/points and resources

    - **Args**:
        - `section_type` (str): Type of section (abstract, introduction, etc.)
        - `section_title` (str): Custom section title
        - `user_prompt` (str): User's writing instructions (supports template syntax)
        - `argument` (ArgumentStructure): The argument tree with points and materials
        - `resources` (SectionResources): External resources (refs, figures, etc.)
        - `constraints` (SectionConstraints): Writing constraints

    - **Template Syntax** (for user_prompt):
        - `{{point:id}}` - Reference a point by ID (expands to point statement)
        - `{{ref:id}}` - Reference a reference by ID (expands to citation)
        - `{{fig:id}}` - Reference a figure by ID
        - `{{eq:id}}` - Reference an equation by ID
        Example: "Focus on {{point:p1}} and ensure {{ref:bordes2013}} is cited."

    - **Resource Linking**:
        - Materials and Points can have `linked_refs`, `linked_figures`, etc.
        - When a material/point is used, its linked resources are automatically cited
        - This gives users explicit control over citation behavior
    """
    # Request metadata
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    user_id: Optional[str] = None
    
    # Section information
    section_type: str  # abstract, introduction, method, etc.
    section_title: str = ""  # Custom title (optional)
    user_prompt: str = ""  # User's specific writing instructions (supports {{}} template syntax)
    
    # Argument structure (replaces context with explicit/implicit nodes)
    argument: ArgumentStructure = Field(default_factory=ArgumentStructure)
    
    # External resources
    resources: SectionResources = Field(default_factory=SectionResources)
    
    # Writing constraints
    constraints: SectionConstraints = Field(default_factory=SectionConstraints)


# =============================================================================
# Section Write Result
# =============================================================================

class SectionWriteResult(BaseModel):
    """
    Result from section writing
    - **Description**:
        - Contains the generated LaTeX content and metadata

    - **Returns**:
        - `request_id` (str): Request identifier
        - `status` (str): 'ok' or 'error'
        - `section_type` (str): Type of section generated
        - `latex_content` (str): Generated LaTeX body content
        - `word_count` (int): Approximate word count
        - `citation_ids` (List[str]): Extracted citation IDs used
        - `figure_ids` (List[str]): Extracted figure IDs used
        - `table_ids` (List[str]): Extracted table IDs used
        - `saved_path` (str, optional): Path where result was saved
        - `error` (str, optional): Error message if failed
    """
    request_id: str
    status: str  # 'ok' or 'error'
    section_type: str = ""
    section_title: str = ""
    latex_content: str = ""
    word_count: int = 0
    citation_ids: List[str] = Field(default_factory=list)
    figure_ids: List[str] = Field(default_factory=list)
    table_ids: List[str] = Field(default_factory=list)
    saved_path: Optional[str] = None  # Path where result was saved (if save=True)
    error: Optional[str] = None


# =============================================================================
# Frontend Request Models (for Backend API)
# =============================================================================

class SectionWriteRequest(BaseModel):
    """
    Request from frontend to write a section
    - **Description**:
        - Simplified request that triggers Commander + Writer pipeline
        - Commander will construct the ArgumentStructure from graph data
    """
    work_id: str
    node_id: str  # Paper Section node ID
    section_type: str
    section_title: Optional[str] = None
    user_prompt: str = ""
    explicit_node_ids: List[str] = Field(default_factory=list)  # FlowGram variable refs
    template_id: Optional[str] = None
    word_count_limit: Optional[int] = None


class SectionTaskStatus(BaseModel):
    """
    Status of a section writing task
    - **Description**:
        - Tracks progress of section generation
    """
    task_id: str
    node_id: str
    status: str  # pending, processing, completed, failed
    progress: Optional[Dict[str, Any]] = None
    result: Optional[SectionWriteResult] = None
    error_message: Optional[str] = None
    created_at: str
    completed_at: Optional[str] = None


# =============================================================================
# Section Type Requirements (based on argument structure)
# =============================================================================

class SectionRequirement(BaseModel):
    """
    Defines requirements for a section type
    - **Description**:
        - Specifies what an argument structure should contain
        - Used for validation before writing
    """
    # Minimum number of main points
    min_points: int = 1
    
    # Required material types (at least one of each)
    required_material_types: List[str] = Field(default_factory=list)
    
    # Recommended material types (warnings if missing)
    recommended_material_types: List[str] = Field(default_factory=list)
    
    # Whether references are required
    references_required: bool = False
    min_references: int = 0
    
    # Whether figures/tables are required
    figures_required: bool = False
    tables_required: bool = False
    
    # Description of what this section needs
    description: str = ""
    
    # Backward compatibility property
    @property
    def min_claims(self) -> int:
        """Deprecated: use min_points instead"""
        return self.min_points


# Detailed requirements for each section type
SECTION_REQUIREMENTS: Dict[str, SectionRequirement] = {
    "abstract": SectionRequirement(
        min_points=1,
        required_material_types=["hypothesis", "method"],
        recommended_material_types=["result", "finding"],
        references_required=False,
        description="Abstract requires points about hypothesis and methodology. Results/findings improve quality."
    ),
    "introduction": SectionRequirement(
        min_points=2,
        required_material_types=["question", "hypothesis"],
        recommended_material_types=["idea", "literature"],
        references_required=True,
        min_references=3,
        description="Introduction requires points about research question and hypothesis. Literature citations expected."
    ),
    "related_work": SectionRequirement(
        min_points=2,
        required_material_types=["literature"],
        recommended_material_types=["concept"],
        references_required=True,
        min_references=5,
        description="Related work requires points supported by literature. Heavy citation expected."
    ),
    "background": SectionRequirement(
        min_points=1,
        required_material_types=["concept"],
        recommended_material_types=["literature", "method"],
        references_required=True,
        min_references=2,
        description="Background requires concept definitions. May include foundational equations."
    ),
    "method": SectionRequirement(
        min_points=1,
        required_material_types=["method"],
        recommended_material_types=["experiment", "data"],
        figures_required=False,
        description="Method requires points about methodology. May include algorithms, equations, diagrams."
    ),
    "experiment": SectionRequirement(
        min_points=1,
        required_material_types=["experiment"],
        recommended_material_types=["data", "metric"],
        tables_required=False,
        description="Experiment requires points about experimental setup. Dataset and metrics expected."
    ),
    "result": SectionRequirement(
        min_points=1,
        required_material_types=["result"],
        recommended_material_types=["finding"],
        figures_required=False,
        tables_required=False,
        description="Results require points about experimental results. Figures and tables recommended."
    ),
    "analysis": SectionRequirement(
        min_points=1,
        required_material_types=["result", "finding"],
        recommended_material_types=[],
        description="Analysis requires points interpreting results and findings."
    ),
    "discussion": SectionRequirement(
        min_points=1,
        required_material_types=["finding"],
        recommended_material_types=["result", "hypothesis"],
        description="Discussion requires points about findings. Should relate back to hypothesis."
    ),
    "conclusion": SectionRequirement(
        min_points=1,
        required_material_types=["finding"],
        recommended_material_types=["result", "hypothesis", "question"],
        description="Conclusion requires points about key findings. Should address original research question."
    ),
    "acknowledgment": SectionRequirement(
        min_points=0,
        required_material_types=[],
        recommended_material_types=[],
        description="Acknowledgment has no required points or materials."
    ),
    "appendix": SectionRequirement(
        min_points=0,
        required_material_types=[],
        recommended_material_types=["method", "experiment", "data"],
        description="Appendix is flexible. Include supplementary materials."
    ),
    "custom": SectionRequirement(
        min_points=0,
        required_material_types=[],
        recommended_material_types=[],
        description="Custom section has no predefined requirements."
    ),
}


# =============================================================================
# Validation Logic
# =============================================================================

class ValidationResult(BaseModel):
    """
    Result of payload validation
    """
    is_valid: bool = True
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    missing_required: List[str] = Field(default_factory=list)
    missing_recommended: List[str] = Field(default_factory=list)


def _collect_all_materials(argument: ArgumentStructure) -> List[Material]:
    """
    Recursively collect all materials from argument structure
    """
    materials = []
    materials.extend(argument.background_context)
    
    def collect_from_points(points: List[Point]):
        for point in points:
            materials.extend(point.supporting_materials)
            materials.extend(point.counter_materials)
            collect_from_points(point.sub_points)
    
    collect_from_points(argument.main_points)
    return materials


def _count_points(argument: ArgumentStructure) -> int:
    """
    Count total number of points (including sub-points)
    """
    count = 0
    
    def count_recursive(points: List[Point]):
        nonlocal count
        for point in points:
            count += 1
            count_recursive(point.sub_points)
    
    count_recursive(argument.main_points)
    return count


# Backward compatibility alias
_count_claims = _count_points


def validate_section_payload(payload: SectionWritePayload) -> ValidationResult:
    """
    Validate a SectionWritePayload against section requirements
    - **Description**:
        - Checks argument structure completeness
        - Verifies required materials are present
        - Validates resource linking (referenced IDs exist)
        - Returns validation result with errors and warnings

    - **Args**:
        - `payload` (SectionWritePayload): The payload to validate

    - **Returns**:
        - `ValidationResult`: Validation result with details
    """
    result = ValidationResult()
    section_type = payload.section_type
    argument = payload.argument
    resources = payload.resources
    
    # Get requirements for this section type
    requirements = SECTION_REQUIREMENTS.get(section_type, SECTION_REQUIREMENTS["custom"])
    
    # Check minimum points
    point_count = _count_points(argument)
    if point_count < requirements.min_points:
        result.errors.append(
            f"Insufficient points: {section_type} section requires at least {requirements.min_points} points, got {point_count}"
        )
        result.missing_required.append("points")
    
    # Build resource ID sets for link validation
    ref_ids = {r.ref_id for r in resources.references}
    fig_ids = {f.figure_id for f in resources.figures}
    table_ids = {t.table_id for t in resources.tables}
    eq_ids = {e.equation_id for e in resources.equations}
    
    # Check that main points have supporting materials
    for point in argument.main_points:
        if not point.supporting_materials and not point.sub_points:
            result.warnings.append(
                f"Unsupported point: '{point.id}' has no supporting materials or sub-points"
            )
        
        # Validate linked_refs exist in resources
        for ref_id in point.linked_refs:
            if ref_id not in ref_ids:
                result.warnings.append(
                    f"Invalid link: Point '{point.id}' references unknown ref_id '{ref_id}'"
                )
    
    # Collect all material types present and validate material links
    all_materials = _collect_all_materials(argument)
    present_material_types = set(m.material_type for m in all_materials)
    
    # Validate material links
    for material in all_materials:
        for ref_id in material.linked_refs:
            if ref_id not in ref_ids:
                result.warnings.append(
                    f"Invalid link: Material '{material.id}' references unknown ref_id '{ref_id}'"
                )
        for fig_id in material.linked_figures:
            if fig_id not in fig_ids:
                result.warnings.append(
                    f"Invalid link: Material '{material.id}' references unknown figure_id '{fig_id}'"
                )
        for table_id in material.linked_tables:
            if table_id not in table_ids:
                result.warnings.append(
                    f"Invalid link: Material '{material.id}' references unknown table_id '{table_id}'"
                )
        for eq_id in material.linked_equations:
            if eq_id not in eq_ids:
                result.warnings.append(
                    f"Invalid link: Material '{material.id}' references unknown equation_id '{eq_id}'"
                )
    
    # Check required material types
    for required_type in requirements.required_material_types:
        if required_type not in present_material_types:
            result.missing_required.append(required_type)
            result.errors.append(
                f"Missing required material: '{required_type}' is required for {section_type} section"
            )
    
    # Check recommended material types (warnings only)
    for recommended_type in requirements.recommended_material_types:
        if recommended_type not in present_material_types:
            result.missing_recommended.append(recommended_type)
            result.warnings.append(
                f"Recommended material missing: '{recommended_type}' is recommended for {section_type} section"
            )
    
    # Check references requirement
    if requirements.references_required:
        ref_count = len(resources.references)
        if ref_count == 0:
            result.errors.append(
                f"References required: {section_type} section requires at least {requirements.min_references} references"
            )
            result.missing_required.append("references")
        elif ref_count < requirements.min_references:
            result.warnings.append(
                f"Few references: {section_type} section typically needs at least {requirements.min_references} references, got {ref_count}"
            )
    
    # Check figures requirement
    if requirements.figures_required and len(resources.figures) == 0:
        result.errors.append(f"Figures required: {section_type} section requires at least one figure")
        result.missing_required.append("figures")
    
    # Check tables requirement
    if requirements.tables_required and len(resources.tables) == 0:
        result.errors.append(f"Tables required: {section_type} section requires at least one table")
        result.missing_required.append("tables")
    
    # Set overall validity
    result.is_valid = len(result.errors) == 0
    
    return result


def get_section_requirements(section_type: str) -> SectionRequirement:
    """
    Get requirements for a section type
    - **Args**:
        - `section_type` (str): The section type

    - **Returns**:
        - `SectionRequirement`: Requirements for this section type
    """
    return SECTION_REQUIREMENTS.get(section_type, SECTION_REQUIREMENTS["custom"])


# =============================================================================
# Backward Compatibility (deprecated, will be removed)
# =============================================================================

# Old model alias for gradual migration
NodeContent = Material  # Deprecated: use Material instead
SectionContext = ArgumentStructure  # Deprecated: use ArgumentStructure instead

# Old requirements format
ResourceRequirement = SectionRequirement  # Deprecated: use SectionRequirement
SECTION_REQUIRED_RESOURCES: Dict[str, List[str]] = {
    section_type: req.required_material_types + req.recommended_material_types
    for section_type, req in SECTION_REQUIREMENTS.items()
}


# =============================================================================
# Paper Assembly Models (for complete paper generation)
# =============================================================================

class SectionChainItem(BaseModel):
    """
    Single section in the assembly chain
    - **Description**:
        - Defines a section to be generated in the paper
        - Can specify either a payload file path or inline payload
        - At least one of payload_file or payload must be provided

    - **Args**:
        - `section_type` (str): Type of section (abstract, introduction, etc.)
        - `payload_file` (str, optional): Path to JSON file containing SectionWritePayload
        - `payload` (SectionWritePayload, optional): Inline payload object
        - `enabled` (bool): Whether to include this section (default: True)
    """
    section_type: str
    payload_file: Optional[str] = None      # Path to JSON file
    payload: Optional["SectionWritePayload"] = None  # Inline payload
    enabled: bool = True                    # Can disable sections without removing


class PaperChainConfig(BaseModel):
    """
    Configuration for paper assembly
    - **Description**:
        - Defines the complete paper structure and generation settings
        - Specifies section order and their payloads
        - Supports shared resources across all sections
        - Supports template-based compilation via uploaded zip files

    - **Args**:
        - `paper_title` (str): Title of the paper
        - `sections` (List[SectionChainItem]): Ordered list of sections to generate
        - `shared_resources` (SectionResources, optional): Resources shared by all sections
        - `output_dir` (str): Directory to save output files
        - `compile_pdf` (bool): Whether to compile to PDF via Typesetter
        - `template_path` (str, optional): Path to LaTeX template zip (e.g., icml2026.zip)
            - When provided, system auto-parses to extract document_class, preamble, etc.
            - This is the PRIMARY way to specify template settings
        - `template_config` (dict, optional): OVERRIDE for parsed template settings
            - Use ONLY to override specific fields from parsed template
            - NOT for specifying full configuration manually
            - Example: {"citation_style": "citep"} to override citation style
            - Available override fields: citation_style, column_format, bib_style, 
              paper_title, paper_authors
        - `base_path` (str, optional): Base path for resolving relative paths
        - `figures_source_dir` (str, optional): Local directory containing figure files
            - Figures will be copied from this directory to the output
            - Example: "./figures" relative to base_path
    """
    paper_title: str = "Untitled Paper"
    sections: List[SectionChainItem] = Field(default_factory=list)
    shared_resources: Optional[SectionResources] = None
    output_dir: str = "./outputs"
    compile_pdf: bool = False
    template_path: Optional[str] = None
    template_config: Optional[Dict[str, Any]] = None  # TemplateConfig as dict
    base_path: Optional[str] = None  # Base path for resolving relative file paths
    figures_source_dir: Optional[str] = None  # Local directory with figure files


class SectionGenerationStatus(BaseModel):
    """
    Status of a single section generation
    """
    section_type: str
    status: str  # 'ok', 'error', 'skipped'
    word_count: int = 0
    latex_content: str = ""
    error: Optional[str] = None


class PaperAssemblyResult(BaseModel):
    """
    Result of paper assembly
    - **Description**:
        - Contains the assembled paper content and generation metadata
        - Includes status for each section and overall result

    - **Returns**:
        - `status` (str): Overall status ('ok', 'partial', 'error')
        - `paper_title` (str): Title of the paper
        - `sections_status` (List[SectionGenerationStatus]): Status of each section
        - `latex_content` (str): Complete assembled LaTeX content
        - `latex_path` (str, optional): Path to saved .tex file
        - `pdf_path` (str, optional): Path to compiled PDF (if compile_pdf=True)
        - `total_word_count` (int): Total word count
        - `errors` (List[str]): List of errors encountered
    """
    status: str  # 'ok', 'partial', 'error'
    paper_title: str
    sections_status: List[SectionGenerationStatus] = Field(default_factory=list)
    latex_content: str = ""
    latex_path: Optional[str] = None
    pdf_path: Optional[str] = None
    total_word_count: int = 0
    errors: List[str] = Field(default_factory=list)


# =============================================================================
# Simple Mode Models (for MetaData Agent)
# =============================================================================

class SimpleSectionInput(BaseModel):
    """
    Simplified section input for MetaData-based generation
    - **Description**:
        - Lightweight alternative to SectionWritePayload
        - Focuses on "what to express" rather than "how to structure"
        - Used by MetaData Agent and can be used by Commander Agent
        
    - **Args**:
        - `section_type` (str): Type of section (introduction, method, etc.)
        - `section_title` (str): Optional custom title
        - `thesis` (str): Core thesis/theme of this section
        - `content_points` (List[str]): Key points to express (natural language)
        - `references` (List[ReferenceInfo]): Available references
        - `figures` (List[FigureInfo]): Available figures
        - `tables` (List[TableInfo]): Available tables
        - `word_limit` (int, optional): Word count limit
        - `style_guide` (str, optional): Target venue style (ICML, NeurIPS, etc.)
        - `intro_context` (str, optional): Introduction content for context
    """
    section_type: str
    section_title: str = ""
    thesis: str = ""
    content_points: List[str] = Field(default_factory=list)
    references: List[ReferenceInfo] = Field(default_factory=list)
    figures: List[FigureInfo] = Field(default_factory=list)
    tables: List[TableInfo] = Field(default_factory=list)
    word_limit: Optional[int] = None
    style_guide: Optional[str] = None
    intro_context: Optional[str] = None  # Introduction content for body sections


class SynthesisSectionInput(BaseModel):
    """
    Input for synthesis sections (Abstract/Conclusion)
    - **Description**:
        - Used for sections that synthesize content from other sections
        - Does not require independent MetaData input
        - Takes already-generated section content as input
        
    - **Args**:
        - `section_type` (str): "abstract" or "conclusion"
        - `paper_title` (str): Title of the paper
        - `prior_sections` (Dict[str, str]): Already generated sections {type: latex_content}
        - `key_contributions` (List[str]): Key contributions extracted from Introduction
        - `word_limit` (int, optional): Word count limit
        - `style_guide` (str, optional): Target venue style
    """
    section_type: str  # "abstract" or "conclusion"
    paper_title: str
    prior_sections: Dict[str, str] = Field(default_factory=dict)
    key_contributions: List[str] = Field(default_factory=list)
    word_limit: Optional[int] = None
    style_guide: Optional[str] = None
