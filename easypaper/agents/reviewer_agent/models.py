"""
Reviewer Agent Models
- **Description**:
    - Defines data models for the review feedback system
    - ReviewContext: Input context for checkers
    - FeedbackResult: Output from individual checkers
    - ReviewResult: Aggregated review result
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from enum import Enum


class Severity(str, Enum):
    """Feedback severity levels"""
    ERROR = "error"      # Must be fixed
    WARNING = "warning"  # Should be fixed
    INFO = "info"        # Informational only


class FeedbackLevel(str, Enum):
    """Hierarchical feedback granularity."""
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"
    SENTENCE = "sentence"


class IssueType(str, Enum):
    """Canonical issue taxonomy for generalized review workflows."""
    TERM_CONSISTENCY = "term_consistency"
    LOGICAL_CONTRADICTION = "logical_contradiction"
    CLAIM_EVIDENCE_GAP = "claim_evidence_gap"
    UNSUPPORTED_GENERALIZATION = "unsupported_generalization"
    STYLE_NOISE = "style_noise"
    LATEX_FORMAT = "latex_format"
    LAYOUT_CONSTRAINT = "layout_constraint"
    STRUCTURE_QUALITY = "structure_quality"
    OTHER = "other"


class FeedbackResult(BaseModel):
    """
    Result from a single feedback checker
    - **Description**:
        - Represents the output of one checker's evaluation
        
    - **Fields**:
        - `checker_name` (str): Name of the checker that produced this feedback
        - `passed` (bool): Whether the check passed
        - `severity` (Severity): Error level of the feedback
        - `message` (str): Human-readable feedback message
        - `details` (Dict): Checker-specific detailed information
        - `suggested_action` (str, optional): Suggested fix action
    """
    checker_name: str
    passed: bool
    severity: Severity = Severity.INFO
    message: str
    details: Dict[str, Any] = Field(default_factory=dict)
    suggested_action: Optional[str] = None


class HierarchicalFeedbackItem(BaseModel):
    """
    Unified hierarchical feedback item.
    - **Description**:
        - Carries review feedback across document/section/paragraph/sentence levels
        - Supports multi-agent aggregation in one structure
    """
    level: FeedbackLevel = FeedbackLevel.SECTION
    agent: str = "reviewer"
    checker: str = ""
    target_id: str = ""  # e.g. "document", "method", "method.p3", "method.p3.s2"
    section_type: Optional[str] = None
    paragraph_index: Optional[int] = None
    sentence_index: Optional[int] = None
    severity: Severity = Severity.INFO
    issue_type: str = ""
    message: str = ""
    suggested_action: Optional[str] = None
    revision_instruction: str = ""
    evidence: Dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0


class RevisionTask(BaseModel):
    """
    Executable revision task produced by review orchestration.
    - **Description**:
        - Encodes what to revise, why, and what to preserve
        - Used by metadata-layer revision planners and exporters
    """
    task_id: str = ""
    section_type: str
    level: FeedbackLevel = FeedbackLevel.SECTION
    target_id: str = ""
    paragraph_indices: List[int] = Field(default_factory=list)
    action: str = "revise"
    priority: int = 5
    rationale: str = ""
    instruction: str = ""
    preserve_claims: List[str] = Field(default_factory=list)
    do_not_change: List[str] = Field(default_factory=list)
    expected_change: str = ""
    source_agents: List[str] = Field(default_factory=list)
    source_checkers: List[str] = Field(default_factory=list)
    issue_type: IssueType = IssueType.OTHER
    acceptance_criteria: List[str] = Field(default_factory=list)


class ConflictResolutionRecord(BaseModel):
    """
    Conflict resolution output for competing suggestions.
    - **Description**:
        - Captures what conflicted and why a decision was selected
    """
    section_type: str = ""
    target_id: str = ""
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    selected_action: str = ""
    selected_source: str = ""
    reason: str = ""
    applied_guardrail: Optional[str] = None


class SemanticCheckRecord(BaseModel):
    """
    Before/after semantic consistency check record.
    """
    section_type: str
    passed: bool = True
    summary: str = ""
    risks: List[str] = Field(default_factory=list)
    action_taken: str = "accepted"
    checker: str = "semantic_guard_llm"


class ParagraphFeedback(BaseModel):
    """
    Feedback for a specific paragraph within a section.
    - **Description**:
        - Provides granular, paragraph-indexed review feedback
        - Enables targeted revision instructions
    """
    paragraph_index: int = 0
    paragraph_preview: str = ""
    issues: List[str] = Field(default_factory=list)
    severity: Severity = Severity.INFO
    suggestion: str = ""


class SectionFeedback(BaseModel):
    """
    Feedback specific to a section
    - **Description**:
        - Contains revision instructions for a specific section
        - action values: 'expand', 'reduce', 'ok', 'fix_latex',
          'resize_figures', 'move_to_appendix'
        - structural_actions carries structured operation descriptors
        - paragraph_feedbacks provides fine-grained per-paragraph feedback
    """
    section_type: str
    current_word_count: int
    target_word_count: int
    action: str  # 'expand', 'reduce', 'ok', 'fix_latex', 'resize_figures', 'move_to_appendix'
    delta_words: int  # positive = add, negative = remove
    revision_prompt: str = ""
    structural_actions: List[str] = Field(default_factory=list)
    paragraph_feedbacks: List[ParagraphFeedback] = Field(default_factory=list)
    # Paragraph-addressable revision support
    target_paragraphs: List[int] = Field(default_factory=list)
    paragraph_instructions: Dict[int, str] = Field(default_factory=dict)
    feedback_level: FeedbackLevel = FeedbackLevel.SECTION
    target_id: str = ""  # e.g. "method" or "method.p3"
    issue_type: IssueType = IssueType.OTHER
    acceptance_criteria: List[str] = Field(default_factory=list)


class ReviewContext(BaseModel):
    """
    Context provided to checkers for evaluation
    - **Description**:
        - Contains all information needed for review
        
    - **Fields**:
        - `sections` (Dict): section_type -> latex_content mapping
        - `word_counts` (Dict): section_type -> word_count mapping
        - `target_pages` (int): Target page count
        - `target_words` (int, optional): Computed target word count
        - `section_targets` (Dict, optional): Per-section word targets from plan
        - `template_path` (str, optional): Path to template file
        - `style_guide` (str, optional): Style guide name (ICML, NeurIPS, etc.)
        - `metadata` (Dict): Original paper metadata
    """
    sections: Dict[str, str] = Field(default_factory=dict)
    word_counts: Dict[str, int] = Field(default_factory=dict)
    target_pages: int = 8
    target_words: Optional[int] = None
    section_targets: Optional[Dict[str, int]] = None  # From PaperPlan
    template_path: Optional[str] = None
    style_guide: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    memory_context: Optional[Dict[str, Any]] = None  # Session memory snapshot
    
    def total_word_count(self) -> int:
        """Calculate total word count across all sections"""
        return sum(self.word_counts.values())
    
    def get_section_target(self, section_type: str) -> Optional[int]:
        """Get target word count for a section (from plan or None)"""
        if self.section_targets:
            return self.section_targets.get(section_type)
        return None


class ReviewResult(BaseModel):
    """
    Aggregated result from all checkers
    - **Description**:
        - Contains combined feedback from all registered checkers
        
    - **Fields**:
        - `passed` (bool): Whether all checks passed
        - `feedbacks` (List): All feedback results
        - `iteration` (int): Current review iteration number
        - `requires_revision` (Dict): section_type -> list of reasons
        - `section_feedbacks` (List): Detailed per-section feedback
    """
    passed: bool = True
    feedbacks: List[FeedbackResult] = Field(default_factory=list)
    iteration: int = 0
    requires_revision: Dict[str, List[str]] = Field(default_factory=dict)
    section_feedbacks: List[SectionFeedback] = Field(default_factory=list)
    hierarchical_feedbacks: List[HierarchicalFeedbackItem] = Field(default_factory=list)
    revision_tasks: List[RevisionTask] = Field(default_factory=list)
    decision_trace: List[Dict[str, Any]] = Field(default_factory=list)
    conflict_resolution: List[ConflictResolutionRecord] = Field(default_factory=list)
    semantic_checks: List[SemanticCheckRecord] = Field(default_factory=list)
    orchestrator_summary: str = ""
    
    def add_feedback(self, feedback: FeedbackResult):
        """Add a feedback result and update passed status"""
        self.feedbacks.append(feedback)
        if not feedback.passed and feedback.severity == Severity.ERROR:
            self.passed = False
    
    def add_section_revision(self, section_type: str, reason: str):
        """Mark a section as requiring revision"""
        if section_type not in self.requires_revision:
            self.requires_revision[section_type] = []
        self.requires_revision[section_type].append(reason)
        self.passed = False

    def add_hierarchical_feedback(self, item: HierarchicalFeedbackItem):
        """Append hierarchical feedback item."""
        self.hierarchical_feedbacks.append(item)

    def add_revision_task(self, task: RevisionTask):
        """Append executable revision task."""
        self.revision_tasks.append(task)


class ReviewRequest(BaseModel):
    """
    Request to the Reviewer Agent
    - **Description**:
        - Input for the review endpoint
    """
    sections: Dict[str, str]
    word_counts: Dict[str, int]
    target_pages: Optional[int] = None
    section_targets: Optional[Dict[str, int]] = None  # Per-section targets from plan
    template_path: Optional[str] = None
    style_guide: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    iteration: int = 0
    memory_context: Optional[Dict[str, Any]] = None  # Session memory snapshot
