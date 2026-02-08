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


class SectionFeedback(BaseModel):
    """
    Feedback specific to a section
    - **Description**:
        - Contains revision instructions for a specific section
    """
    section_type: str
    current_word_count: int
    target_word_count: int
    action: str  # 'expand', 'reduce', 'ok'
    delta_words: int  # positive = add, negative = remove
    revision_prompt: str = ""


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
