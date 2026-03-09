"""
Feedback Checker Base Class
- **Description**:
    - Abstract base class for all feedback checkers
    - Defines the interface that all checkers must implement
    - Supports extensible feedback mechanism
"""
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import ReviewContext, FeedbackResult


class FeedbackChecker(ABC):
    """
    Abstract base class for feedback checkers
    - **Description**:
        - All feedback checkers must inherit from this class
        - Checkers are executed in priority order (lower = earlier)
        
    - **Required Properties**:
        - `name`: Unique identifier for the checker
        - `priority`: Execution order (lower numbers run first)
        
    - **Required Methods**:
        - `check()`: Perform the check and return FeedbackResult
        - `generate_revision_prompt()`: Create prompt for revising content
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique name for this checker
        - Examples: 'word_count', 'layout', 'citation', 'visualization'
        """
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """
        Execution priority (lower numbers run first)
        - Suggested ranges:
            - 0-9: Critical checks (structure, word count)
            - 10-19: Content checks (citations, figures)
            - 20-29: Formatting checks (layout, style)
            - 30+: Optional/informational checks
        """
        pass
    
    @property
    def enabled(self) -> bool:
        """Whether this checker is enabled (can be overridden)"""
        return True
    
    @abstractmethod
    async def check(self, context: "ReviewContext") -> "FeedbackResult":
        """
        Perform the check on the given context
        
        - **Args**:
            - `context` (ReviewContext): The review context with sections and metadata
            
        - **Returns**:
            - `FeedbackResult`: The result of the check
        """
        pass
    
    @abstractmethod
    def generate_revision_prompt(
        self, 
        section_type: str,
        current_content: str,
        feedback: "FeedbackResult",
    ) -> str:
        """
        Generate a revision prompt based on feedback
        
        - **Args**:
            - `section_type` (str): Type of section to revise
            - `current_content` (str): Current section content
            - `feedback` (FeedbackResult): The feedback to address
            
        - **Returns**:
            - `str`: Prompt for the LLM to revise the section
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}' priority={self.priority}>"
