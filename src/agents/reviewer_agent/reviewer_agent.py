"""
Reviewer Agent
- **Description**:
    - Coordinates feedback checkers for paper review
    - Provides iterative feedback loop support
    - Extensible architecture for adding new checkers
"""
import logging
from typing import List, Dict, Any, Optional, Type, TYPE_CHECKING
from fastapi import APIRouter
from openai import AsyncOpenAI

from ..base import BaseAgent
from ...config.schema import ModelConfig
from .models import (
    ReviewContext,
    ReviewResult,
    FeedbackResult,
    Severity,
)
from .checkers.base import FeedbackChecker
from .checkers.word_count import WordCountChecker
from .checkers.style_check import StyleChecker
from .checkers.logic_check import LogicChecker

if TYPE_CHECKING:
    from ...skills.registry import SkillRegistry


logger = logging.getLogger("uvicorn.error")


class ReviewerAgent(BaseAgent):
    """
    Reviewer Agent for paper feedback
    - **Description**:
        - Manages multiple feedback checkers
        - Coordinates review process
        - Generates revision guidance
    """
    
    # Default checkers to register
    DEFAULT_CHECKERS: List[Type[FeedbackChecker]] = [
        WordCountChecker,
    ]
    
    def __init__(
        self,
        config: ModelConfig,
        skill_registry: Optional["SkillRegistry"] = None,
    ):
        """
        Initialize the Reviewer Agent.

        - **Args**:
            - `config` (ModelConfig): Model configuration
            - `skill_registry` (SkillRegistry, optional): Global skill registry
              for loading checker rules and anti-patterns
        """
        self.config = config
        self.model_name = config.model_name
        self._checkers: List[FeedbackChecker] = []
        self._skill_registry = skill_registry
        self._router = self._create_router()
        
        # Register default checkers
        for checker_cls in self.DEFAULT_CHECKERS:
            self.register_checker(checker_cls())
        
        # Register skill-based checkers
        self._register_skill_checkers()
        
        logger.info(
            "ReviewerAgent initialized with %d checkers: %s",
            len(self._checkers),
            [c.name for c in self._checkers]
        )

    def _register_skill_checkers(self) -> None:
        """
        Dynamically register StyleChecker and LogicChecker.

        - **Description**:
            - StyleChecker is always registered (works with or without registry)
            - LogicChecker is registered only when an LLM client can be created
        """
        # StyleChecker: pure rule-based, always available
        self.register_checker(StyleChecker(skill_registry=self._skill_registry))

        # LogicChecker: needs LLM client
        try:
            llm_client = AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
            self.register_checker(
                LogicChecker(
                    llm_client=llm_client,
                    model_name=self.model_name,
                    skill_registry=self._skill_registry,
                )
            )
        except Exception as e:
            logger.warning(
                "ReviewerAgent: could not initialize LogicChecker: %s", e
            )
    
    @property
    def name(self) -> str:
        return "reviewer"
    
    @property
    def description(self) -> str:
        return "Reviews paper content and provides feedback for improvement"
    
    @property
    def router(self) -> APIRouter:
        return self._router
    
    @property
    def endpoints_info(self) -> List[Dict[str, Any]]:
        return [
            {
                "path": "/agent/reviewer/review",
                "method": "POST",
                "description": "Review paper and provide feedback",
            },
            {
                "path": "/agent/reviewer/checkers",
                "method": "GET",
                "description": "List registered checkers",
            },
            {
                "path": "/agent/reviewer/health",
                "method": "GET",
                "description": "Health check",
            },
        ]
    
    def _create_router(self) -> APIRouter:
        """Create FastAPI router"""
        from .router import create_reviewer_router
        return create_reviewer_router(self)
    
    def register_checker(self, checker: FeedbackChecker) -> None:
        """
        Register a feedback checker
        - **Args**:
            - `checker`: FeedbackChecker instance to register
        """
        # Check for duplicate names
        for existing in self._checkers:
            if existing.name == checker.name:
                logger.warning(
                    "Checker '%s' already registered, skipping",
                    checker.name
                )
                return
        
        self._checkers.append(checker)
        # Sort by priority
        self._checkers.sort(key=lambda c: c.priority)
        logger.info("Registered checker: %s (priority=%d)", checker.name, checker.priority)
    
    def unregister_checker(self, name: str) -> bool:
        """
        Unregister a checker by name
        - **Args**:
            - `name`: Name of checker to remove
        - **Returns**:
            - `bool`: True if removed, False if not found
        """
        for i, checker in enumerate(self._checkers):
            if checker.name == name:
                self._checkers.pop(i)
                logger.info("Unregistered checker: %s", name)
                return True
        return False
    
    def get_checkers(self) -> List[Dict[str, Any]]:
        """Get list of registered checkers"""
        return [
            {
                "name": c.name,
                "priority": c.priority,
                "enabled": c.enabled,
                "class": c.__class__.__name__,
            }
            for c in self._checkers
        ]
    
    async def review(
        self,
        context: ReviewContext,
        iteration: int = 0,
    ) -> ReviewResult:
        """
        Run all enabled checkers on the context
        
        - **Args**:
            - `context` (ReviewContext): Review context with paper data
            - `iteration` (int): Current iteration number
            
        - **Returns**:
            - `ReviewResult`: Aggregated review result
        """
        result = ReviewResult(iteration=iteration)
        
        logger.info(
            "reviewer.review iteration=%d sections=%s total_words=%d",
            iteration,
            list(context.sections.keys()),
            context.total_word_count(),
        )
        
        # Run each enabled checker
        for checker in self._checkers:
            if not checker.enabled:
                continue
            
            try:
                feedback = await checker.check(context)
                result.add_feedback(feedback)
                
                logger.info(
                    "reviewer.checker name=%s passed=%s severity=%s",
                    checker.name,
                    feedback.passed,
                    feedback.severity,
                )
                
                # Extract sections needing revision
                if not feedback.passed:
                    sections_to_revise = feedback.details.get("sections_to_revise", {})
                    for section_type, reason in sections_to_revise.items():
                        result.add_section_revision(section_type, reason)
                        
                        # Generate and store revision prompt
                        section_content = context.sections.get(section_type, "")
                        revision_prompt = checker.generate_revision_prompt(
                            section_type,
                            section_content,
                            feedback,
                        )
                        
                        # Find and update section feedback
                        for sf in feedback.details.get("section_feedbacks", []):
                            if sf.get("section_type") == section_type:
                                from .models import SectionFeedback
                                section_fb = SectionFeedback(
                                    section_type=section_type,
                                    current_word_count=sf.get("current_word_count", 0),
                                    target_word_count=sf.get("target_word_count", 0),
                                    action=sf.get("action", "ok"),
                                    delta_words=sf.get("delta_words", 0),
                                    revision_prompt=revision_prompt,
                                )
                                result.section_feedbacks.append(section_fb)
                        
            except Exception as e:
                logger.error("reviewer.checker_error name=%s error=%s", checker.name, str(e))
                result.add_feedback(FeedbackResult(
                    checker_name=checker.name,
                    passed=False,
                    severity=Severity.ERROR,
                    message=f"Checker error: {str(e)}",
                ))
        
        logger.info(
            "reviewer.review.complete passed=%s feedbacks=%d revisions=%d",
            result.passed,
            len(result.feedbacks),
            len(result.requires_revision),
        )
        
        return result
    
    def get_revision_prompt(
        self,
        section_type: str,
        current_content: str,
        review_result: ReviewResult,
    ) -> Optional[str]:
        """
        Get revision prompt for a specific section
        
        - **Args**:
            - `section_type`: Type of section to revise
            - `current_content`: Current section content
            - `review_result`: Review result with feedbacks
            
        - **Returns**:
            - `str`: Revision prompt, or None if no revision needed
        """
        # Find section feedback with revision prompt
        for sf in review_result.section_feedbacks:
            if sf.section_type == section_type and sf.revision_prompt:
                return sf.revision_prompt
        
        return None
