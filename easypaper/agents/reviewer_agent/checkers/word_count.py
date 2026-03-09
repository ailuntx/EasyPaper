"""
Word Count Checker
- **Description**:
    - Checks if paper word count meets target page requirements
    - Provides section-level feedback for expansion/reduction
    - Uses venue-specific configurations
"""
from typing import Dict, List, Optional, TYPE_CHECKING
from .base import FeedbackChecker

if TYPE_CHECKING:
    from ..models import ReviewContext, FeedbackResult


# Venue configurations: pages and approximate words per page
VENUE_CONFIGS: Dict[str, Dict] = {
    # Machine Learning conferences
    "ICML": {"pages": 8, "words_per_page": 850, "abstract_limit": 150},
    "NEURIPS": {"pages": 8, "words_per_page": 700, "abstract_limit": 150},
    "NIPS": {"pages": 8, "words_per_page": 700, "abstract_limit": 150},
    "ICLR": {"pages": 8, "words_per_page": 750, "abstract_limit": 150},
    
    # NLP conferences
    "ACL": {"pages": 8, "words_per_page": 750, "abstract_limit": 150},
    "EMNLP": {"pages": 8, "words_per_page": 750, "abstract_limit": 150},
    "NAACL": {"pages": 8, "words_per_page": 750, "abstract_limit": 150},
    "COLING": {"pages": 8, "words_per_page": 750, "abstract_limit": 200},
    
    # AI conferences
    "AAAI": {"pages": 7, "words_per_page": 800, "abstract_limit": 150},
    "IJCAI": {"pages": 7, "words_per_page": 800, "abstract_limit": 150},
    
    # Vision conferences
    "CVPR": {"pages": 8, "words_per_page": 800, "abstract_limit": 150},
    "ICCV": {"pages": 8, "words_per_page": 800, "abstract_limit": 150},
    "ECCV": {"pages": 14, "words_per_page": 600, "abstract_limit": 150},
    
    # Data Mining
    "KDD": {"pages": 9, "words_per_page": 750, "abstract_limit": 150},
    "WWW": {"pages": 10, "words_per_page": 700, "abstract_limit": 150},
    
    # Default for unknown venues
    "DEFAULT": {"pages": 8, "words_per_page": 750, "abstract_limit": 150},
}

# Fallback section word allocation ratios — ONLY used when no plan targets are
# provided via ReviewContext.section_targets.  The Planner's per-section targets
# are the canonical source of truth; these ratios exist solely as a safety net.
_FALLBACK_SECTION_RATIOS: Dict[str, float] = {
    "abstract": 0.022,
    "introduction": 0.12,
    "related_work": 0.10,
    "method": 0.22,
    "experiment": 0.18,
    "result": 0.15,
    "discussion": 0.05,
    "conclusion": 0.035,
}

# Keep backward-compatible alias (deprecated; prefer plan targets)
SECTION_RATIOS = _FALLBACK_SECTION_RATIOS

# Tolerance for word count checks (percentage)
TOLERANCE_PERCENT = 0.15  # 15% tolerance


class WordCountChecker(FeedbackChecker):
    """
    Checks paper word count against target page requirements
    - **Description**:
        - Evaluates total and per-section word counts
        - Provides specific expansion/reduction guidance
        - Uses venue-specific configurations
    """
    
    @property
    def name(self) -> str:
        return "word_count"
    
    @property
    def priority(self) -> int:
        return 1  # Run early (structure-level check)
    
    def _get_venue_config(self, style_guide: Optional[str]) -> Dict:
        """Get venue configuration from style guide"""
        if not style_guide:
            return VENUE_CONFIGS["DEFAULT"]
        
        # Normalize venue name
        venue_key = style_guide.upper().split()[0]
        return VENUE_CONFIGS.get(venue_key, VENUE_CONFIGS["DEFAULT"])
    
    def _calculate_target_words(
        self, 
        target_pages: Optional[int],
        style_guide: Optional[str],
    ) -> int:
        """Calculate target word count"""
        config = self._get_venue_config(style_guide)
        pages = target_pages or config["pages"]
        return pages * config["words_per_page"]
    
    def _calculate_section_targets(
        self,
        target_words: int,
        sections: Dict[str, str],
    ) -> Dict[str, int]:
        """
        Calculate target word count for each section using fallback ratios.
        - **Description**:
            - Only called when no plan-provided section_targets exist.
            - Uses _FALLBACK_SECTION_RATIOS for known sections.
            - Gives unknown sections a small default allocation.
        """
        targets = {}
        for section_type in sections.keys():
            if section_type in _FALLBACK_SECTION_RATIOS:
                targets[section_type] = int(target_words * _FALLBACK_SECTION_RATIOS[section_type])
            else:
                # Unknown / dynamic section: give proportional allocation
                targets[section_type] = int(target_words * 0.08)
        return targets
    
    async def check(self, context: "ReviewContext") -> "FeedbackResult":
        """
        Check word counts against targets
        
        Returns FeedbackResult with:
        - passed: True if total word count is within tolerance
        - details: Section-level analysis
        
        Uses section_targets from plan if available, otherwise falls back to SECTION_RATIOS.
        """
        from ..models import FeedbackResult, Severity, SectionFeedback
        
        # Calculate targets
        target_words = context.target_words or self._calculate_target_words(
            context.target_pages,
            context.style_guide,
        )
        
        total_words = context.total_word_count()
        
        # Use plan's section_targets if available, otherwise calculate from ratios
        if context.section_targets:
            section_targets = context.section_targets
            # Update total target if section targets sum differently
            section_sum = sum(section_targets.values())
            if section_sum > 0:
                target_words = section_sum
        else:
            section_targets = self._calculate_section_targets(target_words, context.sections)
        
        # Calculate tolerances
        min_words = int(target_words * (1 - TOLERANCE_PERCENT))
        max_words = int(target_words * (1 + TOLERANCE_PERCENT))
        
        # Check total word count
        if total_words < min_words:
            passed = False
            severity = Severity.ERROR
            delta = min_words - total_words
            message = f"Paper is too short: {total_words} words (target: {target_words}, minimum: {min_words}). Need ~{delta} more words."
            suggested_action = "expand"
        elif total_words > max_words:
            passed = False
            severity = Severity.ERROR
            delta = total_words - max_words
            message = f"Paper is too long: {total_words} words (target: {target_words}, maximum: {max_words}). Need to cut ~{delta} words."
            suggested_action = "reduce"
        else:
            passed = True
            severity = Severity.INFO
            message = f"Word count is acceptable: {total_words} words (target: {target_words})"
            suggested_action = None
        
        # Generate per-section feedback
        section_feedbacks = []
        sections_to_revise = {}
        
        for section_type, word_count in context.word_counts.items():
            target = section_targets.get(section_type, 0)
            if target == 0:
                continue
                
            section_min = int(target * (1 - TOLERANCE_PERCENT))
            section_max = int(target * (1 + TOLERANCE_PERCENT))
            
            if word_count < section_min:
                action = "expand"
                delta = target - word_count
                sections_to_revise[section_type] = f"Expand by ~{delta} words"
            elif word_count > section_max:
                action = "reduce"
                delta = word_count - target
                sections_to_revise[section_type] = f"Reduce by ~{delta} words"
            else:
                action = "ok"
                delta = 0
            
            section_feedbacks.append(SectionFeedback(
                section_type=section_type,
                current_word_count=word_count,
                target_word_count=target,
                action=action,
                delta_words=delta if action == "expand" else -delta,
            ))
        
        return FeedbackResult(
            checker_name=self.name,
            passed=passed,
            severity=severity,
            message=message,
            suggested_action=suggested_action,
            details={
                "total_words": total_words,
                "target_words": target_words,
                "min_words": min_words,
                "max_words": max_words,
                "section_targets": section_targets,
                "sections_to_revise": sections_to_revise,
                "section_feedbacks": [sf.model_dump() for sf in section_feedbacks],
            },
        )
    
    def generate_revision_prompt(
        self,
        section_type: str,
        current_content: str,
        feedback: "FeedbackResult",
    ) -> str:
        """
        Generate a revision prompt based on word count feedback
        """
        details = feedback.details
        sections_to_revise = details.get("sections_to_revise", {})
        section_feedbacks = details.get("section_feedbacks", [])
        
        # Find this section's feedback
        section_fb = None
        for sf in section_feedbacks:
            if sf.get("section_type") == section_type:
                section_fb = sf
                break
        
        if not section_fb:
            return ""
        
        action = section_fb.get("action", "ok")
        current_words = section_fb.get("current_word_count", 0)
        target_words = section_fb.get("target_word_count", 0)
        delta = abs(section_fb.get("delta_words", 0))
        
        if action == "expand":
            return f"""Please EXPAND this {section_type} section.

Current word count: {current_words}
Target word count: {target_words}
Words to add: approximately {delta}

Guidelines for expansion:
- Add more detail and depth to existing points
- Include additional examples or evidence
- Expand on methodology or experimental details
- Add more context or background where appropriate
- Maintain academic writing quality

Current content to expand:
{current_content}

Please provide an expanded version that reaches approximately {target_words} words while maintaining coherence and quality."""

        elif action == "reduce":
            return f"""Please REDUCE this {section_type} section.

Current word count: {current_words}
Target word count: {target_words}
Words to remove: approximately {delta}

Guidelines for reduction:
- Remove redundant or repetitive content
- Consolidate similar points
- Use more concise language
- Focus on the most important information
- Maintain key technical details

Current content to reduce:
{current_content}

Please provide a condensed version that reaches approximately {target_words} words while preserving essential information."""

        else:
            return ""  # No revision needed


def get_venue_config(style_guide: Optional[str]) -> Dict:
    """Public helper to get venue configuration"""
    if not style_guide:
        return VENUE_CONFIGS["DEFAULT"]
    venue_key = style_guide.upper().split()[0]
    return VENUE_CONFIGS.get(venue_key, VENUE_CONFIGS["DEFAULT"])


def calculate_target_words(
    target_pages: Optional[int] = None,
    style_guide: Optional[str] = None,
) -> int:
    """Public helper to calculate target words"""
    config = get_venue_config(style_guide)
    pages = target_pages or config["pages"]
    return pages * config["words_per_page"]
