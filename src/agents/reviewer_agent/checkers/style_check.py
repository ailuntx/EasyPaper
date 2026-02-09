"""
Style Checker
- **Description**:
    - Rule-based checker for AI-style language, contractions,
      excessive em-dashes, and formulaic structures
    - Does NOT require LLM calls — purely pattern-based
    - Loads anti-patterns from SkillRegistry if available
"""
import re
import logging
from typing import Dict, List, Optional, TYPE_CHECKING

from .base import FeedbackChecker

if TYPE_CHECKING:
    from ..models import ReviewContext, FeedbackResult
    from ....skills.registry import SkillRegistry

logger = logging.getLogger("uvicorn.error")

# Fallback anti-patterns used when no SkillRegistry is available
_DEFAULT_ANTI_PATTERNS: List[str] = [
    "delve",
    "leverage",
    "tapestry",
    "First and foremost",
    "it is worth noting",
    "rapidly evolving",
    "a testament to",
    "multifaceted",
    "underscores",
    "spearhead",
    "at the forefront",
    "game-changer",
    "revolutionize",
    "groundbreaking",
    "paradigm shift",
    "holistic approach",
    "synergy",
    "plays a crucial role",
    "are poised to",
    "in the realm of",
    "a myriad of",
    "utilize",
    "facilitate",
    "endeavor",
    "elucidate",
    "aforementioned",
]

# Contraction patterns (regex)
_CONTRACTION_RE = re.compile(
    r"\b(it's|don't|can't|won't|we've|hasn't|wouldn't|shouldn't|aren't|isn't|"
    r"couldn't|didn't|doesn't|hadn't|haven't|he's|she's|that's|they're|"
    r"we're|weren't|who's|you're|you've)\b",
    re.IGNORECASE,
)


class StyleChecker(FeedbackChecker):
    """
    Rule-based style checker for AI-style language and formatting.

    - **Description**:
        - Scans sections for anti-pattern words, contractions,
          excessive em-dashes, and "First...Second...Third..." patterns.
        - Loads additional patterns from a SkillRegistry if provided.
        - Returns FeedbackResult with per-section details.
    """

    def __init__(self, skill_registry: Optional["SkillRegistry"] = None):
        self._registry = skill_registry

    @property
    def name(self) -> str:
        return "style_check"

    @property
    def priority(self) -> int:
        return 20  # Formatting-level check

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_anti_patterns(self) -> List[str]:
        """
        Merge default anti-patterns with those from the SkillRegistry.

        - **Returns**:
            - `List[str]`: Combined list of anti-pattern strings
        """
        patterns = list(_DEFAULT_ANTI_PATTERNS)
        if self._registry:
            for skill in self._registry.get_checker_skills():
                if skill.name == "style-check" and skill.anti_patterns:
                    for p in skill.anti_patterns:
                        if p.lower() not in {x.lower() for x in patterns}:
                            patterns.append(p)
        return patterns

    def _load_thresholds(self) -> Dict:
        """
        Load threshold values from the registry skill or use defaults.

        - **Returns**:
            - `Dict`: Threshold configuration
        """
        defaults = {
            "em_dash_density_max": 2.0,
            "three_part_parallel_max": 1,
            "stacked_connectives_max": 2,
        }
        if self._registry:
            for skill in self._registry.get_checker_skills():
                if skill.name == "style-check" and skill.venue_config:
                    defaults.update(skill.venue_config)
        return defaults

    @staticmethod
    def _word_count(text: str) -> int:
        """Rough word count from LaTeX text."""
        clean = re.sub(r"\\[a-zA-Z]+(\{[^}]*\})?", " ", text)
        clean = re.sub(r"[{}\\$%&]", " ", clean)
        return len(clean.split())

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    async def check(self, context: "ReviewContext") -> "FeedbackResult":
        """
        Scan all sections for style issues.

        - **Args**:
            - `context` (ReviewContext): Paper sections and metadata

        - **Returns**:
            - `FeedbackResult`: Pass / fail with detailed issue list
        """
        from ..models import FeedbackResult, Severity

        anti_patterns = self._load_anti_patterns()
        thresholds = self._load_thresholds()

        all_issues: List[Dict] = []
        sections_to_revise: Dict[str, str] = {}

        for section_type, content in context.sections.items():
            section_issues: List[str] = []
            lower_content = content.lower()
            word_count = self._word_count(content)

            # 1. Anti-pattern detection
            matched_patterns: List[str] = []
            for pat in anti_patterns:
                if pat.lower() in lower_content:
                    count = lower_content.count(pat.lower())
                    matched_patterns.append(f"'{pat}' x{count}")
            if matched_patterns:
                section_issues.append(
                    f"AI-style words detected: {', '.join(matched_patterns)}"
                )

            # 2. Contraction detection
            contractions = _CONTRACTION_RE.findall(content)
            if contractions:
                unique = list(set(c.lower() for c in contractions))[:5]
                section_issues.append(
                    f"Contractions found: {', '.join(unique)}"
                )

            # 3. Em-dash density
            em_dash_count = content.count("\u2014") + content.count("---")
            if word_count > 0:
                density = (em_dash_count / word_count) * 1000
                max_density = thresholds["em_dash_density_max"]
                if density > max_density:
                    section_issues.append(
                        f"Em-dash density too high: {density:.1f}/1000 words "
                        f"(max {max_density})"
                    )

            # 4. Three-part parallel ("First,...Second,...Third,...")
            three_part = len(
                re.findall(
                    r"(?i)\bfirst[\s,].*?\bsecond[\s,].*?\bthird[\s,]",
                    content,
                    re.DOTALL,
                )
            )
            max_parallel = thresholds["three_part_parallel_max"]
            if three_part > max_parallel:
                section_issues.append(
                    f"Three-part parallel structure used {three_part} times "
                    f"(max {max_parallel})"
                )

            # 5. Stacked connective adverbs
            connectives = re.findall(
                r"(?i)^(Furthermore|Moreover|Additionally|In addition|Besides|"
                r"Also|Consequently|Hence|Thus|Therefore|Meanwhile|Likewise)[,\s]",
                content,
                re.MULTILINE,
            )
            # Check for consecutive connective-starting sentences
            sentences = re.split(r"(?<=[.!?])\s+", content)
            consecutive = 0
            max_consec = 0
            connective_set = {
                "furthermore", "moreover", "additionally", "in addition",
                "besides", "also", "consequently", "hence", "thus",
                "therefore", "meanwhile", "likewise",
            }
            for sent in sentences:
                first_word = sent.strip().split(",")[0].strip().split()[0].lower() if sent.strip() else ""
                if first_word in connective_set:
                    consecutive += 1
                    max_consec = max(max_consec, consecutive)
                else:
                    consecutive = 0
            max_stacked = thresholds["stacked_connectives_max"]
            if max_consec > max_stacked:
                section_issues.append(
                    f"Stacked connective adverbs: {max_consec} consecutive "
                    f"(max {max_stacked})"
                )

            # Aggregate
            if section_issues:
                all_issues.append({
                    "section": section_type,
                    "issues": section_issues,
                })
                sections_to_revise[section_type] = "; ".join(section_issues)

        # Determine pass / fail
        passed = len(all_issues) == 0
        severity = Severity.WARNING if not passed else Severity.INFO

        if passed:
            message = "Style check passed — no AI-style or formatting issues found."
        else:
            total_issues = sum(len(i["issues"]) for i in all_issues)
            message = (
                f"Style check found {total_issues} issue(s) across "
                f"{len(all_issues)} section(s)."
            )

        return FeedbackResult(
            checker_name=self.name,
            passed=passed,
            severity=severity,
            message=message,
            suggested_action="style_fix" if not passed else None,
            details={
                "section_issues": all_issues,
                "sections_to_revise": sections_to_revise,
            },
        )

    # ------------------------------------------------------------------
    # Revision prompt
    # ------------------------------------------------------------------

    def generate_revision_prompt(
        self,
        section_type: str,
        current_content: str,
        feedback: "FeedbackResult",
    ) -> str:
        """
        Generate a revision prompt to fix style issues.

        - **Args**:
            - `section_type` (str): Section to revise
            - `current_content` (str): Current LaTeX content
            - `feedback` (FeedbackResult): The style check feedback

        - **Returns**:
            - `str`: Revision prompt for the LLM
        """
        section_issues = feedback.details.get("section_issues", [])
        issues_for_section: List[str] = []
        for entry in section_issues:
            if entry.get("section") == section_type:
                issues_for_section = entry.get("issues", [])
                break

        if not issues_for_section:
            return ""

        issues_text = "\n".join(f"- {issue}" for issue in issues_for_section)

        return f"""Please fix the following STYLE issues in this {section_type} section:

{issues_text}

Revision guidelines:
1. Replace AI-style words with concrete, specific academic alternatives
2. Expand all contractions to full forms (it's → it is, don't → do not)
3. Reduce em-dash usage to at most one per paragraph
4. Break up "First...Second...Third..." patterns — vary sentence structure
5. Do NOT stack connective adverbs in consecutive sentences

Current content:
{current_content}

Return the revised LaTeX content only."""
