"""
Structure Checker
- **Description**:
    - Validates section-level structural clarity for long, dense sections.
    - Uses a quality-oriented gate: explicit subsection commands OR clear
      implicit thematic blocks with transitions.
"""
import re
from typing import Dict, List, TYPE_CHECKING

from .base import FeedbackChecker

if TYPE_CHECKING:
    from ..models import ReviewContext, FeedbackResult


class StructureChecker(FeedbackChecker):
    """
    Checker for structural coherence in section drafts.
    """

    @property
    def name(self) -> str:
        return "structure_check"

    @property
    def priority(self) -> int:
        # Run before style checks so structural revisions are planned early.
        return 12

    async def check(self, context: "ReviewContext") -> "FeedbackResult":
        from ..models import FeedbackResult, Severity

        meta = context.metadata or {}
        gate_enabled = bool(meta.get("review_structure_gate_enabled", True))
        if not gate_enabled:
            return FeedbackResult(
                checker_name=self.name,
                passed=True,
                severity=Severity.INFO,
                message="Structure gate disabled by configuration.",
                details={"gate_enabled": False},
            )

        threshold = int(meta.get("structure_gate_min_paragraph_threshold", 5) or 5)
        section_signals = meta.get("section_structure_signals", {}) or {}

        section_feedbacks: List[Dict] = []
        violations: List[Dict] = []
        skipped = {"abstract", "conclusion"}

        for section_type, content in (context.sections or {}).items():
            if section_type in skipped:
                continue
            paragraphs = [p.strip() for p in re.split(r"\n\s*\n", content or "") if p.strip()]
            paragraph_count = len(paragraphs)

            # Only enforce hard gate for dense sections.
            section_signal = section_signals.get(section_type, {}) if isinstance(section_signals, dict) else {}
            recommended = bool(section_signal.get("sectioning_recommended", False))
            if paragraph_count < threshold and not recommended:
                continue

            explicit_count = len(
                re.findall(r"\\subsection\{.+?\}|\\subsubsection\{.+?\}", content or "")
            )
            subsection_titles = re.findall(
                r"\\subsection\{(.+?)\}|\\subsubsection\{(.+?)\}",
                content or "",
            )
            normalized_titles = [
                (a or b or "").strip().lower()
                for a, b in subsection_titles
                if (a or b or "").strip()
            ]
            parent_like_single = False
            if explicit_count == 1 and normalized_titles:
                parent_like_single = (
                    normalized_titles[0] == section_type.strip().lower()
                    or normalized_titles[0] in {f"{section_type.strip().lower()} section", "discussion", "introduction", "results", "methods"}
                )

            transition_markers = 0
            for para in paragraphs[1:]:
                lower = para.lower()
                if lower.startswith((
                    "however", "therefore", "in contrast", "meanwhile", "moreover",
                    "furthermore", "additionally", "by contrast", "in summary",
                )):
                    transition_markers += 1

            # Quality gate: either explicit sectioning, or sufficiently clear implicit blocks.
            implicit_ok = paragraph_count >= 4 and transition_markers >= 1
            explicit_ok = explicit_count >= 1 and not parent_like_single
            # When planner recommends sectioning, we still allow implicit structure,
            # but with a stronger transition requirement to keep quality consistent.
            if recommended:
                passed = explicit_ok or (paragraph_count >= 5 and transition_markers >= 2)
            else:
                passed = explicit_ok or implicit_ok
            if passed:
                continue

            violations.append(
                {
                    "section_type": section_type,
                    "paragraph_count": paragraph_count,
                    "subsection_count": explicit_count,
                    "single_subsection_placeholder": parent_like_single,
                    "transition_markers": transition_markers,
                    "recommended": recommended,
                }
            )
            section_feedbacks.append(
                {
                    "section_type": section_type,
                    "action": "reorganize_blocks",
                    "delta_words": 0,
                    "revision_prompt": (
                        "Improve structural coherence for this dense section. "
                        + (
                            "Planner recommends explicit sectioning for this section; prefer at least one "
                            "\\subsection{} heading. If you keep implicit structure only, add stronger "
                            "transitions between major thematic blocks. "
                            if recommended else
                            "Use either explicit \\subsection{} grouping or clear implicit thematic blocks "
                            "with transition sentences between blocks. "
                        )
                        + (
                            "If you keep explicit sectioning, avoid a single placeholder subsection that repeats the section title. "
                            if parent_like_single else
                            ""
                        )
                    ),
                    "issue_type": "structure_quality",
                    "acceptance_criteria": [
                        "execution_changed",
                        "semantic_preserved",
                        "structure_coherent",
                    ],
                    "target_paragraphs": list(range(paragraph_count)),
                    "paragraph_instructions": {},
                }
            )

        if not violations:
            return FeedbackResult(
                checker_name=self.name,
                passed=True,
                severity=Severity.INFO,
                message="Structure check passed — section organization is adequate.",
                details={
                    "gate_enabled": True,
                    "threshold": threshold,
                    "checked_sections": [
                        s for s in (context.sections or {}).keys() if s not in skipped
                    ],
                },
            )

        return FeedbackResult(
            checker_name=self.name,
            passed=False,
            severity=Severity.ERROR,
            message=(
                f"Structure check failed in {len(violations)} section(s): "
                "dense sections require clearer thematic organization."
            ),
            details={
                "gate_enabled": True,
                "threshold": threshold,
                "violations": violations,
                "section_feedbacks": section_feedbacks,
            },
            suggested_action="reorganize_blocks",
        )

    def generate_revision_prompt(
        self,
        section_type: str,
        current_content: str,
        feedback: "FeedbackResult",
    ) -> str:
        """
        Generate structure-focused revision prompt.
        """
        return (
            f"Revise the {section_type} section to improve structure quality. "
            "Use either explicit subsection commands or clear implicit thematic blocks, "
            "and ensure transitions between blocks."
        )
