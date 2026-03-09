"""
Logic Checker
- **Description**:
    - LLM-based checker for logical consistency, terminology,
      ambiguous references, and Chinglish detection
    - Requires an LLM client to perform analysis
    - Loads prompt templates from SkillRegistry if available
"""
import json
import logging
import re
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from .base import FeedbackChecker

if TYPE_CHECKING:
    from ..models import ReviewContext, FeedbackResult
    from ....skills.registry import SkillRegistry

logger = logging.getLogger("uvicorn.error")

# Default system prompt when no skill is loaded from the registry
_DEFAULT_LOGIC_PROMPT = """You are a meticulous academic paper reviewer focusing on logical consistency.
Analyze the provided paper sections and identify issues at the PARAGRAPH level.

For each section, paragraphs are separated by blank lines. Index them starting from 0.

Issue categories:
1. **Contradictions**: Statements that conflict with each other across sections.
2. **Terminology Inconsistency**: The same concept referred to by different names.
3. **Chinglish / Unnatural Phrasing**: Chinese-grammar-influenced English.
4. **Ambiguous References**: Uses of "it", "this", "they" with unclear antecedents.
5. **Unsupported Claims**: Strong claims without corresponding evidence.

For each issue found, provide:
- The section name
- The paragraph_index (0-based) within that section
- The first ~50 characters of the paragraph (paragraph_preview)
- The problematic text (quoted)
- Why it is a problem
- A suggested fix

Output your analysis as a JSON object:
{
  "issues": [
    {
      "section": "section name",
      "paragraph_index": 0,
      "paragraph_preview": "first ~50 chars...",
      "severity": "high" | "medium" | "low",
      "category": "contradiction" | "terminology" | "chinglish" | "ambiguous_ref" | "unsupported_claim",
      "text": "the problematic text",
      "reason": "explanation",
      "suggestion": "how to fix it"
    }
  ],
  "passed": true | false,
  "summary": "one-sentence overall assessment"
}"""

# Maximum content length sent to LLM (in characters)
_MAX_CONTENT_CHARS = 12000


class LogicChecker(FeedbackChecker):
    """
    LLM-based logic consistency checker.

    - **Description**:
        - Calls an LLM to detect contradictions, terminology issues,
          Chinglish, ambiguous references, and unsupported claims.
        - Optionally loads the analysis prompt from SkillRegistry.
    """

    def __init__(
        self,
        llm_client,
        model_name: str,
        skill_registry: Optional["SkillRegistry"] = None,
    ):
        self._client = llm_client
        self._model = model_name
        self._registry = skill_registry

    @property
    def name(self) -> str:
        return "logic_check"

    @property
    def priority(self) -> int:
        return 10  # Content-level check

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_system_prompt(self) -> str:
        """
        Load the system prompt from the registry or use the default.

        - **Returns**:
            - `str`: The LLM system prompt for logic checking
        """
        if self._registry:
            for skill in self._registry.get_checker_skills():
                if skill.name == "logic-check" and skill.system_prompt_append:
                    return skill.system_prompt_append
        return _DEFAULT_LOGIC_PROMPT

    @staticmethod
    def _format_prior_issues(prior_issues: List[Dict[str, Any]]) -> str:
        """Build a context block describing issues found in previous review iterations."""
        if not prior_issues:
            return ""
        lines = [
            "## Previous Review Issues (check if they have been addressed)",
        ]
        for entry in prior_issues:
            it = entry.get("iteration", "?")
            summary = entry.get("feedback_summary", "")
            passed = entry.get("passed", False)
            status = "PASSED" if passed else "FAILED"
            lines.append(f"- Iteration {it} [{status}]: {summary}")
        lines.append(
            "If any of the above issues are still present, flag them again."
        )
        return "\n".join(lines)

    @staticmethod
    def _assemble_content(sections: Dict[str, str]) -> str:
        """
        Concatenate paper sections into a single string, truncated.

        - **Args**:
            - `sections` (Dict[str, str]): section_type -> LaTeX content

        - **Returns**:
            - `str`: Combined text, truncated to _MAX_CONTENT_CHARS
        """
        parts: List[str] = []
        for stype, content in sections.items():
            parts.append(f"=== Section: {stype} ===\n{content}")
        full = "\n\n".join(parts)
        if len(full) > _MAX_CONTENT_CHARS:
            full = full[:_MAX_CONTENT_CHARS] + "\n\n[... truncated for length ...]"
        return full

    @staticmethod
    def _parse_llm_response(raw: str) -> Dict[str, Any]:
        """
        Extract JSON from the LLM response (handles markdown fences).

        - **Args**:
            - `raw` (str): Raw LLM output

        - **Returns**:
            - `Dict`: Parsed JSON or fallback structure
        """
        # Strip markdown code fences
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("logic_check: failed to parse LLM JSON output")
            return {
                "issues": [],
                "passed": True,
                "summary": "Unable to parse LLM output — treating as pass.",
            }

    # ------------------------------------------------------------------
    # Check
    # ------------------------------------------------------------------

    async def check(self, context: "ReviewContext") -> "FeedbackResult":
        """
        Run LLM-based logic analysis on all paper sections.

        - **Args**:
            - `context` (ReviewContext): Paper sections and metadata

        - **Returns**:
            - `FeedbackResult`: Pass / fail with issues list
        """
        from ..models import FeedbackResult, Severity

        system_prompt = self._get_system_prompt()
        user_content = self._assemble_content(context.sections)

        # Inject prior review issues from memory context so the LLM
        # can verify whether previously identified problems have been fixed
        if context.memory_context and context.memory_context.get("prior_issues"):
            prior_block = self._format_prior_issues(context.memory_context["prior_issues"])
            if prior_block:
                user_content = f"{prior_block}\n\n{user_content}"

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            raw_output = response.choices[0].message.content or ""
            result = self._parse_llm_response(raw_output)
        except Exception as e:
            logger.error("logic_check: LLM call failed: %s", e)
            return FeedbackResult(
                checker_name=self.name,
                passed=True,  # fail-open: don't block pipeline
                severity=Severity.WARNING,
                message=f"Logic check skipped due to LLM error: {e}",
            )

        issues = result.get("issues", [])
        passed = result.get("passed", True)
        summary = result.get("summary", "")
        severity = Severity.WARNING if not passed else Severity.INFO
        if any(i.get("severity") in ("high", "critical") for i in issues):
            severity = Severity.ERROR
        message = summary if summary else (
            f"Logic check found {len(issues)} issue(s)." if issues
            else "Logic check passed."
        )

        # Build sections_to_revise map
        sections_to_revise: Dict[str, str] = {}
        for issue in issues:
            sec = issue.get("section", "unknown")
            if sec not in sections_to_revise:
                sections_to_revise[sec] = issue.get("reason", "logic issue")

        # Build paragraph_feedbacks (grouped by section)
        paragraph_feedbacks: Dict[str, List[Dict]] = {}
        for issue in issues:
            sec = issue.get("section", "unknown")
            if sec not in paragraph_feedbacks:
                paragraph_feedbacks[sec] = []
            paragraph_feedbacks[sec].append({
                "paragraph_index": issue.get("paragraph_index", 0),
                "paragraph_preview": issue.get("paragraph_preview", ""),
                "issues": [f"[{issue.get('category', 'issue')}] {issue.get('text', '')}: {issue.get('reason', '')}"],
                "severity": issue.get("severity", "medium"),
                "suggestion": issue.get("suggestion", ""),
            })

        section_feedbacks: List[Dict] = []
        for sec, pfb in paragraph_feedbacks.items():
            para_indices = []
            para_instructions: Dict[int, str] = {}
            for pf in pfb:
                pidx = int(pf.get("paragraph_index", 0))
                para_indices.append(pidx)
                para_instructions[pidx] = (
                    "Revise this paragraph to resolve logic inconsistencies, clarify references, "
                    "and keep claims aligned with available evidence."
                )
            section_feedbacks.append({
                "section_type": sec,
                "current_word_count": context.word_counts.get(sec, 0),
                "target_word_count": context.get_section_target(sec) or context.word_counts.get(sec, 0),
                "action": "refine_paragraphs" if para_indices else "logic_fix",
                "delta_words": 0,
                "target_paragraphs": sorted(list(set(para_indices))),
                "paragraph_instructions": para_instructions,
            })

        document_feedbacks = []
        if not passed:
            document_feedbacks.append({
                "level": "document",
                "agent": "reviewer",
                "checker": self.name,
                "target_id": "document",
                "severity": severity.value,
                "issue_type": "logical_consistency",
                "message": message,
                "suggested_action": "logic_fix",
            })

        return FeedbackResult(
            checker_name=self.name,
            passed=passed,
            severity=severity,
            message=message,
            suggested_action="logic_fix" if not passed else None,
            details={
                "issues": issues,
                "sections_to_revise": sections_to_revise,
                "paragraph_feedbacks": paragraph_feedbacks,
                "section_feedbacks": section_feedbacks,
                "document_feedbacks": document_feedbacks,
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
        Generate a revision prompt to fix logic issues in a section.

        - **Args**:
            - `section_type` (str): Section to revise
            - `current_content` (str): Current LaTeX content
            - `feedback` (FeedbackResult): Logic check feedback

        - **Returns**:
            - `str`: Revision prompt for the LLM
        """
        issues = feedback.details.get("issues", [])
        relevant = [i for i in issues if i.get("section", "").lower() == section_type.lower()]

        if not relevant:
            return ""

        # Group by paragraph for targeted feedback
        para_issues: Dict[int, List] = {}
        general_issues: List = []
        for i in relevant:
            pidx = i.get("paragraph_index")
            if pidx is not None:
                para_issues.setdefault(pidx, []).append(i)
            else:
                general_issues.append(i)

        parts: List[str] = [f"Please fix the following LOGIC issues in this {section_type} section:\n"]

        for pidx in sorted(para_issues.keys()):
            issues_for_para = para_issues[pidx]
            preview = issues_for_para[0].get("paragraph_preview", "")
            parts.append(f"### Paragraph {pidx}" + (f' (starting with "{preview}...")' if preview else ""))
            for iss in issues_for_para:
                parts.append(
                    f"- [{iss.get('category', 'issue')}] {iss.get('text', '')}: "
                    f"{iss.get('reason', '')} -> Suggestion: {iss.get('suggestion', 'N/A')}"
                )
            parts.append("")

        if general_issues:
            parts.append("### General issues")
            for iss in general_issues:
                parts.append(
                    f"- [{iss.get('category', 'issue')}] {iss.get('text', '')}: "
                    f"{iss.get('reason', '')} -> Suggestion: {iss.get('suggestion', 'N/A')}"
                )

        parts.append("""
Revision guidelines:
1. Resolve any contradictions by aligning claims with evidence
2. Use consistent terminology throughout — pick one term per concept
3. Rewrite Chinglish phrases into natural English
4. Make pronoun references unambiguous — add the noun being referred to
5. Back up strong claims with specific numbers or citations""")

        parts.append(f"\nCurrent content:\n{current_content}")
        parts.append("\nReturn the revised LaTeX content only.")

        return "\n".join(parts)
