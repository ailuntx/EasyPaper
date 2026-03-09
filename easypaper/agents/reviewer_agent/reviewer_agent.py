"""
Reviewer Agent
- **Description**:
    - Coordinates feedback checkers for paper review
    - Provides iterative feedback loop support
    - Extensible architecture for adding new checkers
"""
import logging
import json
import re
from typing import List, Dict, Any, Optional, Type, TYPE_CHECKING
from ..base import BaseAgent
from ..shared.llm_client import LLMClient
from ...config.schema import ModelConfig
from .models import (
    ReviewContext,
    ReviewResult,
    FeedbackResult,
    Severity,
    SectionFeedback,
    ParagraphFeedback,
    HierarchicalFeedbackItem,
    FeedbackLevel,
    RevisionTask,
    IssueType,
)
from .checkers.base import FeedbackChecker
from .checkers.word_count import WordCountChecker
from .checkers.style_check import StyleChecker
from .checkers.logic_check import LogicChecker
from .checkers.structure_check import StructureChecker

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
    
    # Default checkers — WordCountChecker removed; word count is now
    # an informational metric only, not a hard constraint.
    DEFAULT_CHECKERS: List[Type[FeedbackChecker]] = []
    
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
        try:
            self._router = self._create_router()
        except Exception:
            self._router = None
        
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
        self.register_checker(StructureChecker())

        # LogicChecker: needs LLM client
        try:
            llm_client = LLMClient(
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
    def router(self) -> "APIRouter | None":
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
    
    def _create_router(self) -> "APIRouter":
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
    
    async def answer(self, question: str, memory=None) -> str:
        """
        Quick consultation — answer a writing quality or consistency question.
        - **Description**:
            - Uses a lightweight LLM call with focused context extracted
              from SessionMemory.
            - Designed to be called via AskTool during WriterAgent's
              ReAct loop.

        - **Args**:
            - `question` (str): The question to answer
            - `memory` (SessionMemory, optional): Session memory for context

        - **Returns**:
            - `answer` (str): Brief assessment or guidance
        """
        context_parts: List[str] = []
        if memory is not None:
            for stype, content in getattr(memory, "generated_sections", {}).items():
                if content.strip():
                    preview = content[:600] + ("..." if len(content) > 600 else "")
                    context_parts.append(f"[{stype}]: {preview}")
            for rec in getattr(memory, "review_history", [])[-2:]:
                context_parts.append(
                    f"[Review iter {rec.iteration}]: {rec.feedback_summary}"
                )

        context_block = "\n".join(context_parts) if context_parts else "No context available."

        try:
            llm_client = LLMClient(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
            response = await llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an academic paper reviewer providing brief, "
                            "focused feedback. Answer the question concisely based "
                            "on the provided context. Keep your response under 200 words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Paper context:\n{context_block}\n\n"
                            f"Question: {question}"
                        ),
                    },
                ],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content or "No answer generated."
        except Exception as e:
            logger.error("reviewer.answer error: %s", e)
            return f"Could not answer: {e}"

    async def review(
        self,
        context: ReviewContext,
        iteration: int = 0,
        memory=None,
    ) -> ReviewResult:
        """
        Run all enabled checkers on the context.

        - **Args**:
            - `context` (ReviewContext): Review context with paper data
            - `iteration` (int): Current iteration number
            - `memory` (SessionMemory, optional): Shared session memory.
              When provided, checkers can read prior issues and plan
              details directly instead of relying on serialized snapshots.

        - **Returns**:
            - `ReviewResult`: Aggregated review result
        """
        # Inject memory into context for checkers that support it
        if memory is not None and hasattr(context, "memory_context"):
            if context.memory_context is None:
                from ..shared.session_memory import SessionMemory
                if isinstance(memory, SessionMemory):
                    context.memory_context = memory.to_review_context_dict()

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
                    for df in feedback.details.get("document_feedbacks", []):
                        try:
                            result.add_hierarchical_feedback(HierarchicalFeedbackItem(
                                level=FeedbackLevel.DOCUMENT,
                                agent=str(df.get("agent", "reviewer")),
                                checker=str(df.get("checker", checker.name)),
                                target_id=str(df.get("target_id", "document")),
                                severity=Severity(str(df.get("severity", feedback.severity))),
                                issue_type=str(df.get("issue_type", "document_issue")),
                                message=str(df.get("message", feedback.message)),
                                suggested_action=df.get("suggested_action"),
                                evidence=df if isinstance(df, dict) else {},
                            ))
                        except Exception:
                            # Keep review loop resilient even with malformed checker payloads.
                            pass

                    sections_to_revise = feedback.details.get("sections_to_revise", {})
                    section_paragraph_feedbacks = feedback.details.get("paragraph_feedbacks", {})
                    raw_section_feedbacks = feedback.details.get("section_feedbacks", []) or []
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
                        matched_section_feedback = False
                        for sf in raw_section_feedbacks:
                            if sf.get("section_type") == section_type:
                                matched_section_feedback = True
                                raw_para = section_paragraph_feedbacks.get(section_type, []) or []
                                para_feedbacks: List[ParagraphFeedback] = []
                                for p in raw_para:
                                    raw_sev = str(p.get("severity", "warning")).lower()
                                    sev = Severity.WARNING
                                    if raw_sev in ("error", "high", "critical"):
                                        sev = Severity.ERROR
                                    elif raw_sev in ("info", "low"):
                                        sev = Severity.INFO
                                    para_feedbacks.append(ParagraphFeedback(
                                        paragraph_index=int(p.get("paragraph_index", 0)),
                                        paragraph_preview=str(p.get("paragraph_preview", "")),
                                        issues=[str(x) for x in p.get("issues", [])],
                                        severity=sev,
                                        suggestion=str(p.get("suggestion", "")),
                                    ))

                                section_fb = SectionFeedback(
                                    section_type=section_type,
                                    current_word_count=sf.get("current_word_count", 0),
                                    target_word_count=sf.get("target_word_count", 0),
                                    action=sf.get("action", "ok"),
                                    delta_words=sf.get("delta_words", 0),
                                    revision_prompt=revision_prompt,
                                    paragraph_feedbacks=para_feedbacks,
                                    target_paragraphs=sf.get(
                                        "target_paragraphs",
                                        [pf.paragraph_index for pf in para_feedbacks],
                                    ),
                                    paragraph_instructions=sf.get("paragraph_instructions", {}),
                                    feedback_level=FeedbackLevel.SECTION,
                                    target_id=sf.get("target_id", section_type),
                                    issue_type=self._coerce_issue_type(sf.get("issue_type", checker.name)),
                                    acceptance_criteria=self._default_acceptance_criteria(
                                        self._coerce_issue_type(sf.get("issue_type", checker.name))
                                    ),
                                )
                                result.section_feedbacks.append(section_fb)
                                result.add_hierarchical_feedback(HierarchicalFeedbackItem(
                                    level=FeedbackLevel.SECTION,
                                    agent="reviewer",
                                    checker=checker.name,
                                    target_id=section_fb.target_id or section_type,
                                    section_type=section_type,
                                    severity=feedback.severity,
                                    issue_type=checker.name,
                                    message=reason,
                                    suggested_action=sf.get("action", "revise"),
                                    revision_instruction=revision_prompt,
                                    evidence={
                                        "target_paragraphs": section_fb.target_paragraphs,
                                        "paragraph_feedbacks": [
                                            pf.model_dump() for pf in section_fb.paragraph_feedbacks
                                        ],
                                    },
                                ))
                                for pf in section_fb.paragraph_feedbacks:
                                    result.add_hierarchical_feedback(HierarchicalFeedbackItem(
                                        level=FeedbackLevel.PARAGRAPH,
                                        agent="reviewer",
                                        checker=checker.name,
                                        target_id=f"{section_type}.p{pf.paragraph_index}",
                                        section_type=section_type,
                                        paragraph_index=pf.paragraph_index,
                                        severity=pf.severity,
                                        issue_type=checker.name,
                                        message="; ".join(pf.issues)[:500],
                                        suggested_action="refine_paragraph",
                                        revision_instruction=pf.suggestion,
                                        evidence=pf.model_dump(),
                                    ))
                        if not matched_section_feedback:
                            para_feedbacks: List[ParagraphFeedback] = []
                            raw_para = section_paragraph_feedbacks.get(section_type, []) or []
                            para_indices: List[int] = []
                            para_instructions: Dict[int, str] = {}
                            for p in raw_para:
                                pidx = int(p.get("paragraph_index", 0))
                                para_indices.append(pidx)
                                raw_sev = str(p.get("severity", "warning")).lower()
                                sev = Severity.WARNING
                                if raw_sev in ("error", "high", "critical"):
                                    sev = Severity.ERROR
                                elif raw_sev in ("info", "low"):
                                    sev = Severity.INFO
                                para_feedbacks.append(ParagraphFeedback(
                                    paragraph_index=pidx,
                                    paragraph_preview=str(p.get("paragraph_preview", "")),
                                    issues=[str(x) for x in p.get("issues", [])],
                                    severity=sev,
                                    suggestion=str(p.get("suggestion", "")),
                                ))
                                para_instructions[pidx] = str(p.get("suggestion", "")) or reason
                            section_fb = SectionFeedback(
                                section_type=section_type,
                                current_word_count=context.word_counts.get(section_type, 0),
                                target_word_count=context.get_section_target(section_type) or context.word_counts.get(section_type, 0),
                                action="refine_paragraphs" if para_indices else "revise",
                                delta_words=0,
                                revision_prompt=revision_prompt,
                                paragraph_feedbacks=para_feedbacks,
                                target_paragraphs=sorted(list(set(para_indices))),
                                paragraph_instructions=para_instructions,
                                feedback_level=FeedbackLevel.SECTION,
                                target_id=section_type,
                                issue_type=self._coerce_issue_type(checker.name),
                                acceptance_criteria=self._default_acceptance_criteria(self._coerce_issue_type(checker.name)),
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
        await self._orchestrate_reviewer_feedback(context=context, result=result)
        return result

    async def _orchestrate_reviewer_feedback(
        self,
        context: ReviewContext,
        result: ReviewResult,
    ) -> None:
        """
        Synthesize checker outputs into unified hierarchical suggestions.
        - **Description**:
            - Uses an LLM orchestrator for cross-checker consolidation
            - Falls back to deterministic task extraction when LLM fails
        """
        if not result.section_feedbacks and not result.hierarchical_feedbacks:
            return

        prompt_payload = {
            "sections": list(context.sections.keys()),
            "feedbacks": [
                fb.model_dump() if hasattr(fb, "model_dump") else fb
                for fb in result.feedbacks
            ],
            "section_feedbacks": [
                sf.model_dump() if hasattr(sf, "model_dump") else sf
                for sf in result.section_feedbacks
            ],
            "hierarchical_feedbacks": [
                hf.model_dump() if hasattr(hf, "model_dump") else hf
                for hf in result.hierarchical_feedbacks
            ],
        }

        llm_output: Dict[str, Any] = {}
        try:
            llm_client = LLMClient(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
            response = await llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a reviewer orchestrator. Merge checker outputs into "
                            "an executable revision plan. Return JSON only with keys: "
                            "summary, revision_tasks. Each revision task must include "
                            "section_type, level, target_id, paragraph_indices, action, "
                            "priority (1-10), rationale, instruction, preserve_claims, "
                            "do_not_change, expected_change, source_agents, source_checkers, "
                            "issue_type, acceptance_criteria."
                        ),
                    },
                    {"role": "user", "content": json.dumps(prompt_payload, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=2000,
            )
            raw = response.choices[0].message.content or ""
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            llm_output = json.loads(cleaned) if cleaned else {}
        except Exception as e:
            logger.warning("reviewer.orchestrator_llm_failed: %s", e)
            llm_output = {}

        tasks_data = llm_output.get("revision_tasks") if isinstance(llm_output, dict) else None
        if isinstance(tasks_data, list) and tasks_data:
            for idx, task in enumerate(tasks_data):
                try:
                    parsed = RevisionTask(
                        task_id=task.get("task_id") or f"rt_{result.iteration}_{idx}",
                        section_type=str(task.get("section_type", "")),
                        level=FeedbackLevel(str(task.get("level", "section"))),
                        target_id=str(task.get("target_id", task.get("section_type", ""))),
                        paragraph_indices=[int(x) for x in task.get("paragraph_indices", [])],
                        action=str(task.get("action", "revise")),
                        priority=max(1, min(10, int(task.get("priority", 5)))),
                        rationale=str(task.get("rationale", "")),
                        instruction=str(task.get("instruction", "")),
                        preserve_claims=[str(x) for x in task.get("preserve_claims", [])],
                        do_not_change=[str(x) for x in task.get("do_not_change", [])],
                        expected_change=str(task.get("expected_change", "")),
                        source_agents=[str(x) for x in task.get("source_agents", ["reviewer"])],
                        source_checkers=[str(x) for x in task.get("source_checkers", [])],
                        issue_type=IssueType(str(task.get("issue_type", "other"))),
                        acceptance_criteria=[str(x) for x in task.get("acceptance_criteria", [])],
                    )
                    if parsed.section_type:
                        if not parsed.acceptance_criteria:
                            parsed.acceptance_criteria = self._default_acceptance_criteria(parsed.issue_type)
                        result.add_revision_task(parsed)
                except Exception:
                    continue

        if not result.revision_tasks:
            # Deterministic fallback task extraction.
            for idx, sf in enumerate(result.section_feedbacks):
                result.add_revision_task(RevisionTask(
                    task_id=f"rt_fallback_{result.iteration}_{idx}",
                    section_type=sf.section_type,
                    level=FeedbackLevel.PARAGRAPH if sf.target_paragraphs else FeedbackLevel.SECTION,
                    target_id=sf.target_id or sf.section_type,
                    paragraph_indices=sf.target_paragraphs or [],
                    action=sf.action or "revise",
                    priority=8 if sf.action in ("fix_latex", "logic_fix") else 6,
                    rationale=sf.revision_prompt[:240],
                    instruction=sf.revision_prompt,
                    preserve_claims=["Preserve factual claims and citations"],
                    do_not_change=["Do not introduce unsupported claims"],
                    expected_change="Address reviewer issues with minimal regression",
                    source_agents=["reviewer"],
                    source_checkers=list({
                        hf.checker for hf in result.hierarchical_feedbacks
                        if hf.section_type == sf.section_type and hf.checker
                    }),
                    issue_type=IssueType.OTHER,
                    acceptance_criteria=self._default_acceptance_criteria(IssueType.OTHER),
                ))

        result.orchestrator_summary = str(llm_output.get("summary", "")).strip() if isinstance(llm_output, dict) else ""
        if not result.orchestrator_summary:
            result.orchestrator_summary = (
                f"Built {len(result.revision_tasks)} executable revision task(s) from checker evidence."
            )

    async def verify_execution(
        self,
        section_type: str,
        task_contract: Dict[str, Any],
        before_text: str,
        after_text: str,
        semantic_passed: bool,
        semantic_summary: str,
    ) -> Dict[str, Any]:
        """
        Build reviewer-side acceptance conclusion for one revision task.
        - **Description**:
            - Reviewer owns acceptance judgment; Writer only returns execution receipts
            - Semantic consistency is treated as a hard acceptance gate
        """
        changed = (before_text or "").strip() != (after_text or "").strip()
        expected_change = str(task_contract.get("expected_change", "")).strip().lower()
        allow_noop = bool(task_contract.get("allow_noop", False)) or expected_change in {
            "",
            "none",
            "no_change",
            "no-op",
            "already_satisfied",
        }
        resolved_or_not_applicable = bool(semantic_passed and (changed or allow_noop))
        passed = bool(resolved_or_not_applicable)
        criteria = [str(x) for x in (task_contract.get("acceptance_criteria", []) or [])]
        gate_map: Dict[str, bool] = {
            "execution_changed": changed,
            "semantic_preserved": bool(semantic_passed),
            "contradiction_resolved": bool(semantic_passed and changed),
            "evidence_sufficient": bool(semantic_passed and changed),
            "structure_coherent": bool(semantic_passed and changed),
            "resolved_or_not_applicable": resolved_or_not_applicable,
        }
        gate_results = [
            {
                "gate": gate,
                "passed": bool(gate_map.get(gate, True)),
            }
            for gate in criteria
        ]
        if criteria:
            passed = all(bool(item.get("passed", False)) for item in gate_results)
        return {
            "section_type": section_type,
            "target": str(task_contract.get("target") or section_type),
            "instruction": str(task_contract.get("instruction") or ""),
            "constraints": task_contract.get("constraints", {}),
            "passed": passed,
            "changed": changed,
            "semantic_passed": bool(semantic_passed),
            "acceptance_criteria": criteria,
            "acceptance_gates": gate_results,
            "summary": (
                "Accepted: acceptance criteria satisfied."
                if passed
                else "Rejected: acceptance criteria not satisfied."
            ),
            "reason": semantic_summary,
            "source_agent": "reviewer",
            "source_stage": "reviewer_verification",
        }

    @staticmethod
    def _default_acceptance_criteria(issue_type: IssueType) -> List[str]:
        """
        Return default acceptance gates by issue type.
        - **Description**:
            - Provides generalized, reusable verification gates
            - Keeps orchestration robust even when LLM omits criteria
        """
        base = ["semantic_preserved", "resolved_or_not_applicable"]
        if issue_type == IssueType.LOGICAL_CONTRADICTION:
            return base + ["execution_changed", "contradiction_resolved"]
        if issue_type in (IssueType.CLAIM_EVIDENCE_GAP, IssueType.UNSUPPORTED_GENERALIZATION):
            return base + ["execution_changed", "evidence_sufficient"]
        if issue_type == IssueType.STRUCTURE_QUALITY:
            return base + ["execution_changed", "structure_coherent"]
        return base

    @staticmethod
    def _coerce_issue_type(raw: Any) -> IssueType:
        """Convert raw issue type string to canonical enum with fallback."""
        value = str(raw or "").strip().lower()
        aliases = {
            "logic": IssueType.LOGICAL_CONTRADICTION,
            "logic_check": IssueType.LOGICAL_CONTRADICTION,
            "style": IssueType.STYLE_NOISE,
            "style_check": IssueType.STYLE_NOISE,
            "fix_latex": IssueType.LATEX_FORMAT,
            "layout": IssueType.LAYOUT_CONSTRAINT,
            "structure": IssueType.STRUCTURE_QUALITY,
            "structure_check": IssueType.STRUCTURE_QUALITY,
        }
        if value in aliases:
            return aliases[value]
        try:
            return IssueType(value)
        except Exception:
            return IssueType.OTHER
    
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
