"""
MetaData Agent - Simple Mode Paper Generation
- **Description**:
    - Generates complete papers from simplified MetaData input
    - Multi-phase generation with persistent ReferencePool:
        0. Planning - creates detailed paper plan
        1. Introduction (Leader) - sets tone, extracts contributions
        2. Body Sections - Method, Experiment, Results, Related Work
        3. Synthesis Sections - Abstract and Conclusion from prior sections
        3.5. Review Loop - iterative feedback and revision
        4. PDF Compilation - via Typesetter Agent
    - Two-phase content generation pattern:
        - Phase A (Judgment + Search): LLM judges whether the section needs
          additional references; if yes, PaperSearchTool is called system-side
          and results are merged into the ReferencePool before writing.
        - Phase B (Pure Writing): LLM generates content with all available
          refs (core + discovered) in the prompt, no tools attached.
    - Fixed-sequence review:
        - Mini-review executes citation validation, word count, and key point
          checks in deterministic order (Type 2 tools).
    - ReferencePool accumulates references across all phases:
        user's core refs + discovered refs -> final .bib
    - Independent API, no frontend dependency
"""
import asyncio
import json
import re
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path

import httpx
from pydantic import BaseModel
from fastapi import APIRouter

from ..react_base import ReActAgent
from ...config.schema import ModelConfig, ToolsConfig
from ..shared.reference_pool import ReferencePool
from .models import (
    PaperMetaData,
    PaperGenerationRequest,
    PaperGenerationResult,
    SectionResult,
    SectionGenerationRequest,
    BODY_SECTION_SOURCES,
    SYNTHESIS_SECTIONS,
    DEFAULT_SECTION_ORDER,
    CodeRepoOnError,
)
from ..shared.prompt_compiler import (
    compile_introduction_prompt,
    compile_body_section_prompt,
    compile_synthesis_prompt,
    extract_contributions_from_intro,
    SECTION_PROMPTS,
)
from ..writer_agent.section_models import (
    ReferenceInfo,
    SimpleSectionInput,
    SynthesisSectionInput,
)
from ..planner_agent.models import (
    PaperPlan,
    SectionPlan,
    PlanRequest,
    calculate_total_words,
)
from ..shared.table_converter import convert_tables
from ..shared.session_memory import SessionMemory, ReviewRecord
from ..shared.code_context import (
    CodeContextBuilder,
    format_code_context_for_prompt,
    format_code_context_for_planner,
    render_code_repository_summary_markdown,
)
from ..reviewer_agent.models import (
    ReviewResult,
    FeedbackResult,
    Severity,
    SectionFeedback,
    RevisionTask,
    ConflictResolutionRecord,
    SemanticCheckRecord,
)
from .models import FigureSpec, TableSpec, StructuralAction, SpaceEstimate


# System prompt for pure content generation (no tools attached)
GENERATION_SYSTEM_PROMPT = """\
You are an expert academic writer specializing in research paper composition.
Use present tense for methods, no contractions (it is, do not, cannot),
no possessives on method names (the performance of X, not X's performance).
Place key information at sentence end. Output pure LaTeX only.

CITATION RULES:
- Use ONLY the citation keys listed in the provided references.
- NEVER invent or hallucinate citation keys.
- If a claim needs a reference but none of the provided keys fit, omit the \\cite command.

OUTPUT FORMAT:
Return ONLY the LaTeX content for the section. Do not include explanations outside the LaTeX."""


# System prompt for pre-generation search judgment
SEARCH_JUDGMENT_PROMPT = """\
You are an academic research assistant. Your task is to analyze whether the existing \
references are sufficient for writing a specific section of a research paper, or whether \
additional references should be searched.

Given the following section context, analyze whether additional references are needed.

SECTION: {section_type} ({section_title})
PAPER TITLE: {paper_title}
KEY POINTS TO COVER:
{key_points}

EXISTING REFERENCES ({n_refs} papers):
{ref_summaries}

RULES:
- If existing references adequately cover the section's claims, set need_search to false.
- If there are gaps (e.g., missing baselines, no empirical evidence for a claim, \
a sub-topic with zero refs), set need_search to true.
- Provide 1-3 focused search queries using specific keywords, not broad topics.
  Good: "R&D tax credit firm productivity panel data"
  Bad: "innovation economics"
- Each query should target a specific gap you identified.

Respond with ONLY a JSON object, no other text:
{{"need_search": true/false, "reason": "...", "queries": ["...", "..."]}}\
"""


class MetaDataAgent(ReActAgent):
    """
    MetaData Agent for simple-mode paper generation

    - **Description**:
        - Inherits from ReActAgent for access to react_loop and setup_tools.
        - Accepts 5 natural language fields + BibTeX references.
        - Manages a persistent ReferencePool throughout paper generation:
            - User's core references (~5 papers) initialize the pool.
            - During content generation, LLM may call search_papers (ReAct)
              to discover additional references.
            - Discovered papers undergo two-layer validation (LLM judgment +
              system cross-reference) before being added to the pool.
            - The pool's valid_citation_keys grows across phases.
        - Dual-mode tool invocation:
            - Type 1 (ReAct): _generate_introduction / _generate_body_section
              use react_loop with search_papers for autonomous reference search.
            - Type 2 (Delegated): WriterAgent handles iterative mini-review
              (citation validation, word count, key point coverage) internally.
        - Independent API, can be called directly via curl/Postman.
    """

    def __init__(self, config: ModelConfig, tools_config: Optional[ToolsConfig] = None):
        # Use default tools config if not provided
        if tools_config is None:
            tools_config = ToolsConfig(
                enabled=True,
                available_tools=[
                    "validate_citations",
                    "count_words",
                    "check_key_points",
                    "search_papers",
                ],
                max_react_iterations=3,
            )
        super().__init__(config, tools_config)
        self.results_dir = Path(__file__).parent.parent.parent.parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._router = self._create_router()
        # Skill registry — injected post-construction by agents/__init__.py
        self._skill_registry = None
        # Peer agent references — injected post-construction via set_peers()
        self._writer = None
        self._reviewer = None
        self._planner = None
        self._vlm_reviewer = None

    def set_peers(self, agents: Dict[str, "BaseAgent"]) -> None:
        """
        Inject references to peer agents for direct method calls.
        - **Description**:
            - Called after all agents are initialized in initialize_agents().
            - Enables MetaDataAgent to call WriterAgent, ReviewerAgent, etc.
              directly instead of via HTTP.

        - **Args**:
            - `agents` (Dict[str, BaseAgent]): The full agent dictionary.
        """
        self._writer = agents.get("writer")
        self._reviewer = agents.get("reviewer")
        self._planner = agents.get("planner")
        self._vlm_reviewer = agents.get("vlm_review")
    
    @property
    def name(self) -> str:
        """Agent name identifier"""
        return "metadata"
    
    @property
    def description(self) -> str:
        """Agent description"""
        return "MetaData-based paper generation (Simple Mode) - generates complete papers from 5 natural language fields + BibTeX references"
    
    @property
    def router(self) -> APIRouter:
        """Return the FastAPI router for this agent"""
        return self._router
    
    @property
    def endpoints_info(self) -> List[Dict[str, Any]]:
        """Return endpoint metadata for list_agents"""
        return [
            {
                "path": "/metadata/generate",
                "method": "POST",
                "description": "Generate complete paper from MetaData (5 fields + references)",
            },
            {
                "path": "/metadata/generate/section",
                "method": "POST",
                "description": "Generate a single section (for debugging or incremental generation)",
            },
            {
                "path": "/metadata/health",
                "method": "GET",
                "description": "Health check endpoint",
            },
            {
                "path": "/metadata/schema",
                "method": "GET",
                "description": "Get input schema for paper generation",
            },
        ]
    
    def _create_router(self) -> APIRouter:
        """Create FastAPI router for this agent"""
        from .router import create_metadata_router
        return create_metadata_router(self)

    def _get_active_skills(self, section_type: str, style_guide: str = None):
        """
        Retrieve active writing skills from the skill registry.

        - **Args**:
            - `section_type` (str): Current section being generated
            - `style_guide` (str, optional): Venue name for venue-profile matching

        - **Returns**:
            - `list` of WritingSkill or None
        """
        if self._skill_registry is None or len(self._skill_registry) == 0:
            return None
        # Runtime decision: select skills using the current request's style_guide.
        # This avoids binding venue profiles at service startup time.
        return self._skill_registry.get_writing_skills(
            section_type=section_type,
            venue=style_guide,
        )

    async def _build_code_repository_context(
        self,
        metadata: PaperMetaData,
    ) -> Optional[Dict[str, Any]]:
        """
        Build code repository context for section-aware writing.
        - **Description**:
            - Resolves and scans optional code repository input.
            - Produces repository overview, section evidence packs, and index.

        - **Args**:
            - `metadata` (PaperMetaData): User metadata payload.

        - **Returns**:
            - `Dict[str, Any] | None`: Built context or None when not provided.
        """
        if not metadata.code_repository:
            return None

        builder = CodeContextBuilder(workspace_root=str(Path.cwd()))
        return await builder.build(
            code_repo=metadata.code_repository,
            paper_title=metadata.title,
        )

    def _retrieve_runtime_code_evidence(
        self,
        code_context: Optional[Dict[str, Any]],
        section_type: str,
        metadata: PaperMetaData,
        contributions: Optional[List[str]] = None,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve section-specific runtime evidence as fallback.
        - **Description**:
            - Uses lightweight query bundles per section.
            - Runs before writing to enrich prompt context.

        - **Args**:
            - `code_context` (Dict | None): Built repository context.
            - `section_type` (str): Target section.
            - `metadata` (PaperMetaData): User metadata.
            - `contributions` (List[str], optional): Known contributions.
            - `top_k` (int): Max snippets to return.

        - **Returns**:
            - `List[Dict]`: Retrieved evidence list.
        """
        if not code_context:
            return []

        query_bundle: List[str] = []
        if section_type == "method":
            query_bundle = ["algorithm", "model", "module", "implementation", "forward", "pipeline"]
            query_bundle.extend(metadata.method.split()[:12])
        elif section_type == "experiment":
            query_bundle = ["train", "eval", "dataset", "metric", "configuration", "ablation"]
            query_bundle.extend(metadata.data.split()[:12])
            query_bundle.extend(metadata.experiments.split()[:12])
        elif section_type == "result":
            query_bundle = ["result", "analysis", "benchmark", "compare", "metric", "report"]
            query_bundle.extend(metadata.experiments.split()[:12])
        else:
            query_bundle = ["project", "workflow", "pipeline"]
            query_bundle.extend(metadata.idea_hypothesis.split()[:10])

        if contributions:
            query_bundle.extend(" ".join(contributions[:2]).split()[:8])

        builder = CodeContextBuilder(workspace_root=str(Path.cwd()))
        return builder.retrieve_for_section(
            context=code_context,
            section_type=section_type,
            query_bundle=query_bundle,
            top_k=top_k,
        )

    def _format_research_context_for_prompt(
        self,
        research_context: Optional[Dict[str, Any]],
        section_type: str,
    ) -> str:
        """
        Format research context into a compact writer-consumable brief.
        """
        if not research_context:
            return ""

        lines: List[str] = ["## Research Context Brief"]
        area = str(research_context.get("research_area", "")).strip()
        summary = str(research_context.get("summary", "")).strip()
        if area:
            lines.append(f"- Area: {area}")
        if summary:
            lines.append(f"- Landscape: {summary}")

        trends = research_context.get("research_trends", []) or []
        if trends:
            lines.append("- Trends:")
            for t in trends[:3]:
                lines.append(f"  - {t}")

        gaps = research_context.get("gaps", []) or []
        if gaps:
            lines.append("- Gaps to address:")
            for g in gaps[:3]:
                lines.append(f"  - {g}")

        key_papers = research_context.get("key_papers", []) or []
        if key_papers:
            lines.append("- Key papers and why they matter:")
            for kp in key_papers[:5]:
                title = kp.get("title", "")
                contribution = kp.get("contribution", "")
                if title:
                    lines.append(f"  - {title}: {contribution}")

        claim_matrix = research_context.get("claim_evidence_matrix", []) or []
        section_claims = [
            c for c in claim_matrix
            if c.get("section_type") in {section_type, "global", "", None}
        ]
        if section_claims:
            lines.append("- Claim-evidence priorities:")
            for c in section_claims[:6]:
                claim = c.get("claim", "")
                refs = c.get("support_refs", []) or []
                priority = c.get("priority", "")
                reason = c.get("reason", "")
                ref_text = ", ".join(refs[:4]) if refs else "none"
                lines.append(
                    f"  - [{priority}] Claim: {claim} | Evidence refs: {ref_text} | Why: {reason}"
                )

        ranking = research_context.get("contribution_ranking", {}) or {}
        if ranking:
            lines.append("- Contribution ranking:")
            for band in ("P0", "P1", "P2"):
                items = ranking.get(band, []) or []
                if not items:
                    continue
                for item in items[:3]:
                    contribution = item.get("contribution", "")
                    why = item.get("why_it_matters", "")
                    sections = item.get("suggested_sections", []) or []
                    section_hint = ", ".join(sections[:3]) if sections else "n/a"
                    lines.append(
                        f"  - {band}: {contribution} | Why: {why} | Suggested sections: {section_hint}"
                    )

        if len(lines) <= 1:
            return ""
        return "\n".join(lines)

    def _collect_section_citation_budget_usage(
        self,
        *,
        section_type: str,
        content: str,
        section_plan: Optional[SectionPlan],
        writer_valid_keys: List[str],
    ) -> Optional[Dict[str, Any]]:
        """
        Collect soft-cap citation usage stats for one section.
        """
        if not section_plan:
            return None

        valid_set = set(writer_valid_keys or [])
        _, _, used_keys = self._validate_and_fix_citations(
            content,
            valid_set,
            remove_invalid=False,
        )
        selected_refs = list(
            section_plan.budget_selected_refs
            or section_plan.assigned_refs
            or []
        )
        selected_set = set(selected_refs)
        used_budget_keys = [k for k in used_keys if k in selected_set]
        overflow_keys = [k for k in used_keys if k not in selected_set]
        budget = section_plan.citation_budget or {}
        return {
            "section_type": section_type,
            "min_refs": budget.get("min_refs"),
            "target_refs": budget.get("target_refs"),
            "max_refs": budget.get("max_refs"),
            "selected_refs": selected_refs,
            "reserve_refs": list(section_plan.budget_reserve_refs or []),
            "writer_valid_keys_count": len(valid_set),
            "used_keys": used_keys,
            "used_count": len(used_keys),
            "used_budget_keys": used_budget_keys,
            "used_budget_count": len(used_budget_keys),
            "overflow_keys": overflow_keys,
            "overflow_count": len(overflow_keys),
        }

    @staticmethod
    def _upsert_section_budget_usage(
        usage_rows: List[Dict[str, Any]],
        usage_row: Optional[Dict[str, Any]],
    ) -> None:
        """
        Upserts one section citation usage row by section_type.
        - **Description**:
         - Updates existing section stats if present, otherwise appends a new row.
         - Ensures one section appears once in exported citation_budget_usage.

        - **Args**:
         - `usage_rows` (List[Dict[str, Any]]): Collected usage rows.
         - `usage_row` (Optional[Dict[str, Any]]): New usage row to merge.

        - **Returns**:
         - `None` (None): In-place update only.
        """
        if not usage_row:
            return
        section_type = usage_row.get("section_type")
        if not section_type:
            return
        for idx, row in enumerate(usage_rows):
            if row.get("section_type") == section_type:
                usage_rows[idx] = usage_row
                return
        usage_rows.append(usage_row)

    @staticmethod
    def _build_citation_plan_alignment_stats(
        paper_plan: Optional[PaperPlan],
        usage_rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Builds citation alignment stats between plan and final manuscript.
        - **Description**:
         - Compares planned citation budget and selected keys with actual used keys.
         - Produces per-section and overall metrics for citation usage auditing.

        - **Args**:
         - `paper_plan` (Optional[PaperPlan]): Planner output containing citation budgets.
         - `usage_rows` (List[Dict[str, Any]]): Real usage extracted from final section content.

        - **Returns**:
         - `stats` (Dict[str, Any]): Aggregated alignment report payload.
        """
        usage_by_section: Dict[str, Dict[str, Any]] = {
            str(r.get("section_type")): r
            for r in usage_rows
            if r.get("section_type")
        }

        per_section: List[Dict[str, Any]] = []
        total_selected = 0
        total_used = 0
        total_used_budget = 0
        total_overflow = 0
        total_missing = 0
        sections_with_budget = 0
        sections_meeting_min = 0

        for sp in (paper_plan.sections if paper_plan else []):
            budget = sp.citation_budget or {}
            selected_refs = list(sp.budget_selected_refs or sp.assigned_refs or [])
            selected_set = set(selected_refs)
            usage = usage_by_section.get(sp.section_type, {})
            used_keys = list(usage.get("used_keys", []) or [])
            used_budget_keys = list(usage.get("used_budget_keys", []) or [])
            overflow_keys = list(usage.get("overflow_keys", []) or [])
            missing_selected_refs = [k for k in selected_refs if k not in set(used_budget_keys)]

            min_refs = int(budget.get("min_refs") or 0)
            target_refs = int(budget.get("target_refs") or 0)
            max_refs = int(budget.get("max_refs") or 0)

            used_count = len(used_keys)
            used_budget_count = len(used_budget_keys)
            overflow_count = len(overflow_keys)
            selected_count = len(selected_refs)
            coverage_rate = (
                round(used_budget_count / selected_count, 4) if selected_count > 0 else None
            )
            min_met = (used_count >= min_refs) if min_refs > 0 else True
            target_met = (used_count >= target_refs) if target_refs > 0 else True
            max_exceeded = (used_count > max_refs) if max_refs > 0 else False

            if selected_count > 0:
                sections_with_budget += 1
            if min_met:
                sections_meeting_min += 1

            total_selected += selected_count
            total_used += used_count
            total_used_budget += used_budget_count
            total_overflow += overflow_count
            total_missing += len(missing_selected_refs)

            per_section.append(
                {
                    "section_type": sp.section_type,
                    "plan": {
                        "min_refs": min_refs,
                        "target_refs": target_refs,
                        "max_refs": max_refs,
                        "selected_count": selected_count,
                        "selected_refs": selected_refs,
                    },
                    "final": {
                        "used_count": used_count,
                        "used_keys": used_keys,
                        "used_budget_count": used_budget_count,
                        "used_budget_keys": used_budget_keys,
                        "overflow_count": overflow_count,
                        "overflow_keys": overflow_keys,
                    },
                    "delta": {
                        "coverage_rate": coverage_rate,
                        "missing_selected_count": len(missing_selected_refs),
                        "missing_selected_refs": missing_selected_refs,
                        "extra_non_budget_count": overflow_count,
                    },
                    "status": {
                        "min_met": min_met,
                        "target_met": target_met,
                        "max_exceeded": max_exceeded,
                    },
                }
            )

        overall = {
            "sections_total": len(per_section),
            "sections_with_budget": sections_with_budget,
            "sections_meeting_min": sections_meeting_min,
            "total_selected_refs": total_selected,
            "total_used_refs": total_used,
            "total_used_budget_refs": total_used_budget,
            "total_overflow_refs": total_overflow,
            "total_missing_selected_refs": total_missing,
            "overall_budget_coverage_rate": (
                round(total_used_budget / total_selected, 4) if total_selected > 0 else None
            ),
        }

        return {
            "overall": overall,
            "sections": per_section,
        }

    def _build_structure_alignment_stats(
        self,
        *,
        paper_plan: Optional[PaperPlan],
        generated_sections: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Build plan-vs-final structure alignment statistics.
        """
        threshold = int(
            getattr(self.tools_config, "structure_gate_min_paragraph_threshold", 5)
        ) if self.tools_config else 5
        per_section: List[Dict[str, Any]] = []
        gate_expected_count = 0
        gate_pass_count = 0

        for sp in (paper_plan.sections if paper_plan else []):
            final_content = generated_sections.get(sp.section_type, "") or ""
            paragraphs = [
                p.strip() for p in re.split(r"\n\s*\n", final_content) if p.strip()
            ]
            paragraph_count = len(paragraphs)
            subsection_count = len(
                re.findall(r"\\subsection\{.+?\}|\\subsubsection\{.+?\}", final_content)
            )
            transition_count = 0
            for para in paragraphs[1:]:
                lower = para.lower()
                if lower.startswith((
                    "however", "therefore", "in contrast", "meanwhile",
                    "moreover", "furthermore", "additionally", "by contrast",
                    "in summary",
                )):
                    transition_count += 1

            expected_gate = bool(
                sp.sectioning_recommended
                or paragraph_count >= threshold
            ) and sp.section_type not in {"abstract", "conclusion"}
            passed_gate = bool(
                subsection_count > 0
                or (paragraph_count >= 4 and transition_count >= 1)
            )
            if expected_gate:
                gate_expected_count += 1
                if passed_gate:
                    gate_pass_count += 1

            per_section.append(
                {
                    "section_type": sp.section_type,
                    "plan": {
                        "paragraph_count": len(sp.paragraphs or []),
                        "topic_clusters": list(sp.topic_clusters or []),
                        "transition_intents": list(sp.transition_intents or []),
                        "sectioning_recommended": bool(sp.sectioning_recommended),
                    },
                    "final": {
                        "paragraph_count": paragraph_count,
                        "explicit_subsection_count": subsection_count,
                        "transition_marker_count": transition_count,
                    },
                    "status": {
                        "structure_gate_expected": expected_gate,
                        "structure_gate_passed": passed_gate if expected_gate else None,
                    },
                }
            )

        return {
            "overall": {
                "sections_total": len(per_section),
                "gate_expected_sections": gate_expected_count,
                "gate_passed_sections": gate_pass_count,
                "gate_pass_rate": (
                    round(gate_pass_count / gate_expected_count, 4)
                    if gate_expected_count > 0
                    else None
                ),
            },
            "sections": per_section,
        }

    def _build_paragraph_feedback_alignment_report(
        self,
        *,
        memory: Optional[SessionMemory],
        generated_sections: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Build paragraph feedback alignment report against final section paragraphs.
        """
        if memory is None or not getattr(memory, "review_history", None):
            return {"sections": [], "overall": {"records": 0}}

        latest = memory.review_history[-1]
        rows: List[Dict[str, Any]] = []
        total_targets = 0
        total_mapped = 0
        total_out_of_range = 0

        for section_type, feedback in (latest.section_feedbacks or {}).items():
            if not isinstance(feedback, dict):
                continue
            target_paragraphs = [
                int(x) for x in (feedback.get("target_paragraphs", []) or [])
                if str(x).strip().lstrip("-").isdigit()
            ]
            final_paragraphs = self._split_section_paragraphs(
                generated_sections.get(section_type, "") or ""
            )
            final_count = len(final_paragraphs)
            mapped = []
            out_of_range = []
            for pidx in target_paragraphs:
                if 0 <= pidx < final_count:
                    mapped.append({"from": pidx, "to": [pidx], "strategy": "identity"})
                elif final_count > 0:
                    nearest = min(max(pidx, 0), final_count - 1)
                    mapped.append({"from": pidx, "to": [nearest], "strategy": "clamped_nearest"})
                    out_of_range.append(pidx)
                else:
                    mapped.append({"from": pidx, "to": [], "strategy": "no_paragraphs"})
                    out_of_range.append(pidx)

            total_targets += len(target_paragraphs)
            total_mapped += len([m for m in mapped if m.get("to")])
            total_out_of_range += len(out_of_range)

            rows.append(
                {
                    "section_type": section_type,
                    "target_paragraphs": target_paragraphs,
                    "final_paragraph_count": final_count,
                    "mappings": mapped,
                    "out_of_range_targets": out_of_range,
                }
            )

        return {
            "overall": {
                "records": len(rows),
                "total_targets": total_targets,
                "mapped_targets": total_mapped,
                "out_of_range_targets": total_out_of_range,
            },
            "sections": rows,
        }

    def _rebuild_citation_budget_usage_from_final_sections(
        self,
        *,
        paper_plan: Optional[PaperPlan],
        generated_sections: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Rebuild citation budget usage from final post-review section contents.
        - **Description**:
         - Extracts citation usage from final manuscript sections after review iterations.
         - Ensures exported usage/alignment reflects the final delivered content.

        - **Args**:
         - `paper_plan` (Optional[PaperPlan]): Planner output containing section plans.
         - `generated_sections` (Dict[str, str]): Final section content map.

        - **Returns**:
         - `usage_rows` (List[Dict[str, Any]]): Final per-section citation usage rows.
        """
        if not paper_plan:
            return []

        usage_rows: List[Dict[str, Any]] = []
        for sp in paper_plan.sections:
            final_content = generated_sections.get(sp.section_type, "")
            writer_valid_keys = list(
                dict.fromkeys(
                    list(sp.assigned_refs or [])
                    + list(sp.budget_reserve_refs or [])
                )
            )
            usage_row = self._collect_section_citation_budget_usage(
                section_type=sp.section_type,
                content=final_content,
                section_plan=sp,
                writer_valid_keys=writer_valid_keys,
            )
            self._upsert_section_budget_usage(usage_rows, usage_row)
        return usage_rows

    def _build_reviewer_acceptance_stats(
        self,
        *,
        memory: Optional[SessionMemory],
    ) -> Dict[str, Any]:
        """
        Build acceptance statistics from reviewer verification records.
        """
        if memory is None or not getattr(memory, "review_history", None):
            return {"overall": {"total": 0}, "by_iteration": []}

        by_iteration: List[Dict[str, Any]] = []
        total = 0
        passed = 0
        failed = 0
        noop_accepted = 0
        changed_accepted = 0

        for rec in memory.review_history:
            verifications = list(getattr(rec, "reviewer_verification", []) or [])
            iter_total = len(verifications)
            iter_passed = 0
            iter_failed = 0
            iter_noop_accepted = 0
            iter_changed_accepted = 0
            for v in verifications:
                if not isinstance(v, dict):
                    continue
                ok = bool(v.get("passed", False))
                changed_flag = bool(v.get("changed", False))
                if ok:
                    iter_passed += 1
                    if changed_flag:
                        iter_changed_accepted += 1
                    else:
                        iter_noop_accepted += 1
                else:
                    iter_failed += 1

            total += iter_total
            passed += iter_passed
            failed += iter_failed
            noop_accepted += iter_noop_accepted
            changed_accepted += iter_changed_accepted
            by_iteration.append(
                {
                    "iteration": int(getattr(rec, "iteration", 0)),
                    "total": iter_total,
                    "passed": iter_passed,
                    "failed": iter_failed,
                    "noop_accepted": iter_noop_accepted,
                    "changed_accepted": iter_changed_accepted,
                }
            )

        return {
            "overall": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "pass_rate": round(passed / total, 4) if total > 0 else None,
                "noop_accepted": noop_accepted,
                "changed_accepted": changed_accepted,
            },
            "by_iteration": by_iteration,
        }

    def _build_citation_repair_stats(
        self,
        *,
        memory: Optional[SessionMemory],
    ) -> Dict[str, Any]:
        """
        Build invalid-citation removal statistics from decision traces.
        """
        if memory is None or not getattr(memory, "review_history", None):
            return {"overall": {"removed_total": 0}, "by_section": {}, "events": []}

        by_section: Dict[str, int] = {}
        events: List[Dict[str, Any]] = []
        removed_total = 0
        for rec in memory.review_history:
            trace_rows = list(getattr(rec, "decision_trace", []) or [])
            for row in trace_rows:
                if not isinstance(row, dict):
                    continue
                if str(row.get("decision", "")) != "removed_invalid_citations":
                    continue
                section_type = str(row.get("section_type", "unknown"))
                count = int(row.get("count", 0) or 0)
                removed_total += count
                by_section[section_type] = by_section.get(section_type, 0) + count
                events.append(
                    {
                        "iteration": int(getattr(rec, "iteration", 0)),
                        "section_type": section_type,
                        "count": count,
                        "keys": list(row.get("keys", []) or []),
                    }
                )
        return {
            "overall": {
                "removed_total": removed_total,
                "events": len(events),
            },
            "by_section": by_section,
            "events": events,
        }

    def _build_explicit_subsection_coverage(
        self,
        *,
        paper_plan: Optional[PaperPlan],
        generated_sections: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Build explicit subsection coverage for recommended sections.
        """
        rows: List[Dict[str, Any]] = []
        recommended_total = 0
        recommended_with_explicit = 0
        for sp in (paper_plan.sections if paper_plan else []):
            if sp.section_type in {"abstract", "conclusion"}:
                continue
            if not bool(sp.sectioning_recommended):
                continue
            recommended_total += 1
            content = generated_sections.get(sp.section_type, "") or ""
            explicit_count = len(
                re.findall(r"\\subsection\{.+?\}|\\subsubsection\{.+?\}", content)
            )
            has_explicit = explicit_count > 0
            if has_explicit:
                recommended_with_explicit += 1
            rows.append(
                {
                    "section_type": sp.section_type,
                    "section_title": sp.section_title,
                    "explicit_subsection_count": explicit_count,
                    "has_explicit_subsection": has_explicit,
                }
            )
        return {
            "overall": {
                "recommended_sections": recommended_total,
                "recommended_with_explicit_subsection": recommended_with_explicit,
                "coverage_rate": (
                    round(recommended_with_explicit / recommended_total, 4)
                    if recommended_total > 0
                    else None
                ),
            },
            "sections": rows,
        }

    def _build_core_refs_for_research_context(
        self,
        ref_pool: ReferencePool,
    ) -> List[Dict[str, Any]]:
        """
        Convert core references into research-context paper records.
        """
        rows: List[Dict[str, Any]] = []
        for ref in ref_pool.core_refs:
            ref_id = ref.get("ref_id", "")
            if not ref_id:
                continue
            rows.append(
                {
                    "ref_id": ref_id,
                    "bibtex": ref.get("bibtex", ""),
                    "title": ref.get("title", ""),
                    "year": ref.get("year"),
                    "abstract": ref.get("abstract", ""),
                    "venue": ref.get("venue", ""),
                    "citation_count": ref.get("citation_count"),
                    "source": ref.get("source", "core_reference"),
                }
            )
        return rows
    
    async def generate_paper(
        self,
        metadata: PaperMetaData,
        output_dir: Optional[str] = None,
        save_output: bool = True,
        compile_pdf: bool = True,
        template_path: Optional[str] = None,
        figures_source_dir: Optional[str] = None,
        target_pages: Optional[int] = None,
        enable_review: bool = True,
        max_review_iterations: int = 3,
        enable_planning: bool = True,
        enable_vlm_review: bool = False,
    ) -> PaperGenerationResult:
        """
        Generate complete paper from MetaData
        
        Seven-phase process:
        0. Planning - creates detailed paper plan (structure, word budgets, guidance)
        1. Introduction (Leader) - sets tone, extracts contributions
        2. Body Sections (can be parallel) - Method, Experiment, Results, Related Work
        3. Synthesis Sections - Abstract and Conclusion
        3.5. Review Loop - iterative feedback and revision
        4. PDF Compilation (if template provided) - via Typesetter Agent
        5. VLM Review (if enabled) - check page overflow and layout issues
        
        Args:
            metadata: Paper metadata with 5 fields + references
            output_dir: Directory for output files
            save_output: Whether to save output to disk
            compile_pdf: Whether to compile PDF (requires template_path)
            template_path: Path to .zip template file
            figures_source_dir: Directory containing figure files
            target_pages: Target page count (uses venue default if not set)
            enable_review: Whether to enable review loop
            max_review_iterations: Maximum number of review iterations
            enable_planning: Whether to create a paper plan before generation
            enable_vlm_review: Whether to run VLM-based PDF review after compilation
        """
        # Use template_path from metadata if not provided
        if template_path is None:
            template_path = metadata.template_path
        
        # Use target_pages from metadata if not provided
        if target_pages is None:
            target_pages = metadata.target_pages
        errors = []
        sections_results = []
        generated_sections: Dict[str, str] = {}
        paper_plan: Optional[PaperPlan] = None
        review_iterations = 0
        target_word_count = None
        research_context: Optional[Dict[str, Any]] = None
        research_context_v1: Optional[Dict[str, Any]] = None
        research_context_v2: Optional[Dict[str, Any]] = None
        discovered_references_by_section: Dict[str, List[Dict[str, Any]]] = {}
        seed_discovered_references: List[Dict[str, Any]] = []
        core_refs_for_context: List[Dict[str, Any]] = []
        citation_budget_usage: List[Dict[str, Any]] = []
        code_context: Optional[Dict[str, Any]] = None
        code_summary_markdown: Optional[str] = None
        prompt_traces: List[Dict[str, Any]] = []
        
        # Initialize Session Memory for cross-agent coordination
        memory = SessionMemory()
        memory.log("metadata", "init", "session_started",
                    narrative=f"Started paper generation session for '{metadata.title}' targeting {target_pages} pages.",
                    title=metadata.title, target_pages=target_pages)
        
        # Initialize persistent reference pool from user's core references.
        # Plain-text refs are resolved via search before falling back to heuristic.
        search_cfg_for_pool = {}
        if self.tools_config and self.tools_config.paper_search:
            ps = self.tools_config.paper_search
            search_cfg_for_pool = {
                "serpapi_api_key": ps.serpapi_api_key,
                "semantic_scholar_api_key": ps.semantic_scholar_api_key,
                "timeout": ps.timeout,
                "semantic_scholar_min_results_before_fallback": ps.semantic_scholar_min_results_before_fallback,
                "enable_query_cache": ps.enable_query_cache,
                "cache_ttl_hours": ps.cache_ttl_hours,
            }
        ref_pool = await ReferencePool.create(
            metadata.references,
            paper_search_config=search_cfg_for_pool,
        )
        print(f"[MetaDataAgent] Reference pool initialized: {ref_pool.summary()}")
        print(f"[MetaDataAgent] Core citation keys: {ref_pool.valid_citation_keys}")
        
        # Keep parsed_refs for backward-compatible methods that still need it
        parsed_refs = ref_pool.get_all_refs()
        
        # Validate figure/table file paths before proceeding
        validation_errors = self._validate_file_paths(metadata)
        if validation_errors:
            print("[MetaDataAgent] File validation errors:")
            for err in validation_errors:
                print(f"  - {err}")
            return PaperGenerationResult(
                status="error",
                paper_title=metadata.title,
                errors=validation_errors,
            )

        # Convert non-LaTeX figure formats (e.g. TIFF, BMP, WEBP) to PDF/PNG
        if metadata.figures:
            n_converted = self._convert_figures_for_latex(metadata)
            if n_converted:
                print(f"[MetaDataAgent] Converted {n_converted} figure(s) to LaTeX-compatible format")

        # Create output directory
        if save_output:
            if output_dir:
                paper_dir = Path(output_dir)
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_title = re.sub(r'[^\w\-]', '_', metadata.title)[:50]
                paper_dir = self.results_dir / f"{safe_title}_{timestamp}"
            paper_dir.mkdir(parents=True, exist_ok=True)
        else:
            paper_dir = None
        
        try:
            # =================================================================
            # Phase 0e-pre: Code repository understanding (optional, pre-plan)
            # =================================================================
            if metadata.code_repository:
                print("[MetaDataAgent] Phase 0e-pre: Building code repository context...")
                try:
                    code_context = await self._build_code_repository_context(metadata)
                    if code_context:
                        code_summary_markdown = render_code_repository_summary_markdown(code_context)
                        stats = code_context.get("scan_stats", {})
                        print(
                            "[MetaDataAgent] Code context ready (pre-plan): "
                            f"{stats.get('indexed_files', 0)} files indexed"
                        )
                except Exception as e:
                    on_error = metadata.code_repository.on_error
                    msg = f"Code repository ingestion failed: {e}"
                    print(f"[MetaDataAgent] Warning: {msg}")
                    if (
                        metadata.code_repository.type.value == "local_dir"
                        and isinstance(e, (FileNotFoundError, ValueError))
                    ):
                        return PaperGenerationResult(
                            status="error",
                            paper_title=metadata.title,
                            errors=[msg],
                        )
                    if on_error == CodeRepoOnError.STRICT:
                        return PaperGenerationResult(
                            status="error",
                            paper_title=metadata.title,
                            errors=[msg],
                        )
                    errors.append(msg)
                    code_context = None
                    code_summary_markdown = None

            # =================================================================
            # Phase 0: Planning (if enabled)
            # =================================================================
            if enable_planning:
                print(f"[MetaDataAgent] Phase 0: Creating Paper Plan...")
                search_cfg = {}
                if self.tools_config and self.tools_config.paper_search:
                    ps = self.tools_config.paper_search
                    search_cfg = {
                        "serpapi_api_key": ps.serpapi_api_key,
                        "semantic_scholar_api_key": ps.semantic_scholar_api_key,
                        "timeout": ps.timeout,
                        "search_results_per_round": ps.search_results_per_round,
                        "planner_max_queries_per_section": ps.planner_max_queries_per_section,
                        "planner_inter_round_delay_sec": ps.planner_inter_round_delay_sec,
                        "planner_min_target_papers_per_section": ps.planner_min_target_papers_per_section,
                        "semantic_scholar_min_results_before_fallback": ps.semantic_scholar_min_results_before_fallback,
                        "enable_query_cache": ps.enable_query_cache,
                        "cache_ttl_hours": ps.cache_ttl_hours,
                        "citation_budget_enabled": ps.citation_budget_enabled,
                        "citation_budget_soft_cap": ps.citation_budget_soft_cap,
                        "citation_budget_export": ps.citation_budget_export,
                        "citation_budget_reserve_size": ps.citation_budget_reserve_size,
                        "style_guide": metadata.style_guide,
                    }

                rc_cfg = (
                    self.tools_config.research_context
                    if self.tools_config and self.tools_config.research_context
                    else None
                )
                rc_first_enabled = rc_cfg.research_context_first_enabled if rc_cfg else False
                rc_enabled = rc_cfg.enabled if rc_cfg else False
                if rc_enabled and not core_refs_for_context:
                    core_refs_for_context = self._build_core_refs_for_research_context(ref_pool)
                    core_with_abstract = sum(
                        1 for r in core_refs_for_context if (r.get("abstract") or "").strip()
                    )
                    print(
                        "[MetaDataAgent] Core refs prepared for context: "
                        f"{len(core_refs_for_context)} total, {core_with_abstract} with abstract"
                    )

                # Phase 0a: build research context before planning decisions
                if rc_enabled and rc_first_enabled:
                    print("[MetaDataAgent] Phase 0a: Building research context v1 (pre-plan)...")
                    try:
                        core_refs_for_context = self._build_core_refs_for_research_context(ref_pool)
                        core_with_abstract = sum(
                            1 for r in core_refs_for_context if (r.get("abstract") or "").strip()
                        )
                        print(
                            "[MetaDataAgent] Core refs in pre-plan context: "
                            f"{len(core_refs_for_context)} total, {core_with_abstract} with abstract"
                        )
                        seed_discovered_references = await self._planner.discover_seed_references(
                            title=metadata.title,
                            idea_hypothesis=metadata.idea_hypothesis,
                            method=metadata.method,
                            data=metadata.data,
                            experiments=metadata.experiments,
                            existing_ref_keys=list(ref_pool.valid_citation_keys),
                            paper_search_config=search_cfg,
                        )
                        for paper in seed_discovered_references:
                            ref_pool.add_discovered(
                                paper.get("ref_id", ""),
                                paper.get("bibtex", ""),
                                source="planner_seed_discovery",
                            )
                        preplan_context_pool: List[Dict[str, Any]] = []
                        preplan_context_pool.extend(core_refs_for_context)
                        preplan_context_pool.extend(seed_discovered_references)
                        if preplan_context_pool:
                            seed_plan = PaperPlan(title=metadata.title, sections=[])
                            research_context_v1 = await self._planner._generate_research_context(
                                plan=seed_plan,
                                discovered={"global": preplan_context_pool},
                            )
                            research_context = research_context_v1
                            print(
                                "[MetaDataAgent] Research context v1 ready: "
                                f"{research_context_v1.get('research_area', 'N/A')}"
                            )
                    except Exception as e:
                        print(f"[MetaDataAgent] Warning: Failed to build research context v1: {e}")

                paper_plan = await self._create_paper_plan(
                    metadata=metadata,
                    target_pages=target_pages,
                    style_guide=metadata.style_guide,
                    research_context=research_context_v1,
                    code_context=code_context,
                )
                if paper_plan:
                    memory.plan = paper_plan
                    if self.tools_config and not getattr(
                        self.tools_config,
                        "planner_structure_signals_enabled",
                        True,
                    ):
                        for sp in paper_plan.sections:
                            sp.topic_clusters = []
                            sp.transition_intents = []
                            sp.sectioning_recommended = False
                    memory.log("planner", "phase0", "plan_created",
                               narrative=f"Planner created a {len(paper_plan.sections)}-section paper plan with ~{paper_plan.get_total_estimated_words()} estimated words.",
                               sections=len(paper_plan.sections),
                               estimated_words=paper_plan.get_total_estimated_words())
                    print(f"[MetaDataAgent] Plan created: {len(paper_plan.sections)} sections, ~{paper_plan.get_total_estimated_words()} words (est.)")
                    
                    # Apply auto-detected wide flags from plan to metadata figures/tables
                    if paper_plan.wide_figures:
                        print(f"[MetaDataAgent] Applying wide flag to figures: {paper_plan.wide_figures}")
                        for fig in metadata.figures:
                            if fig.id in paper_plan.wide_figures and not fig.wide:
                                fig.wide = True
                    
                    if paper_plan.wide_tables:
                        print(f"[MetaDataAgent] Applying wide flag to tables: {paper_plan.wide_tables}")
                        for tbl in metadata.tables:
                            if tbl.id in paper_plan.wide_tables and not tbl.wide:
                                tbl.wide = True
                    
                    # Save plan to output directory
                    if save_output and paper_dir:
                        planning_dir = paper_dir / "analysis" / "planning"
                        planning_dir.mkdir(parents=True, exist_ok=True)
                        plan_path = planning_dir / "paper_plan.json"
                        plan_path.write_text(
                            paper_plan.model_dump_json(indent=2),
                            encoding="utf-8",
                        )

                    # Reference discovery: Planner searches for relevant papers
                    print("[MetaDataAgent] Phase 0b: Discovering references...")
                    discovered = await self._planner.discover_references(
                        plan=paper_plan,
                        existing_ref_keys=list(ref_pool.valid_citation_keys),
                        paper_search_config=search_cfg,
                    )
                    discovered_references_by_section = discovered
                    disc_count = 0
                    for sec_type, papers in discovered.items():
                        for paper in papers:
                            added = ref_pool.add_discovered(
                                paper["ref_id"], paper["bibtex"], source="planner_discovery",
                            )
                            if added:
                                disc_count += 1
                    if disc_count:
                        print(f"[MetaDataAgent] Discovered {disc_count} new references (pool: {ref_pool.summary()})")
                        memory.log("planner", "phase0b", "references_discovered",
                                   narrative=f"Planner discovered {disc_count} additional references via academic search to support the paper plan.",
                                   count=disc_count)

                    # Phase 0c: Assign references to sections
                    print("[MetaDataAgent] Phase 0c: Assigning references to sections...")
                    self._planner.assign_references(
                        plan=paper_plan,
                        discovered=discovered,
                        core_ref_keys=list(ref_pool.valid_citation_keys
                                           - {p["ref_id"] for papers in discovered.values() for p in papers}),
                        paper_search_config=search_cfg,
                    )
                    for sp in paper_plan.sections:
                        if sp.assigned_refs:
                            print(f"  [{sp.section_type}] {len(sp.assigned_refs)} refs assigned")

                    # Phase 0d: Generate research context (if enabled)
                    if rc_enabled and discovered:
                        print("[MetaDataAgent] Phase 0d: Generating research context...")
                        try:
                            merged_discovered: Dict[str, List[Dict[str, Any]]] = dict(discovered)
                            if seed_discovered_references:
                                merged_discovered["global"] = seed_discovered_references
                            if core_refs_for_context:
                                merged_discovered["global_core"] = core_refs_for_context
                            research_context_v2 = await self._planner._generate_research_context(
                                plan=paper_plan,
                                discovered=merged_discovered,
                            )
                            if research_context_v2:
                                research_context = research_context_v2
                                print(
                                    "[MetaDataAgent] Research context v2 generated: "
                                    f"{research_context_v2.get('research_area', 'N/A')}"
                                )
                        except Exception as e:
                            print(f"[MetaDataAgent] Warning: Failed to generate research context: {e}")
                else:
                    print(f"[MetaDataAgent] Planning skipped or failed, using defaults")

            # =================================================================
            # Phase 0e: Code repository understanding (optional)
            # =================================================================
            if metadata.code_repository and not code_context:
                print("[MetaDataAgent] Phase 0e: Building code repository context...")
                try:
                    code_context = await self._build_code_repository_context(metadata)
                    if code_context:
                        code_summary_markdown = render_code_repository_summary_markdown(code_context)
                        stats = code_context.get("scan_stats", {})
                        print(
                            "[MetaDataAgent] Code context ready: "
                            f"{stats.get('indexed_files', 0)} files indexed"
                        )
                except Exception as e:
                    on_error = metadata.code_repository.on_error
                    msg = f"Code repository ingestion failed: {e}"
                    print(f"[MetaDataAgent] Warning: {msg}")
                    # Always fail fast when user provides a local directory but path is invalid.
                    # This avoids silently generating without code context when filesystem input is wrong.
                    if (
                        metadata.code_repository.type.value == "local_dir"
                        and isinstance(e, (FileNotFoundError, ValueError))
                    ):
                        return PaperGenerationResult(
                            status="error",
                            paper_title=metadata.title,
                            errors=[msg],
                        )
                    if on_error == CodeRepoOnError.STRICT:
                        return PaperGenerationResult(
                            status="error",
                            paper_title=metadata.title,
                            errors=[msg],
                        )
                    errors.append(msg)
                    code_context = None
                    code_summary_markdown = None
            
            # =================================================================
            # Phase 0.5: Convert Tables (if any)
            # =================================================================
            converted_tables: Dict[str, str] = {}
            if metadata.tables:
                print(f"[MetaDataAgent] Phase 0.5: Converting {len(metadata.tables)} tables...")
                # Determine base path for resolving file paths
                base_path = None
                if save_output and paper_dir:
                    base_path = str(paper_dir.parent)  # Parent of output dir
                
                converted_tables = await convert_tables(
                    tables=metadata.tables,
                    llm_client=self.client,
                    model_name=self.model_name,
                    base_path=base_path,
                )
                print(f"[MetaDataAgent] Converted {len(converted_tables)} tables to LaTeX")
            
            # =================================================================
            # Phase 1: Introduction (Leader Section)
            # =================================================================
            print(f"[MetaDataAgent] Phase 1: Generating Introduction...")
            intro_plan = paper_plan.get_section("introduction") if paper_plan else None
            intro_result = await self._generate_introduction(
                metadata, ref_pool, section_plan=intro_plan,
                figures=metadata.figures, tables=metadata.tables,
                code_context=code_context,
                research_context=research_context,
                prompt_traces=prompt_traces,
                memory=memory,
            )
            sections_results.append(intro_result)
            print(f"[MetaDataAgent] After introduction: {ref_pool.summary()}")
            
            if intro_result.status == "ok":
                generated_sections["introduction"] = intro_result.latex_content
                memory.log("metadata", "phase1", "introduction_generated",
                           narrative=f"Writer completed the introduction section ({intro_result.word_count} words).",
                           word_count=intro_result.word_count)
                if intro_plan:
                    intro_valid_keys = list(
                        dict.fromkeys(
                            list(intro_plan.assigned_refs or [])
                            + list(intro_plan.budget_reserve_refs or [])
                        )
                    )
                    intro_budget_usage = self._collect_section_citation_budget_usage(
                        section_type="introduction",
                        content=intro_result.latex_content,
                        section_plan=intro_plan,
                        writer_valid_keys=intro_valid_keys,
                    )
                    self._upsert_section_budget_usage(citation_budget_usage, intro_budget_usage)
                # Extract contributions for consistency
                contributions = extract_contributions_from_intro(intro_result.latex_content)
                if not contributions:
                    contributions = [
                        f"We propose {metadata.title}",
                        f"Novel approach: {metadata.method[:100]}...",
                    ]
            else:
                errors.append(f"Introduction generation failed: {intro_result.error}")
                contributions = []
                if memory is not None:
                    memory.log(
                        "metadata",
                        "phase1",
                        "introduction_failed",
                        narrative=f"Introduction failed and generation stopped: {intro_result.error}",
                        status="error",
                        error=intro_result.error,
                    )
                return PaperGenerationResult(
                    status="error",
                    paper_title=metadata.title,
                    sections=sections_results,
                    errors=errors,
                )
            
            # Use contributions from plan if available
            if paper_plan and paper_plan.contributions:
                contributions = paper_plan.contributions
            
            # Store contributions in memory for cross-agent coordination
            memory.contributions = contributions
            
            # =================================================================
            # Phase 2: Body Sections (can be parallel)
            # =================================================================
            print(f"[MetaDataAgent] Phase 2: Generating Body Sections...")
            # Dynamic: read body section types from the plan (no hardcoding)
            if paper_plan:
                body_section_types = paper_plan.get_body_section_types()
            else:
                body_section_types = ["related_work", "method", "experiment", "result"]
            print(f"[MetaDataAgent] Body sections from plan: {body_section_types}")
            
            # Generate body sections sequentially (ref_pool accumulates across sections)
            for section_type in body_section_types:
                section_plan = paper_plan.get_section(section_type) if paper_plan else None
                # Filter figures/tables for this section
                section_figures = [f for f in metadata.figures if f.section == section_type or not f.section]
                section_tables = [t for t in metadata.tables if t.section == section_type or not t.section]
                
                try:
                    result = await self._generate_body_section(
                        section_type=section_type,
                        metadata=metadata,
                        intro_context=generated_sections.get("introduction", ""),
                        contributions=contributions,
                        ref_pool=ref_pool,
                        section_plan=section_plan,
                        figures=section_figures,
                        tables=section_tables,
                        converted_tables=converted_tables,
                        code_context=code_context,
                        research_context=research_context,
                        prompt_traces=prompt_traces,
                        memory=memory,
                    )
                except Exception as e:
                    result = SectionResult(
                        section_type=section_type,
                        status="error",
                        error=str(e),
                    )
                
                sections_results.append(result)
                if result.status == "ok":
                    generated_sections[section_type] = result.latex_content
                    memory.log("metadata", "phase2", f"{section_type}_generated",
                               narrative=f"Writer completed the {section_type} section ({result.word_count} words).",
                               word_count=result.word_count)
                    if section_plan:
                        section_valid_keys = list(
                            dict.fromkeys(
                                list(section_plan.assigned_refs or [])
                                + list(section_plan.budget_reserve_refs or [])
                            )
                        )
                        section_budget_usage = self._collect_section_citation_budget_usage(
                            section_type=section_type,
                            content=result.latex_content,
                            section_plan=section_plan,
                            writer_valid_keys=section_valid_keys,
                        )
                        self._upsert_section_budget_usage(citation_budget_usage, section_budget_usage)
                    print(f"[MetaDataAgent] After {section_type}: {ref_pool.summary()}")
                else:
                    errors.append(f"{section_type} generation failed: {result.error}")
            
            # =================================================================
            # Phase 3: Synthesis Sections (Abstract + Conclusion)
            # =================================================================
            print(f"[MetaDataAgent] Phase 3: Generating Synthesis Sections...")
            
            # Generate Abstract
            abstract_result = await self._generate_synthesis_section(
                section_type="abstract",
                paper_title=metadata.title,
                prior_sections=generated_sections,
                contributions=contributions,
                style_guide=metadata.style_guide,
                section_plan=paper_plan.get_section("abstract") if paper_plan else None,
                prompt_traces=prompt_traces,
                memory=memory,
            )
            sections_results.insert(0, abstract_result)  # Abstract goes first
            if abstract_result.status == "ok":
                generated_sections["abstract"] = abstract_result.latex_content
                memory.log("metadata", "phase3", "abstract_generated",
                           narrative=f"Writer completed the abstract ({abstract_result.word_count} words).",
                           word_count=abstract_result.word_count)
            else:
                errors.append(f"Abstract generation failed: {abstract_result.error}")
            
            # Generate Conclusion only when planner includes it in the final structure.
            should_generate_conclusion = bool(
                paper_plan and paper_plan.get_section("conclusion") is not None
            )
            if should_generate_conclusion:
                conclusion_result = await self._generate_synthesis_section(
                    section_type="conclusion",
                    paper_title=metadata.title,
                    prior_sections=generated_sections,
                    contributions=contributions,
                    style_guide=metadata.style_guide,
                    section_plan=paper_plan.get_section("conclusion") if paper_plan else None,
                    prompt_traces=prompt_traces,
                    memory=memory,
                )
                sections_results.append(conclusion_result)
                if conclusion_result.status == "ok":
                    generated_sections["conclusion"] = conclusion_result.latex_content
                    memory.log("metadata", "phase3", "conclusion_generated",
                               narrative=f"Writer completed the conclusion ({conclusion_result.word_count} words).",
                               word_count=conclusion_result.word_count)
                else:
                    errors.append(f"Conclusion generation failed: {conclusion_result.error}")
            
            # =================================================================
            # Reference Usage Validation
            # =================================================================
            self._validate_ref_usage(generated_sections, ref_pool)

            # =================================================================
            # Unified Review Orchestration (Reviewer + VLM)
            # =================================================================
            (
                generated_sections,
                sections_results,
                review_iterations,
                target_word_count,
                pdf_path,
                orchestration_errors,
            ) = await self._run_review_orchestration(
                generated_sections=generated_sections,
                sections_results=sections_results,
                metadata=metadata,
                parsed_refs=ref_pool.get_all_refs(),
                paper_plan=paper_plan,
                template_path=template_path,
                figures_source_dir=figures_source_dir,
                converted_tables=converted_tables,
                max_review_iterations=max_review_iterations,
                enable_review=enable_review,
                compile_pdf=compile_pdf,
                enable_vlm_review=enable_vlm_review,
                target_pages=target_pages,
                paper_dir=paper_dir,
                memory=memory,
            )
            if orchestration_errors:
                errors.extend(orchestration_errors)

            # Recompute citation usage from final post-review content so exports align
            # with the delivered manuscript instead of pre-review drafts.
            if paper_plan:
                citation_budget_usage = self._rebuild_citation_budget_usage_from_final_sections(
                    paper_plan=paper_plan,
                    generated_sections=generated_sections,
                )
            
            # =================================================================
            # Assemble Paper
            # =================================================================
            print(f"[MetaDataAgent] Assembling paper...")
            latex_content = self._assemble_paper(
                title=metadata.title,
                sections=generated_sections,
                references=ref_pool.get_all_refs(),
                valid_citation_keys=ref_pool.valid_citation_keys,
            )
            
            # Calculate total word count
            total_words = sum(r.word_count for r in sections_results if r.word_count)
            
            # Save output if requested
            output_path = None
            if save_output and paper_dir:
                output_path = str(paper_dir)
                analysis_dir = paper_dir / "analysis"
                planning_dir = analysis_dir / "planning"
                research_dir = analysis_dir / "research_context"
                citation_dir = analysis_dir / "citations"
                structure_dir = analysis_dir / "structure"
                review_dir = analysis_dir / "review"
                references_dir = analysis_dir / "references"
                code_dir = analysis_dir / "code_context"
                logs_dir = paper_dir / "logs"
                traces_dir = logs_dir / "traces"
                for d in (
                    analysis_dir,
                    planning_dir,
                    research_dir,
                    citation_dir,
                    structure_dir,
                    review_dir,
                    references_dir,
                    code_dir,
                    logs_dir,
                    traces_dir,
                ):
                    d.mkdir(parents=True, exist_ok=True)
                
                # Save main.tex
                tex_path = paper_dir / "main.tex"
                tex_path.write_text(latex_content, encoding="utf-8")
                
                # Save references.bib (uses ref_pool for all accumulated refs)
                bib_content = ref_pool.to_bibtex()
                bib_path = paper_dir / "references.bib"
                bib_path.write_text(bib_content, encoding="utf-8")
                
                # Save metadata.json
                meta_path = paper_dir / "metadata.json"
                meta_path.write_text(
                    json.dumps(metadata.model_dump(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )

                # Save research_context.json (if generated)
                if research_context:
                    rc_path = research_dir / "research_context.json"
                    context_output = dict(research_context)
                    context_output.pop("_llm_raw_outputs", None)
                    core_ref_ids = [r.get("ref_id", "") for r in core_refs_for_context if r.get("ref_id")]
                    core_refs_with_abstract_count = sum(
                        1 for r in core_refs_for_context if (r.get("abstract") or "").strip()
                    )
                    context_output["core_refs_included_count"] = len(core_ref_ids)
                    context_output["core_ref_ids"] = core_ref_ids
                    context_output["core_refs_with_abstract_count"] = core_refs_with_abstract_count
                    context_output["generated_at"] = datetime.now().isoformat()
                    rc_path.write_text(
                        json.dumps(context_output, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"[MetaDataAgent] Research context saved to: {rc_path}")

                if research_context_v1:
                    rc_v1_path = research_dir / "research_context_v1.json"
                    rc_v1_payload = dict(research_context_v1)
                    raw_outputs_v1 = rc_v1_payload.pop("_llm_raw_outputs", None)
                    core_ref_ids = [r.get("ref_id", "") for r in core_refs_for_context if r.get("ref_id")]
                    core_refs_with_abstract_count = sum(
                        1 for r in core_refs_for_context if (r.get("abstract") or "").strip()
                    )
                    rc_v1_payload["core_refs_included_count"] = len(core_ref_ids)
                    rc_v1_payload["core_ref_ids"] = core_ref_ids
                    rc_v1_payload["core_refs_with_abstract_count"] = core_refs_with_abstract_count
                    rc_v1_payload["generated_at"] = datetime.now().isoformat()
                    rc_v1_path.write_text(
                        json.dumps(rc_v1_payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"[MetaDataAgent] Research context v1 saved to: {rc_v1_path}")
                    if raw_outputs_v1:
                        raw_path = research_dir / "research_context_v1_raw.txt"
                        raw_path.write_text(
                            "\n\n---\n\n".join(raw_outputs_v1),
                            encoding="utf-8",
                        )
                        print(f"[MetaDataAgent] Research context v1 raw LLM outputs saved to: {raw_path}")

                if research_context_v2:
                    rc_v2_path = research_dir / "research_context_v2.json"
                    rc_v2_payload = dict(research_context_v2)
                    raw_outputs_v2 = rc_v2_payload.pop("_llm_raw_outputs", None)
                    core_ref_ids = [r.get("ref_id", "") for r in core_refs_for_context if r.get("ref_id")]
                    core_refs_with_abstract_count = sum(
                        1 for r in core_refs_for_context if (r.get("abstract") or "").strip()
                    )
                    rc_v2_payload["core_refs_included_count"] = len(core_ref_ids)
                    rc_v2_payload["core_ref_ids"] = core_ref_ids
                    rc_v2_payload["core_refs_with_abstract_count"] = core_refs_with_abstract_count
                    rc_v2_payload["generated_at"] = datetime.now().isoformat()
                    rc_v2_path.write_text(
                        json.dumps(rc_v2_payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"[MetaDataAgent] Research context v2 saved to: {rc_v2_path}")
                    if raw_outputs_v2:
                        raw_path = research_dir / "research_context_v2_raw.txt"
                        raw_path.write_text(
                            "\n\n---\n\n".join(raw_outputs_v2),
                            encoding="utf-8",
                        )
                        print(f"[MetaDataAgent] Research context v2 raw LLM outputs saved to: {raw_path}")

                if research_context:
                    cem = research_context.get("claim_evidence_matrix", [])
                    cr = research_context.get("contribution_ranking", {"P0": [], "P1": [], "P2": []})
                    cem_path = research_dir / "claim_evidence_matrix.json"
                    cem_path.write_text(
                        json.dumps(
                            {"claim_evidence_matrix": cem, "generated_at": datetime.now().isoformat()},
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    cr_path = research_dir / "contribution_ranking.json"
                    cr_path.write_text(
                        json.dumps(
                            {"contribution_ranking": cr, "generated_at": datetime.now().isoformat()},
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    rc_cfg_for_export = (
                        self.tools_config.research_context
                        if self.tools_config and self.tools_config.research_context
                        else None
                    )
                    if (
                        rc_cfg_for_export
                        and rc_cfg_for_export.export_planning_decision_trace
                        and research_context.get("planning_decision_trace")
                    ):
                        pdt_path = research_dir / "planning_decision_trace.json"
                        pdt_path.write_text(
                            json.dumps(
                                {
                                    "planning_decision_trace": research_context.get(
                                        "planning_decision_trace", []
                                    ),
                                    "generated_at": datetime.now().isoformat(),
                                },
                                indent=2,
                                ensure_ascii=False,
                            ),
                            encoding="utf-8",
                        )
                        print(f"[MetaDataAgent] Planning decision trace saved to: {pdt_path}")

                ps_cfg_for_export = (
                    self.tools_config.paper_search
                    if self.tools_config and self.tools_config.paper_search
                    else None
                )
                budget_export_enabled = bool(
                    ps_cfg_for_export and ps_cfg_for_export.citation_budget_export
                )
                if budget_export_enabled and paper_plan:
                    # Keep usage rows in plan section order for stable auditing.
                    usage_by_section = {
                        row.get("section_type"): row
                        for row in citation_budget_usage
                        if row.get("section_type")
                    }
                    ordered_usage: List[Dict[str, Any]] = []
                    for sp in paper_plan.sections:
                        section_row = usage_by_section.get(sp.section_type)
                        if section_row:
                            ordered_usage.append(section_row)

                    budget_plan_rows: List[Dict[str, Any]] = []
                    for sp in paper_plan.sections:
                        if not sp.citation_budget:
                            continue
                        budget_plan_rows.append(
                            {
                                "section_type": sp.section_type,
                                "citation_budget": sp.citation_budget,
                                "budget_selected_refs": sp.budget_selected_refs,
                                "budget_reserve_refs": sp.budget_reserve_refs,
                                "budget_must_use_refs": sp.budget_must_use_refs,
                                "assigned_refs": sp.assigned_refs,
                            }
                        )
                    budget_plan_path = citation_dir / "citation_budget_plan.json"
                    plan_export = {
                        "citation_strategy": paper_plan.citation_strategy if paper_plan else {},
                        "sections": budget_plan_rows,
                        "generated_at": datetime.now().isoformat(),
                    }
                    budget_plan_path.write_text(
                        json.dumps(plan_export, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"[MetaDataAgent] Citation budget plan saved to: {budget_plan_path}")

                    budget_usage_path = citation_dir / "citation_budget_usage.json"
                    budget_usage_path.write_text(
                        json.dumps(
                            {
                                "sections": ordered_usage,
                                "generated_at": datetime.now().isoformat(),
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    print(f"[MetaDataAgent] Citation budget usage saved to: {budget_usage_path}")

                    alignment_stats = self._build_citation_plan_alignment_stats(
                        paper_plan=paper_plan,
                        usage_rows=ordered_usage,
                    )
                    alignment_stats["generated_at"] = datetime.now().isoformat()
                    alignment_path = citation_dir / "citation_plan_alignment.json"
                    alignment_path.write_text(
                        json.dumps(alignment_stats, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"[MetaDataAgent] Citation plan alignment saved to: {alignment_path}")
                    citation_repair_stats = self._build_citation_repair_stats(
                        memory=memory,
                    )
                    citation_repair_stats["generated_at"] = datetime.now().isoformat()
                    citation_repair_path = citation_dir / "citation_repair_stats.json"
                    citation_repair_path.write_text(
                        json.dumps(citation_repair_stats, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(
                        "[MetaDataAgent] Citation repair stats saved to: "
                        f"{citation_repair_path}"
                    )

                if paper_plan:
                    structure_alignment = self._build_structure_alignment_stats(
                        paper_plan=paper_plan,
                        generated_sections=generated_sections,
                    )
                    structure_alignment["generated_at"] = datetime.now().isoformat()
                    structure_alignment_path = structure_dir / "structure_alignment.json"
                    structure_alignment_path.write_text(
                        json.dumps(structure_alignment, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(
                        "[MetaDataAgent] Structure alignment saved to: "
                        f"{structure_alignment_path}"
                    )
                    paragraph_alignment = self._build_paragraph_feedback_alignment_report(
                        memory=memory,
                        generated_sections=generated_sections,
                    )
                    paragraph_alignment["generated_at"] = datetime.now().isoformat()
                    paragraph_alignment_path = (
                        structure_dir / "paragraph_feedback_alignment.json"
                    )
                    paragraph_alignment_path.write_text(
                        json.dumps(paragraph_alignment, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(
                        "[MetaDataAgent] Paragraph feedback alignment saved to: "
                        f"{paragraph_alignment_path}"
                    )
                    subsection_coverage = self._build_explicit_subsection_coverage(
                        paper_plan=paper_plan,
                        generated_sections=generated_sections,
                    )
                    subsection_coverage["generated_at"] = datetime.now().isoformat()
                    subsection_coverage_path = (
                        structure_dir / "explicit_subsection_coverage.json"
                    )
                    subsection_coverage_path.write_text(
                        json.dumps(subsection_coverage, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(
                        "[MetaDataAgent] Explicit subsection coverage saved to: "
                        f"{subsection_coverage_path}"
                    )

                reviewer_acceptance_stats = self._build_reviewer_acceptance_stats(
                    memory=memory,
                )
                reviewer_acceptance_stats["generated_at"] = datetime.now().isoformat()
                reviewer_acceptance_path = review_dir / "reviewer_acceptance_stats.json"
                reviewer_acceptance_path.write_text(
                    json.dumps(reviewer_acceptance_stats, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                print(
                    "[MetaDataAgent] Reviewer acceptance stats saved to: "
                    f"{reviewer_acceptance_path}"
                )

                # Save per-paper planner analysis for auditability
                if discovered_references_by_section:
                    core_ref_id_set = {
                        r.get("ref_id", "") for r in core_refs_for_context if r.get("ref_id")
                    }
                    analysis_rows: List[Dict[str, Any]] = []
                    for section_type, papers in discovered_references_by_section.items():
                        for paper in papers:
                            ref_id = paper.get("ref_id", "")
                            analysis_rows.append({
                                "section_type": section_type,
                                "ref_id": ref_id,
                                "title": paper.get("title", ""),
                                "year": paper.get("year"),
                                "venue": paper.get("venue", ""),
                                "source": paper.get("source", ""),
                                "citation_count": paper.get("citation_count"),
                                "abstract": paper.get("abstract", ""),
                                "relevance_score": paper.get("relevance_score"),
                                "relevance_reason": paper.get("relevance_reason", ""),
                                "is_core_reference": ref_id in core_ref_id_set,
                            })

                    analysis_payload = {
                        "total_sections": len(discovered_references_by_section),
                        "total_papers": len(analysis_rows),
                        "papers": analysis_rows,
                        "generated_at": datetime.now().isoformat(),
                    }
                    analysis_path = references_dir / "discovered_papers_analysis.json"
                    analysis_path.write_text(
                        json.dumps(analysis_payload, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"[MetaDataAgent] Discovered papers analysis saved to: {analysis_path}")

                # Save code repository summary markdown (if generated)
                if code_context:
                    graph_path = code_dir / "code_evidence_graph.json"
                    graph_path.write_text(
                        json.dumps(
                            {
                                "code_evidence_graph": code_context.get("code_evidence_graph", []),
                                "generated_at": datetime.now().isoformat(),
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    assets_path = code_dir / "writing_assets.json"
                    assets_path.write_text(
                        json.dumps(
                            {
                                "writing_assets": code_context.get("writing_assets", {}),
                                "section_asset_packs": code_context.get("section_asset_packs", {}),
                                "claim_support_candidates": code_context.get("claim_support_candidates", []),
                                "generated_at": datetime.now().isoformat(),
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    if paper_plan:
                        focus_rows: List[Dict[str, Any]] = []
                        for sp in paper_plan.sections:
                            if not getattr(sp, "code_focus", None):
                                continue
                            focus_rows.append(
                                {
                                    "section_type": sp.section_type,
                                    "section_title": sp.section_title,
                                    "code_focus": sp.code_focus,
                                }
                            )
                        focus_path = code_dir / "section_code_focus_plan.json"
                        focus_path.write_text(
                            json.dumps(
                                {
                                    "sections": focus_rows,
                                    "generated_at": datetime.now().isoformat(),
                                },
                                indent=2,
                                ensure_ascii=False,
                            ),
                            encoding="utf-8",
                        )
                    print(f"[MetaDataAgent] Code analysis artifacts saved to: {code_dir}")

                if code_summary_markdown:
                    crs_path = code_dir / "code_repository_summary.md"
                    crs_path.write_text(code_summary_markdown, encoding="utf-8")
                    print(f"[MetaDataAgent] Code repository summary saved to: {crs_path}")

                # Save prompt/evidence traces (optional)
                if metadata.export_prompt_traces and prompt_traces:
                    traces_path = traces_dir / "prompt_traces.json"
                    traces_path.write_text(
                        json.dumps(prompt_traces, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"[MetaDataAgent] Prompt traces saved to: {traces_path}")

                # Persist session memory (review history + agent logs)
                memory.log("metadata", "final", "paper_assembled",
                           narrative=f"Paper assembled successfully with {total_words} total words.",
                           total_words=total_words, status="assembled")
                memory.persist_all(paper_dir)
                
                print(f"[MetaDataAgent] Output saved to: {output_path}")
            
            # Determine overall status
            if not errors:
                status = "ok"
            elif len(errors) < len(sections_results):
                status = "partial"
            else:
                status = "error"
            
            return PaperGenerationResult(
                status=status,
                paper_title=metadata.title,
                sections=sections_results,
                latex_content=latex_content,
                output_path=output_path,
                pdf_path=pdf_path,
                total_word_count=total_words,
                target_word_count=target_word_count,
                review_iterations=review_iterations,
                errors=errors,
            )
            
        except Exception as e:
            print(f"[MetaDataAgent] Error: {e}")
            return PaperGenerationResult(
                status="error",
                paper_title=metadata.title,
                sections=sections_results,
                errors=[str(e)],
            )
    
    async def generate_single_section(
        self,
        request: SectionGenerationRequest,
    ) -> SectionResult:
        """Generate a single section (for debugging or incremental generation)"""
        metadata = request.metadata
        ref_pool = ReferencePool(metadata.references)
        
        if request.section_type == "introduction":
            return await self._generate_introduction(metadata, ref_pool)
        elif request.section_type in SYNTHESIS_SECTIONS:
            prior = request.prior_sections or {}
            contributions = extract_contributions_from_intro(prior.get("introduction", ""))
            return await self._generate_synthesis_section(
                section_type=request.section_type,
                paper_title=metadata.title,
                prior_sections=prior,
                contributions=contributions,
                style_guide=metadata.style_guide,
            )
        else:
            contributions = []
            if request.intro_context:
                contributions = extract_contributions_from_intro(request.intro_context)
            return await self._generate_body_section(
                section_type=request.section_type,
                metadata=metadata,
                intro_context=request.intro_context or "",
                contributions=contributions,
                ref_pool=ref_pool,
            )
    
    # =========================================================================
    # Phase 1: Introduction Generation
    # =========================================================================
    
    async def _generate_introduction(
        self,
        metadata: PaperMetaData,
        ref_pool: ReferencePool,
        section_plan: Optional[SectionPlan] = None,
        figures: Optional[List[FigureSpec]] = None,
        tables: Optional[List[TableSpec]] = None,
        code_context: Optional[Dict[str, Any]] = None,
        research_context: Optional[Dict[str, Any]] = None,
        prompt_traces: Optional[List[Dict[str, Any]]] = None,
        memory: Optional[SessionMemory] = None,
    ) -> SectionResult:
        """
        Generate Introduction section — delegates to WriterAgent.

        - **Description**:
            - Prompt compilation, then WriterAgent.run() with
              ReAct AskTool + iterative mini-review.
            - Reference discovery is done centrally in Planner (Phase 0b).

        - **Args**:
            - `metadata` (PaperMetaData): Paper metadata.
            - `ref_pool` (ReferencePool): Persistent reference pool.
            - `section_plan` (SectionPlan, optional): Plan for this section.
            - `figures` (List[FigureSpec], optional): Figures.
            - `tables` (List[TableSpec], optional): Tables.
            - `memory` (SessionMemory, optional): Shared session memory.

        - **Returns**:
            - `SectionResult`: Generation result.
        """
        try:
            key_points: List[str] = []
            if section_plan:
                key_points = section_plan.get_key_points()

            # Compile prompt then delegate to WriterAgent
            intro_runtime_evidence = self._retrieve_runtime_code_evidence(
                code_context=code_context,
                section_type="introduction",
                metadata=metadata,
                top_k=2,
            )
            intro_code_context = format_code_context_for_prompt(
                context=code_context,
                section_type="introduction",
                retrieved_evidence=intro_runtime_evidence,
                top_k=4,
            )
            intro_research_context = self._format_research_context_for_prompt(
                research_context=research_context,
                section_type="introduction",
            )
            prompt = compile_introduction_prompt(
                paper_title=metadata.title,
                idea_hypothesis=metadata.idea_hypothesis,
                method_summary=metadata.method,
                data_summary=metadata.data,
                experiments_summary=metadata.experiments,
                references=ref_pool.get_all_refs(),
                style_guide=metadata.style_guide,
                section_plan=section_plan,
                figures=figures,
                tables=tables,
                active_skills=self._get_active_skills("introduction", metadata.style_guide),
                code_context=intro_code_context,
                research_context=intro_research_context,
                enable_structure_contract=bool(
                    self.tools_config is None
                    or getattr(self.tools_config, "writer_structure_contract_enabled", True)
                ),
            )
            if prompt_traces is not None:
                prompt_traces.append(
                    {
                        "section_type": "introduction",
                        "phase": "generation",
                        "code_context_used": bool(intro_code_context),
                        "research_context_used": bool(intro_research_context),
                        "runtime_evidence": intro_runtime_evidence,
                        "prompt": prompt,
                    }
                )

            # Use section-assigned refs if available, else fall back to full pool
            section_keys = (
                section_plan.assigned_refs
                if section_plan and section_plan.assigned_refs
                else list(ref_pool.valid_citation_keys)
            )
            reserve_keys = (
                section_plan.budget_reserve_refs
                if section_plan and section_plan.budget_reserve_refs
                else []
            )
            writer_valid_keys = list(dict.fromkeys(list(section_keys) + list(reserve_keys)))
            result = await self._writer.run(
                system_prompt=GENERATION_SYSTEM_PROMPT,
                user_prompt=prompt,
                section_type="introduction",
                valid_citation_keys=writer_valid_keys,
                key_points=key_points or None,
                memory=memory,
                peers={"planner": self._planner, "reviewer": self._reviewer},
            )
            content = result.get("generated_content", "")
            if not content.strip():
                return SectionResult(
                    section_type="introduction",
                    section_title=section_plan.section_title if section_plan else "Introduction",
                    status="error",
                    error="Writer returned empty introduction content",
                )
            word_count = len(content.split())

            return SectionResult(
                section_type="introduction",
                section_title=section_plan.section_title if section_plan else "Introduction",
                status="ok",
                latex_content=content,
                word_count=word_count,
            )
        except Exception as e:
            return SectionResult(
                section_type="introduction",
                status="error",
                error=str(e),
            )
    
    # =========================================================================
    # Phase 2: Body Section Generation
    # =========================================================================
    
    async def _generate_body_section(
        self,
        section_type: str,
        metadata: PaperMetaData,
        intro_context: str,
        contributions: List[str],
        ref_pool: ReferencePool,
        section_plan: Optional[SectionPlan] = None,
        figures: Optional[List[FigureSpec]] = None,
        tables: Optional[List[TableSpec]] = None,
        converted_tables: Optional[Dict[str, str]] = None,
        code_context: Optional[Dict[str, Any]] = None,
        research_context: Optional[Dict[str, Any]] = None,
        prompt_traces: Optional[List[Dict[str, Any]]] = None,
        memory: Optional[SessionMemory] = None,
    ) -> SectionResult:
        """
        Generate a body section using two-phase pattern.

        - **Description**:
            - Phase A (Judgment): LLM analyses whether existing references
              cover the section needs and suggests search queries if not.
            - Phase A (Search): If needed, PaperSearchTool is called directly
              (system-side) and results are merged into ref_pool.
            - Phase B (Writing): Pure LLM call with no tools. All available
              refs (core + newly discovered) are included in the prompt.

        - **Args**:
            - `section_type` (str): Type of section (method, experiment, etc.).
            - `metadata` (PaperMetaData): Paper metadata.
            - `intro_context` (str): Introduction content for context.
            - `contributions` (List[str]): Paper contributions.
            - `ref_pool` (ReferencePool): Persistent reference pool.
            - `section_plan` (SectionPlan, optional): Plan for this section.
            - `figures` (List[FigureSpec], optional): Figures for this section.
            - `tables` (List[TableSpec], optional): Tables for this section.
            - `converted_tables` (Dict[str, str], optional): Converted table LaTeX.

        - **Returns**:
            - `SectionResult`: Generation result.
        """
        try:
            key_points = []
            if section_plan:
                key_points = section_plan.get_key_points()

            section_title_str = (
                section_plan.section_title
                if section_plan and section_plan.section_title
                else section_type.replace("_", " ").title()
            )
            # Get relevant content from metadata based on section type
            if section_plan and section_plan.content_sources:
                sources = section_plan.content_sources
            else:
                sources = BODY_SECTION_SOURCES.get(section_type, [])

            content_parts = []
            for source in sources:
                if source == "references":
                    continue  # References are handled separately
                value = getattr(metadata, source, "")
                if value:
                    content_parts.append(f"### {source.title()}\n{value}")

            metadata_content = "\n\n".join(content_parts) if content_parts else metadata.method

            # Build memory context for cross-section awareness
            memory_context = memory.get_writing_context(section_type) if memory else ""

            runtime_evidence = self._retrieve_runtime_code_evidence(
                code_context=code_context,
                section_type=section_type,
                metadata=metadata,
                contributions=contributions,
                top_k=3,
            )
            section_code_context = format_code_context_for_prompt(
                context=code_context,
                section_type=section_type,
                retrieved_evidence=runtime_evidence,
                top_k=6,
            )
            section_research_context = self._format_research_context_for_prompt(
                research_context=research_context,
                section_type=section_type,
            )

            # Build prompt AFTER search so it includes any new refs
            prompt = compile_body_section_prompt(
                section_type=section_type,
                metadata_content=metadata_content,
                intro_context=intro_context,
                contributions=contributions,
                references=ref_pool.get_all_refs(),
                style_guide=metadata.style_guide,
                section_plan=section_plan,
                figures=figures,
                tables=tables,
                converted_tables=converted_tables,
                active_skills=self._get_active_skills(section_type, metadata.style_guide),
                memory_context=memory_context,
                code_context=section_code_context,
                research_context=section_research_context,
                enable_structure_contract=bool(
                    self.tools_config is None
                    or getattr(self.tools_config, "writer_structure_contract_enabled", True)
                ),
            )
            if prompt_traces is not None:
                prompt_traces.append(
                    {
                        "section_type": section_type,
                        "phase": "generation",
                        "code_context_used": bool(section_code_context),
                        "research_context_used": bool(section_research_context),
                        "runtime_evidence": runtime_evidence,
                        "prompt": prompt,
                    }
                )

            section_keys = (
                section_plan.assigned_refs
                if section_plan and section_plan.assigned_refs
                else list(ref_pool.valid_citation_keys)
            )
            reserve_keys = (
                section_plan.budget_reserve_refs
                if section_plan and section_plan.budget_reserve_refs
                else []
            )
            writer_valid_keys = list(dict.fromkeys(list(section_keys) + list(reserve_keys)))
            result = await self._writer.run(
                system_prompt=GENERATION_SYSTEM_PROMPT,
                user_prompt=prompt,
                section_type=section_type,
                valid_citation_keys=writer_valid_keys,
                key_points=key_points or None,
                memory=memory,
                peers={"planner": self._planner, "reviewer": self._reviewer},
            )
            content = result.get("generated_content", "")
            word_count = len(content.split())

            # Use plan's title if available
            if section_plan and section_plan.section_title:
                section_title = section_plan.section_title
            else:
                titles = {
                    "related_work": "Related Work",
                    "method": "Methodology",
                    "experiment": "Experiments",
                    "result": "Results",
                    "discussion": "Discussion",
                }
                section_title = titles.get(section_type, section_type.title())

            return SectionResult(
                section_type=section_type,
                section_title=section_title,
                status="ok",
                latex_content=content,
                word_count=word_count,
            )
        except Exception as e:
            return SectionResult(
                section_type=section_type,
                status="error",
                error=str(e),
            )
    
    # =========================================================================
    # Phase 3: Synthesis Section Generation
    # =========================================================================
    
    async def _generate_synthesis_section(
        self,
        section_type: str,
        paper_title: str,
        prior_sections: Dict[str, str],
        contributions: List[str],
        style_guide: Optional[str] = None,
        section_plan: Optional[SectionPlan] = None,
        prompt_traces: Optional[List[Dict[str, Any]]] = None,
        memory: Optional[SessionMemory] = None,
    ) -> SectionResult:
        """Generate synthesis section (Abstract or Conclusion) via WriterAgent."""
        try:
            memory_context = memory.get_cross_section_summary() if memory else ""

            prompt = compile_synthesis_prompt(
                section_type=section_type,
                paper_title=paper_title,
                prior_sections=prior_sections,
                key_contributions=contributions,
                style_guide=style_guide,
                section_plan=section_plan,
                active_skills=self._get_active_skills(section_type, style_guide),
                memory_context=memory_context,
            )
            if prompt_traces is not None:
                prompt_traces.append(
                    {
                        "section_type": section_type,
                        "phase": "generation",
                        "code_context_used": False,
                        "runtime_evidence": [],
                        "prompt": prompt,
                    }
                )

            synthesis_system = (
                "You are an expert academic writer. Use present tense for methods, "
                "no contractions (it is, do not, cannot), no possessives on method "
                "names (the performance of X, not X's performance). "
                "Place key information at sentence end. Output pure LaTeX only."
            )

            result = await self._writer.run(
                system_prompt=synthesis_system,
                user_prompt=prompt,
                section_type=section_type,
                enable_review=False,
                memory=memory,
                peers={"planner": self._planner, "reviewer": self._reviewer},
            )
            content = result.get("generated_content", "")

            # Hard rule: strip ALL citations and cross-references from
            # abstract and conclusion — these must be self-contained.
            if section_type in ("abstract", "conclusion"):
                content = re.sub(r'~?\\cite\{[^}]*\}', '', content)
                # Strip "Figure~\ref{...}", "Table~\ref{...}", "Section~\ref{...}"
                content = re.sub(
                    r'(?:Figure|Fig\.|Table|Tab\.|Section|Sec\.|Equation|Eq\.)~?\\ref\{[^}]*\}',
                    '', content,
                )
                # Strip any remaining bare \ref{...}
                content = re.sub(r'~?\\ref\{[^}]*\}', '', content)
                # Clean orphaned parentheses like "(, )" or "( )"
                content = re.sub(r'\(\s*[,;]?\s*\)', '', content)
                content = re.sub(r'  +', ' ', content)

            word_count = len(content.split())
            
            # Use plan's title if available
            if section_plan and section_plan.section_title:
                section_title = section_plan.section_title
            else:
                section_title = "Abstract" if section_type == "abstract" else "Conclusion"
            
            return SectionResult(
                section_type=section_type,
                section_title=section_title,
                status="ok",
                latex_content=content,
                word_count=word_count,
            )
        except Exception as e:
            return SectionResult(
                section_type=section_type,
                status="error",
                error=str(e),
            )
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _parse_references(self, bibtex_list: List[str]) -> List[Dict[str, Any]]:
        """Parse BibTeX entries into structured format"""
        parsed = []
        
        for bibtex in bibtex_list:
            try:
                # Extract key fields using regex
                ref_id_match = re.search(r'@\w+{([^,]+),', bibtex)
                title_match = re.search(r'title\s*=\s*[{"]([^}"]+)[}"]', bibtex, re.IGNORECASE)
                author_match = re.search(r'author\s*=\s*[{"]([^}"]+)[}"]', bibtex, re.IGNORECASE)
                year_match = re.search(r'year\s*=\s*[{"]?(\d{4})[}"]?', bibtex, re.IGNORECASE)
                
                ref = {
                    "ref_id": ref_id_match.group(1) if ref_id_match else f"ref_{len(parsed)+1}",
                    "title": title_match.group(1) if title_match else "",
                    "authors": author_match.group(1) if author_match else "",
                    "year": int(year_match.group(1)) if year_match else None,
                    "bibtex": bibtex,
                }
                parsed.append(ref)
            except Exception:
                # If parsing fails, create a minimal entry
                parsed.append({
                    "ref_id": f"ref_{len(parsed)+1}",
                    "bibtex": bibtex,
                })
        
        return parsed
    
    @staticmethod
    def _validate_ref_usage(
        generated_sections: Dict[str, str],
        ref_pool: "ReferencePool",
    ) -> Dict[str, Any]:
        """
        Check that every reference in the pool is cited at least once.
        Logs warnings for uncited references and returns structured coverage.
        """
        all_content = "\n".join(generated_sections.values())
        cited_keys = ReferencePool.extract_cite_keys(all_content)
        pool_keys = ref_pool.valid_citation_keys
        uncited = pool_keys - cited_keys
        if uncited:
            print(f"[MetaDataAgent] WARNING: {len(uncited)} uncited reference(s): "
                  + ", ".join(sorted(uncited)[:10])
                  + ("..." if len(uncited) > 10 else ""))
        else:
            print(f"[MetaDataAgent] All {len(pool_keys)} pooled references are cited.")
        return {
            "cited_keys": sorted(cited_keys),
            "pool_keys": sorted(pool_keys),
            "uncited_keys": sorted(uncited),
            "coverage": (len(cited_keys & pool_keys) / len(pool_keys)) if pool_keys else 1.0,
        }

    async def _enforce_reference_coverage(
        self,
        generated_sections: Dict[str, str],
        sections_results: List[SectionResult],
        paper_plan: Optional[PaperPlan],
        metadata: PaperMetaData,
        valid_ref_keys: Set[str],
        memory: Optional[SessionMemory] = None,
        max_sections_to_revise: int = 2,
    ) -> Set[str]:
        """
        Reference coverage fix pass.
        - **Description**:
            - Finds uncited pooled references.
            - Routes each missing key to one section that has it in assigned_refs.
            - Applies targeted revision prompts to integrate those citations.
        """
        if not paper_plan:
            return set()

        all_content = "\n".join(generated_sections.values())
        cited_keys = ReferencePool.extract_cite_keys(all_content)
        uncited_keys = set(valid_ref_keys) - set(cited_keys)
        if not uncited_keys:
            return set()

        # Map section -> missing keys it can cite.
        missing_by_section: Dict[str, List[str]] = {}
        for sp in paper_plan.sections:
            st = sp.section_type
            if st in ("abstract", "conclusion"):
                continue
            if st not in generated_sections:
                continue
            assigned = set(getattr(sp, "assigned_refs", []) or [])
            missing = sorted(list(assigned & uncited_keys))
            if missing:
                missing_by_section[st] = missing

        if not missing_by_section:
            return set()

        revised_sections: Set[str] = set()
        # Prioritize sections that can absorb the most missing refs.
        targets = sorted(
            missing_by_section.items(),
            key=lambda kv: len(kv[1]),
            reverse=True,
        )[:max_sections_to_revise]

        for section_type, missing_keys in targets:
            prompt = (
                f"Reference coverage fix for section '{section_type}'.\n"
                f"Integrate the following citation keys naturally into relevant claims: "
                f"{', '.join(missing_keys[:6])}.\n"
                "Rules:\n"
                "- Preserve technical meaning and paragraph structure.\n"
                "- Use ONLY these keys via \\cite{key}.\n"
                "- Do not add citations in abstract or conclusion.\n"
                "- Do not fabricate facts; attach citations to existing statements where appropriate."
            )
            revised = await self._revise_section(
                section_type=section_type,
                current_content=generated_sections[section_type],
                revision_prompt=prompt,
                metadata=metadata,
                memory=memory,
            )
            if revised and revised.strip():
                generated_sections[section_type] = revised
                # Sync word_count in section results
                for sr in sections_results:
                    if sr.section_type == section_type and sr.status == "ok":
                        sr.latex_content = revised
                        sr.word_count = len(revised.split())
                        break
                revised_sections.add(section_type)

        if revised_sections:
            post_cited = ReferencePool.extract_cite_keys("\n".join(generated_sections.values()))
            post_coverage = (len(set(valid_ref_keys) & set(post_cited)) / len(valid_ref_keys)) if valid_ref_keys else 1.0
            print(
                "[MetaDataAgent] Ref coverage pass revised="
                f"{sorted(revised_sections)} coverage={post_coverage:.0%}"
            )
        return revised_sections

    def _validate_file_paths(self, metadata: PaperMetaData) -> List[str]:
        """
        Validate that all provided file paths exist before generation.
        
        Args:
            metadata: Paper metadata with figures and tables
            
        Returns:
            List of error messages (empty if all valid)
        """
        errors = []
        base_path = os.getcwd()
        
        # Validate figure file paths
        for fig in metadata.figures:
            if fig.auto_generate:
                continue  # Skip auto-generate figures
            if fig.file_path:
                # Resolve relative paths
                if os.path.isabs(fig.file_path):
                    resolved_path = fig.file_path
                else:
                    resolved_path = os.path.join(base_path, fig.file_path)
                resolved_path = os.path.normpath(resolved_path)
                
                if not os.path.exists(resolved_path):
                    errors.append(f"Figure file not found: {fig.file_path} (resolved: {resolved_path})")
        
        # Validate table file paths
        for tbl in metadata.tables:
            if tbl.auto_generate:
                continue  # Skip auto-generate tables
            if tbl.file_path:
                # Resolve relative paths
                if os.path.isabs(tbl.file_path):
                    resolved_path = tbl.file_path
                else:
                    resolved_path = os.path.join(base_path, tbl.file_path)
                resolved_path = os.path.normpath(resolved_path)
                
                if not os.path.exists(resolved_path):
                    errors.append(f"Table file not found: {tbl.file_path} (resolved: {resolved_path})")
            # Tables without file_path should have content
            elif not tbl.content and not tbl.auto_generate:
                errors.append(f"Table {tbl.id} has no file_path or content")
        
        return errors

    @staticmethod
    def _convert_figures_for_latex(metadata: "PaperMetaData") -> int:
        """
        Convert figure files to LaTeX-compatible formats (PDF preferred, then PNG).
        Mutates FigureSpec.file_path in-place if conversion is performed.

        - **Returns**:
            - `int`: Number of figures converted.
        """
        from PIL import Image as PILImage

        LATEX_OK = {".pdf", ".png", ".jpg", ".jpeg", ".eps"}
        converted = 0
        base_path = os.getcwd()

        for fig in metadata.figures:
            if fig.auto_generate or not fig.file_path:
                continue
            resolved = (
                fig.file_path if os.path.isabs(fig.file_path)
                else os.path.join(base_path, fig.file_path)
            )
            resolved = os.path.normpath(resolved)
            if not os.path.exists(resolved):
                continue
            ext = os.path.splitext(resolved)[1].lower()
            if ext in LATEX_OK:
                continue

            # Convert to PDF (preferred); fall back to PNG if PDF conversion fails
            pdf_path = os.path.splitext(resolved)[0] + ".pdf"
            try:
                img = PILImage.open(resolved)
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                img.save(pdf_path, "PDF")
                # Update FigureSpec so downstream sees the new path
                if os.path.isabs(fig.file_path):
                    fig.file_path = pdf_path
                else:
                    fig.file_path = os.path.relpath(pdf_path, base_path)
                converted += 1
                print(f"[MetaDataAgent] Converted figure {fig.id}: {ext} -> .pdf")
            except Exception as exc:
                png_path = os.path.splitext(resolved)[0] + ".png"
                try:
                    img = PILImage.open(resolved)
                    img.save(png_path, "PNG")
                    if os.path.isabs(fig.file_path):
                        fig.file_path = png_path
                    else:
                        fig.file_path = os.path.relpath(png_path, base_path)
                    converted += 1
                    print(f"[MetaDataAgent] Converted figure {fig.id}: {ext} -> .png")
                except Exception as png_exc:
                    print(f"[MetaDataAgent] WARNING: Cannot convert {fig.id} "
                          f"({ext}): pdf={exc}, png={png_exc}")
        return converted

    def _collect_figure_paths(
        self, 
        figures: List[FigureSpec], 
        base_path: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Collect figure file paths from FigureSpec list
        
        Args:
            figures: List of FigureSpec objects
            base_path: Base directory for resolving relative paths
                       (typically the directory containing the metadata JSON)
            
        Returns:
            Dict mapping figure ID to absolute file path
        """
        import os
        paths = {}
        for fig in figures:
            if fig.auto_generate:
                print(f"[MetaDataAgent] Figure auto-generation not implemented: {fig.id}")
                continue
            if fig.file_path:
                # Resolve relative paths against base_path
                if base_path and not os.path.isabs(fig.file_path):
                    resolved_path = os.path.join(base_path, fig.file_path)
                else:
                    resolved_path = fig.file_path
                
                # Normalize the path
                resolved_path = os.path.normpath(resolved_path)
                
                if os.path.exists(resolved_path):
                    paths[fig.id] = resolved_path
                else:
                    print(f"[MetaDataAgent] Warning: Figure file not found: {resolved_path}")
        return paths
    
    def _assemble_paper(
        self,
        title: str,
        sections: Dict[str, str],
        references: List[Dict[str, Any]],
        valid_citation_keys: set = None,
    ) -> str:
        """Assemble complete LaTeX document with final citation validation"""
        
        # Extract valid keys if not provided
        if valid_citation_keys is None:
            valid_citation_keys = self._extract_valid_citation_keys(references)
        
        # Basic LaTeX template
        latex = r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{natbib}

\title{""" + self._escape_latex(title) + r"""}
\author{Author Names}
\date{\today}

\begin{document}

\maketitle

"""
        # Add abstract
        if "abstract" in sections:
            latex += r"\begin{abstract}" + "\n"
            latex += sections["abstract"] + "\n"
            latex += r"\end{abstract}" + "\n\n"
        
        # Add sections in order (handle dynamic sections)
        _default_order = ["introduction", "related_work", "method", "experiment", "result", "discussion", "conclusion"]
        _default_titles = {
            "introduction": "Introduction",
            "related_work": "Related Work",
            "method": "Methodology",
            "experiment": "Experiments",
            "result": "Results",
            "discussion": "Discussion",
            "conclusion": "Conclusion",
        }
        # Build final order: known sections first, then any extras
        section_order = [s for s in _default_order if s in sections]
        for s in sections:
            if s != "abstract" and s not in section_order:
                section_order.append(s)
        
        for section_type in section_order:
            if section_type in sections and sections[section_type]:
                title = _default_titles.get(section_type, section_type.replace("_", " ").title())
                latex += f"\\section{{{title}}}\n"
                # Fix common LaTeX reference syntax errors
                content = self._fix_latex_references(sections[section_type])
                # Global citation validation - remove any remaining invalid citations
                content, invalid, valid = self._validate_and_fix_citations(
                    content, valid_citation_keys, remove_invalid=True
                )
                if invalid:
                    print(f"[Assemble] Removed {len(invalid)} invalid citations from {section_type}: {invalid[:5]}")
                latex += content + "\n\n"
        
        # Add bibliography
        latex += r"""
\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
"""
        
        return latex
    
    def _fix_latex_references(self, content: str) -> str:
        """
        Fix common LaTeX reference syntax errors generated by LLM.
        
        Fixes:
        - \\reftab{id} -> \\ref{tab:id}
        - \\reffig{id} -> \\ref{fig:id}
        - \\ref{id} (without prefix) -> \\ref{tab:id} or \\ref{fig:id} based on context
        - \\%---... comment dividers (escaped percent) -> removed entirely
        """
        import re
        
        # Remove escaped comment dividers (\%--- or \%===)
        # These appear as literal text in PDF because \% is an escaped percent sign
        content = re.sub(r'\\%[-=]+\s*\n?', '', content)
        
        # Also remove lines that are just escaped percent signs with dashes
        content = re.sub(r'^\s*\\%[-=]+.*$', '', content, flags=re.MULTILINE)
        
        # Clean up multiple blank lines that may result
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Fix \reftab{id} -> \ref{tab:id}
        content = re.sub(r'\\reftab\{([^}]+)\}', r'\\ref{tab:\1}', content)
        
        # Fix \reffig{id} -> \ref{fig:id}
        content = re.sub(r'\\reffig\{([^}]+)\}', r'\\ref{fig:\1}', content)
        
        # Fix Table~\ref{id} -> Table~\ref{tab:id} (when missing tab: prefix)
        # Only if not already prefixed with tab: or fig:
        content = re.sub(
            r'(Table~?\\ref\{)(?!tab:|fig:)([^}]+)\}',
            r'\1tab:\2}',
            content
        )
        
        # Fix Figure~\ref{id} -> Figure~\ref{fig:id} (when missing fig: prefix)
        content = re.sub(
            r'(Figure~?\\ref\{)(?!tab:|fig:)([^}]+)\}',
            r'\1fig:\2}',
            content
        )
        
        return content
    
    def _extract_valid_citation_keys(self, parsed_refs: List[Dict[str, Any]]) -> set:
        """
        Extract valid citation keys from parsed references.
        
        Args:
            parsed_refs: List of parsed reference dictionaries
            
        Returns:
            Set of valid citation keys
        """
        keys = set()
        for ref in parsed_refs:
            ref_id = ref.get("ref_id", "")
            if ref_id:
                keys.add(ref_id)
        return keys
    
    def _validate_and_fix_citations(
        self, 
        content: str, 
        valid_keys: set,
        remove_invalid: bool = True
    ) -> tuple:
        """
        Validate citations in content and optionally remove invalid ones.
        
        Args:
            content: LaTeX content with \cite{} commands
            valid_keys: Set of valid citation keys
            remove_invalid: Whether to remove invalid citations
            
        Returns:
            Tuple of (fixed_content, list_of_invalid_keys, list_of_valid_keys_used)
        """
        import re
        
        # Find all citation keys in the content
        # Handle \cite{key1, key2}, \citep{key}, \citet{key}, etc.
        # Captures group(1) = command name, group(2) = keys string
        cite_pattern = r'\\(cite[pt]?|citeauthor|citeyear|citealt|citealp)\{([^}]+)\}'
        
        invalid_keys = []
        valid_keys_used = []
        
        def process_cite(match):
            cmd_name = match.group(1)   # e.g. "cite", "citep", "citet"
            cite_content = match.group(2)
            keys = [k.strip() for k in cite_content.split(',')]
            
            valid_in_cite = []
            for key in keys:
                if key in valid_keys:
                    valid_in_cite.append(key)
                    if key not in valid_keys_used:
                        valid_keys_used.append(key)
                else:
                    if key not in invalid_keys:
                        invalid_keys.append(key)
            
            if remove_invalid:
                if valid_in_cite:
                    return f'\\{cmd_name}{{{", ".join(valid_in_cite)}}}'
                else:
                    # All keys invalid, remove entire citation
                    return ''
            else:
                return match.group(0)
        
        fixed_content = re.sub(cite_pattern, process_cite, content)
        
        # Clean up empty citations and dangling text
        # Remove empty citation commands: \cite{}, \citep{}, \citet{}, etc.
        fixed_content = re.sub(r'\\(?:cite[pt]?|citeauthor|citeyear|citealt|citealp)\{\s*\}', '', fixed_content)
        # Clean up double spaces
        fixed_content = re.sub(r'  +', ' ', fixed_content)
        # Clean up space before punctuation
        fixed_content = re.sub(r' +([.,;:])', r'\1', fixed_content)
        
        return fixed_content, invalid_keys, valid_keys_used
    
    def _ensure_figures_defined(
        self,
        generated_sections: Dict[str, str],
        paper_plan: Optional[PaperPlan],
        figures: Optional[List[FigureSpec]],
    ) -> Dict[str, str]:
        """
        Ensure all figures assigned for definition have their environments created.
        
        If a figure is in section_plan.figures_to_define but no \\begin{figure} 
        exists with matching label, inject the figure environment.
        
        Args:
            generated_sections: Dict of section_type -> latex_content
            paper_plan: Paper plan with figure assignments
            figures: List of figure specifications
            
        Returns:
            Updated generated_sections dict
        """
        import re
        
        if not paper_plan or not figures:
            return generated_sections
        
        # Build figure lookup
        figure_map = {fig.id: fig for fig in figures}
        
        # Pre-scan ALL sections (including appendix) to find which figures
        # already have their environments defined somewhere in the paper.
        # This prevents re-injecting figures that were moved to the appendix
        # by structural overflow actions.
        globally_defined_figs: set = set()
        globally_used_targets: set = set()
        all_content = "\n".join(generated_sections.values())
        target_pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
        for tgt in re.findall(target_pattern, all_content):
            norm = str(tgt).strip()
            if norm:
                globally_used_targets.add(norm)
                globally_used_targets.add(Path(norm).name)
                globally_used_targets.add(Path(norm).stem)
        for fig in figures:
            fig_pattern = rf'\\begin{{figure\*?}}.*?\\label{{{re.escape(fig.id)}}}.*?\\end{{figure\*?}}'
            if re.search(fig_pattern, all_content, re.DOTALL):
                globally_defined_figs.add(fig.id)
        
        for section in paper_plan.sections:
            section_type = section.section_type
            figures_to_define = section.get_figure_ids_to_define()
            
            if not figures_to_define or section_type not in generated_sections:
                continue
            
            content = generated_sections[section_type]
            
            for fig_id in figures_to_define:
                fig = figure_map.get(fig_id)
                if not fig:
                    continue
                
                # Skip if this figure is already defined ANYWHERE in the paper
                # (including appendix — it may have been moved there deliberately)
                if fig_id in globally_defined_figs:
                    continue
                # Also skip if the same underlying image target already appears
                # via includegraphics with a different label/alias.
                aliases = {fig_id, Path(fig_id).name, Path(fig_id).stem}
                if getattr(fig, "file_path", ""):
                    fp = str(fig.file_path)
                    aliases.update({Path(fp).name, Path(fp).stem})
                if any(a for a in aliases if a in globally_used_targets):
                    continue
                
                # Figure not defined anywhere - inject it
                print(f"[EnsureFigures] Injecting missing figure '{fig_id}' in '{section_type}'")
                
                # Determine environment and width based on wide flag
                env_name = "figure*" if fig.wide else "figure"
                width = "\\textwidth" if fig.wide else "0.9\\linewidth"
                
                # Build figure LaTeX
                figure_latex = f"""
\\begin{{{env_name}}}[htbp]
\\centering
\\includegraphics[width={width}]{{{fig_id}}}
\\caption{{{fig.caption}}}\\label{{{fig_id}}}
\\end{{{env_name}}}
"""
                
                # Find a good insertion point:
                # 1. After the first paragraph that mentions this figure
                # 2. Or at the end of the section
                ref_pattern = rf'(Figure~?\\ref{{{re.escape(fig_id)}}}[^.]*\.)'
                match = re.search(ref_pattern, content)
                if match:
                    # Insert after the sentence that references the figure
                    insert_pos = match.end()
                    content = content[:insert_pos] + "\n" + figure_latex + content[insert_pos:]
                else:
                    # Insert at the beginning of the section
                    content = figure_latex + "\n" + content
                
                generated_sections[section_type] = content
        
        return generated_sections

    @staticmethod
    def _deduplicate_figure_environments(
        generated_sections: Dict[str, str],
        section_order: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Remove duplicate figure/table environments across and within sections.
        - **Description**:
            - Scans all sections for \\begin{figure}...\\end{figure} blocks
            - Keeps only the FIRST occurrence of each \\label{...}
            - Removes subsequent duplicates (both within and across sections)
            - Preserves \\ref{} references to figures defined elsewhere

        - **Args**:
            - `generated_sections` (Dict[str, str]): section_type -> LaTeX content
            - `section_order` (List[str], optional): Preferred ordering for priority

        - **Returns**:
            - Updated generated_sections with duplicates removed
        """
        import re

        # Determine processing order: use section_order if given
        if section_order:
            ordered_keys = [k for k in section_order if k in generated_sections]
            ordered_keys += [k for k in generated_sections if k not in ordered_keys]
        else:
            ordered_keys = list(generated_sections.keys())

        seen_labels: set = set()
        total_removed = 0

        for section_type in ordered_keys:
            content = generated_sections.get(section_type, "")
            if not content:
                continue

            # Match figure and figure* environments
            env_pattern = re.compile(
                r'\\begin\{(figure\*?)\}.*?\\end\{\1\}',
                re.DOTALL,
            )

            new_content = content
            offset = 0

            for m in env_pattern.finditer(content):
                block = m.group(0)
                labels = re.findall(r'\\label\{([^}]+)\}', block)
                label = labels[0] if labels else None

                if label and label in seen_labels:
                    # Duplicate — remove the entire figure environment
                    start = m.start() + offset
                    end = m.end() + offset
                    new_content = new_content[:start] + new_content[end:]
                    offset += -(m.end() - m.start())
                    total_removed += 1
                elif label:
                    seen_labels.add(label)

            # Also remove within-section duplicates for unlabeled figures
            # by checking includegraphics targets
            generated_sections[section_type] = new_content.strip()

        if total_removed > 0:
            print(f"[DeduplicateFigures] Removed {total_removed} duplicate figure environments")

        return generated_sections

    @staticmethod
    def _strip_code_path_references(
        generated_sections: Dict[str, str],
    ) -> Dict[str, str]:
        """
        Remove raw code file path references from all generated sections.
        - **Description**:
            - Strips \\texttt{code/...} and \\texttt{*.py/.c/.cpp/.R/.jl} patterns
            - Removes surrounding prose like "implemented in \\texttt{...}"
            - Acts as a safety net in case the Writer ignores prompt rules

        - **Args**:
            - `generated_sections` (Dict[str, str]): section_type -> LaTeX content

        - **Returns**:
            - Updated generated_sections with code paths removed
        """
        import re as _re

        total_stripped = 0
        for section_type in list(generated_sections.keys()):
            content = generated_sections[section_type]
            original = content

            # Pattern 1: \texttt{code/...} or \texttt{path/to/file.py}
            content = _re.sub(
                r'(?:,?\s*(?:implemented|defined|derived|found|coded|written|specified|described)'
                r'\s+(?:in|from|within|via|using)\s+)?'
                r'\\texttt\{[^}]*(?:\.py|\.c|\.cc|\.cpp|\.R|\.jl|\.ipynb|code/)[^}]*\}',
                '', content,
                flags=_re.IGNORECASE,
            )
            # Pattern 2: \texttt{function_name} derived from \texttt{code/file.py}
            content = _re.sub(
                r'\s*\((?:derived\s+from|from|in|see)\s+\\texttt\{[^}]*(?:\.py|\.c|\.cpp|code/)[^}]*\}\)',
                '', content,
                flags=_re.IGNORECASE,
            )
            # Clean double spaces and orphaned commas
            content = _re.sub(r'\s*,\s*,', ',', content)
            content = _re.sub(r'  +', ' ', content)

            if content != original:
                total_stripped += 1
                generated_sections[section_type] = content

        if total_stripped > 0:
            print(f"[StripCodePaths] Cleaned code path references in {total_stripped} section(s)")

        return generated_sections

    def _ensure_tables_defined(
        self,
        generated_sections: Dict[str, str],
        paper_plan: Optional[PaperPlan],
        tables: Optional[List[TableSpec]],
        converted_tables: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:
        """
        Ensure all tables assigned for definition have their environments created.
        - **Description**:
            - Mirrors _ensure_figures_defined for tables.
            - If a table is in section_plan.tables_to_define but no \\begin{table}
              exists with matching label, inject the table environment using
              pre-converted LaTeX from converted_tables when available.

        - **Args**:
            - `generated_sections` (Dict[str, str]): Section contents keyed by type
            - `paper_plan` (Optional[PaperPlan]): Paper plan with table assignments
            - `tables` (Optional[List[TableSpec]]): Table specifications
            - `converted_tables` (Optional[Dict[str, str]]): table_id -> LaTeX code

        - **Returns**:
            - `generated_sections` (Dict[str, str]): Updated sections dict
        """
        import re

        if not paper_plan or not tables:
            return generated_sections

        _converted = converted_tables or {}

        # Build table lookup
        table_map = {tbl.id: tbl for tbl in tables}

        # Pre-scan ALL sections to find which tables are already defined
        # (prevents re-injecting tables moved to appendix)
        globally_defined_tables: set = set()
        all_content = "\n".join(generated_sections.values())
        for tbl in tables:
            tbl_pattern = rf'\\begin{{table\*?}}.*?\\label{{{re.escape(tbl.id)}}}.*?\\end{{table\*?}}'
            if re.search(tbl_pattern, all_content, re.DOTALL):
                globally_defined_tables.add(tbl.id)

        for section in paper_plan.sections:
            section_type = section.section_type
            tables_to_define = section.get_table_ids_to_define()

            if not tables_to_define or section_type not in generated_sections:
                continue

            content = generated_sections[section_type]

            for tbl_id in tables_to_define:
                tbl = table_map.get(tbl_id)
                if not tbl:
                    continue

                # Skip if already defined anywhere in the paper
                if tbl_id in globally_defined_tables:
                    continue

                # Table not defined — inject it
                print(f"[EnsureTables] Injecting missing table '{tbl_id}' in '{section_type}'")

                env_name = "table*" if tbl.wide else "table"

                if tbl_id in _converted:
                    # Use pre-converted LaTeX (best quality)
                    table_latex = _converted[tbl_id]
                    # Ensure it has a \label
                    if f"\\label{{{tbl_id}}}" not in table_latex:
                        # Insert label before \end{table...}
                        label_str = f"\\label{{{tbl_id}}}"
                        table_latex = re.sub(
                            rf'(\\end{{{env_name}}})',
                            lambda m: f"{label_str}\n{m.group(1)}",
                            table_latex,
                        )
                else:
                    # Generate a placeholder table
                    caption = tbl.caption or tbl_id
                    table_latex = (
                        f"\\begin{{{env_name}}}[htbp]\n"
                        f"\\centering\n"
                        f"\\caption{{{caption}}}\\label{{{tbl_id}}}\n"
                        f"\\begin{{tabular}}{{lcc}}\n"
                        f"\\hline\n"
                        f"Column 1 & Column 2 & Column 3 \\\\\n"
                        f"\\hline\n"
                        f"-- & -- & -- \\\\\n"
                        f"\\hline\n"
                        f"\\end{{tabular}}\n"
                        f"\\end{{{env_name}}}"
                    )

                # Find a good insertion point:
                # 1. After a sentence referencing this table
                # 2. Or at the end of the section
                ref_pattern = rf'(Table~?\\ref{{{re.escape(tbl_id)}}}[^.]*\.)'
                match = re.search(ref_pattern, content)
                if match:
                    insert_pos = match.end()
                    content = content[:insert_pos] + "\n" + table_latex + "\n" + content[insert_pos:]
                else:
                    # Append at the end of the section
                    content = content + "\n" + table_latex + "\n"

                generated_sections[section_type] = content

        return generated_sections

    @staticmethod
    def _normalize_float_placement(content: str) -> str:
        """
        Normalize figure/table placement hints to reduce end-of-document float piles.
        - **Description**:
            - Rewrites strict top-only placement ([t]) to [htbp].
            - Leaves existing broader placement options unchanged.

        - **Args**:
            - `content` (str): Section LaTeX content.

        - **Returns**:
            - `str`: Content with normalized float placement.
        """
        if not content:
            return content
        content = re.sub(r'\\begin\{figure\*?\}\[t\]', lambda m: m.group(0).replace('[t]', '[htbp]'), content)
        content = re.sub(r'\\begin\{table\*?\}\[t\]', lambda m: m.group(0).replace('[t]', '[htbp]'), content)
        return content

    def _collect_typesetter_figure_ids(
        self,
        generated_sections: Dict[str, str],
        figures: Optional[List[FigureSpec]],
        figure_paths: Optional[Dict[str, str]],
    ) -> List[str]:
        """
        Collect figure identifiers needed by the typesetter.
        - **Description**:
            - Combines IDs from metadata, includegraphics tags, and figure-path keys.
            - Keeps only likely IDs (not explicit file paths).

        - **Args**:
            - `generated_sections` (Dict[str, str]): Section contents.
            - `figures` (Optional[List[FigureSpec]]): Figure specs from metadata.
            - `figure_paths` (Optional[Dict[str, str]]): Explicit figure-id to file-path mapping.

        - **Returns**:
            - `figure_ids` (List[str]): Deduplicated figure IDs for typesetter resource resolution.
        """
        ids: set[str] = set()
        for fig in (figures or []):
            if getattr(fig, "id", None):
                ids.add(str(fig.id))
        for key in (figure_paths or {}).keys():
            if key:
                ids.add(str(key))

        pattern = r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}"
        for content in generated_sections.values():
            for raw in re.findall(pattern, content or ""):
                token = str(raw).strip()
                if not token:
                    continue
                if "/" in token or token.startswith(".") or token.endswith(
                    (".png", ".jpg", ".jpeg", ".pdf", ".svg")
                ):
                    continue
                ids.add(token)
        return sorted(ids)

    # =========================================================================
    # Pre-Generation Search Judgment (Phase A)
    # =========================================================================

    async def _judge_search_need(
        self,
        section_type: str,
        section_title: str,
        paper_title: str,
        key_points: List[str],
        ref_pool: ReferencePool,
    ) -> Dict[str, Any]:
        """
        Ask the LLM whether additional references are needed for a section.

        - **Description**:
            - Phase A of the two-phase generation pattern.
            - Sends a lightweight prompt asking the LLM to analyse the gap
              between the section's requirements and the existing references.
            - The LLM returns a structured JSON response with need_search,
              reason, and queries fields.
            - Does NOT use Function Calling — the LLM directly outputs JSON.

        - **Args**:
            - `section_type` (str): E.g. "introduction", "method".
            - `section_title` (str): Human-readable section title.
            - `paper_title` (str): Title of the paper being generated.
            - `key_points` (List[str]): Key points to cover in this section.
            - `ref_pool` (ReferencePool): Current reference pool.

        - **Returns**:
            - `dict`: {"need_search": bool, "reason": str, "queries": [str]}
              On any error, returns {"need_search": False, ...}.
        """
        # Build reference summaries for the prompt
        ref_summaries_parts = []
        for ref in ref_pool.get_all_refs():
            ref_id = ref.get("ref_id", "unknown")
            title = ref.get("title", "Untitled")
            year = ref.get("year", "?")
            ref_summaries_parts.append(f"  - [{ref_id}] {title} ({year})")
        ref_summaries = "\n".join(ref_summaries_parts) if ref_summaries_parts else "  (none)"

        key_points_str = "\n".join(f"  - {kp}" for kp in key_points) if key_points else "  (not specified)"

        prompt = SEARCH_JUDGMENT_PROMPT.format(
            section_type=section_type,
            section_title=section_title,
            paper_title=paper_title,
            key_points=key_points_str,
            n_refs=len(ref_pool.get_all_refs()),
            ref_summaries=ref_summaries,
        )

        print(f"[SearchJudge] Judging search need for {section_type} ({section_title})...")

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an academic research assistant. Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )
            raw = response.choices[0].message.content or ""
            # Strip markdown code fences if present
            raw = raw.strip()
            if raw.startswith("```"):
                # Remove opening fence (e.g. ```json)
                raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()

            result = json.loads(raw)

            need = result.get("need_search", False)
            reason = result.get("reason", "")
            queries = result.get("queries", [])

            # Sanitize
            if not isinstance(queries, list):
                queries = []
            queries = [q for q in queries if isinstance(q, str) and len(q.strip()) > 0]

            print(f"[SearchJudge] need_search={need}, reason={reason}")
            if queries:
                print(f"[SearchJudge] Suggested queries: {queries}")

            return {"need_search": bool(need), "reason": reason, "queries": queries}

        except (json.JSONDecodeError, KeyError) as e:
            print(f"[SearchJudge] Failed to parse LLM response: {e}")
            print(f"[SearchJudge] Raw response: {raw[:300]}")
            return {"need_search": False, "reason": f"parse error: {e}", "queries": []}
        except Exception as e:
            print(f"[SearchJudge] LLM call failed: {e}")
            return {"need_search": False, "reason": f"error: {e}", "queries": []}

    async def _execute_pre_searches(
        self,
        queries: List[str],
        ref_pool: ReferencePool,
    ) -> int:
        """
        Execute paper searches and merge valid results into the reference pool.

        - **Description**:
            - Phase A continuation: after the LLM provides search queries,
              this method directly calls PaperSearchTool.execute() for each
              query (without going through the ReAct loop).
            - Found papers are validated and added to the ref_pool.
            - Returns the count of newly added references.

        - **Args**:
            - `queries` (List[str]): Search queries from the judgment step.
            - `ref_pool` (ReferencePool): Persistent reference pool to update.

        - **Returns**:
            - `int`: Number of new references added to the pool.
        """
        from ..shared.tools.paper_search import PaperSearchTool

        paper_search_cfg = self.tools_config.paper_search if self.tools_config else None
        tool = PaperSearchTool(
            serpapi_api_key=(
                paper_search_cfg.serpapi_api_key if paper_search_cfg else None
            ),
            semantic_scholar_api_key=(
                paper_search_cfg.semantic_scholar_api_key if paper_search_cfg else None
            ),
            timeout=paper_search_cfg.timeout if paper_search_cfg else 10,
            semantic_scholar_min_results_before_fallback=(
                paper_search_cfg.semantic_scholar_min_results_before_fallback
                if paper_search_cfg else 3
            ),
            enable_query_cache=paper_search_cfg.enable_query_cache if paper_search_cfg else True,
            cache_ttl_hours=paper_search_cfg.cache_ttl_hours if paper_search_cfg else 24,
        )

        added_count = 0
        for i, query in enumerate(queries):
            # Add delay between queries to avoid rate limiting
            if i > 0:
                print("[PreSearch] Waiting 1.5s between queries to avoid rate limits...")
                await asyncio.sleep(1.5)

            print(f"[PreSearch] Executing search ({i+1}/{len(queries)}): '{query}'")
            try:
                per_round = paper_search_cfg.search_results_per_round if paper_search_cfg else 5
                result = await tool.execute(query=query, max_results=per_round)
                if not result.success:
                    print(f"[PreSearch] Search failed: {result.message}")
                    continue

                papers = result.data.get("papers", []) if result.data else []
                print(f"[PreSearch] Found {len(papers)} papers for '{query}'")

                for paper in papers:
                    bibtex = paper.get("bibtex", "")
                    cite_key = paper.get("bibtex_key", "") or paper.get("ref_id", "")
                    if not bibtex or not cite_key:
                        continue
                    if ref_pool.has_key(cite_key):
                        continue
                    added = ref_pool.add_discovered(cite_key, bibtex, source="pre_search")
                    if added:
                        added_count += 1
                        title = paper.get("title", "?")
                        print(f"[PreSearch] Added to ref_pool: [{cite_key}] {title}")

            except Exception as e:
                print(f"[PreSearch] Error searching '{query}': {e}")

        print(f"[PreSearch] Total new references added: {added_count} "
              f"(pool: {ref_pool.summary()})")
        return added_count

    def _validate_and_merge_new_references(
        self,
        content: str,
        msg_history: List[Dict[str, Any]],
        ref_pool: ReferencePool,
    ) -> str:
        """
        Two-layer validation of references discovered via search_papers.

        - **Description**:
            - Layer 1 (LLM judgment): The LLM already decided which papers
              to cite during the ReAct loop. We extract those decisions.
            - Layer 2 (System cross-reference): We verify that every \\cite{}
              key in the generated content either:
              (a) already exists in ref_pool (core or previously discovered), or
              (b) matches a BibTeX entry returned by search_papers — in which
                  case we add it to ref_pool as a discovered reference.
            - Keys that match neither (hallucinated) are removed from the content.

        - **Args**:
            - `content` (str): Generated LaTeX content.
            - `msg_history` (List[dict]): Full message history from react_loop.
            - `ref_pool` (ReferencePool): Persistent reference pool to update.

        - **Returns**:
            - `str`: Content with hallucinated citations removed.
        """
        # Extract BibTeX entries from search_papers tool results in message history
        search_results = ReferencePool.extract_search_results_from_history(msg_history)
        if search_results:
            print(f"[ValidateRefs] Found {len(search_results)} papers from search results")

        # Extract all \cite{} keys from the generated content
        cited_keys = ReferencePool.extract_cite_keys(content)
        if not cited_keys:
            return content

        print(f"[ValidateRefs] Content cites {len(cited_keys)} keys: {cited_keys}")

        # Check each cited key
        hallucinated_keys = []
        for key in cited_keys:
            if ref_pool.has_key(key):
                # Already in pool (core or previously discovered)
                continue

            if key in search_results:
                # Found in search results — add to pool
                added = ref_pool.add_discovered(key, search_results[key], source="search")
                if added:
                    print(f"[ValidateRefs] Added discovered ref: {key}")
                else:
                    print(f"[ValidateRefs] Duplicate discovered ref (skipped): {key}")
            else:
                # Not in pool and not from search — hallucinated
                hallucinated_keys.append(key)
                print(f"[ValidateRefs] Hallucinated key removed: {key}")

        # Remove hallucinated citations from content
        for key in hallucinated_keys:
            content = ReferencePool.remove_citation(content, key)

        if hallucinated_keys:
            print(f"[ValidateRefs] Removed {len(hallucinated_keys)} hallucinated citations")

        return content

    def _generate_bib_file(self, references: List[Dict[str, Any]]) -> str:
        """Generate .bib file content from parsed references"""
        bib_entries = []
        for ref in references:
            if ref.get("bibtex"):
                bib_entries.append(ref["bibtex"])
            else:
                # Generate a minimal entry
                ref_id = ref.get("ref_id", "unknown")
                title = ref.get("title", "Unknown Title")
                authors = ref.get("authors", "Unknown Author")
                year = ref.get("year", 2024)
                
                entry = f"""@article{{{ref_id},
  title = {{{title}}},
  author = {{{authors}}},
  year = {{{year}}},
}}"""
                bib_entries.append(entry)
        
        return "\n\n".join(bib_entries)
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters"""
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text
    
    # =========================================================================
    # Phase 4: PDF Compilation
    # =========================================================================
    
    async def _compile_pdf(
        self,
        generated_sections: Dict[str, str],
        template_path: str,
        references: List[Dict[str, Any]],
        output_dir: Path,
        paper_title: str,
        figures_source_dir: Optional[str] = None,
        figure_paths: Optional[Dict[str, str]] = None,
        converted_tables: Optional[Dict[str, str]] = None,
        paper_plan: Optional[PaperPlan] = None,
        figures: Optional[List[FigureSpec]] = None,
        metadata_tables: Optional[List[TableSpec]] = None,
    ) -> Tuple[Optional[str], Optional[str], List[str], Dict[str, List[str]]]:
        """
        Compile PDF using Typesetter Agent (multi-file mode).
        - **Description**:
            - Passes sections as a dict to the TypesetterAgent, which writes each
              section to its own .tex file and uses \\input{} in main.tex
            - This enables precise error-to-section mapping from LaTeX logs

        - **Args**:
            - `generated_sections` (Dict[str, str]): Section contents
            - `template_path` (str): Path to .zip template file
            - `references` (List[Dict[str, Any]]): Parsed reference list
            - `output_dir` (Path): Output directory
            - `paper_title` (str): Paper title
            - `figures_source_dir` (Optional[str]): Directory with figure files (legacy)
            - `figure_paths` (Optional[Dict[str, str]]): figure_id -> file_path
            - `converted_tables` (Optional[Dict[str, str]]): table_id -> LaTeX table code
            - `paper_plan` (Optional[PaperPlan]): Paper plan with figure/table assignments
            - `figures` (Optional[List[FigureSpec]]): Figure specifications
            - `metadata_tables` (Optional[List[TableSpec]]): Table specifications

        - **Returns**:
            - Tuple of (pdf_path, latex_path, compile_errors, section_errors)
            - On success: (pdf_path, latex_path, [], {})
            - On failure: (None, None, [error1, ...], {"section_type": [errors]})
        """
        print(f"[MetaDataAgent] Phase 4: Compiling PDF with template: {template_path}")
        if not (paper_title or "").strip():
            return None, None, ["missing_or_empty_paper_title"], {}
        
        try:
            # Dynamic: read section order and titles from the plan
            if paper_plan:
                section_order = paper_plan.get_compile_section_order()
                section_titles = paper_plan.get_section_titles()
            else:
                section_order = ["introduction", "related_work", "method", "experiment", "result", "conclusion"]
                section_titles = {
                    "introduction": "Introduction",
                    "related_work": "Related Work",
                    "method": "Methodology",
                    "experiment": "Experiments",
                    "result": "Results",
                    "conclusion": "Conclusion",
                }
            # Include any generated sections not in plan (e.g. appendix from review loop)
            for st in generated_sections:
                if st != "abstract" and st not in section_order:
                    section_order.append(st)
                if st not in section_titles:
                    section_titles[st] = st.replace("_", " ").title()
            
            # Final citation validation pass (safety net)
            valid_citation_keys = self._extract_valid_citation_keys(references)
            total_invalid_removed = 0
            for section_type in list(generated_sections.keys()):
                content = generated_sections[section_type]
                content = self._fix_latex_references(content)
                content = self._normalize_float_placement(content)
                fixed_content, invalid, valid = self._validate_and_fix_citations(
                    content, valid_citation_keys, remove_invalid=True
                )
                if invalid:
                    print(f"[CompilePDF] Removed {len(invalid)} invalid citations from {section_type}: {invalid[:3]}{'...' if len(invalid) > 3 else ''}")
                    total_invalid_removed += len(invalid)
                generated_sections[section_type] = fixed_content
            
            if total_invalid_removed > 0:
                print(f"[CompilePDF] Total invalid citations removed: {total_invalid_removed}")

            # Strip raw code file path references from all sections
            generated_sections = self._strip_code_path_references(generated_sections)

            # Ensure all assigned figures have their environments created
            if paper_plan and figures:
                generated_sections = self._ensure_figures_defined(
                    generated_sections=generated_sections,
                    paper_plan=paper_plan,
                    figures=figures,
                )

            # Deduplicate figure environments (cross-section and within-section)
            generated_sections = self._deduplicate_figure_environments(
                generated_sections,
                section_order=section_order,
            )

            # Ensure all assigned tables have their environments created
            if paper_plan and metadata_tables:
                generated_sections = self._ensure_tables_defined(
                    generated_sections=generated_sections,
                    paper_plan=paper_plan,
                    tables=metadata_tables,
                    converted_tables=converted_tables,
                )
            
            # Prepare references for Typesetter
            typesetter_refs = []
            for ref in references:
                if ref.get("bibtex"):
                    typesetter_refs.append({
                        "ref_id": ref.get("ref_id", ""),
                        "bibtex": ref.get("bibtex"),
                    })
            figure_ids = self._collect_typesetter_figure_ids(
                generated_sections=generated_sections,
                figures=figures,
                figure_paths=figure_paths,
            )
            
            # Call Typesetter Agent API with multi-file sections dict
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{os.getenv('AGENTSYS_SELF_URL', 'http://127.0.0.1:8000')}/agent/typesetter/compile",
                    json={
                        "request_id": str(uuid.uuid4()),
                        "payload": {
                            "sections": generated_sections,
                            "section_order": section_order,
                            "section_titles": section_titles,
                            "template_path": template_path,
                            "template_config": {
                                "paper_title": paper_title,
                                "paper_authors": "EasyPaper",
                            },
                            "references": typesetter_refs,
                            "figure_ids": figure_ids,
                            "output_dir": str(output_dir),
                            "figures_source_dir": figures_source_dir,
                            "figure_paths": figure_paths or {},
                            "converted_tables": converted_tables or {},
                        }
                    }
                )
                
                if response.status_code != 200:
                    print(f"[MetaDataAgent] Typesetter error: {response.status_code} - {response.text}")
                    return None, None, [f"Typesetter HTTP {response.status_code}"], {}
                
                result = response.json()
                
                if result.get("status") == "ok" and result.get("result"):
                    compilation_result = result["result"]
                    pdf_path = compilation_result.get("pdf_path")
                    latex_path = compilation_result.get("source_path")
                    compile_warnings = compilation_result.get("warnings", [])
                    section_errors = compilation_result.get("section_errors", {})
                    
                    if pdf_path:
                        print(f"[MetaDataAgent] PDF compiled successfully: {pdf_path}")
                        if compile_warnings:
                            print(f"[MetaDataAgent] Compile warnings: {compile_warnings[:5]}")
                        if section_errors:
                            print(f"[MetaDataAgent] Section errors (on success): {section_errors}")
                    else:
                        print(f"[MetaDataAgent] PDF compilation failed: compilation result has no pdf_path")

                    # Guard: ensure final main.tex contains required structure.
                    if latex_path:
                        main_tex_path = Path(latex_path) / "main.tex"
                        structure_errors = self._validate_main_tex_structure(main_tex_path)
                        if structure_errors:
                            print(f"[MetaDataAgent] main.tex structure validation failed: {structure_errors}")
                            return None, None, structure_errors, section_errors

                    return pdf_path, latex_path, [], section_errors
                else:
                    # Compilation failed - extract errors from result
                    compile_errors: List[str] = []
                    section_errors: Dict[str, List[str]] = {}
                    error_msg = result.get("error", "Unknown error")
                    if result.get("result"):
                        compile_errors = result["result"].get("errors", [])
                        section_errors = result["result"].get("section_errors", {})
                    if not compile_errors:
                        compile_errors = [e.strip() for e in error_msg.split(";") if e.strip()]
                    print(f"[MetaDataAgent] PDF compilation failed: {error_msg}")
                    print(f"[Typesetter] Compile errors: {compile_errors}")
                    if section_errors:
                        print(f"[Typesetter] Section errors: {section_errors}")
                    return None, None, compile_errors, section_errors
                
        except httpx.ConnectError:
            print("[MetaDataAgent] Error: Could not connect to Typesetter Agent")
            return None, None, ["Could not connect to Typesetter Agent"], {}
        except Exception as e:
            print(f"[MetaDataAgent] PDF compilation error: {e}")
            return None, None, [str(e)], {}

    @staticmethod
    def _validate_main_tex_structure(main_tex_path: Path) -> List[str]:
        """
        Validate that compiled main.tex contains non-empty title/abstract.
        - **Returns**:
            - `List[str]`: Validation errors; empty list means pass.
        """
        if not main_tex_path.exists():
            return [f"main.tex not found: {main_tex_path}"]
        try:
            text = main_tex_path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            return [f"cannot read main.tex: {e}"]

        errors: List[str] = []

        # Title
        title_match = re.search(r'\\title(?:\[[^\]]*\])?\{([^}]*)\}', text, flags=re.DOTALL)
        if not title_match or not title_match.group(1).strip():
            errors.append("missing_or_empty_title")

        # Abstract content: supports \abstract{...} and \begin{abstract}...\end{abstract}
        abstract_cmd = re.search(r'\\abstract\{([^}]*)\}', text, flags=re.DOTALL)
        abstract_env = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', text, flags=re.DOTALL)
        abstract_text = ""
        if abstract_cmd:
            abstract_text = abstract_cmd.group(1)
        elif abstract_env:
            abstract_text = abstract_env.group(1)
        if not abstract_text.strip():
            errors.append("missing_or_empty_abstract")

        return errors
    
    # =========================================================================
    # Phase 5: VLM Review
    # =========================================================================
    
    async def _call_vlm_review(
        self,
        pdf_path: str,
        page_limit: int = 8,
        template_type: str = "ICML",
        sections_info: Optional[Dict[str, Any]] = None,
        memory: Optional[SessionMemory] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Call VLM Review Agent directly (no HTTP) to check the PDF.
        - **Description**:
            - Builds a VLMReviewRequest and calls self._vlm_reviewer.review().
            - Memory context is injected automatically inside review().

        - **Args**:
            - `pdf_path` (str): Path to compiled PDF
            - `page_limit` (int): Maximum allowed pages
            - `template_type` (str): Template type for context
            - `sections_info` (Dict, optional): Section word counts
            - `memory` (SessionMemory, optional): Shared session memory

        - **Returns**:
            - VLM review result dict or None on failure
        """
        if self._vlm_reviewer is None:
            print("[MetaDataAgent] VLM Review Agent not available, skipping")
            return None
        try:
            from ..vlm_review_agent.models import VLMReviewRequest

            request = VLMReviewRequest(
                pdf_path=pdf_path,
                page_limit=page_limit,
                template_type=template_type,
                check_overflow=True,
                check_underfill=True,
                check_layout=False,
                sections_info=sections_info or {},
            )

            result = await self._vlm_reviewer.review(request, memory=memory)
            return result.model_dump()

        except Exception as e:
            print(f"[MetaDataAgent] VLM Review error: {e}")
            return None
    
    # =========================================================================
    # Phase 0: Planning Methods
    # =========================================================================
    
    async def _create_paper_plan(
        self,
        metadata: PaperMetaData,
        target_pages: Optional[int],
        style_guide: Optional[str],
        research_context: Optional[Dict[str, Any]] = None,
        code_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[PaperPlan]:
        """
        Create a paper plan by calling the Planner Agent
        
        Args:
            metadata: Paper metadata
            target_pages: Target page count
            style_guide: Writing style guide (e.g., "ICML")
            
        Returns:
            PaperPlan or None if planning fails
        """
        try:
            # Prepare figure info for planner
            figures_info = []
            for fig in metadata.figures:
                figures_info.append({
                    "id": fig.id,
                    "caption": fig.caption,
                    "description": fig.description,
                    "section": fig.section,
                    "wide": fig.wide,
                    "file_path": getattr(fig, "file_path", None) or "",
                })
            
            # Prepare table info for planner
            tables_info = []
            for tbl in metadata.tables:
                tables_info.append({
                    "id": tbl.id,
                    "caption": tbl.caption,
                    "description": tbl.description,
                    "section": tbl.section,
                    "wide": tbl.wide,
                    "file_path": getattr(tbl, "file_path", None) or "",
                })
            
            from ..planner_agent.models import PlanRequest, FigureInfo, TableInfo
            planner_code_brief = format_code_context_for_planner(
                context=code_context,
                style_guide=style_guide,
            ) if code_context else ""

            plan_request = PlanRequest(
                title=metadata.title,
                idea_hypothesis=metadata.idea_hypothesis,
                method=metadata.method,
                data=metadata.data,
                experiments=metadata.experiments,
                references=metadata.references,
                research_context=research_context,
                code_context=code_context,
                code_writing_assets={
                    **((code_context or {}).get("writing_assets", {}) or {}),
                    "planner_brief": planner_code_brief,
                } if code_context else None,
                figures=[FigureInfo(**fi) for fi in figures_info],
                tables=[TableInfo(**ti) for ti in tables_info],
                target_pages=target_pages,
                style_guide=style_guide,
            )

            paper_plan = await self._planner.create_plan(plan_request)
            return paper_plan

        except Exception as e:
            print(f"[MetaDataAgent] Planning error: {e}")
            return None
    
    # =========================================================================
    # Phase 3.5: Review Loop Methods
    # =========================================================================

    def _perform_baseline_gap_audit(
        self,
        generated_sections: Dict[str, str],
        enable_review: bool,
        enable_vlm_review: bool,
    ) -> Dict[str, Any]:
        """
        Build a module-level baseline gap audit snapshot.
        - **Description**:
            - Captures current capabilities against the LLM-upgrade goals
            - Stored into review export for explainability/audit trails
        """
        has_paragraph_content = any("\n\n" in (c or "") for c in generated_sections.values())
        return {
            "reviewer": {
                "has_hierarchical_feedback": True,
                "has_llm_orchestrator": True,
                "gap": "needs stronger cross-agent conflict arbitration",
            },
            "writer": {
                "writes_at_section_level": True,
                "has_paragraph_units": has_paragraph_content,
                "gap": "needs explicit revision plan execution constraints",
            },
            "metadata": {
                "has_unified_review_loop": bool(enable_review),
                "has_conflict_guardrails": True,
                "gap": "needs explainable LLM arbitration traces",
            },
            "vlm": {
                "enabled": bool(enable_vlm_review),
                "has_structural_strategy": True,
                "gap": "needs text-structure co-planning translation",
            },
            "export": {
                "iteration_centric": True,
                "has_agent_hierarchy": True,
                "gap": "decision/reason traces should be first-class fields",
            },
        }

    async def _llm_plan_revision_tasks(
        self,
        review_result: ReviewResult,
    ) -> List[Dict[str, Any]]:
        """
        Generate paragraph-level executable revision plans from tasks.
        """
        task_payload = [
            t.model_dump() if hasattr(t, "model_dump") else t
            for t in (review_result.revision_tasks or [])
        ]
        if not task_payload:
            return []

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a paragraph revision planner. Convert tasks into JSON-only "
                            "revision_plan entries with keys: task_id, section_type, target_paragraphs, "
                            "paragraph_instructions, preserve_claims, do_not_change, expected_change, priority, "
                            "issue_type, acceptance_criteria."
                        ),
                    },
                    {"role": "user", "content": json.dumps({"tasks": task_payload}, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=1800,
            )
            raw = response.choices[0].message.content or ""
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            parsed = json.loads(cleaned) if cleaned else {}
            plan = parsed.get("revision_plan", [])
            if isinstance(plan, list):
                return plan
        except Exception as e:
            print(f"[ReviewPlanner] LLM planner fallback: {e}")

        # Deterministic fallback
        fallback: List[Dict[str, Any]] = []
        for idx, task in enumerate(task_payload):
            section_type = str(task.get("section_type", ""))
            if not section_type:
                continue
            tps = [int(x) for x in task.get("paragraph_indices", []) if isinstance(x, int) or str(x).isdigit()]
            instruction = str(task.get("instruction", "")) or str(task.get("rationale", "Revise this section."))
            paragraph_instructions = {p: instruction for p in tps}
            fallback.append({
                "task_id": task.get("task_id", f"fallback_{idx}"),
                "section_type": section_type,
                "target_paragraphs": tps,
                "paragraph_instructions": paragraph_instructions,
                "preserve_claims": task.get("preserve_claims", []),
                "do_not_change": task.get("do_not_change", []),
                "expected_change": task.get("expected_change", ""),
                "priority": int(task.get("priority", 5)),
                "issue_type": str(task.get("issue_type", "other")),
                "acceptance_criteria": [str(x) for x in task.get("acceptance_criteria", [])],
            })
        return fallback

    def _apply_revision_plan_to_feedbacks(
        self,
        review_result: ReviewResult,
        revision_plan: List[Dict[str, Any]],
    ) -> None:
        """
        Merge revision-plan paragraph instructions into section feedbacks.
        """
        if not revision_plan:
            return
        by_section = {sf.section_type: sf for sf in review_result.section_feedbacks}
        for plan_item in revision_plan:
            section_type = str(plan_item.get("section_type", ""))
            if not section_type:
                continue
            sf = by_section.get(section_type)
            if sf is None:
                sf = SectionFeedback(
                    section_type=section_type,
                    current_word_count=0,
                    target_word_count=0,
                    action="refine_paragraphs",
                    delta_words=0,
                    revision_prompt=str(plan_item.get("expected_change", "")),
                    target_paragraphs=[],
                    paragraph_instructions={},
                )
                review_result.section_feedbacks.append(sf)
                by_section[section_type] = sf
            targets = self._normalize_target_paragraphs(plan_item.get("target_paragraphs", []))
            sf.target_paragraphs = sorted(list(set((sf.target_paragraphs or []) + targets)))
            normalized_instructions = self._normalize_paragraph_instructions(
                plan_item.get("paragraph_instructions", {}),
                target_paragraphs=sf.target_paragraphs,
                fallback_instruction=str(plan_item.get("expected_change", "")).strip(),
            )
            for k, v in normalized_instructions.items():
                try:
                    sf.paragraph_instructions[int(k)] = str(v)
                except Exception:
                    continue
            if sf.target_paragraphs and sf.action == "revise":
                sf.action = "refine_paragraphs"
            if plan_item.get("issue_type"):
                try:
                    sf.issue_type = plan_item.get("issue_type")
                except Exception:
                    pass
            criteria = plan_item.get("acceptance_criteria", []) or []
            if criteria:
                sf.acceptance_criteria = [str(x) for x in criteria]
            preserve_claims = plan_item.get("preserve_claims", [])
            do_not_change = plan_item.get("do_not_change", [])
            expected_change = plan_item.get("expected_change", "")
            extra = []
            if preserve_claims:
                extra.append(f"Preserve claims: {', '.join([str(x) for x in preserve_claims])}.")
            if do_not_change:
                extra.append(f"Do not change: {', '.join([str(x) for x in do_not_change])}.")
            if expected_change:
                extra.append(f"Expected change: {expected_change}.")
            if extra:
                sf.revision_prompt = (sf.revision_prompt + "\n\n" + " ".join(extra)).strip()

    @staticmethod
    def _normalize_target_paragraphs(raw_targets: Any) -> List[int]:
        """
        Normalizes paragraph targets from heterogeneous payloads.
        - **Description**:
         - Normalizes paragraph targets from heterogeneous payload types (int/str/list) into a deduplicated integer list.

        - **Args**:
         - `raw_targets` (Any): Raw paragraph target payload from planner/reviewer output.

        - **Returns**:
         - `targets` (List[int]): Deduplicated paragraph indices in ascending order.
        """
        if raw_targets is None:
            return []
        if isinstance(raw_targets, (int, str)):
            raw_targets = [raw_targets]
        if not isinstance(raw_targets, list):
            return []
        targets: List[int] = []
        for item in raw_targets:
            if isinstance(item, int):
                targets.append(item)
            elif isinstance(item, str) and item.strip().isdigit():
                targets.append(int(item.strip()))
        return sorted(list(set(targets)))

    @staticmethod
    def _normalize_paragraph_instructions(
        raw_instructions: Any,
        target_paragraphs: Optional[List[int]] = None,
        fallback_instruction: str = "",
    ) -> Dict[int, str]:
        """
        Normalizes paragraph instructions into a stable mapping.
        - **Description**:
         - Converts reviewer/planner outputs from `dict`, `list`, or `str` into `{paragraph_index: instruction}`.
         - Supports JSON-like string payloads and generic text instructions with paragraph-target fallback.

        - **Args**:
         - `raw_instructions` (Any): Raw paragraph instruction payload from planner/reviewer output.
         - `target_paragraphs` (Optional[List[int]]): Candidate paragraph indices used for fallback fan-out.
         - `fallback_instruction` (str): Backup instruction applied when parsing fails but targets exist.

        - **Returns**:
         - `normalized` (Dict[int, str]): Normalized paragraph instruction mapping.
        """
        normalized: Dict[int, str] = {}
        targets = target_paragraphs or []

        # dict form
        if isinstance(raw_instructions, dict):
            for k, v in raw_instructions.items():
                if str(k).isdigit():
                    normalized[int(k)] = str(v).strip()
            return normalized

        # list form
        if isinstance(raw_instructions, list):
            for item in raw_instructions:
                if isinstance(item, dict):
                    pidx = item.get("paragraph_index")
                    ins = item.get("instruction") or item.get("suggestion") or item.get("text")
                    if (isinstance(pidx, int) or str(pidx).isdigit()) and ins:
                        normalized[int(pidx)] = str(ins).strip()
                elif isinstance(item, str) and targets:
                    for t in targets:
                        normalized[t] = item.strip()
            return normalized

        # string form (generic or JSON-like)
        if isinstance(raw_instructions, str):
            text = raw_instructions.strip()
            if not text:
                return normalized

            # Try JSON object first
            if text.startswith("{") and text.endswith("}"):
                try:
                    parsed = json.loads(text)
                    if isinstance(parsed, dict):
                        for k, v in parsed.items():
                            if str(k).isdigit():
                                normalized[int(k)] = str(v).strip()
                        if normalized:
                            return normalized
                except Exception:
                    pass

            # Parse "3: xxx; 4: yyy" pattern
            pairs = re.findall(r"(\d+)\s*[:=-]\s*([^;]+)", text)
            for pidx, ins in pairs:
                normalized[int(pidx)] = ins.strip()
            if normalized:
                return normalized

            # Generic sentence => apply to all targets when available
            if targets:
                for t in targets:
                    normalized[t] = text
                return normalized

        # Fallback: if we have targets, use fallback instruction
        if targets and fallback_instruction:
            for t in targets:
                normalized[t] = fallback_instruction
        return normalized

    @staticmethod
    def _default_acceptance_criteria(issue_type: str) -> List[str]:
        """
        Build default acceptance gates for generalized issue handling.
        - **Description**:
         - Uses issue taxonomy to derive validation gates
         - Keeps contracts stable when reviewer does not supply criteria
        """
        base = ["execution_changed", "semantic_preserved"]
        normalized = str(issue_type or "other").strip().lower()
        if normalized == "logical_contradiction":
            return base + ["contradiction_resolved"]
        if normalized in ("claim_evidence_gap", "unsupported_generalization"):
            return base + ["evidence_sufficient"]
        return base

    async def _run_semantic_consistency_guard(
        self,
        section_type: str,
        before_text: str,
        after_text: str,
        revision_prompt: str,
    ) -> SemanticCheckRecord:
        """
        Check semantic consistency of before/after revisions.
        """
        if before_text.strip() == after_text.strip():
            return SemanticCheckRecord(
                section_type=section_type,
                passed=True,
                summary="No semantic delta detected.",
                risks=[],
                action_taken="accepted",
            )
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a semantic consistency guard for academic revisions. "
                            "Return JSON only with keys: passed (bool), summary (str), risks (list[str]). "
                            "Fail if key claims are dropped or contradictions are introduced."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "section_type": section_type,
                                "revision_prompt": revision_prompt[:1200],
                                "before": before_text[:7000],
                                "after": after_text[:7000],
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=600,
            )
            raw = response.choices[0].message.content or ""
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            parsed = json.loads(cleaned) if cleaned else {}
            passed = bool(parsed.get("passed", True))
            summary = str(parsed.get("summary", "Semantic check completed."))
            risks = [str(x) for x in parsed.get("risks", [])]
            return SemanticCheckRecord(
                section_type=section_type,
                passed=passed,
                summary=summary,
                risks=risks,
                action_taken="accepted" if passed else "rollback",
            )
        except Exception as e:
            return SemanticCheckRecord(
                section_type=section_type,
                passed=False,
                summary=f"Semantic guard failed due to error: {e}",
                risks=["semantic_guard_unavailable"],
                action_taken="rollback",
            )

    async def _translate_vlm_to_revision_plan(
        self,
        vlm_result: Dict[str, Any],
        generated_sections: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """
        Translate VLM section advice into joint text/structure revision tasks.
        """
        if not vlm_result:
            return []
        payload = {
            "summary": vlm_result.get("summary", ""),
            "needs_trim": bool(vlm_result.get("needs_trim", False)),
            "needs_expand": bool(vlm_result.get("needs_expand", False)),
            "section_recommendations": {
                sec: (advice.model_dump() if hasattr(advice, "model_dump") else advice)
                for sec, advice in (vlm_result.get("section_recommendations", {}) or {}).items()
            },
            "sections": {k: v[:1800] for k, v in generated_sections.items()},
        }
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You translate VLM layout feedback into executable JSON revision tasks. "
                            "Return JSON only with key vlm_revision_plan as list of objects: "
                            "section_type, action, delta_words, paragraph_instructions, rationale."
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ],
                temperature=0.2,
                max_tokens=1200,
            )
            raw = response.choices[0].message.content or ""
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)
            parsed = json.loads(cleaned) if cleaned else {}
            plan = parsed.get("vlm_revision_plan", [])
            if isinstance(plan, list):
                return plan
        except Exception as e:
            print(f"[VLMTranslate] LLM translator fallback: {e}")

        fallback = []
        for section_type, advice in (vlm_result.get("section_recommendations", {}) or {}).items():
            action = getattr(advice, "recommended_action", None) or advice.get("recommended_action")
            target_change = getattr(advice, "target_change", None) or advice.get("target_change") or 0
            fallback.append({
                "section_type": section_type,
                "action": "reduce" if action == "trim" else ("expand" if action == "expand" else "ok"),
                "delta_words": int(target_change or 0),
                "paragraph_instructions": {},
                "rationale": "VLM fallback translation",
            })
        return fallback

    async def _resolve_conflicts_with_llm(
        self,
        reviewer_feedbacks: List[SectionFeedback],
        external_feedbacks: List[SectionFeedback],
    ) -> Tuple[List[SectionFeedback], List[ConflictResolutionRecord]]:
        """
        Resolve conflicts between reviewer and external (typesetter/VLM) feedback.
        """
        guardrail_priority = {
            "fix_latex": 100,
            "reduce": 80,
            "expand": 20,
            "revise": 30,
            "refine_paragraphs": 40,
            "ok": 0,
        }
        merged = {sf.section_type: sf for sf in reviewer_feedbacks}
        conflict_records: List[ConflictResolutionRecord] = []
        for ext in external_feedbacks:
            existing = merged.get(ext.section_type)
            if existing is None:
                merged[ext.section_type] = ext
                continue
            if existing.action == ext.action:
                merged[ext.section_type] = self._merge_section_feedbacks(
                    [existing],
                    [ext],
                    prefer_vlm=False,
                )[0]
                continue
            # Hard guardrails first.
            if guardrail_priority.get(ext.action, 0) > guardrail_priority.get(existing.action, 0):
                chosen, rejected = ext, existing
                guardrail = "rule_guardrail_priority"
            elif guardrail_priority.get(ext.action, 0) < guardrail_priority.get(existing.action, 0):
                chosen, rejected = existing, ext
                guardrail = "rule_guardrail_priority"
            else:
                # LLM arbitration for same-priority soft conflicts.
                chosen = existing
                rejected = ext
                guardrail = None
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You arbitrate conflicting section revision actions. "
                                    "Prefer preserving compilation validity and page limits. "
                                    "Return JSON only: {selected: 'a'|'b', reason: '...'}."
                                ),
                            },
                            {
                                "role": "user",
                                "content": json.dumps(
                                    {
                                        "section": ext.section_type,
                                        "candidate_a": existing.model_dump(),
                                        "candidate_b": ext.model_dump(),
                                    },
                                    ensure_ascii=False,
                                ),
                            },
                        ],
                        temperature=0.1,
                        max_tokens=300,
                    )
                    raw = response.choices[0].message.content or ""
                    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
                    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
                    arb = json.loads(cleaned) if cleaned else {}
                    if str(arb.get("selected", "a")).lower() == "b":
                        chosen, rejected = ext, existing
                    reason = str(arb.get("reason", "LLM arbitration selected safer action."))
                except Exception:
                    reason = "Fallback arbitration kept existing action."
                conflict_records.append(ConflictResolutionRecord(
                    section_type=ext.section_type,
                    target_id=ext.target_id or existing.target_id or ext.section_type,
                    candidates=[existing.model_dump(), ext.model_dump()],
                    selected_action=chosen.action,
                    selected_source="llm_arbiter",
                    reason=reason,
                    applied_guardrail=guardrail,
                ))
                merged[ext.section_type] = chosen
                continue

            merged[ext.section_type] = chosen
            conflict_records.append(ConflictResolutionRecord(
                section_type=ext.section_type,
                target_id=ext.target_id or existing.target_id or ext.section_type,
                candidates=[existing.model_dump(), ext.model_dump()],
                selected_action=chosen.action,
                selected_source="rule_guardrail",
                reason=f"Selected '{chosen.action}' over '{rejected.action}' by guardrail priority.",
                applied_guardrail=guardrail,
            ))
        return list(merged.values()), conflict_records
    
    def _build_vlm_feedback(
        self,
        vlm_result: Dict[str, Any],
        structural_actions: Optional[List[StructuralAction]] = None,
    ) -> Tuple[List[FeedbackResult], List[SectionFeedback]]:
        """
        Build feedback and section revisions from VLM result.
        - **Description**:
            - Converts VLM review output into Reviewer-compatible feedback
            - Maps overflow/underfill to FeedbackResult and section advice to SectionFeedback
            - When structural_actions are provided, enriches revision prompts with
              global strategy context (e.g. what was moved to appendix, resized)

        - **Args**:
            - `vlm_result` (Dict[str, Any]): Raw VLM review result dict
            - `structural_actions` (Optional[List[StructuralAction]]): Planned/executed actions

        - **Returns**:
            - `feedbacks` (List[FeedbackResult]): Aggregated feedback results
            - `section_feedbacks` (List[SectionFeedback]): Per-section revision guidance
        """
        feedbacks: List[FeedbackResult] = []
        section_feedbacks: List[SectionFeedback] = []
        
        if not vlm_result:
            return feedbacks, section_feedbacks

        # try:
        #     print("[VLMReview] RawResult:\n" + json.dumps(vlm_result, ensure_ascii=False, indent=2))
        # except Exception as e:
        #     print(f"[VLMReview] RawResult log failed: {e}")
        
        overflow_pages = vlm_result.get("overflow_pages", 0)
        needs_trim = vlm_result.get("needs_trim", False)
        needs_expand = vlm_result.get("needs_expand", False)
        
        if overflow_pages > 0 or needs_trim:
            feedbacks.append(FeedbackResult(
                checker_name="vlm_review",
                passed=False,
                severity=Severity.ERROR,
                message=vlm_result.get("summary", "Page overflow detected"),
                details={
                    "overflow_pages": overflow_pages,
                    "needs_trim": True,
                    "source": "vlm_review",
                },
            ))
        elif needs_expand:
            feedbacks.append(FeedbackResult(
                checker_name="vlm_review",
                passed=False,
                severity=Severity.WARNING,
                message=vlm_result.get("summary", "Underfill detected"),
                details={
                    "needs_expand": True,
                    "source": "vlm_review",
                },
            ))
        else:
            feedbacks.append(FeedbackResult(
                checker_name="vlm_review",
                passed=True,
                severity=Severity.INFO,
                message=vlm_result.get("summary", "VLM review passed"),
                details={"source": "vlm_review"},
            ))
        
        print(
            "[VLMReview] Summary: "
            f"{vlm_result.get('summary', 'No summary')} | "
            f"overflow_pages={overflow_pages} needs_trim={needs_trim} needs_expand={needs_expand}"
        )
        
        # Build structural context string for enriched revision prompts
        structural_context = None
        if structural_actions:
            ctx_parts = [
                f"This paper exceeds the page limit by {overflow_pages:.1f} pages. "
                "The following structural adjustments have been applied:"
            ]
            moved_count = sum(1 for a in structural_actions if a.action_type in ("move_figure", "move_table"))
            resized_count = sum(1 for a in structural_actions if a.action_type == "resize_figure")
            downgraded_count = sum(1 for a in structural_actions if a.action_type == "downgrade_wide")
            appendix_created = any(a.action_type == "create_appendix" for a in structural_actions)

            if appendix_created:
                ctx_parts.append("- An Appendix section has been created.")
            if moved_count > 0:
                moved_ids = [a.target_id for a in structural_actions if a.action_type in ("move_figure", "move_table")]
                ctx_parts.append(f"- {moved_count} figure(s)/table(s) moved to Appendix: {', '.join(moved_ids)}.")
            if downgraded_count > 0:
                ctx_parts.append(f"- {downgraded_count} wide figure(s) converted to single-column.")
            if resized_count > 0:
                ctx_parts.append(f"- {resized_count} figure(s) resized to smaller width.")

            total_saved = sum(a.estimated_savings for a in structural_actions)
            remaining_trim = max(0, overflow_pages - total_saved)
            if remaining_trim > 0:
                ctx_parts.append(
                    f"After these adjustments, an estimated {remaining_trim:.1f} pages of word-level "
                    "trimming is still needed across sections."
                )
            structural_context = " ".join(ctx_parts)

        section_recommendations = vlm_result.get("section_recommendations", {}) or {}
        for section_type, advice in section_recommendations.items():
            if section_type == "appendix":
                continue
            recommended_action = getattr(advice, "recommended_action", None) or advice.get("recommended_action")
            target_change = getattr(advice, "target_change", None) or advice.get("target_change")
            guidance = getattr(advice, "specific_guidance", None) or advice.get("specific_guidance")
            
            if recommended_action == "trim":
                action = "reduce"
                delta_words = -abs(target_change) if target_change else 0
            elif recommended_action == "expand":
                action = "expand"
                delta_words = abs(target_change) if target_change else 0
            else:
                action = "ok"
                delta_words = 0
            
            # Build per-section structural action descriptors
            section_struct_actions = []
            if structural_actions:
                section_struct_actions = [
                    f"{a.action_type}:{a.target_id}"
                    for a in structural_actions
                    if a.section == section_type
                ]
            
            revision_prompt = self._build_vlm_revision_prompt(
                section_type=section_type,
                action=action,
                delta_words=delta_words,
                guidance=guidance,
                structural_context=structural_context if action == "reduce" else None,
            )
            
            section_feedbacks.append(SectionFeedback(
                section_type=section_type,
                current_word_count=0,
                target_word_count=0,
                action=action,
                delta_words=delta_words,
                revision_prompt=revision_prompt,
                structural_actions=section_struct_actions,
            ))
            print(
                "[VLMReview] Advice: "
                f"section={section_type} action={action} delta_words={delta_words} "
                f"guidance={guidance or 'n/a'}"
                + (f" structural_actions={section_struct_actions}" if section_struct_actions else "")
            )
        
        return feedbacks, section_feedbacks
    
    def _build_vlm_revision_prompt(
        self,
        section_type: str,
        action: str,
        delta_words: int,
        guidance: Optional[str] = None,
        structural_context: Optional[str] = None,
    ) -> str:
        """
        Build revision prompt from VLM guidance with optional structural context.
        - **Description**:
            - Produces a detailed revision instruction for the writer agent
            - When structural_context is provided, includes information about
              overall page-reduction strategy so the writer knows what has changed

        - **Args**:
            - `section_type` (str): Section name to revise
            - `action` (str): "reduce" or "expand"
            - `delta_words` (int): Target word change (+/-)
            - `guidance` (Optional[str]): Extra VLM guidance
            - `structural_context` (Optional[str]): Global strategy summary

        - **Returns**:
            - `prompt` (str): Revision instruction prompt
        """
        action_text = "reduce" if action == "reduce" else "expand"
        delta = abs(delta_words) if delta_words else 0

        parts = []

        # Global strategy context (if structural actions were planned)
        if structural_context:
            parts.append(structural_context)

        parts.append(
            f"Revise the {section_type} section to {action_text} by approximately {delta} words."
        )

        if action == "reduce":
            parts.append(
                "Prioritize: (1) removing redundant explanations of figures/tables that may "
                "have been moved to the Appendix, (2) condensing verbose passages, "
                "(3) merging overlapping sentences. Preserve factual consistency, "
                "citations, equations, and LaTeX formatting."
            )
        else:
            parts.append(
                "Prioritize expanding with additional evidence, details, or analysis. "
                "Preserve factual consistency, citations, and LaTeX formatting."
            )

        if guidance:
            parts.append(f"Additional guidance: {guidance}")

        return " ".join(parts)
    
    # =====================================================================
    # Smart page-limit control: structural overflow strategy
    # =====================================================================

    # Estimated page cost for non-text elements
    _ELEMENT_PAGE_COST = {
        "figure*": 0.4,
        "figure": 0.2,
        "table*": 0.3,
        "table": 0.15,
    }

    def _estimate_section_space(
        self,
        section_type: str,
        content: str,
    ) -> SpaceEstimate:
        """
        Estimate non-text space usage in a section.
        - **Description**:
            - Counts figure/table environments and their LaTeX labels
            - Returns a SpaceEstimate with per-type counts and total page cost

        - **Args**:
            - `section_type` (str): Section name (for logging)
            - `content` (str): LaTeX content of the section

        - **Returns**:
            - `SpaceEstimate` with element counts, ids, and estimated pages
        """
        est = SpaceEstimate()

        # Count wide/narrow figures
        est.wide_figures = len(re.findall(r"\\begin\{figure\*\}", content))
        est.narrow_figures = (
            len(re.findall(r"\\begin\{figure\}", content)) - est.wide_figures
        )
        if est.narrow_figures < 0:
            est.narrow_figures = 0

        # Count wide/narrow tables
        est.wide_tables = len(re.findall(r"\\begin\{table\*\}", content))
        est.narrow_tables = (
            len(re.findall(r"\\begin\{table\}", content)) - est.wide_tables
        )
        if est.narrow_tables < 0:
            est.narrow_tables = 0

        # Extract labels
        est.figure_ids = re.findall(r"\\label\{(fig:[^}]+)\}", content)
        est.table_ids = re.findall(r"\\label\{(tab:[^}]+)\}", content)

        # Estimate total pages consumed by non-text elements
        est.total_pages = (
            est.wide_figures * self._ELEMENT_PAGE_COST["figure*"]
            + est.narrow_figures * self._ELEMENT_PAGE_COST["figure"]
            + est.wide_tables * self._ELEMENT_PAGE_COST["table*"]
            + est.narrow_tables * self._ELEMENT_PAGE_COST["table"]
        )
        return est

    def _plan_overflow_strategy(
        self,
        overflow_pages: float,
        generated_sections: Dict[str, str],
        paper_plan: Optional[PaperPlan],
        figures: Optional[List[FigureSpec]],
    ) -> List[StructuralAction]:
        """
        Decide which structural actions to take based on overflow severity.
        - **Description**:
            - Level 1 (< 0.5 pages): word trimming only (no structural actions)
            - Level 2 (0.5-1.5 pages): downgrade figure* -> figure, resize images
            - Level 3 (> 1.5 pages): create appendix, move low-priority figures/tables
            - IMPORTANT: appendix section is excluded from scanning — it is the
              destination for moved elements, not a source.
            - Each figure/table ID is acted upon at most once (deduplication).

        - **Args**:
            - `overflow_pages` (float): How many pages over the limit
            - `generated_sections` (Dict[str, str]): Section contents
            - `paper_plan` (Optional[PaperPlan]): Paper plan (for targets)
            - `figures` (Optional[List[FigureSpec]]): Figure specs from metadata

        - **Returns**:
            - List of StructuralAction to execute before word-level revisions
        """
        from ..vlm_review_agent.models import SECTION_TRIM_PRIORITY

        actions: List[StructuralAction] = []
        remaining = overflow_pages

        if remaining <= 0:
            return actions

        # Step 1: gather space estimates per BODY section only (exclude appendix)
        space_map: Dict[str, SpaceEstimate] = {}
        for sec, content in generated_sections.items():
            if sec == "appendix":
                continue  # Never scan appendix — it is the move destination
            space_map[sec] = self._estimate_section_space(sec, content)

        print(
            f"[OverflowStrategy] overflow={overflow_pages:.1f} pages, "
            f"Level={'1' if overflow_pages < 0.5 else '2' if overflow_pages <= 1.5 else '3'}"
        )
        for sec, est in space_map.items():
            if est.total_pages > 0:
                print(
                    f"  {sec}: {est.wide_figures} figure*, {est.narrow_figures} figure, "
                    f"{est.wide_tables} table*, {est.narrow_tables} table  "
                    f"=> ~{est.total_pages:.2f} pages"
                )

        # Level 1: < 0.5 pages — word trim only (handled outside, return empty)
        if overflow_pages < 0.5:
            return actions

        # Track processed IDs to avoid duplicate actions on the same element
        processed_ids: set = set()

        # =================================================================
        # Level 2 actions: downgrade wide figures/tables + resize
        # =================================================================
        # Sort sections by trim priority (high -> low) for deterministic order
        sorted_sections = sorted(
            space_map.keys(),
            key=lambda s: SECTION_TRIM_PRIORITY.get(s, 5),
            reverse=True,
        )

        # 2a. Downgrade figure* -> figure (only sections that actually have wide figures)
        for sec in sorted_sections:
            est = space_map.get(sec)
            if not est or est.wide_figures == 0:
                continue
            for fid in est.figure_ids:
                if fid in processed_ids:
                    continue
                savings = 0.2
                actions.append(StructuralAction(
                    action_type="downgrade_wide",
                    target_id=fid,
                    section=sec,
                    estimated_savings=savings,
                ))
                processed_ids.add(fid)
                remaining -= savings
                print(f"  -> downgrade_wide figure {fid} in {sec} (save ~{savings:.1f}p)")
                if remaining <= 0:
                    break
            if remaining <= 0:
                break

        # 2b. Resize remaining figures (shrink width) — only if not already processed
        if remaining > 0:
            for sec in sorted_sections:
                est = space_map.get(sec)
                if not est or (est.wide_figures + est.narrow_figures) == 0:
                    continue
                for fid in est.figure_ids:
                    resize_key = f"resize:{fid}"
                    if resize_key in processed_ids:
                        continue
                    savings = 0.05
                    actions.append(StructuralAction(
                        action_type="resize_figure",
                        target_id=fid,
                        section=sec,
                        params={"width": "0.8\\linewidth"},
                        estimated_savings=savings,
                    ))
                    processed_ids.add(resize_key)
                    remaining -= savings
                    if remaining <= 0:
                        break
                if remaining <= 0:
                    break

        # =================================================================
        # Level 3 actions: create appendix + move low-priority figures/tables
        # =================================================================
        if overflow_pages > 1.5 and remaining > 0:
            if "appendix" not in generated_sections:
                actions.append(StructuralAction(
                    action_type="create_appendix",
                    estimated_savings=0,
                ))

            for sec in sorted_sections:
                est = space_map.get(sec)
                if not est:
                    continue
                # Move figures
                for fid in est.figure_ids:
                    move_key = f"move:{fid}"
                    if move_key in processed_ids:
                        continue
                    savings = (
                        self._ELEMENT_PAGE_COST["figure*"]
                        if est.wide_figures > 0
                        else self._ELEMENT_PAGE_COST["figure"]
                    )
                    actions.append(StructuralAction(
                        action_type="move_figure",
                        target_id=fid,
                        section=sec,
                        estimated_savings=savings,
                    ))
                    processed_ids.add(move_key)
                    remaining -= savings
                    print(f"  -> move_figure {fid} from {sec} to appendix (save ~{savings:.1f}p)")
                    if remaining <= 0:
                        break
                # Move tables
                if remaining > 0:
                    for tid in est.table_ids:
                        move_key = f"move:{tid}"
                        if move_key in processed_ids:
                            continue
                        savings = (
                            self._ELEMENT_PAGE_COST["table*"]
                            if est.wide_tables > 0
                            else self._ELEMENT_PAGE_COST["table"]
                        )
                        actions.append(StructuralAction(
                            action_type="move_table",
                            target_id=tid,
                            section=sec,
                            estimated_savings=savings,
                        ))
                        processed_ids.add(move_key)
                        remaining -= savings
                        print(f"  -> move_table {tid} from {sec} to appendix (save ~{savings:.1f}p)")
                        if remaining <= 0:
                            break
                if remaining <= 0:
                    break

        total_savings = sum(a.estimated_savings for a in actions)
        print(
            f"[OverflowStrategy] Planned {len(actions)} structural actions, "
            f"estimated savings ~{total_savings:.1f} pages"
        )
        return actions

    # =====================================================================
    # Structural operation execution methods
    # =====================================================================

    def _resize_figures_in_section(
        self,
        section_content: str,
        actions: List[StructuralAction],
    ) -> str:
        """
        Apply resize and downgrade-wide actions to a section's LaTeX content.
        - **Description**:
            - downgrade_wide: figure* -> figure, end{figure*} -> end{figure}
            - resize_figure: adjusts includegraphics width parameter
            - Pure regex operations, no LLM call needed

        - **Args**:
            - `section_content` (str): Current LaTeX content of the section
            - `actions` (List[StructuralAction]): Actions targeting this section

        - **Returns**:
            - Modified section content with figure adjustments applied
        """
        modified = section_content

        for act in actions:
            if act.action_type == "downgrade_wide":
                # Replace figure* -> figure (all occurrences in this section)
                modified = modified.replace("\\begin{figure*}", "\\begin{figure}")
                modified = modified.replace("\\end{figure*}", "\\end{figure}")
                print(f"  [Structural] Downgraded figure* -> figure for {act.target_id}")

            elif act.action_type == "resize_figure":
                target_width = act.params.get("width", "0.8\\linewidth")
                # Use lambda replacements to avoid regex interpreting
                # backslashes in LaTeX commands like \linewidth
                modified = re.sub(
                    r"\\includegraphics\[([^\]]*?)width\s*=\s*\\textwidth",
                    lambda m: f"\\includegraphics[{m.group(1)}width={target_width}",
                    modified,
                )
                modified = re.sub(
                    r"\\includegraphics\[([^\]]*?)width\s*=\s*\\linewidth",
                    lambda m: f"\\includegraphics[{m.group(1)}width={target_width}",
                    modified,
                )
                modified = re.sub(
                    r"\\includegraphics\[([^\]]*?)width\s*=\s*\\columnwidth",
                    lambda m: f"\\includegraphics[{m.group(1)}width={target_width}",
                    modified,
                )
                print(f"  [Structural] Resized figures to {target_width} for {act.target_id}")

        return modified

    def _move_figures_to_appendix(
        self,
        generated_sections: Dict[str, str],
        actions: List[StructuralAction],
    ) -> None:
        """
        Move figure/table environments from source sections to the appendix.
        - **Description**:
            - Extracts the full \\begin{figure}...\\end{figure} block containing the target label
            - Replaces it in the source section with a cross-reference note
            - Appends the extracted block to generated_sections["appendix"]
            - Operates in-place on generated_sections dict

        - **Args**:
            - `generated_sections` (Dict[str, str]): All section contents (mutated)
            - `actions` (List[StructuralAction]): move_figure / move_table actions
        """
        # Ensure appendix section exists
        if "appendix" not in generated_sections:
            generated_sections["appendix"] = ""

        appendix_parts: List[str] = []
        if generated_sections["appendix"]:
            appendix_parts.append(generated_sections["appendix"])

        # Track labels already in appendix to prevent duplicates
        existing_labels = set(re.findall(
            r'\\label\{([^}]+)\}', generated_sections["appendix"],
        ))

        for act in actions:
            if act.action_type not in ("move_figure", "move_table"):
                continue

            sec = act.section
            content = generated_sections.get(sec, "")
            if not content:
                continue

            target_label = act.target_id  # e.g. "fig:arch"

            if target_label in existing_labels:
                print(f"  [Structural] Skipping {target_label} — already in appendix")
                continue

            # Determine environment type
            if act.action_type == "move_figure":
                env_names = ["figure\\*", "figure"]
                ref_text = f"Figure~\\\\ref{{{target_label}}}"
            else:
                env_names = ["table\\*", "table"]
                ref_text = f"Table~\\\\ref{{{target_label}}}"

            extracted = False
            for env in env_names:
                # Match the environment that contains this specific label
                pattern = re.compile(
                    r"(\\begin\{" + env + r"\}.*?\\label\{" + re.escape(target_label) + r"\}.*?\\end\{" + env + r"\})",
                    re.DOTALL,
                )
                m = pattern.search(content)
                if m:
                    block = m.group(1)
                    # Replace the block with a cross-reference
                    replacement = f"% [{target_label} moved to Appendix]\n(see {ref_text} in the Appendix)"
                    content = content[:m.start()] + replacement + content[m.end():]
                    generated_sections[sec] = content
                    appendix_parts.append(block)
                    existing_labels.add(target_label)
                    extracted = True
                    print(f"  [Structural] Moved {target_label} from {sec} to appendix")
                    break

            if not extracted:
                # Try reverse order: label might appear before the begin
                for env in env_names:
                    pattern = re.compile(
                        r"(\\begin\{" + env + r"\}.*?\\end\{" + env + r"\})",
                        re.DOTALL,
                    )
                    for m in pattern.finditer(content):
                        if target_label in m.group(1):
                            block = m.group(1)
                            replacement = f"% [{target_label} moved to Appendix]\n(see {ref_text} in the Appendix)"
                            content = content[:m.start()] + replacement + content[m.end():]
                            generated_sections[sec] = content
                            appendix_parts.append(block)
                            existing_labels.add(target_label)
                            extracted = True
                            print(f"  [Structural] Moved {target_label} from {sec} to appendix (alt)")
                            break
                    if extracted:
                        break

            if not extracted:
                print(f"  [Structural] WARNING: Could not find {target_label} in {sec}")

        generated_sections["appendix"] = "\n\n".join(appendix_parts)

    def _create_appendix_section(
        self,
        generated_sections: Dict[str, str],
        section_order: List[str],
    ) -> None:
        """
        Create an appendix section if it doesn't already exist.
        - **Description**:
            - Adds "appendix" key to generated_sections
            - Inserts "appendix" after "conclusion" in section_order

        - **Args**:
            - `generated_sections` (Dict[str, str]): Section dict (mutated)
            - `section_order` (List[str]): Section ordering list (mutated)
        """
        if "appendix" not in generated_sections:
            generated_sections["appendix"] = ""
            print("[Structural] Created appendix section")

        if "appendix" not in section_order:
            # Insert after conclusion if it exists, else at the end
            if "conclusion" in section_order:
                idx = section_order.index("conclusion") + 1
                section_order.insert(idx, "appendix")
            else:
                section_order.append("appendix")
            print(f"[Structural] Added appendix to section_order at position {section_order.index('appendix')}")

    def _execute_structural_actions(
        self,
        actions: List[StructuralAction],
        generated_sections: Dict[str, str],
        section_order: List[str],
    ) -> None:
        """
        Execute all structural actions in order: create appendix, then move, then resize.
        - **Description**:
            - Groups actions by type and applies them in the correct order
            - Mutates generated_sections and section_order in place

        - **Args**:
            - `actions` (List[StructuralAction]): Planned structural actions
            - `generated_sections` (Dict[str, str]): Section contents (mutated)
            - `section_order` (List[str]): Section ordering (mutated)
        """
        if not actions:
            return

        print(f"[Structural] Executing {len(actions)} structural actions...")

        # Phase 1: create appendix if needed
        create_actions = [a for a in actions if a.action_type == "create_appendix"]
        if create_actions:
            self._create_appendix_section(generated_sections, section_order)

        # Phase 2: move figures/tables to appendix
        move_actions = [a for a in actions if a.action_type in ("move_figure", "move_table")]
        if move_actions:
            # Ensure appendix exists before moving anything
            if "appendix" not in generated_sections:
                self._create_appendix_section(generated_sections, section_order)
            self._move_figures_to_appendix(generated_sections, move_actions)

        # Phase 3: resize/downgrade in remaining sections
        resize_actions = [a for a in actions if a.action_type in ("resize_figure", "downgrade_wide")]
        if resize_actions:
            # Group by section
            by_section: Dict[str, List[StructuralAction]] = {}
            for a in resize_actions:
                by_section.setdefault(a.section, []).append(a)
            for sec, sec_actions in by_section.items():
                if sec in generated_sections:
                    generated_sections[sec] = self._resize_figures_in_section(
                        generated_sections[sec], sec_actions
                    )

        print("[Structural] All structural actions executed")

    # LaTeX error patterns mapped to fix instructions
    LATEX_ERROR_FIXES: Dict[str, str] = {
        "ended by": (
            "Fix unclosed LaTeX environment. Ensure every \\begin{{...}} has a matching \\end{{...}}. "
            "Check figure, table, and equation environments."
        ),
        "misplaced alignment tab character &": (
            "Escape all literal '&' characters as '\\&' in regular text. "
            "Only use bare '&' inside tabular/align environments."
        ),
        "unicode character": (
            "Replace Unicode characters with LaTeX equivalents. "
            "For example: use \\textendash for –, $-$ or $\\minus$ for −, "
            "\\% for %, \\& for &."
        ),
        "missing $ inserted": (
            "Fix math mode errors. Wrap mathematical symbols like _, ^, "
            "\\alpha, \\beta in $...$ when used outside math environments."
        ),
        "undefined control sequence": (
            "Remove or replace undefined LaTeX commands. "
            "Check for typos in command names or missing package imports."
        ),
        "not in outer par mode": (
            "Move float environments (figure, table) out of restricted contexts. "
            "Floats cannot appear inside minipage, parbox, or other floats."
        ),
        "file not found": (
            "Remove or comment out \\includegraphics for missing figure files. "
            "Replace with a placeholder comment if needed."
        ),
        "no output pdf file produced": (
            "Fix critical LaTeX errors that prevent PDF generation. "
            "Check for unclosed environments, invalid commands, and encoding issues."
        ),
    }
    
    def _build_typesetter_feedback(
        self,
        compile_errors: List[str],
        generated_sections: Dict[str, str],
        section_errors: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[List[FeedbackResult], List[SectionFeedback]]:
        """
        Build feedback from LaTeX compilation errors.
        - **Description**:
            - Converts compilation errors into reviewer-compatible feedback
            - When section_errors is provided (multi-file mode), uses precise
              error-to-section mapping from the LaTeX log file tracking
            - Falls back to heuristic content matching when section_errors is not available

        - **Args**:
            - `compile_errors` (List[str]): Error messages from LaTeX compiler
            - `generated_sections` (Dict[str, str]): Current section contents
            - `section_errors` (Dict[str, List[str]], optional): Pre-mapped section -> errors

        - **Returns**:
            - `feedbacks` (List[FeedbackResult]): Aggregated feedback results
            - `section_feedbacks` (List[SectionFeedback]): Per-section revision guidance
        """
        feedbacks: List[FeedbackResult] = []
        section_feedbacks: List[SectionFeedback] = []
        
        if not compile_errors and not section_errors:
            return feedbacks, section_feedbacks
        
        total_errors = len(compile_errors) if compile_errors else sum(
            len(v) for v in (section_errors or {}).values()
        )
        print(f"[Typesetter] Building feedback from {total_errors} compile errors")
        
        # Build a combined feedback result
        feedbacks.append(FeedbackResult(
            checker_name="typesetter",
            passed=False,
            severity=Severity.ERROR,
            message=f"LaTeX compilation failed with {total_errors} error(s): {'; '.join(compile_errors[:3]) if compile_errors else 'see section_errors'}",
            details={
                "source": "typesetter",
                "compile_errors": compile_errors or [],
                "section_errors": section_errors or {},
            },
        ))
        
        # =====================================================================
        # Multi-file mode: precise section_errors mapping available
        # =====================================================================
        if section_errors:
            for sec_type, sec_errs in section_errors.items():
                if sec_type not in generated_sections:
                    continue
                if not sec_errs:
                    continue
                
                revision_parts = [
                    "Fix the following LaTeX compilation errors in this section:\n"
                ]
                for err in sec_errs:
                    err_lower = err.lower()
                    matched_fix = False
                    for pattern, fix in self.LATEX_ERROR_FIXES.items():
                        if pattern in err_lower:
                            revision_parts.append(f"- {err}: {fix}")
                            matched_fix = True
                            break
                    if not matched_fix:
                        revision_parts.append(f"- {err}: Review and correct this LaTeX error.")
                revision_parts.append(
                    "\nOutput ONLY valid LaTeX. Do NOT use unescaped special characters "
                    "(&, %, $, #, _, {, }) in regular text."
                )
                
                section_feedbacks.append(SectionFeedback(
                    section_type=sec_type,
                    current_word_count=len(generated_sections.get(sec_type, "").split()),
                    target_word_count=0,
                    action="fix_latex",
                    delta_words=0,
                    revision_prompt="\n".join(revision_parts),
                ))
                print(f"[Typesetter] Targeted fix (multi-file): section={sec_type} errors={sec_errs[:3]}")
            
            # Handle any errors not attributed to a specific section
            attributed_errors = set()
            for errs in section_errors.values():
                attributed_errors.update(errs)
            unattributed = [e for e in (compile_errors or []) if e not in attributed_errors]
            if unattributed and not section_feedbacks:
                # If no section feedbacks were created from section_errors,
                # fall through to the heuristic/broadcast path below
                compile_errors = unattributed
            elif section_feedbacks:
                return feedbacks, section_feedbacks
        
        # =====================================================================
        # Fallback: heuristic matching or broadcast
        # =====================================================================
        if not compile_errors:
            return feedbacks, section_feedbacks
        
        # Collect fix instructions for all errors
        fix_instructions: List[str] = []
        for error in compile_errors:
            error_lower = error.lower()
            for pattern, fix in self.LATEX_ERROR_FIXES.items():
                if pattern in error_lower:
                    fix_instructions.append(f"Error: {error}\nFix: {fix}")
                    break
            else:
                fix_instructions.append(f"Error: {error}\nFix: Review and correct this LaTeX error.")
        
        # Try to locate errors to specific sections by scanning content
        section_error_map: Dict[str, List[str]] = {}
        for error in compile_errors:
            error_lower = error.lower()
            matched_section = None
            
            if "figure" in error_lower or "includegraphics" in error_lower:
                for section_type, content in generated_sections.items():
                    if "\\begin{figure" in content or "\\includegraphics" in content:
                        matched_section = section_type
                        break
            elif "tabular" in error_lower or "alignment tab" in error_lower:
                for section_type, content in generated_sections.items():
                    if "\\begin{tabular" in content or "\\begin{table" in content or "&" in content:
                        matched_section = section_type
                        break
            
            if matched_section:
                if matched_section not in section_error_map:
                    section_error_map[matched_section] = []
                section_error_map[matched_section].append(error)
        
        if section_error_map:
            for section_type, sec_errs in section_error_map.items():
                revision_parts = [
                    "Fix the following LaTeX compilation errors in this section:\n"
                ]
                for err in sec_errs:
                    err_lower = err.lower()
                    for pattern, fix in self.LATEX_ERROR_FIXES.items():
                        if pattern in err_lower:
                            revision_parts.append(f"- {err}: {fix}")
                            break
                    else:
                        revision_parts.append(f"- {err}: Review and correct.")
                revision_parts.append(
                    "\nOutput ONLY valid LaTeX. Do NOT use unescaped special characters "
                    "(&, %, $, #, _, {, }) in regular text."
                )
                
                section_feedbacks.append(SectionFeedback(
                    section_type=section_type,
                    current_word_count=len(generated_sections.get(section_type, "").split()),
                    target_word_count=0,
                    action="fix_latex",
                    delta_words=0,
                    revision_prompt="\n".join(revision_parts),
                ))
                print(f"[Typesetter] Targeted fix (heuristic): section={section_type} errors={sec_errs}")
        else:
            # Cannot locate to specific section - broadcast to all sections
            all_fix_prompt = (
                "Fix the following LaTeX compilation errors in this section:\n"
                + "\n".join(f"- {inst}" for inst in fix_instructions)
                + "\n\nOutput ONLY valid LaTeX. Ensure all environments are properly closed. "
                "Escape special characters (&, %, $, #, _, {, }) in regular text."
            )
            for section_type in generated_sections:
                if section_type == "appendix":
                    continue
                section_feedbacks.append(SectionFeedback(
                    section_type=section_type,
                    current_word_count=len(generated_sections.get(section_type, "").split()),
                    target_word_count=0,
                    action="fix_latex",
                    delta_words=0,
                    revision_prompt=all_fix_prompt,
                ))
            print(f"[Typesetter] Broadcast fix to all {len(generated_sections)} sections (excl. appendix)")
        
        return feedbacks, section_feedbacks
    
    def _merge_section_feedbacks(
        self,
        base_feedbacks: List[SectionFeedback],
        vlm_feedbacks: List[SectionFeedback],
        prefer_vlm: bool,
    ) -> List[SectionFeedback]:
        """
        Merge section feedbacks with conflict resolution.
        - **Description**:
            - Merges reviewer and VLM section feedback
            - Resolves conflicts based on prefer_vlm flag
        
        - **Args**:
            - `base_feedbacks` (List[SectionFeedback]): Reviewer-driven feedback
            - `vlm_feedbacks` (List[SectionFeedback]): VLM-driven feedback
            - `prefer_vlm` (bool): Whether to override conflicts with VLM advice
        
        - **Returns**:
            - `merged` (List[SectionFeedback]): Merged section feedback list
        """
        merged: Dict[str, SectionFeedback] = {fb.section_type: fb for fb in base_feedbacks}
        
        for fb in vlm_feedbacks:
            existing = merged.get(fb.section_type)
            if not existing:
                merged[fb.section_type] = fb
                continue
            
            # fix_latex always takes priority — compilation must succeed first
            if fb.action == "fix_latex" and existing.action != "fix_latex":
                merged[fb.section_type] = fb
            elif existing.action == "fix_latex":
                # Keep existing fix_latex; append new prompt if also fix_latex
                if fb.action == "fix_latex":
                    existing.revision_prompt += "\n\n" + fb.revision_prompt
            elif existing.action != fb.action and prefer_vlm:
                merged[fb.section_type] = fb
            elif existing.action == fb.action and abs(fb.delta_words) > abs(existing.delta_words):
                merged[fb.section_type] = fb
            else:
                # Preserve paragraph-level targets/instructions from both sources.
                merged_targets = sorted(
                    list(set((existing.target_paragraphs or []) + (fb.target_paragraphs or [])))
                )
                existing.target_paragraphs = merged_targets
                if fb.paragraph_instructions:
                    existing.paragraph_instructions.update(fb.paragraph_instructions)
                if fb.paragraph_feedbacks:
                    existing.paragraph_feedbacks.extend(fb.paragraph_feedbacks)
                if fb.revision_prompt and fb.revision_prompt not in (existing.revision_prompt or ""):
                    if existing.revision_prompt:
                        existing.revision_prompt += "\n\n" + fb.revision_prompt
                    else:
                        existing.revision_prompt = fb.revision_prompt
        
        return list(merged.values())
    
    def _resolve_section_feedbacks(
        self,
        section_feedbacks: List[SectionFeedback],
        revised_sections: set,
        review_result: ReviewResult,
    ) -> None:
        """
        Mark section feedbacks as resolved after revision.
        - **Description**:
            - Clears revision prompts for sections already revised
            - Updates review_result.requires_revision accordingly
        
        - **Args**:
            - `section_feedbacks` (List[SectionFeedback]): Feedback list to update
            - `revised_sections` (set): Sections that were revised
            - `review_result` (ReviewResult): Review result to update
        
        - **Returns**:
            - `None`
        """
        if not revised_sections:
            return
        
        for sf in section_feedbacks:
            if sf.section_type in revised_sections:
                sf.action = "ok"
                sf.delta_words = 0
                sf.revision_prompt = ""
        
        for section_type in list(review_result.requires_revision.keys()):
            if section_type in revised_sections:
                review_result.requires_revision.pop(section_type, None)
    
    async def _apply_revisions(
        self,
        review_result: ReviewResult,
        generated_sections: Dict[str, str],
        sections_results: List[SectionResult],
        valid_citation_keys: set,
        metadata: PaperMetaData,
        memory: Optional[SessionMemory] = None,
        semantic_checks: Optional[List[SemanticCheckRecord]] = None,
        decision_trace: Optional[List[Dict[str, Any]]] = None,
        writer_response_section: Optional[List[Dict[str, Any]]] = None,
        writer_response_paragraph: Optional[List[Dict[str, Any]]] = None,
        reviewer_verification: Optional[List[Dict[str, Any]]] = None,
    ) -> set:
        """
        Apply revisions based on a unified review result.
        - **Description**:
            - Uses review_result.section_feedbacks to revise sections
            - Updates generated_sections and sections_results in place
            - Injects revision history from memory to prevent regression
        
        - **Args**:
            - `review_result` (ReviewResult): Unified review result
            - `generated_sections` (Dict[str, str]): Section contents
            - `sections_results` (List[SectionResult]): Section results to update
            - `valid_citation_keys` (set): Valid citation keys
            - `metadata` (PaperMetaData): Original metadata for context
            - `memory` (SessionMemory, optional): Session memory for revision context
        
        - **Returns**:
            - `revised_sections` (set): Section types that were revised
        """
        revised_sections: set = set()
        if not review_result or not review_result.section_feedbacks:
            return revised_sections
        
        for sf in review_result.section_feedbacks:
            if sf.action == "ok":
                continue
            if sf.section_type not in generated_sections:
                continue
            
            revision_prompt = sf.revision_prompt
            if not revision_prompt:
                continue
            task_contract = {
                "target": sf.target_id or sf.section_type,
                "target_level": "paragraph" if sf.target_paragraphs else "section",
                "instruction": revision_prompt,
                "issue_type": str(getattr(sf, "issue_type", "other") or "other"),
                "constraints": {
                    "preserve_claims": ["Preserve factual claims and citations."],
                    "do_not_change": [],
                },
                "acceptance_criteria": self._default_acceptance_criteria(
                    str(getattr(sf, "issue_type", "other") or "other")
                ),
                "target_paragraphs": sf.target_paragraphs or [],
                "paragraph_instructions": sf.paragraph_instructions or {},
            }
            if getattr(sf, "acceptance_criteria", None):
                task_contract["acceptance_criteria"] = [str(x) for x in (sf.acceptance_criteria or [])]
            if memory is not None:
                issue_ctx = memory.get_issue_context(limit=60)
                unresolved = issue_ctx.get("unresolved_issues", [])
                hard_rules = []
                soft_hints = []
                for issue in unresolved:
                    sec = str(issue.get("section_type", ""))
                    target_id = str(issue.get("target_id", ""))
                    if sec not in ("", sf.section_type) and not target_id.startswith(f"{sf.section_type}."):
                        continue
                    msg = str(issue.get("message", ""))[:220]
                    if str(issue.get("locked_mode", "soft")) == "hard":
                        hard_rules.append(msg)
                    else:
                        soft_hints.append(msg)
                if hard_rules:
                    revision_prompt = (
                        "HARD LOCK ISSUES (must remain resolved; do not reintroduce):\n"
                        + "\n".join([f"- {x}" for x in hard_rules[:8]])
                        + "\n\n"
                        + revision_prompt
                    )
                if soft_hints:
                    revision_prompt = (
                        revision_prompt
                        + "\n\nSOFT LOCK GUIDANCE (avoid regression if possible):\n"
                        + "\n".join([f"- {x}" for x in soft_hints[:6]])
                    )
            
            print(
                "[ReviewLoop] Applying revision: "
                f"section={sf.section_type} action={sf.action} delta_words={sf.delta_words}"
            )
            
            current_content = generated_sections[sf.section_type]
            before_content = current_content

            # Paragraph-first revision path (higher precision than whole-section rewrite).
            if sf.target_paragraphs:
                revised_content = await self._revise_section_paragraphs(
                    section_type=sf.section_type,
                    current_content=current_content,
                    target_paragraphs=sf.target_paragraphs,
                    paragraph_instructions=sf.paragraph_instructions or {},
                    fallback_prompt=revision_prompt,
                    metadata=metadata,
                    memory=memory,
                    task_contract_base=task_contract,
                    valid_citation_keys=valid_citation_keys,
                )
            else:
                revised_content = await self._revise_section(
                    section_type=sf.section_type,
                    current_content=current_content,
                    revision_prompt=revision_prompt,
                    metadata=metadata,
                    memory=memory,
                    task_contract=task_contract,
                    valid_citation_keys=valid_citation_keys,
                )
            
            if revised_content:
                revised_content = self._fix_latex_references(revised_content)
                revised_content, invalid_citations, _ = self._validate_and_fix_citations(
                    revised_content, valid_citation_keys, remove_invalid=True
                )
                if invalid_citations:
                    print(f"[ReviewLoop] Removed {len(invalid_citations)} invalid citations from {sf.section_type}: {invalid_citations[:3]}{'...' if len(invalid_citations) > 3 else ''}")
                    if decision_trace is not None:
                        decision_trace.append(
                            {
                                "section_type": sf.section_type,
                                "decision": "removed_invalid_citations",
                                "count": len(invalid_citations),
                                "keys": list(invalid_citations),
                            }
                        )
                semantic_record = await self._run_semantic_consistency_guard(
                    section_type=sf.section_type,
                    before_text=before_content,
                    after_text=revised_content,
                    revision_prompt=revision_prompt,
                )
                if semantic_checks is not None:
                    semantic_checks.append(semantic_record)
                if not semantic_record.passed:
                    # Roll back unsafe revisions, then attempt one fallback section rewrite.
                    if decision_trace is not None:
                        decision_trace.append({
                            "section_type": sf.section_type,
                            "decision": "rollback_revision",
                            "reason": semantic_record.summary,
                            "risks": semantic_record.risks,
                        })
                    fallback_content = await self._revise_section(
                        section_type=sf.section_type,
                        current_content=before_content,
                        revision_prompt=revision_prompt,
                        metadata=metadata,
                        memory=memory,
                        task_contract=task_contract,
                        valid_citation_keys=valid_citation_keys,
                    )
                    if fallback_content:
                        fallback_content = self._fix_latex_references(fallback_content)
                        fallback_content, _, _ = self._validate_and_fix_citations(
                            fallback_content, valid_citation_keys, remove_invalid=True
                        )
                        fallback_semantic = await self._run_semantic_consistency_guard(
                            section_type=sf.section_type,
                            before_text=before_content,
                            after_text=fallback_content,
                            revision_prompt=revision_prompt,
                        )
                        if semantic_checks is not None:
                            semantic_checks.append(fallback_semantic)
                        if fallback_semantic.passed:
                            revised_content = fallback_content
                            if decision_trace is not None:
                                decision_trace.append({
                                    "section_type": sf.section_type,
                                    "decision": "accept_fallback_section_rewrite",
                                    "reason": fallback_semantic.summary,
                                })
                        else:
                            if decision_trace is not None:
                                decision_trace.append({
                                    "section_type": sf.section_type,
                                    "decision": "keep_original_after_failed_semantic_checks",
                                    "reason": fallback_semantic.summary,
                                    "risks": fallback_semantic.risks,
                                })
                            continue
                    else:
                        continue
                
                generated_sections[sf.section_type] = revised_content
                new_word_count = len(revised_content.split())
                if writer_response_section is not None:
                    writer_response_section.append({
                        "target_id": sf.target_id or sf.section_type,
                        "section_type": sf.section_type,
                        "target": task_contract["target"],
                        "instruction": task_contract["instruction"],
                        "constraints": task_contract["constraints"],
                        "disposition": "executed",
                        "source_agent": "writer",
                        "evidence": {
                            "before_words": len(before_content.split()),
                            "after_words": new_word_count,
                        },
                    })
                if writer_response_paragraph is not None:
                    final_paragraphs = self._split_section_paragraphs(revised_content)
                    final_para_count = len(final_paragraphs)
                    for pidx in (sf.target_paragraphs or []):
                        mapped_to: List[int] = []
                        mapping_strategy = "identity"
                        if 0 <= int(pidx) < final_para_count:
                            mapped_to = [int(pidx)]
                        elif final_para_count > 0:
                            mapped_to = [min(max(int(pidx), 0), final_para_count - 1)]
                            mapping_strategy = "clamped_nearest"
                        else:
                            mapping_strategy = "no_paragraphs"
                        writer_response_paragraph.append({
                            "target_id": f"{sf.section_type}.p{int(pidx)}",
                            "section_type": sf.section_type,
                            "paragraph_index": int(pidx),
                            "target": f"{sf.section_type}.p{int(pidx)}",
                            "disposition": "executed",
                            "instruction": str((task_contract.get("paragraph_instructions", {}) or {}).get(int(pidx), "")),
                            "constraints": task_contract["constraints"],
                            "source_agent": "writer",
                            "evidence": {
                                "content_changed": True,
                                "mapped_to_paragraph_indices": mapped_to,
                                "mapping_strategy": mapping_strategy,
                                "final_paragraph_count": final_para_count,
                            },
                        })
                if reviewer_verification is not None:
                    verify_result: Dict[str, Any] = {
                        "section_type": sf.section_type,
                        "target": task_contract["target"],
                        "passed": semantic_record.passed,
                        "summary": semantic_record.summary,
                        "checker": "reviewer_verifier",
                    }
                    if hasattr(self._reviewer, "verify_execution"):
                        try:
                            verify_result = await self._reviewer.verify_execution(
                                section_type=sf.section_type,
                                task_contract=task_contract,
                                before_text=before_content,
                                after_text=revised_content,
                                semantic_passed=semantic_record.passed,
                                semantic_summary=semantic_record.summary,
                            )
                        except Exception:
                            pass
                    reviewer_verification.append(verify_result)
                
                for sr in sections_results:
                    if sr.section_type == sf.section_type:
                        sr.latex_content = revised_content
                        sr.word_count = new_word_count
                        break
                
                revised_sections.add(sf.section_type)
                print(f"[MetaDataAgent] Revised {sf.section_type}: {new_word_count} words")
        
        return revised_sections

    @staticmethod
    def _split_section_paragraphs(content: str) -> List[str]:
        """
        Split section content into paragraph units.
        - **Description**:
            - Uses blank lines as paragraph boundaries
            - Preserves paragraph order for stable paragraph IDs
        """
        if not content:
            return []
        return [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]

    @staticmethod
    def _join_section_paragraphs(paragraphs: List[str]) -> str:
        """Join paragraph units back into section LaTeX text."""
        return "\n\n".join([p for p in paragraphs if p is not None]).strip()

    async def _revise_section_paragraphs(
        self,
        section_type: str,
        current_content: str,
        target_paragraphs: List[int],
        paragraph_instructions: Dict[int, str],
        fallback_prompt: str,
        metadata: PaperMetaData,
        memory: Optional[SessionMemory] = None,
        task_contract_base: Optional[Dict[str, Any]] = None,
        valid_citation_keys: Optional[set] = None,
    ) -> Optional[str]:
        """
        Revise selected paragraphs, then reassemble the section.
        - **Description**:
            - Applies targeted paragraph-level revision prompts
            - Falls back to whole-section revision if paragraph pass fails
        """
        paragraphs = self._split_section_paragraphs(current_content)
        if not paragraphs:
            return await self._revise_section(
                section_type=section_type,
                current_content=current_content,
                revision_prompt=fallback_prompt,
                metadata=metadata,
                memory=memory,
                task_contract=task_contract_base,
                valid_citation_keys=valid_citation_keys,
            )

        revised_any = False
        for pidx in sorted(set(target_paragraphs)):
            if pidx < 0 or pidx >= len(paragraphs):
                continue
            instruction = paragraph_instructions.get(
                pidx,
                "Improve this paragraph according to reviewer feedback while preserving factual correctness.",
            )
            revised_paragraph = await self._revise_paragraph(
                section_type=section_type,
                paragraph_index=pidx,
                paragraph_text=paragraphs[pidx],
                instruction=instruction,
                memory=memory,
                task_contract=task_contract_base,
                valid_citation_keys=valid_citation_keys,
            )
            if revised_paragraph and revised_paragraph.strip():
                paragraphs[pidx] = revised_paragraph.strip()
                revised_any = True

        if revised_any:
            return self._join_section_paragraphs(paragraphs)

        # Fallback to section-level rewrite if paragraph-level pass yields no change.
        return await self._revise_section(
            section_type=section_type,
            current_content=current_content,
            revision_prompt=fallback_prompt,
            metadata=metadata,
            memory=memory,
            task_contract=task_contract_base,
            valid_citation_keys=valid_citation_keys,
        )
    
    def _get_sections_fingerprint(self, sections: Dict[str, str]) -> str:
        """
        Build a stable fingerprint for section content.
        - **Description**:
            - Generates a hash string from section contents
            - Used to detect no-op revisions
        
        - **Args**:
            - `sections` (Dict[str, str]): Section contents
        
        - **Returns**:
            - `fingerprint` (str): SHA-256 fingerprint
        """
        import hashlib
        hasher = hashlib.sha256()
        for section_type in sorted(sections.keys()):
            hasher.update(section_type.encode("utf-8"))
            hasher.update(b"\n")
            hasher.update(sections[section_type].encode("utf-8"))
            hasher.update(b"\n")
        return hasher.hexdigest()
    
    async def _run_review_orchestration(
        self,
        generated_sections: Dict[str, str],
        sections_results: List[SectionResult],
        metadata: PaperMetaData,
        parsed_refs: List[Dict[str, Any]],
        paper_plan: Optional[PaperPlan],
        template_path: Optional[str],
        figures_source_dir: Optional[str],
        converted_tables: Dict[str, str],
        max_review_iterations: int,
        enable_review: bool,
        compile_pdf: bool,
        enable_vlm_review: bool,
        target_pages: Optional[int],
        paper_dir: Optional[Path],
        memory: Optional[SessionMemory] = None,
    ) -> Tuple[Dict[str, str], List[SectionResult], int, Optional[int], Optional[str], List[str]]:
        """
        Run unified review orchestration across reviewer and VLM.
        - **Description**:
            - Executes reviewer checks and VLM review in a single loop
            - Applies revisions, recompiles, and rechecks until pass or limit
        
        - **Args**:
            - `generated_sections` (Dict[str, str]): Section contents
            - `sections_results` (List[SectionResult]): Section result list
            - `metadata` (PaperMetaData): Paper metadata
            - `parsed_refs` (List[Dict[str, Any]]): Parsed references
            - `paper_plan` (Optional[PaperPlan]): Paper plan with targets
            - `template_path` (Optional[str]): Template path for PDF
            - `figures_source_dir` (Optional[str]): Figure source directory
            - `converted_tables` (Dict[str, str]): Converted table LaTeX
            - `max_review_iterations` (int): Maximum review iterations
            - `enable_review` (bool): Enable ReviewerAgent checks
            - `compile_pdf` (bool): Compile PDF if template_path is provided
            - `enable_vlm_review` (bool): Enable VLM-based PDF review
            - `target_pages` (Optional[int]): Target page count
            - `paper_dir` (Optional[Path]): Output directory
        
        - **Returns**:
            - `generated_sections` (Dict[str, str]): Updated sections
            - `sections_results` (List[SectionResult]): Updated results
            - `review_iterations` (int): Iteration count used
            - `target_word_count` (Optional[int]): Target word count from reviewer
            - `pdf_path` (Optional[str]): Latest compiled PDF path
            - `errors` (List[str]): Orchestration errors
        """
        errors: List[str] = []
        review_iterations = 0
        target_word_count = None
        pdf_path = None
        last_fingerprint = self._get_sections_fingerprint(generated_sections)
        last_compiled_fingerprint = last_fingerprint  # track what was last compiled
        last_vlm_result = None
        
        if enable_review:
            print(f"[MetaDataAgent] Unified Review Loop (max {max_review_iterations} iterations)...")
        baseline_gap_audit = self._perform_baseline_gap_audit(
            generated_sections=generated_sections,
            enable_review=enable_review,
            enable_vlm_review=enable_vlm_review,
        )
        if memory is not None:
            memory.log(
                "metadata",
                "review_init",
                "baseline_gap_audit",
                narrative="Captured baseline capability-gap audit for LLM enhancement rollout.",
                audit=baseline_gap_audit,
            )
        
        for iteration in range(max_review_iterations):
            review_iterations = iteration + 1
            iteration_decision_trace: List[Dict[str, Any]] = []
            iteration_conflicts: List[ConflictResolutionRecord] = []
            iteration_semantic_checks: List[SemanticCheckRecord] = []
            iteration_revision_plan: List[Dict[str, Any]] = []
            iteration_writer_response_section: List[Dict[str, Any]] = []
            iteration_writer_response_paragraph: List[Dict[str, Any]] = []
            iteration_reviewer_verification: List[Dict[str, Any]] = []
            print(f"[MetaDataAgent] Review iteration {review_iterations}/{max_review_iterations}")
            print(
                "[MetaDataAgent] Review context: "
                f"target_pages={target_pages or 8} enable_review={enable_review} enable_vlm_review={enable_vlm_review}"
            )
            
            word_counts = {
                sr.section_type: sr.word_count
                for sr in sections_results
                if sr.status == "ok"
            }
            
            review_result = ReviewResult(iteration=iteration)

            # Reference coverage pass: proactively integrate uncited assigned refs.
            ref_revised_sections: Set[str] = set()
            if paper_plan:
                ref_revised_sections = await self._enforce_reference_coverage(
                    generated_sections=generated_sections,
                    sections_results=sections_results,
                    paper_plan=paper_plan,
                    metadata=metadata,
                    valid_ref_keys=self._extract_valid_citation_keys(parsed_refs),
                    memory=memory,
                    max_sections_to_revise=2,
                )
                if ref_revised_sections:
                    if memory:
                        memory.log(
                            "metadata",
                            f"review_iter_{review_iterations}",
                            "reference_coverage_revised",
                            narrative=(
                                "Applied targeted revisions to improve citation coverage "
                                f"in sections: {', '.join(sorted(ref_revised_sections))}."
                            ),
                            revised_sections=sorted(ref_revised_sections),
                        )

            if enable_review:
                # Build section_targets from plan so Reviewer uses
                # the same targets as the Planner (unified ratios).
                section_targets = None
                if paper_plan and paper_plan.sections:
                    section_targets = {
                        s.section_type: s.get_estimated_words()
                        for s in paper_plan.sections
                    }
                section_structure_signals = None
                if paper_plan and paper_plan.sections:
                    section_structure_signals = {
                        s.section_type: {
                            "topic_clusters": list(s.topic_clusters or []),
                            "transition_intents": list(s.transition_intents or []),
                            "sectioning_recommended": bool(s.sectioning_recommended),
                            "paragraph_count": len(s.paragraphs or []),
                        }
                        for s in paper_plan.sections
                    }
                reviewer_result, target_word_count = await self._call_reviewer(
                    sections=generated_sections,
                    word_counts=word_counts,
                    target_pages=target_pages,
                    style_guide=metadata.style_guide,
                    template_path=template_path,
                    iteration=iteration,
                    section_targets=section_targets,
                    section_structure_signals=section_structure_signals,
                    memory=memory,
                )
                if reviewer_result is None:
                    print("[MetaDataAgent] Reviewer not available, skipping content review")
                else:
                    review_result = ReviewResult(**reviewer_result)
                    iteration_decision_trace.append({
                        "source": "reviewer_orchestrator",
                        "summary": review_result.orchestrator_summary or "Reviewer orchestration completed.",
                        "num_tasks": len(review_result.revision_tasks),
                    })
                    # Build paragraph-level executable plan and project it to section feedbacks.
                    iteration_revision_plan = await self._llm_plan_revision_tasks(review_result)
                    self._apply_revision_plan_to_feedbacks(review_result, iteration_revision_plan)
                    print(
                        "[Reviewer] Result: "
                        f"passed={review_result.passed} "
                        f"sections_to_revise={list(review_result.requires_revision.keys())}"
                    )
                    
                    # Conflict resolution: if the previous VLM iteration detected
                    # page overflow, suppress any Reviewer "expand" feedback.
                    # Page limits are a hard constraint; word count targets are soft.
                    if last_vlm_result and last_vlm_result.get("needs_trim"):
                        suppressed = [
                            sf.section_type
                            for sf in review_result.section_feedbacks
                            if sf.action == "expand"
                        ]
                        if suppressed:
                            review_result.section_feedbacks = [
                                sf for sf in review_result.section_feedbacks
                                if sf.action != "expand"
                            ]
                            for sec in suppressed:
                                review_result.requires_revision.pop(sec, None)
                            print(
                                f"[ReviewLoop] Suppressed Reviewer 'expand' for "
                                f"{suppressed} (VLM overflow active)"
                            )
            
            reviewer_revised_sections = await self._apply_revisions(
                review_result=review_result,
                generated_sections=generated_sections,
                sections_results=sections_results,
                valid_citation_keys=self._extract_valid_citation_keys(parsed_refs),
                metadata=metadata,
                memory=memory,
                semantic_checks=iteration_semantic_checks,
                decision_trace=iteration_decision_trace,
                writer_response_section=iteration_writer_response_section,
                writer_response_paragraph=iteration_writer_response_paragraph,
                reviewer_verification=iteration_reviewer_verification,
            )
            if reviewer_revised_sections:
                print(f"[ReviewLoop] Reviewer revised: {sorted(reviewer_revised_sections)}")
            self._resolve_section_feedbacks(
                section_feedbacks=review_result.section_feedbacks,
                revised_sections=reviewer_revised_sections,
                review_result=review_result,
            )
            word_counts = {
                sr.section_type: sr.word_count
                for sr in sections_results
                if sr.status == "ok"
            }
            print(f"[ReviewLoop] Word counts: {word_counts}")
            
            # Compile PDF and run VLM review if enabled
            compile_succeeded = False
            last_compiled_fingerprint = self._get_sections_fingerprint(generated_sections)
            if compile_pdf and template_path and paper_dir:
                iteration_dir = paper_dir / f"iteration_{review_iterations:02d}"
                iteration_dir.mkdir(parents=True, exist_ok=True)
                print(f"[ReviewLoop] PDF output dir: {iteration_dir}")
                figure_base_path = os.getcwd()
                figure_paths = self._collect_figure_paths(metadata.figures, base_path=figure_base_path)
                pdf_result_path, _, compile_errors, section_errors = await self._compile_pdf(
                    generated_sections=generated_sections,
                    template_path=template_path,
                    references=parsed_refs,
                    output_dir=iteration_dir,
                    paper_title=metadata.title,
                    figures_source_dir=figures_source_dir,
                    figure_paths=figure_paths,
                    converted_tables=converted_tables,
                    paper_plan=paper_plan,
                    figures=metadata.figures,
                    metadata_tables=metadata.tables,
                )
                if pdf_result_path:
                    pdf_path = pdf_result_path
                    compile_succeeded = True
                else:
                    # Compilation failed -> treat as review feedback, not a hard exit
                    print(f"[ReviewLoop] PDF compilation failed, treating as Typesetter review feedback")
                    ts_feedbacks, ts_section_feedbacks = self._build_typesetter_feedback(
                        compile_errors=compile_errors,
                        section_errors=section_errors,
                        generated_sections=generated_sections,
                    )
                    for fb in ts_feedbacks:
                        review_result.add_feedback(fb)
                    
                    # Merge typesetter section feedbacks
                    merged_seed = self._merge_section_feedbacks(
                        review_result.section_feedbacks,
                        ts_section_feedbacks,
                        prefer_vlm=False,  # Typesetter fixes take priority via action="fix_latex"
                    )
                    merged_section_feedbacks, ts_conflicts = await self._resolve_conflicts_with_llm(
                        reviewer_feedbacks=review_result.section_feedbacks,
                        external_feedbacks=merged_seed,
                    )
                    iteration_conflicts.extend(ts_conflicts)
                    review_result.section_feedbacks = merged_section_feedbacks
                    for sf in ts_section_feedbacks:
                        review_result.add_section_revision(sf.section_type, "Typesetter LaTeX fix")
                    
                    review_result.passed = False
                    # Skip VLM review this iteration (no PDF to review)
                
                if compile_succeeded and enable_vlm_review and pdf_path:
                    print(f"[MetaDataAgent] VLM Review: pdf_path={pdf_path}")
                    last_vlm_result = await self._call_vlm_review(
                        pdf_path=pdf_path,
                        page_limit=target_pages or 8,
                        template_type=metadata.style_guide or "ICML",
                        sections_info={
                            sr.section_type: {"word_count": sr.word_count}
                            for sr in sections_results if sr.word_count
                        },
                        memory=memory,
                    )
                    if last_vlm_result:
                        print(
                            "[VLMReview] Result: "
                            f"passed={last_vlm_result.get('passed', False)} "
                            f"overflow_pages={last_vlm_result.get('overflow_pages', 0)} "
                            f"needs_trim={last_vlm_result.get('needs_trim', False)} "
                            f"needs_expand={last_vlm_result.get('needs_expand', False)} "
                            f"sections={list((last_vlm_result.get('section_recommendations') or {}).keys())}"
                        )
                        
                        # ==========================================================
                        # Smart page-limit control: plan & execute structural actions
                        # BEFORE building VLM feedback so context is in prompts
                        # ==========================================================
                        planned_structural_actions: List[StructuralAction] = []
                        if last_vlm_result.get("needs_trim"):
                            overflow = last_vlm_result.get("overflow_pages", 0)
                            if overflow > 0:
                                section_order = list(generated_sections.keys())
                                if paper_plan and paper_plan.sections:
                                    section_order = [s.section_type for s in paper_plan.sections]
                                
                                planned_structural_actions = self._plan_overflow_strategy(
                                    overflow_pages=overflow,
                                    generated_sections=generated_sections,
                                    paper_plan=paper_plan,
                                    figures=metadata.figures,
                                )
                                if planned_structural_actions:
                                    self._execute_structural_actions(
                                        planned_structural_actions,
                                        generated_sections,
                                        section_order,
                                    )
                                    # Update sections_results after structural changes
                                    for sr in sections_results:
                                        if sr.section_type in generated_sections:
                                            sr.latex_content = generated_sections[sr.section_type]
                                            sr.word_count = len(generated_sections[sr.section_type].split())
                                    print(
                                        f"[ReviewLoop] Structural actions applied: "
                                        f"{len(planned_structural_actions)} actions, "
                                        f"total estimated savings ~"
                                        f"{sum(a.estimated_savings for a in planned_structural_actions):.1f} pages"
                                    )
                        
                        # Build VLM feedback with structural context baked in
                        vlm_feedbacks, vlm_section_feedbacks = self._build_vlm_feedback(
                            last_vlm_result,
                            structural_actions=planned_structural_actions or None,
                        )
                        # LLM translation: VLM -> executable text/structure revision tasks
                        translated_vlm_plan = await self._translate_vlm_to_revision_plan(
                            vlm_result=last_vlm_result,
                            generated_sections=generated_sections,
                        )
                        for p in translated_vlm_plan:
                            st = str(p.get("section_type", ""))
                            if not st:
                                continue
                            action = str(p.get("action", "ok"))
                            delta = int(p.get("delta_words", 0) or 0)
                            prompt = str(p.get("rationale", "")) or "Apply VLM-guided layout refinements."
                            section_struct_actions = []
                            if planned_structural_actions:
                                section_struct_actions = [
                                    f"{a.action_type}:{a.target_id}"
                                    for a in planned_structural_actions
                                    if a.section == st
                                ]
                            vlm_section_feedbacks.append(SectionFeedback(
                                section_type=st,
                                current_word_count=0,
                                target_word_count=0,
                                action=action,
                                delta_words=delta,
                                revision_prompt=prompt,
                                structural_actions=section_struct_actions,
                                target_paragraphs=self._normalize_target_paragraphs(
                                    p.get("target_paragraphs", [])
                                ),
                                paragraph_instructions=self._normalize_paragraph_instructions(
                                    p.get("paragraph_instructions", {}),
                                    target_paragraphs=self._normalize_target_paragraphs(
                                        p.get("target_paragraphs", [])
                                    ),
                                    fallback_instruction=prompt,
                                ),
                            ))
                        if translated_vlm_plan:
                            iteration_decision_trace.append({
                                "source": "vlm_translator_llm",
                                "summary": f"Translated {len(translated_vlm_plan)} VLM recommendation(s) into revision plan.",
                            })
                        for fb in vlm_feedbacks:
                            review_result.add_feedback(fb)
                        
                        prefer_vlm = bool(last_vlm_result.get("needs_trim") or last_vlm_result.get("needs_expand"))
                        merged_seed = self._merge_section_feedbacks(
                            review_result.section_feedbacks,
                            vlm_section_feedbacks,
                            prefer_vlm=prefer_vlm,
                        )
                        merged_section_feedbacks, conflicts = await self._resolve_conflicts_with_llm(
                            reviewer_feedbacks=review_result.section_feedbacks,
                            external_feedbacks=merged_seed,
                        )
                        iteration_conflicts.extend(conflicts)
                        review_result.section_feedbacks = merged_section_feedbacks
                        for sf in review_result.section_feedbacks:
                            if sf.section_type in word_counts:
                                sf.current_word_count = word_counts.get(sf.section_type, 0)
                            if paper_plan:
                                section_plan = paper_plan.get_section(sf.section_type)
                                if section_plan:
                                    est = section_plan.get_estimated_words()
                                    if est > 0:
                                        sf.target_word_count = est
                        
                        for sf in merged_section_feedbacks:
                            if sf.action != "ok":
                                review_result.add_section_revision(sf.section_type, "VLM adjustment")
                    else:
                        print("[MetaDataAgent] VLM review unavailable, skipping")
            elif enable_vlm_review:
                errors.append("VLM review skipped: PDF not compiled (missing template or output path)")
            
            # Apply revisions from VLM feedback and/or Typesetter LaTeX-fix feedback
            post_compile_revised = await self._apply_revisions(
                review_result=review_result,
                generated_sections=generated_sections,
                sections_results=sections_results,
                valid_citation_keys=self._extract_valid_citation_keys(parsed_refs),
                metadata=metadata,
                memory=memory,
                semantic_checks=iteration_semantic_checks,
                decision_trace=iteration_decision_trace,
                writer_response_section=iteration_writer_response_section,
                writer_response_paragraph=iteration_writer_response_paragraph,
                reviewer_verification=iteration_reviewer_verification,
            )
            if post_compile_revised:
                sources = "VLM/Typesetter" if not compile_succeeded else "VLM"
                print(f"[ReviewLoop] {sources} revised: {sorted(post_compile_revised)}")
            self._resolve_section_feedbacks(
                section_feedbacks=review_result.section_feedbacks,
                revised_sections=post_compile_revised,
                review_result=review_result,
            )
            review_result.conflict_resolution = iteration_conflicts
            review_result.semantic_checks = iteration_semantic_checks
            review_result.decision_trace = iteration_decision_trace
            
            # Record review iteration in session memory
            if memory is not None:
                # Build section feedback summary for the record
                section_fb_dict = {}
                for sf in review_result.section_feedbacks:
                    section_fb_dict[sf.section_type] = {
                        "action": sf.action,
                        "delta_words": sf.delta_words,
                        "target_paragraphs": sf.target_paragraphs,
                        "paragraph_instructions": sf.paragraph_instructions,
                        "paragraph_feedbacks": [
                            pf.model_dump() if hasattr(pf, "model_dump") else pf
                            for pf in (sf.paragraph_feedbacks if hasattr(sf, "paragraph_feedbacks") else [])
                        ],
                    }
                agent_feedbacks: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
                for fb in review_result.feedbacks:
                    agent_name = str(fb.checker_name or "reviewer")
                    if agent_name not in agent_feedbacks:
                        agent_feedbacks[agent_name] = {
                            "document_feedbacks": [],
                            "section_feedbacks": [],
                            "paragraph_feedbacks": [],
                            "sentence_feedbacks": [],
                        }
                    agent_feedbacks[agent_name]["document_feedbacks"].append({
                        "level": "document",
                        "agent": agent_name,
                        "checker": fb.checker_name,
                        "target_id": "document",
                        "severity": fb.severity.value if hasattr(fb.severity, "value") else str(fb.severity),
                        "issue_type": fb.suggested_action or fb.checker_name,
                        "message": fb.message,
                        "details": fb.details,
                    })
                for hf in (review_result.hierarchical_feedbacks or []):
                    item = hf.model_dump() if hasattr(hf, "model_dump") else hf
                    agent_name = str(item.get("agent", "reviewer"))
                    item["source_agent"] = str(item.get("source_agent") or agent_name)
                    item["source_stage"] = str(item.get("source_stage") or item["source_agent"])
                    if agent_name not in agent_feedbacks:
                        agent_feedbacks[agent_name] = {
                            "document_feedbacks": [],
                            "section_feedbacks": [],
                            "paragraph_feedbacks": [],
                            "sentence_feedbacks": [],
                        }
                    level = str(item.get("level", "section"))
                    bucket = f"{level}_feedbacks"
                    if bucket not in agent_feedbacks[agent_name]:
                        bucket = "section_feedbacks"
                    agent_feedbacks[agent_name][bucket].append(item)
                for sf in review_result.section_feedbacks:
                    source_agent = "reviewer"
                    if sf.action == "fix_latex":
                        source_agent = "typesetter"
                    elif sf.structural_actions:
                        source_agent = "vlm_review"
                    elif "VLM" in (sf.revision_prompt or "") or "page limit" in (sf.revision_prompt or "").lower():
                        source_agent = "vlm_review"
                    if source_agent not in agent_feedbacks:
                        agent_feedbacks[source_agent] = {
                            "document_feedbacks": [],
                            "section_feedbacks": [],
                            "paragraph_feedbacks": [],
                            "sentence_feedbacks": [],
                        }
                    sec_item = {
                        "level": "section",
                        "agent": source_agent,
                        "source_agent": source_agent,
                        "source_stage": source_agent,
                        "checker": source_agent,
                        "target_id": sf.target_id or sf.section_type,
                        "section_type": sf.section_type,
                        "severity": "warning",
                        "issue_type": sf.action,
                        "message": sf.revision_prompt[:400],
                        "target_paragraphs": sf.target_paragraphs,
                    }
                    agent_feedbacks[source_agent]["section_feedbacks"].append(sec_item)
                    for pf in sf.paragraph_feedbacks or []:
                        pf_item = pf.model_dump() if hasattr(pf, "model_dump") else pf
                        agent_feedbacks[source_agent]["paragraph_feedbacks"].append({
                            "level": "paragraph",
                            "agent": source_agent,
                            "source_agent": source_agent,
                            "source_stage": source_agent,
                            "checker": source_agent,
                            "target_id": f"{sf.section_type}.p{pf_item.get('paragraph_index', 0)}",
                            "section_type": sf.section_type,
                            "paragraph_index": pf_item.get("paragraph_index", 0),
                            "severity": str(pf_item.get("severity", "warning")),
                            "issue_type": sf.action,
                            "message": "; ".join(pf_item.get("issues", []))[:400],
                            "suggestion": pf_item.get("suggestion", ""),
                        })
                actions_taken = sorted(
                    reviewer_revised_sections | post_compile_revised
                )
                lifecycle_result = memory.update_issue_lifecycle(
                    iteration=review_iterations,
                    hierarchical_feedbacks=[
                        hf.model_dump() if hasattr(hf, "model_dump") else hf
                        for hf in (review_result.hierarchical_feedbacks or [])
                    ],
                    writer_response_section=iteration_writer_response_section,
                    writer_response_paragraph=iteration_writer_response_paragraph,
                )
                word_snapshot = {
                    sr.section_type: sr.word_count
                    for sr in sections_results if sr.status == "ok"
                }
                record = ReviewRecord(
                    iteration=review_iterations,
                    reviewer="unified",
                    passed=review_result.passed,
                    feedback_summary="; ".join(
                        f.message for f in review_result.feedbacks if not f.passed
                    )[:500],
                    section_feedbacks=section_fb_dict,
                    hierarchical_feedbacks=[
                        hf.model_dump() if hasattr(hf, "model_dump") else hf
                        for hf in (review_result.hierarchical_feedbacks or [])
                    ],
                    agent_feedbacks=agent_feedbacks,
                    decision_trace=iteration_decision_trace + [
                        dt if isinstance(dt, dict) else {"event": str(dt)}
                        for dt in (review_result.decision_trace or [])
                    ],
                    revision_plan=iteration_revision_plan + [
                        t.model_dump() if hasattr(t, "model_dump") else t
                        for t in (review_result.revision_tasks or [])
                    ],
                    before_after_semantic_check=[
                        s.model_dump() if hasattr(s, "model_dump") else s
                        for s in iteration_semantic_checks
                    ],
                    conflict_resolution=[
                        c.model_dump() if hasattr(c, "model_dump") else c
                        for c in iteration_conflicts
                    ],
                    baseline_gap_audit=baseline_gap_audit,
                    issue_lifecycle=lifecycle_result.get("issue_lifecycle", []),
                    writer_response_section=iteration_writer_response_section,
                    writer_response_paragraph=iteration_writer_response_paragraph,
                    reviewer_verification=iteration_reviewer_verification,
                    regression_report=lifecycle_result.get("regression_report", {}),
                    actions_taken=[f"revised:{s}" for s in actions_taken],
                    result_snapshot=word_snapshot,
                )
                memory.add_review(record)
                review_narr = f"Review iteration {review_iterations}: "
                if review_result.passed:
                    review_narr += "All checks passed."
                else:
                    failed_msgs = [f.message for f in review_result.feedbacks if not f.passed]
                    review_narr += f"Found {len(failed_msgs)} issue(s). "
                    if failed_msgs:
                        review_narr += failed_msgs[0][:150]
                if actions_taken:
                    review_narr += f" Revised sections: {', '.join(actions_taken)}."
                memory.log("metadata", f"review_iter_{review_iterations}",
                           "review_completed",
                           narrative=review_narr,
                           passed=review_result.passed,
                           revised=actions_taken)
                # Update memory sections with latest content
                for stype, content in generated_sections.items():
                    memory.update_section(stype, content)

            current_fingerprint = self._get_sections_fingerprint(generated_sections)
            if current_fingerprint == last_fingerprint:
                if review_result.passed and (not last_vlm_result or last_vlm_result.get("passed", True)):
                    print("[MetaDataAgent] Review passed with no further changes (fingerprint stable)")
                elif not compile_succeeded:
                    errors.append("LaTeX compilation failed and revisions did not change content")
                    print("[MetaDataAgent] Exiting: compilation failed with no effective revisions (fingerprint stable)")
                else:
                    if last_vlm_result and not last_vlm_result.get("passed", True):
                        errors.append(last_vlm_result.get("summary", "VLM review failed"))
                        print("[MetaDataAgent] Exiting due to VLM failure with no changes")
                break
            
            last_fingerprint = current_fingerprint
            if not reviewer_revised_sections and not post_compile_revised and review_result.passed and (not last_vlm_result or last_vlm_result.get("passed", True)):
                print("[MetaDataAgent] Exiting: no revisions needed and review passed")
                break
        
        # =====================================================================
        # Final compilation pass
        # After the loop, revisions may have been applied (VLM/Typesetter)
        # that were never compiled.  Run one more compile if content changed.
        # =====================================================================
        if compile_pdf and template_path and paper_dir:
            final_fp = self._get_sections_fingerprint(generated_sections)
            if final_fp != last_compiled_fingerprint:
                print("[MetaDataAgent] Final pass: content changed since last compile — recompiling")
                final_dir = paper_dir / f"iteration_{review_iterations:02d}_final"
                final_dir.mkdir(parents=True, exist_ok=True)
                figure_base_path = os.getcwd()
                figure_paths = self._collect_figure_paths(metadata.figures, base_path=figure_base_path)
                final_pdf, _, final_errors, _ = await self._compile_pdf(
                    generated_sections=generated_sections,
                    template_path=template_path,
                    references=parsed_refs,
                    output_dir=final_dir,
                    paper_title=metadata.title,
                    figures_source_dir=figures_source_dir,
                    figure_paths=figure_paths,
                    converted_tables=converted_tables,
                    paper_plan=paper_plan,
                    figures=metadata.figures,
                    metadata_tables=metadata.tables,
                )
                if final_pdf:
                    pdf_path = final_pdf
                    print(f"[MetaDataAgent] Final pass compiled: {final_pdf}")
                elif final_errors:
                    errors.extend(final_errors)
                    print(f"[MetaDataAgent] Final pass compile errors: {final_errors[:2]}")
            else:
                print("[MetaDataAgent] Final pass: no content changes since last compile — skipping")
        
        return generated_sections, sections_results, review_iterations, target_word_count, pdf_path, errors
    
    async def _call_reviewer(
        self,
        sections: Dict[str, str],
        word_counts: Dict[str, int],
        target_pages: Optional[int],
        style_guide: Optional[str],
        template_path: Optional[str],
        iteration: int,
        section_targets: Optional[Dict[str, int]] = None,
        section_structure_signals: Optional[Dict[str, Any]] = None,
        memory: Optional[SessionMemory] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """
        Call Reviewer Agent directly (no HTTP) to check the paper.
        - **Description**:
            - Builds a ReviewContext and calls self._reviewer.review() directly.
            - Passes memory so checkers can read prior issues natively.

        - **Returns**:
            - Tuple of (review_result_dict, target_word_count) or (None, None) on failure
        """
        try:
            from ..reviewer_agent.models import ReviewContext as RC

            review_ctx = RC(
                sections=sections,
                word_counts=word_counts,
                target_pages=target_pages or 8,
                style_guide=style_guide,
                template_path=template_path,
                metadata={},
            )
            review_ctx.metadata["review_structure_gate_enabled"] = bool(
                getattr(self.tools_config, "review_structure_gate_enabled", True)
            ) if self.tools_config else True
            review_ctx.metadata["structure_gate_min_paragraph_threshold"] = int(
                getattr(self.tools_config, "structure_gate_min_paragraph_threshold", 5)
            ) if self.tools_config else 5
            if section_structure_signals:
                review_ctx.metadata["section_structure_signals"] = section_structure_signals
            if memory is not None:
                review_ctx.metadata["issue_memory"] = memory.get_issue_context(limit=40)
            if section_targets:
                review_ctx.section_targets = section_targets

            review_result = await self._reviewer.review(
                context=review_ctx,
                iteration=iteration,
                memory=memory,
            )

            result = review_result.model_dump()

            target_word_count = None
            for fb in result.get("feedbacks", []):
                details = fb.get("details", {})
                if "target_words" in details:
                    target_word_count = details["target_words"]
                    break

            return result, target_word_count

        except Exception as e:
            print(f"[MetaDataAgent] Review error: {e}")
            return None, None
    
    async def _revise_section(
        self,
        section_type: str,
        current_content: str,
        revision_prompt: str,
        metadata: PaperMetaData,
        memory: Optional[SessionMemory] = None,
        task_contract: Optional[Dict[str, Any]] = None,
        valid_citation_keys: Optional[set] = None,
    ) -> Optional[str]:
        """
        Revise a section based on feedback — delegates to WriterAgent.
        - **Description**:
            - Packs the revision instructions + current content as a user_prompt
              and delegates to WriterAgent, which can consult memory/planner/reviewer
              via AskTool during the ReAct loop.

        - **Args**:
            - `section_type` (str): Type of section to revise
            - `current_content` (str): Current section LaTeX content
            - `revision_prompt` (str): Instructions for the revision
            - `metadata` (PaperMetaData): Paper metadata for context
            - `memory` (SessionMemory, optional): Session memory

        - **Returns**:
            - Revised content string, or None on failure
        """
        try:
            system_prompt = (
                "You are an expert academic writer revising a paper section.\n"
                "Follow the revision instructions carefully to improve the content.\n"
                "Maintain academic writing quality.\n"
                "Output ONLY the revised LaTeX content, no explanations or preamble."
            )

            # Build user message with revision context from memory
            revision_ctx = ""
            if memory:
                revision_ctx = memory.get_revision_context(section_type)
                issue_ctx = memory.get_issue_context(limit=20)
                unresolved = issue_ctx.get("unresolved_issues", [])
                if unresolved:
                    pinned = []
                    for item in unresolved[:8]:
                        if str(item.get("section_type", "")) in ("", section_type):
                            pinned.append(
                                f"- [{item.get('locked_mode', 'soft')}] {item.get('target_id', '')}: {str(item.get('message', ''))[:180]}"
                            )
                    if pinned:
                        revision_ctx = (
                            revision_ctx
                            + ("\n\n" if revision_ctx else "")
                            + "## Unresolved Memory Issues\n"
                            + "\n".join(pinned)
                        )

            if "Current content" in revision_prompt or current_content in revision_prompt:
                user_message = revision_prompt
            else:
                user_message = (
                    f"{revision_prompt}\n\n"
                    f"Current content of the {section_type} section to revise:\n"
                    f"{current_content}"
                )

            if revision_ctx:
                user_message = f"{revision_ctx}\n\n{user_message}"
            # Strong citation preservation guard for revision mode.
            user_message += (
                "\n\nCitation guardrails:\n"
                "- Preserve all existing valid \\cite{...} commands unless explicitly instructed to remove invalid ones.\n"
                "- Do not drop citations during style or structure edits.\n"
            )

            result = await self._writer.run(
                system_prompt=system_prompt,
                user_prompt=user_message,
                section_type=section_type,
                enable_review=False,
                valid_citation_keys=sorted(list(valid_citation_keys or [])),
                mode="revision",
                current_content=current_content,
                revision_plan={
                    **(task_contract or {}),
                    "section_type": section_type,
                    "target": str((task_contract or {}).get("target") or section_type),
                    "target_level": "section",
                    "instruction": str((task_contract or {}).get("instruction") or revision_prompt),
                    "constraints": (
                        (task_contract or {}).get("constraints")
                        or {
                            "preserve_claims": ["Preserve factual claims and citations."],
                            "do_not_change": [],
                        }
                    ),
                },
                memory=memory,
                peers={"planner": self._planner, "reviewer": self._reviewer},
            )
            revised = result.get("generated_content", "")
            return revised if revised else None

        except Exception as e:
            print(f"[MetaDataAgent] Revision error for {section_type}: {e}")
            return None

    async def _revise_paragraph(
        self,
        section_type: str,
        paragraph_index: int,
        paragraph_text: str,
        instruction: str,
        memory: Optional[SessionMemory] = None,
        task_contract: Optional[Dict[str, Any]] = None,
        valid_citation_keys: Optional[set] = None,
    ) -> Optional[str]:
        """
        Revise a single paragraph using WriterAgent.
        - **Description**:
            - Uses a focused prompt to avoid unnecessary edits to other paragraphs
            - Returns revised paragraph text only
        """
        try:
            system_prompt = (
                "You are an expert academic editor revising one paragraph.\n"
                "Keep the same scientific meaning and preserve LaTeX correctness.\n"
                "Return ONLY the revised paragraph text."
            )
            revision_ctx = ""
            if memory:
                revision_ctx = memory.get_revision_context(section_type)
                issue_ctx = memory.get_issue_context(limit=20)
                unresolved = issue_ctx.get("unresolved_issues", [])
                if unresolved:
                    pinned = []
                    for item in unresolved[:12]:
                        target_id = str(item.get("target_id", ""))
                        if target_id.endswith(f".p{paragraph_index}") or str(item.get("section_type", "")) == section_type:
                            pinned.append(
                                f"- [{item.get('locked_mode', 'soft')}] {target_id}: {str(item.get('message', ''))[:180]}"
                            )
                    if pinned:
                        revision_ctx = (
                            revision_ctx
                            + ("\n\n" if revision_ctx else "")
                            + "## Memory Issues For This Paragraph\n"
                            + "\n".join(pinned)
                        )

            user_message = (
                f"Section: {section_type}\n"
                f"Paragraph index: {paragraph_index}\n"
                f"Instruction: {instruction}\n\n"
                f"Current paragraph:\n{paragraph_text}"
            )
            if revision_ctx:
                user_message = f"{revision_ctx}\n\n{user_message}"
            user_message += (
                "\n\nCitation guardrails:\n"
                "- Preserve existing valid citations in this paragraph.\n"
                "- Only remove citations when they are explicitly marked invalid.\n"
            )

            result = await self._writer.run(
                system_prompt=system_prompt,
                user_prompt=user_message,
                section_type=section_type,
                enable_review=False,
                valid_citation_keys=sorted(list(valid_citation_keys or [])),
                mode="revision",
                current_content=paragraph_text,
                revision_plan={
                    **(task_contract or {}),
                    "section_type": section_type,
                    "target": f"{section_type}.p{paragraph_index}",
                    "target_level": "paragraph",
                    "instruction": instruction,
                    "constraints": (
                        (task_contract or {}).get("constraints")
                        or {
                            "preserve_claims": ["Preserve the original scientific claim and evidence links."],
                            "do_not_change": [],
                        }
                    ),
                    "target_paragraphs": [paragraph_index],
                    "paragraph_instructions": {paragraph_index: instruction},
                },
                memory=memory,
                peers={"planner": self._planner, "reviewer": self._reviewer},
            )
            revised = result.get("generated_content", "")
            return revised if revised else None
        except Exception as e:
            print(
                f"[MetaDataAgent] Paragraph revision error for {section_type}.p{paragraph_index}: {e}"
            )
            return None
