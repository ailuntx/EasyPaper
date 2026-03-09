"""
Planner Agent
- **Description**:
    - Creates detailed paper plans before generation
    - Paragraph-level planning with VLM-informed figure/table placement
    - Outputs PaperPlan to guide Writers and Reviewers
"""
import json
import logging
import re
import random
from typing import List, Dict, Any, Optional

from ..base import BaseAgent
from ..shared.llm_client import LLMClient
from ...config.schema import ModelConfig
from .models import (
    PaperPlan,
    SectionPlan,
    PlanRequest,
    PlanResult,
    ParagraphPlan,
    FigurePlacement,
    TablePlacement,
    PaperType,
    NarrativeStyle,
    DEFAULT_EMPIRICAL_SECTIONS,
    ELEMENT_PAGE_COST,
    WORDS_PER_SENTENCE,
    WORDS_PER_PARAGRAPH,
    calculate_total_words,
    estimate_target_paragraphs,
)


logger = logging.getLogger("uvicorn.error")


# =========================================================================
# LLM Prompts
# =========================================================================

# =========================================================================
# Multi-Step Planner Prompts
# =========================================================================

# --- STEP 1: Structure Decision ---
STEP1_STRUCTURE_SYSTEM = """You are an expert academic paper planner.
Given a paper's metadata and target venue, decide the high-level structure.
Output ONLY a JSON object. No markdown, no explanation."""

STEP1_STRUCTURE_USER = """Decide the structure for this paper:

**Title**: {title}
**Idea/Hypothesis**: {idea_hypothesis}
**Method summary**: {method}
**Data**: {data}
**Experiments summary**: {experiments}
**Target venue/style**: {style_guide}
**Target pages**: {target_pages}
**Research Context**: {research_context_summary}
**Code Assets**: {code_writing_assets_summary}

Output JSON:
{{
  "paper_type": "empirical|theoretical|survey|position|system|benchmark",
  "contributions": ["Contribution 1", "Contribution 2", ...],
  "narrative_style": "technical|tutorial|concise|comprehensive",
  "sections": [
    {{"section_type": "abstract", "section_title": "Abstract"}},
    {{"section_type": "introduction", "section_title": "Introduction"}},
    ...
  ],
  "structure_rationale": "Why this structure suits the venue and content",
  "abstract_focus": "What the abstract should emphasize"
}}

IMPORTANT:
- "abstract" is always required.
- Choose sections appropriate for {style_guide}. Use your knowledge of venue norms.
- Each section needs section_type (lowercase, e.g. "method", "result") and section_title.
- For empirical studies, consider whether a dedicated Method section is needed.
- Conclusion is optional; for Nature-style, it may be integrated into Discussion.
Output valid JSON only."""

# --- STEP 2: Citation Strategy ---
STEP2_CITATION_SYSTEM = """You are an expert academic citation strategist.
Given a paper's structure and venue, decide the total citation count and
per-section allocation. Output ONLY a JSON object."""

STEP2_CITATION_USER = """Decide the citation strategy for this paper:

**Title**: {title}
**Venue**: {style_guide}
**Target pages**: {target_pages}
**Sections**: {section_list}
**Available reference keys**: {reference_keys}

Output JSON:
{{
  "total_target": <int>,
  "rationale": "Why this total is appropriate for the venue and paper scope",
  "section_allocation": {{
    "<section_type>": {{
      "target_refs": <int>,
      "rationale": "Why this section needs this many"
    }},
    ...
  }}
}}

Use your knowledge of academic publishing norms to decide appropriate totals.
Sections that carry literature-review duties need more citations.
Abstract and conclusion typically need 0 citations.
Output valid JSON only."""

# --- STEP 3: Section Planning (called per section) ---
STEP3_SECTION_SYSTEM = """You are an expert academic section planner.
Given a section's role in the paper, plan its paragraphs in detail.
Output ONLY a JSON object."""

STEP3_SECTION_USER = """Plan the **{section_title}** ({section_type}) section:

**Paper title**: {title}
**Paper type**: {paper_type}
**Venue**: {style_guide}
**Space budget for this section**: ~{section_words} words (~{section_paragraphs} paragraphs)
**Contributions**: {contributions}

**Available figures**: {figure_info}
**Available tables**: {table_info}
**Available references**: {reference_keys}
**Code assets**: {code_writing_assets_summary}

**Content sources**:
- Idea/Hypothesis: {idea_hypothesis}
- Method: {method}
- Data: {data}
- Experiments: {experiments}

Output JSON:
{{
  "paragraphs": [
    {{
      "key_point": "The main argument of this paragraph",
      "supporting_points": ["Evidence 1", "Evidence 2"],
      "approx_sentences": 5,
      "role": "motivation|problem_statement|definition|evidence|comparison|transition|summary",
      "references_to_cite": ["ref_key1"],
      "figures_to_reference": [],
      "tables_to_reference": []
    }}
  ],
  "figures": [
    {{"figure_id": "fig:X", "position_hint": "early|middle|late", "caption_guidance": "..."}}
  ],
  "tables": [],
  "topic_clusters": ["Theme A", "Theme B"],
  "transition_intents": ["From X to Y"],
  "sectioning_recommended": false,
  "code_focus": {{
    "must_use_evidence_ids": [],
    "key_assets": [],
    "allowed_claim_scope": "",
    "notes": ""
  }},
  "writing_guidance": "Specific guidance for the writer"
}}

IMPORTANT:
- Plan enough paragraphs to fill ~{section_words} words. Each paragraph is ~150-250 words.
- Each paragraph should have 3-8 sentences.
- For narrative sections (introduction, discussion), plan **4-8 cohesive paragraphs**
  with substantial depth each, rather than 10+ short fragmented paragraphs.
  An academic paragraph should develop a complete idea, not a single bullet point.
- Use sectioning_recommended sparingly; prefer false for narrative sections
  like introduction or discussion where continuous prose flow is expected.
- Do NOT set sectioning_recommended to true for introduction or discussion
  unless the section is exceptionally long and structurally complex.
- FIGURE PLACEMENT RULES:
  - Only include figures marked "DEFINE in this section" in the "figures" array.
  - For figures marked "REFERENCE ONLY", use them in paragraphs' "figures_to_reference"
    but do NOT add them to the "figures" array (they are defined elsewhere).
  - NEVER create a \\begin{{figure}} environment for a REFERENCE ONLY figure.
Output valid JSON only."""


# =========================================================================
# Planner Agent
# =========================================================================

class PlannerAgent(BaseAgent):
    """
    Planner Agent for paper planning.
    - **Description**:
        - Creates comprehensive paragraph-level plans
        - Optionally uses VLM for intelligent figure/table analysis
        - Directly encapsulates all planning logic (no Strategy pattern)
    """

    def __init__(
        self,
        config: ModelConfig,
        vlm_service: Optional[Any] = None,
    ):
        """
        Initialize the Planner Agent.

        - **Args**:
            - `config` (ModelConfig): LLM configuration
            - `vlm_service` (VLMService, optional): Shared VLM service for figure analysis
        """
        self.config = config
        self.model_name = config.model_name
        self.client = LLMClient(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.vlm_service = vlm_service
        self._last_plan: Optional[PaperPlan] = None
        try:
            self._router = self._create_router()
        except Exception:
            self._router = None

        logger.info("PlannerAgent initialized (vlm=%s)", vlm_service is not None)

    @property
    def name(self) -> str:
        return "planner"

    @property
    def description(self) -> str:
        return "Creates detailed paragraph-level paper plans"

    @property
    def router(self) -> "APIRouter | None":
        return self._router

    @property
    def endpoints_info(self) -> List[Dict[str, Any]]:
        return [
            {
                "path": "/agent/planner/plan",
                "method": "POST",
                "description": "Create a paper plan from metadata",
            },
            {
                "path": "/agent/planner/health",
                "method": "GET",
                "description": "Health check",
            },
        ]

    def _create_router(self) -> "APIRouter":
        from .router import create_planner_router
        return create_planner_router(self)

    @staticmethod
    def _normalize_section_type_name(section_type: str) -> str:
        """
        Normalize common plural/alias section names.
        """
        st = (section_type or "").strip().lower()
        alias_map = {
            "methods": "method",
            "methodology": "method",
            "experiments": "experiment",
            "results": "result",
            "intro": "introduction",
        }
        return alias_map.get(st, st)

    # =====================================================================
    # AskTool consultation interface
    # =====================================================================

    async def answer(self, question: str) -> str:
        """
        Two-stage answer about the paper plan.
        - **Description**:
            - Stage 1: Rule-based keyword filtering over the cached
              PaperPlan to gather compact candidate snippets.
            - Stage 2: LLM refinement — passes the candidates + question
              to ``self.client`` for a concise, semantically precise answer.
            - If the LLM call fails, falls back to Stage 1 output.

        - **Args**:
            - `question` (str): Natural-language question about the plan

        - **Returns**:
            - `result` (str): Precise answer about the plan
        """
        if self._last_plan is None:
            return "No plan available yet."

        candidates = self._gather_plan_candidates(question)
        if not candidates:
            return "No matching plan information found."

        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a paper-planning assistant. Based on the "
                            "plan context below, answer the question concisely "
                            "and precisely. Keep your response under 200 words."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Plan context:\n{candidates}\n\n"
                            f"Question: {question}"
                        ),
                    },
                ],
                temperature=0.3,
                max_tokens=300,
            )
            return response.choices[0].message.content or candidates
        except Exception as e:
            logger.warning("planner.answer LLM refine failed: %s", e)
            return candidates

    def _gather_plan_candidates(self, question: str) -> str:
        """
        Stage 1: Gather compact plan snippets via keyword matching.
        - **Description**:
            - Produces token-budgeted output: section_type + paragraph
              count + guidance first sentence + key point summaries.
            - Roughly ~50 tokens per matching section.

        - **Args**:
            - `question` (str): The question driving keyword extraction

        - **Returns**:
            - `text` (str): Compact matching plan context
        """
        keywords = [w.lower() for w in question.split() if len(w) > 2]
        hits: List[str] = []

        for sp in self._last_plan.sections:
            stype = sp.section_type
            guidance = sp.writing_guidance or ""
            para_texts = " ".join(
                getattr(p, "key_point", "") for p in (sp.paragraphs or [])
            )
            fig_texts = " ".join(
                getattr(fp, "figure_id", "") + " " + getattr(fp, "purpose", "")
                for fp in (sp.figure_placements or [])
            )
            tbl_texts = " ".join(
                getattr(tp, "table_id", "") + " " + getattr(tp, "purpose", "")
                for tp in (sp.table_placements or [])
            )
            full = f"{stype} {guidance} {para_texts} {fig_texts} {tbl_texts}".lower()

            if not keywords or any(kw in full for kw in keywords):
                guidance_snippet = guidance.split(".")[0] if guidance else ""
                n_paras = len(sp.paragraphs or [])
                kp_list = ", ".join(
                    getattr(p, "key_point", "")[:60]
                    for p in (sp.paragraphs or [])[:4]
                )
                line = f"- {stype}: {n_paras} paragraphs"
                est = sp.get_estimated_words() if hasattr(sp, "get_estimated_words") else 0
                if est:
                    line += f", ~{est} words"
                if guidance_snippet:
                    line += f", guidance: \"{guidance_snippet}\""
                if kp_list:
                    line += f", key points: [{kp_list}]"
                for fp in (sp.figure_placements or []):
                    fid = getattr(fp, "figure_id", "")
                    purpose = getattr(fp, "purpose", "")[:60]
                    line += f", fig {fid}: {purpose}"
                for tp in (sp.table_placements or []):
                    tid = getattr(tp, "table_id", "")
                    purpose = getattr(tp, "purpose", "")[:60]
                    line += f", tbl {tid}: {purpose}"
                hits.append(line)

        return "\n".join(hits) if hits else ""

    # =====================================================================
    # Core planning
    # =====================================================================

    def _format_research_context_for_planning(
        self,
        research_context: Optional[Dict[str, Any]],
    ) -> str:
        """
        Format compact research context for planning input.
        """
        if not research_context:
            return "Not available."

        lines: List[str] = []
        area = str(research_context.get("research_area", "")).strip()
        summary = str(research_context.get("summary", "")).strip()
        if area:
            lines.append(f"- Research area: {area}")
        if summary:
            lines.append(f"- Landscape summary: {summary}")

        trends = research_context.get("research_trends", []) or []
        if trends:
            lines.append("- Key trends:")
            for t in trends[:3]:
                lines.append(f"  - {t}")

        gaps = research_context.get("gaps", []) or []
        if gaps:
            lines.append("- Key gaps/opportunities:")
            for g in gaps[:3]:
                lines.append(f"  - {g}")

        ranking = research_context.get("contribution_ranking", {}) or {}
        if ranking:
            lines.append("- Contribution ranking hints:")
            for band in ("P0", "P1", "P2"):
                items = ranking.get(band, []) or []
                if not items:
                    continue
                top_text = ", ".join(
                    [str(x.get("contribution", "")).strip() for x in items[:3] if isinstance(x, dict)]
                )
                if top_text:
                    lines.append(f"  - {band}: {top_text}")

        return "\n".join(lines) if lines else "Not available."

    def _format_code_assets_for_planning(
        self,
        code_context: Optional[Dict[str, Any]],
        code_writing_assets: Optional[Dict[str, Any]],
    ) -> str:
        """
        Format compact code-driven writing assets for planner decisions.
        """
        assets = code_writing_assets or {}
        if not assets and code_context:
            assets = code_context.get("writing_assets", {}) or {}

        section_packs = {}
        if code_context:
            section_packs = code_context.get("section_asset_packs", {}) or {}
        evidence_graph = (code_context or {}).get("code_evidence_graph", []) or []

        if not assets and not section_packs and not evidence_graph:
            return "Not available."

        lines: List[str] = [f"- Evidence nodes extracted: {len(evidence_graph)}"]
        planner_brief = str(assets.get("planner_brief", "")).strip() if isinstance(assets, dict) else ""
        if planner_brief:
            lines.append("- Planner brief:")
            for chunk in planner_brief.splitlines()[:10]:
                lines.append(f"  {chunk}")
        for key, label in (
            ("method_pipeline", "Method assets"),
            ("experiment_protocol", "Experiment assets"),
            ("result_readouts", "Result assets"),
            ("risk_limitations", "Risk assets"),
        ):
            rows = assets.get(key, []) or []
            if not rows:
                continue
            lines.append(f"- {label}:")
            for row in rows[:4]:
                title = str(row.get("title", "")).strip()
                if title:
                    lines.append(f"  - {title}")
        for sec in ("introduction", "method", "experiment", "result", "discussion"):
            pack = section_packs.get(sec, {}) or {}
            evidence_ids = [str(x).strip() for x in (pack.get("evidence_ids", []) or []) if str(x).strip()]
            if evidence_ids:
                lines.append(f"- Suggested evidence IDs for {sec}: {', '.join(evidence_ids[:6])}")
            guardrails = [str(x).strip() for x in (pack.get("claim_guardrails", []) or []) if str(x).strip()]
            if guardrails:
                lines.append(f"- Claim guardrails for {sec}:")
                for guardrail in guardrails[:2]:
                    lines.append(f"  - {guardrail}")
        return "\n".join(lines) if lines else "Not available."

    @staticmethod
    def _normalize_code_focus(raw: Any) -> Dict[str, Any]:
        """
        Normalize LLM-provided code_focus object for each section.
        """
        if not isinstance(raw, dict):
            return {}
        must_use = [str(x).strip() for x in (raw.get("must_use_evidence_ids", []) or []) if str(x).strip()]
        key_assets = [str(x).strip() for x in (raw.get("key_assets", []) or []) if str(x).strip()]
        allowed_scope = str(raw.get("allowed_claim_scope", "")).strip()
        notes = str(raw.get("notes", "")).strip()
        out: Dict[str, Any] = {}
        if must_use:
            out["must_use_evidence_ids"] = must_use[:10]
        if key_assets:
            out["key_assets"] = key_assets[:8]
        if allowed_scope:
            out["allowed_claim_scope"] = allowed_scope[:320]
        if notes:
            out["notes"] = notes[:320]
        return out

    @staticmethod
    def _strip_code_fence(text: str) -> str:
        """
        Strip common markdown code fences from model outputs.
        """
        raw = (text or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        return raw.strip()

    @staticmethod
    def _extract_balanced_json_block(text: str, start_char: str) -> Optional[str]:
        """
        Extract first balanced JSON object/array block from text.
        """
        end_char = "}" if start_char == "{" else "]"
        start_idx = text.find(start_char)
        if start_idx < 0:
            return None

        depth = 0
        in_string = False
        escaped = False
        for i in range(start_idx, len(text)):
            ch = text[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == "\"":
                    in_string = False
                continue

            if ch == "\"":
                in_string = True
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[start_idx:i + 1]
        return None

    @classmethod
    def _safe_load_json(
        cls,
        raw: str,
        expected: Optional[type] = None,
    ) -> Optional[Any]:
        """
        Parse JSON robustly from model outputs with optional type check.
        """
        cleaned = cls._strip_code_fence(raw)
        candidates: List[str] = [cleaned]
        obj_block = cls._extract_balanced_json_block(cleaned, "{")
        arr_block = cls._extract_balanced_json_block(cleaned, "[")
        if obj_block:
            candidates.append(obj_block)
        if arr_block:
            candidates.append(arr_block)

        for cand in candidates:
            if not cand:
                continue
            try:
                parsed = json.loads(cand)
                if expected is not None and not isinstance(parsed, expected):
                    continue
                return parsed
            except Exception:
                continue
        return None

    # -----------------------------------------------------------------
    # LLM call helper with retry
    # -----------------------------------------------------------------

    async def _llm_json_call(
        self,
        system_prompt: str,
        user_prompt: str,
        label: str,
        max_retries: int = 2,
        temperature: float = 0.3,
    ) -> Dict[str, Any]:
        """
        Call LLM and parse the result as JSON, with retry on parse failure.

        - **Args**:
          - `label` (str): Log label for this call (e.g. "step1_structure").
          - `max_retries` (int): Number of retry attempts on JSON parse failure.

        - **Returns**:
          - `Dict[str, Any]`: Parsed JSON object, or empty dict on total failure.
        """
        for attempt in range(1 + max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                )
                text = response.choices[0].message.content.strip()
                parsed = self._safe_load_json(text, expected=dict)
                if parsed:
                    logger.info("planner.%s ok (attempt=%d)", label, attempt)
                    return parsed
                logger.warning(
                    "planner.%s json_parse_failed attempt=%d", label, attempt,
                )
            except Exception as e:
                logger.warning(
                    "planner.%s error attempt=%d: %s", label, attempt, e,
                )
        logger.error("planner.%s all_attempts_failed", label)
        return {}

    # -----------------------------------------------------------------
    # Main entry point — multi-step planning
    # -----------------------------------------------------------------

    async def create_plan(self, request: PlanRequest) -> PaperPlan:
        """
        Create a paper plan from metadata using a multi-step approach.

        - **Description**:
          - Step 1: Structure decision (paper_type, sections, contributions)
          - Step 2: Citation strategy (total target + per-section allocation)
          - Step 3: Per-section paragraph planning
          - Each step produces a small, simple JSON output that is easy for
            the LLM to generate correctly, reducing parse failures.

        - **Args**:
          - `request` (PlanRequest): Planning request with metadata.

        - **Returns**:
          - `PaperPlan`: Complete paragraph-level paper plan.
        """
        n_figures = len(request.figures) if request.figures else 0
        n_tables = len(request.tables) if request.tables else 0
        n_wide_figures = sum(
            1 for f in (request.figures or []) if self._should_be_wide_figure(f)
        )
        n_wide_tables = sum(
            1 for t in (request.tables or []) if self._should_be_wide_table(t)
        )

        total_words = calculate_total_words(
            request.target_pages,
            request.style_guide,
            n_figures=n_figures,
            n_tables=n_tables,
            n_wide_figures=n_wide_figures,
            n_wide_tables=n_wide_tables,
        )
        target_pages = request.target_pages or 10
        style_guide = request.style_guide or "DEFAULT"
        total_paragraphs = estimate_target_paragraphs(total_words)

        # VLM analysis for figures and tables
        figure_analyses: Dict[str, Any] = {}
        table_analyses: Dict[str, Any] = {}
        if self.vlm_service:
            figure_analyses = await self._analyze_figures(request.figures or [])
            table_analyses = await self._analyze_tables(request.tables or [])

        reference_keys = self._extract_reference_keys(request.references)
        figure_info = self._format_figure_info(request.figures or [], figure_analyses)
        table_info = self._format_table_info(request.tables or [], table_analyses)
        rc_summary = self._format_research_context_for_planning(request.research_context)
        code_summary = self._format_code_assets_for_planning(
            code_context=request.code_context,
            code_writing_assets=request.code_writing_assets,
        )

        logger.info(
            "planner.create_plan title=%s words=%d paragraphs=%d vlm=%s",
            request.title[:30], total_words, total_paragraphs,
            bool(figure_analyses or table_analyses),
        )

        # =============================================================
        # STEP 1: Structure Decision
        # =============================================================
        step1_prompt = STEP1_STRUCTURE_USER.format(
            title=request.title,
            idea_hypothesis=request.idea_hypothesis[:2000],
            method=request.method[:1500],
            data=request.data[:1000],
            experiments=request.experiments[:1500],
            style_guide=style_guide,
            target_pages=target_pages,
            research_context_summary=rc_summary,
            code_writing_assets_summary=code_summary,
        )
        structure = await self._llm_json_call(
            STEP1_STRUCTURE_SYSTEM, step1_prompt, "step1_structure",
        )

        # Extract structure decisions
        paper_type_str = structure.get("paper_type", "empirical")
        try:
            paper_type = PaperType(paper_type_str.lower())
        except ValueError:
            paper_type = PaperType.EMPIRICAL

        style_str = structure.get("narrative_style", "technical")
        try:
            narrative_style = NarrativeStyle(style_str.lower())
        except ValueError:
            narrative_style = NarrativeStyle.TECHNICAL

        contributions = structure.get("contributions", [])
        structure_rationale = structure.get("structure_rationale", "")
        abstract_focus = structure.get("abstract_focus", "")

        raw_sections = structure.get("sections", [])
        section_order: List[Dict[str, str]] = []
        for s in raw_sections:
            if isinstance(s, dict) and s.get("section_type"):
                st = self._normalize_section_type_name(str(s["section_type"]))
                section_order.append({
                    "section_type": st,
                    "section_title": s.get("section_title", self._get_section_title(st)),
                })

        if not section_order or len(section_order) < 3:
            section_order = [
                {"section_type": st, "section_title": self._get_section_title(st)}
                for st in DEFAULT_EMPIRICAL_SECTIONS
            ]
            logger.warning("planner.step1_fallback using default sections")

        if not any(s["section_type"] == "abstract" for s in section_order):
            section_order.insert(0, {"section_type": "abstract", "section_title": "Abstract"})

        # Deduplicate section_types: if the LLM produces multiple sections
        # with the same type (e.g. 3 "result" sections), append _2, _3, etc.
        # to create unique keys while preserving semantic meaning.
        type_counts: Dict[str, int] = {}
        for sec in section_order:
            st = sec["section_type"]
            type_counts[st] = type_counts.get(st, 0) + 1
            if type_counts[st] > 1:
                sec["section_type"] = f"{st}_{type_counts[st]}"

        section_types_str = ", ".join(s["section_type"] for s in section_order)
        logger.info(
            "planner.step1_done paper_type=%s sections=[%s]",
            paper_type.value, section_types_str,
        )

        # =============================================================
        # STEP 2: Citation Strategy
        # =============================================================
        step2_prompt = STEP2_CITATION_USER.format(
            title=request.title,
            style_guide=style_guide,
            target_pages=target_pages,
            section_list=section_types_str,
            reference_keys=", ".join(reference_keys) if reference_keys else "None",
        )
        citation_strategy = await self._llm_json_call(
            STEP2_CITATION_SYSTEM, step2_prompt, "step2_citation",
        )

        if not citation_strategy.get("total_target"):
            total_paras = total_paragraphs
            body_count = sum(
                1 for s in section_order
                if s["section_type"] not in ("abstract", "conclusion")
            )
            citation_strategy = {
                "total_target": self._estimate_total_citations(
                    style_guide, body_count, total_paras,
                ),
                "rationale": "Fallback estimation",
                "section_allocation": {},
            }
        logger.info(
            "planner.step2_done total_target=%s",
            citation_strategy.get("total_target"),
        )

        # =============================================================
        # STEP 3: Per-Section Planning
        # =============================================================
        n_body = sum(
            1 for s in section_order
            if s["section_type"] not in ("abstract", "conclusion")
        )
        sections: List[SectionPlan] = []

        # Pre-assign figures to sections so each figure is DEFINED in
        # exactly one section.  Other sections may still REFERENCE it.
        figure_assignment = self._assign_figures_to_sections(
            request.figures or [], section_order,
        )

        for order, sec_info in enumerate(section_order):
            section_type = sec_info["section_type"]
            section_title = sec_info["section_title"]

            # Abstract and conclusion are synthesis sections generated
            # separately; skip Step 3 planning (no figures, no paragraphs).
            if section_type in ("abstract", "conclusion"):
                sections.append(SectionPlan(
                    section_type=section_type,
                    section_title=section_title,
                    paragraphs=[],
                    figures=[],
                    tables=[],
                    content_sources=self._get_default_sources(section_type),
                    depends_on=self._get_dependencies(section_type),
                    citation_budget={"target_refs": 0, "min_refs": 0, "max_refs": 0},
                    order=order,
                ))
                logger.info("planner.step3_skip section=%s (synthesis)", section_type)
                continue

            # Allocate word budget per section proportionally
            alloc = (citation_strategy.get("section_allocation") or {}).get(section_type, {})
            if isinstance(alloc, dict) and alloc.get("target_refs"):
                total_target = int(citation_strategy.get("total_target", 1) or 1)
                share = int(alloc.get("target_refs", 0)) / max(1, total_target)
                section_words = max(400, int(total_words * max(share, 0.1)))
            else:
                section_words = max(400, total_words // max(1, n_body))

            section_paragraphs = max(1, section_words // WORDS_PER_PARAGRAPH)

            # Build per-section figure info distinguishing DEFINE vs REFERENCE
            section_figure_info = self._format_section_figure_info(
                request.figures or [], figure_analyses or {},
                section_type, figure_assignment,
            )

            step3_prompt = STEP3_SECTION_USER.format(
                section_type=section_type,
                section_title=section_title,
                title=request.title,
                paper_type=paper_type.value,
                style_guide=style_guide,
                section_words=section_words,
                section_paragraphs=section_paragraphs,
                contributions=", ".join(contributions) if contributions else "Not specified",
                figure_info=section_figure_info,
                table_info=table_info,
                reference_keys=", ".join(reference_keys) if reference_keys else "None",
                code_writing_assets_summary=code_summary,
                idea_hypothesis=request.idea_hypothesis[:1500],
                method=request.method[:1500],
                data=request.data[:1000],
                experiments=request.experiments[:1500],
            )
            section_data = await self._llm_json_call(
                STEP3_SECTION_SYSTEM, step3_prompt,
                f"step3_{section_type}",
            )

            # Parse paragraphs
            raw_paragraphs = section_data.get("paragraphs", [])
            paragraphs = self._parse_paragraph_plans(raw_paragraphs)
            if not paragraphs:
                default_sents = max(3, section_words // WORDS_PER_SENTENCE)
                paragraphs = self._generate_default_paragraphs(
                    section_type, default_sents, section_data,
                )

            # Parse figure/table placements
            figure_placements = self._build_figure_placements(
                section_data.get("figures", []), figure_analyses or {},
            )
            table_placements = self._build_table_placements(
                section_data.get("tables", []), table_analyses or {},
            )

            # Citation budget from Step 2
            alloc = (citation_strategy.get("section_allocation") or {}).get(section_type, {})
            if isinstance(alloc, dict):
                citation_budget = {
                    "target_refs": int(alloc.get("target_refs", 0) or 0),
                    "rationale": alloc.get("rationale", ""),
                }
            else:
                citation_budget = {}

            sections.append(SectionPlan(
                section_type=section_type,
                section_title=section_title,
                paragraphs=paragraphs,
                figures=figure_placements,
                tables=table_placements,
                figures_to_reference=section_data.get("figures_to_reference", []),
                tables_to_reference=section_data.get("tables_to_reference", []),
                content_sources=section_data.get(
                    "content_sources", self._get_default_sources(section_type),
                ),
                depends_on=self._get_dependencies(section_type),
                citation_budget=citation_budget,
                topic_clusters=self._normalize_string_list(
                    section_data.get("topic_clusters", []), max_items=4,
                ),
                transition_intents=self._normalize_string_list(
                    section_data.get("transition_intents", []), max_items=3,
                ),
                sectioning_recommended=self._coerce_bool(
                    section_data.get("sectioning_recommended", False),
                ),
                code_focus=self._normalize_code_focus(
                    section_data.get("code_focus", {}),
                ),
                writing_guidance=section_data.get("writing_guidance", ""),
                order=order,
            ))

            logger.info(
                "planner.step3_section section=%s paragraphs=%d sentences=%d",
                section_type, len(paragraphs),
                sum(p.approx_sentences for p in paragraphs),
            )

        # Whole-plan paragraph budget validation
        target_paras = estimate_target_paragraphs(total_words)
        llm_total_paras = sum(len(sp.paragraphs) for sp in sections)
        if llm_total_paras > 0 and llm_total_paras < target_paras * 0.5:
            scale = target_paras / max(1, llm_total_paras)
            for sp in sections:
                if sp.section_type in ("abstract", "conclusion"):
                    continue
                section_target_sents = int(
                    sum(p.approx_sentences for p in sp.paragraphs) * scale
                )
                sp.paragraphs = self._expand_paragraph_plan(
                    sp.paragraphs, section_target_sents, sp.section_type,
                )
            expanded_total = sum(len(sp.paragraphs) for sp in sections)
            logger.info(
                "planner.plan_budget_expansion llm_paras=%d target=%d expanded=%d",
                llm_total_paras, target_paras, expanded_total,
            )

        paper_plan = PaperPlan(
            title=request.title,
            paper_type=paper_type,
            sections=sections,
            contributions=contributions,
            narrative_style=narrative_style,
            terminology=structure.get("terminology", {}),
            structure_rationale=structure_rationale,
            abstract_focus=abstract_focus,
            citation_strategy=citation_strategy,
        )

        self._assign_figure_table_definitions(paper_plan, request, figure_analyses, table_analyses)

        logger.info(
            "planner.plan_created sections=%d sentences=%d",
            len(paper_plan.sections), paper_plan.get_total_sentences(),
        )
        self._last_plan = paper_plan
        return paper_plan

    async def discover_seed_references(
        self,
        title: str,
        idea_hypothesis: str,
        method: str,
        data: str,
        experiments: str,
        existing_ref_keys: List[str],
        paper_search_config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Discover global references before section-level planning.
        """
        import asyncio
        from ..shared.tools.paper_search import PaperSearchTool

        cfg = paper_search_config or {}
        tool = PaperSearchTool(
            serpapi_api_key=cfg.get("serpapi_api_key"),
            semantic_scholar_api_key=cfg.get("semantic_scholar_api_key"),
            timeout=cfg.get("timeout", 10),
            semantic_scholar_min_results_before_fallback=cfg.get(
                "semantic_scholar_min_results_before_fallback", 3
            ),
            enable_query_cache=cfg.get("enable_query_cache", True),
            cache_ttl_hours=cfg.get("cache_ttl_hours", 24),
        )

        max_queries = max(3, min(6, int(cfg.get("planner_max_queries_per_section", 5))))
        per_round = max(3, int(cfg.get("search_results_per_round", 5)))
        delay_sec = max(0.5, float(cfg.get("planner_inter_round_delay_sec", 1.5)))

        seeds = [
            title,
            f"{title} {idea_hypothesis[:160]}",
            f"{title} {method[:180]}",
            f"{title} {data[:160]}",
            f"{title} {experiments[:180]}",
            f"{method[:120]} {experiments[:120]}",
        ]
        queries: List[str] = []
        seen_q = set()
        for q in seeds:
            qq = " ".join(str(q).split()).strip()
            if len(qq) < 8 or qq in seen_q:
                continue
            seen_q.add(qq)
            queries.append(qq)
            if len(queries) >= max_queries:
                break

        discovered: List[Dict[str, Any]] = []
        seen_keys = set(existing_ref_keys)
        for i, query in enumerate(queries):
            if i > 0:
                await asyncio.sleep(delay_sec)
            try:
                result = await tool.execute(query=query, max_results=per_round)
                if not result.success:
                    continue
                papers = result.data.get("papers", []) if result.data else []
                for paper in papers:
                    bkey = paper.get("bibtex_key", "")
                    bibtex = paper.get("bibtex", "")
                    if bkey and bibtex and bkey not in seen_keys:
                        seen_keys.add(bkey)
                        discovered.append(
                            {
                                "ref_id": bkey,
                                "bibtex": bibtex,
                                "title": paper.get("title", ""),
                                "year": paper.get("year"),
                                "abstract": paper.get("abstract", ""),
                                "venue": paper.get("venue", ""),
                                "citation_count": paper.get("citation_count"),
                                "source": paper.get("source", ""),
                            }
                        )
            except Exception as e:
                logger.warning("planner.seed_search_error query='%s': %s", query[:80], e)

        logger.info("planner.seed_reference_discovery count=%d", len(discovered))
        return discovered

    # =====================================================================
    # Reference discovery
    # =====================================================================

    async def discover_references(
        self,
        plan: PaperPlan,
        existing_ref_keys: List[str],
        paper_search_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Discover additional references for each section based on the plan.
        - **Description**:
            - Analyzes each section's key points and generates search queries.
            - Executes multi-round searches via PaperSearchTool.
            - Supports loop searching until target count is reached or no more queries.
            - Returns discovered papers grouped by section_type.
            - Called once during planning, replacing per-section search judgment.

        - **Args**:
            - `plan` (PaperPlan): The paper plan with section structures.
            - `existing_ref_keys` (List[str]): Already-available citation keys.
            - `paper_search_config` (dict, optional): PaperSearchTool config.

        - **Returns**:
            - `Dict[str, List[Dict]]`: section_type -> list of discovered papers
              (each with ref_id, bibtex, title, etc.)
        """
        import asyncio
        from ..shared.tools.paper_search import PaperSearchTool

        cfg = paper_search_config or {}
        tool = PaperSearchTool(
            serpapi_api_key=cfg.get("serpapi_api_key"),
            semantic_scholar_api_key=cfg.get("semantic_scholar_api_key"),
            timeout=cfg.get("timeout", 10),
            semantic_scholar_min_results_before_fallback=cfg.get(
                "semantic_scholar_min_results_before_fallback", 3
            ),
            enable_query_cache=cfg.get("enable_query_cache", True),
            cache_ttl_hours=cfg.get("cache_ttl_hours", 24),
        )

        # Read configuration for multi-round search
        results_per_round = cfg.get("search_results_per_round", 5)
        max_queries_per_section = cfg.get("planner_max_queries_per_section", 5)
        inter_round_delay = cfg.get("planner_inter_round_delay_sec", 1.5)
        min_target_papers = cfg.get("planner_min_target_papers_per_section", 3)

        # Build search queries from plan — multiple per section for multi-round search
        section_queries: Dict[str, List[str]] = {}
        section_targets: Dict[str, int] = {}  # Target paper count per section

        # --- Top-down citation target computation ---
        # Priority: 1) per-section citation_budget from Planner
        #           2) global citation_strategy from Planner
        #           3) venue-aware estimation + proportional distribution
        body_sections = [
            sp for sp in plan.sections
            if sp.section_type not in ("abstract", "conclusion")
        ]

        strategy = plan.citation_strategy if isinstance(plan.citation_strategy, dict) else {}
        global_total = int(strategy.get("total_target", 0) or 0)
        section_allocation = strategy.get("section_allocation")

        if global_total <= 0:
            # Planner did not provide global strategy; estimate from venue + scale
            total_paras = sum(len(sp.paragraphs) for sp in body_sections)
            global_total = self._estimate_total_citations(
                style_guide=cfg.get("style_guide"),
                n_body_sections=len(body_sections),
                total_paragraphs=total_paras,
            )
            section_allocation = None
            logger.info(
                "planner.citation_strategy fallback: estimated total_target=%d "
                "(venue=%s, body_sections=%d, paragraphs=%d)",
                global_total, cfg.get("style_guide", "unknown"),
                len(body_sections), total_paras,
            )
        else:
            logger.info(
                "planner.citation_strategy from_planner: total_target=%d",
                global_total,
            )

        topdown_targets = self._distribute_citations_topdown(
            total_target=global_total,
            body_sections=body_sections,
            section_allocation=section_allocation,
        )

        for sp in plan.sections:
            if sp.section_type in ("abstract", "conclusion"):
                continue
            key_points = sp.get_key_points()
            if not key_points:
                continue

            # Generate multiple search queries for this section
            queries = await self._generate_search_queries(
                sp.section_type, key_points, existing_ref_keys, plan.title,
            )

            # Store up to N queries per section for multi-round search
            if queries:
                section_queries[sp.section_type] = queries[:max_queries_per_section]
                # Priority: 1) planner per-section budget, 2) top-down allocation
                planner_budget = sp.citation_budget if isinstance(sp.citation_budget, dict) else {}
                planner_target = planner_budget.get("target_refs")
                if planner_target is not None:
                    try:
                        section_targets[sp.section_type] = max(1, int(planner_target))
                    except Exception:
                        section_targets[sp.section_type] = topdown_targets.get(
                            sp.section_type, min_target_papers,
                        )
                else:
                    section_targets[sp.section_type] = topdown_targets.get(
                        sp.section_type, min_target_papers,
                    )

        discovered: Dict[str, List[Dict[str, Any]]] = {}
        seen_keys: set = set(existing_ref_keys)

        for section_type, queries in section_queries.items():
            section_papers: List[Dict[str, Any]] = []
            target_count = section_targets.get(section_type, 3)
            round_num = 0

            # Collect candidates from all planned rounds, then filter/select to target_count.
            while round_num < len(queries):
                query = queries[round_num]

                if round_num > 0:
                    # Rate limiting between rounds with small jitter to reduce burst contention
                    jitter = random.uniform(0, 0.4)
                    await asyncio.sleep(max(0.0, inter_round_delay + jitter))

                try:
                    result = await tool.execute(query=query, max_results=results_per_round)
                    if not result.success:
                        round_num += 1
                        continue

                    papers = result.data.get("papers", []) if result.data else []

                    for paper in papers:
                        bkey = paper.get("bibtex_key", "")
                        bibtex = paper.get("bibtex", "")
                        if bkey and bibtex and bkey not in seen_keys:
                            seen_keys.add(bkey)
                            section_papers.append({
                                "ref_id": bkey,
                                "bibtex": bibtex,
                                "title": paper.get("title", ""),
                                "year": paper.get("year"),
                                "abstract": paper.get("abstract", ""),
                                "venue": paper.get("venue", ""),
                                "citation_count": paper.get("citation_count"),
                            })

                    logger.info(
                        "planner.search_round section=%s round=%d query='%s' found=%d total=%d",
                        section_type, round_num, query[:50], len(papers), len(section_papers),
                    )
                except Exception as e:
                    logger.warning("planner.search_error query='%s': %s", query, e)

                round_num += 1

            # Filter papers by relevance before storing
            if section_papers:
                raw_count = len(section_papers)
                # Get key points for this section from the plan
                section_key_points = []
                for sp in plan.sections:
                    if sp.section_type == section_type:
                        section_key_points = sp.get_key_points()
                        break

                # Filter by relevance using LLM
                filtered_papers = await self._filter_papers_by_relevance(
                    papers=section_papers,
                    section_type=section_type,
                    key_points=section_key_points,
                    paper_title=plan.title,
                )
                filtered_count = len(filtered_papers)

                # Keep exactly N when possible: select top target_count by quality.
                filtered_sorted = sorted(
                    filtered_papers,
                    key=lambda p: (
                        float(p.get("relevance_score") or 0.0),
                        int(p.get("citation_count") or 0),
                        int(p.get("year") or 0),
                    ),
                    reverse=True,
                )
                selected_papers = filtered_sorted[:target_count] if target_count > 0 else filtered_sorted

                if selected_papers:
                    discovered[section_type] = selected_papers
                    logger.info(
                        "planner.discovered_refs section=%s target=%d raw=%d filtered=%d selected=%d",
                        section_type, target_count, raw_count, filtered_count, len(selected_papers),
                    )

        total = sum(len(v) for v in discovered.values())
        logger.info("planner.reference_discovery_complete total=%d", total)
        return discovered

    def assign_references(
        self,
        plan: "PaperPlan",
        discovered: Dict[str, List[Dict[str, Any]]],
        core_ref_keys: List[str],
        paper_search_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Distribute references to sections, populating SectionPlan.assigned_refs.

        - **Description**:
            - Discovered refs are assigned to the section they were found for.
            - Core (user-provided) refs are assigned to all body sections
              so every section can cite them.
            - Abstract and conclusion get NO refs (citations forbidden there).
            - A single ref can appear in multiple sections.

        - **Args**:
            - `plan` (PaperPlan): The paper plan to mutate in-place.
            - `discovered` (Dict[str, List[Dict]]): section_type -> papers from
              discover_references().
            - `core_ref_keys` (List[str]): Citation keys of user-provided refs.
        """
        cfg = paper_search_config or {}
        budget_enabled = cfg.get("citation_budget_enabled", True)
        soft_cap = cfg.get("citation_budget_soft_cap", True)
        reserve_size = max(1, int(cfg.get("citation_budget_reserve_size", 4)))

        no_cite_sections = {"abstract", "conclusion"}

        # Compute top-down allocation for fallback
        body_sections = [
            sp for sp in plan.sections
            if sp.section_type not in no_cite_sections
        ]
        strategy = plan.citation_strategy if isinstance(plan.citation_strategy, dict) else {}
        global_total = int(strategy.get("total_target", 0) or 0)
        section_allocation = strategy.get("section_allocation")

        if global_total <= 0:
            total_paras = sum(len(sp.paragraphs) for sp in body_sections)
            global_total = self._estimate_total_citations(
                style_guide=cfg.get("style_guide"),
                n_body_sections=len(body_sections),
                total_paragraphs=total_paras,
            )
            section_allocation = None

        topdown_targets = self._distribute_citations_topdown(
            total_target=global_total,
            body_sections=body_sections,
            section_allocation=section_allocation,
        )

        for sp in plan.sections:
            if sp.section_type in no_cite_sections:
                sp.assigned_refs = []
                sp.budget_selected_refs = []
                sp.budget_reserve_refs = []
                sp.budget_must_use_refs = []
                sp.citation_budget = {
                    "enabled": budget_enabled,
                    "min_refs": 0,
                    "target_refs": 0,
                    "max_refs": 0,
                    "candidate_count": 0,
                    "selected_count": 0,
                    "soft_cap": soft_cap,
                }
                continue

            discovered_for_section = discovered.get(sp.section_type, [])
            discovered_ranked = self._rank_references_for_section(discovered_for_section)
            planner_hint_refs = [r for r in sp.get_all_references() if r]
            budget = self._infer_section_citation_budget(
                section_type=sp.section_type,
                paragraph_count=len(sp.paragraphs),
                candidate_refs=discovered_ranked,
                planner_hint_refs=planner_hint_refs,
                core_ref_keys=core_ref_keys,
                planner_budget=sp.citation_budget if isinstance(sp.citation_budget, dict) else {},
                topdown_target=topdown_targets.get(sp.section_type),
            )

            if not budget_enabled:
                refs: List[str] = list(core_ref_keys)
                for paper in discovered_ranked:
                    rid = paper.get("ref_id", "")
                    if rid and rid not in refs:
                        refs.append(rid)
                sp.assigned_refs = refs
                sp.budget_selected_refs = refs
                sp.budget_reserve_refs = []
                sp.budget_must_use_refs = planner_hint_refs[:3]
                budget["enabled"] = False
                budget["selected_count"] = len(refs)
                sp.citation_budget = budget
                continue

            selected_refs = list(budget.get("selected_refs", []))
            reserve_refs = list(budget.get("reserve_refs", []))
            must_use_refs = list(budget.get("must_use_refs", []))

            if not selected_refs:
                fallback = [k for k in planner_hint_refs if k in core_ref_keys]
                selected_refs = fallback[: max(1, budget.get("target_refs", 1))]
            sp.assigned_refs = selected_refs
            sp.budget_selected_refs = selected_refs
            sp.budget_reserve_refs = reserve_refs[:reserve_size]
            sp.budget_must_use_refs = must_use_refs
            budget["enabled"] = True
            budget["selected_count"] = len(selected_refs)
            budget["soft_cap"] = soft_cap
            sp.citation_budget = budget

        assigned_counts = {
            sp.section_type: len(sp.assigned_refs)
            for sp in plan.sections if sp.assigned_refs
        }
        logger.info("planner.assign_references result=%s", assigned_counts)

    @staticmethod
    def _estimate_total_citations(
        style_guide: Optional[str],
        n_body_sections: int,
        total_paragraphs: int,
    ) -> int:
        """
        Estimate total citations for the paper when Planner omits
        citation_strategy, using venue conventions and paper scale.

        - **Args**:
          - `style_guide` (str | None): Venue hint (e.g. "Nature", "NeurIPS").
          - `n_body_sections` (int): Number of body sections (excl abstract/conclusion).
          - `total_paragraphs` (int): Total paragraph count across body sections.

        - **Returns**:
          - `int`: Estimated total citation target.
        """
        sg = (style_guide or "").lower()

        # Venue-aware base range (mid-point used as default)
        if any(v in sg for v in ("nature", "science", "cell", "lancet", "nejm")):
            base = 35
        elif any(v in sg for v in ("neurips", "icml", "iclr", "aaai", "cvpr", "acl", "emnlp")):
            base = 30
        elif any(v in sg for v in ("journal", "tpami", "jmlr", "tkde", "tac")):
            base = 45
        elif "workshop" in sg:
            base = 18
        else:
            base = 30

        # Scale by paper complexity
        scale_factor = max(1.0, total_paragraphs / 15.0)
        return max(15, int(base * min(scale_factor, 2.0)))

    @staticmethod
    def _distribute_citations_topdown(
        total_target: int,
        body_sections: List["SectionPlan"],
        section_allocation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, int]:
        """
        Distribute total citation target across body sections by weight.

        - **Description**:
          - If `section_allocation` is available (from Planner's citation_strategy),
            use share_pct to derive per-section targets.
          - Otherwise, distribute proportionally by paragraph count, giving
            each section a fair share of the total.

        - **Args**:
          - `total_target` (int): Total citation target for the paper.
          - `body_sections` (List[SectionPlan]): Body section plans.
          - `section_allocation` (dict | None): Planner-provided share_pct map.

        - **Returns**:
          - `Dict[str, int]`: section_type -> target citation count.
        """
        targets: Dict[str, int] = {}

        if section_allocation:
            allocated = 0
            for sp in body_sections:
                alloc = section_allocation.get(sp.section_type, {})
                if not isinstance(alloc, dict):
                    alloc = {}
                # Prefer explicit target_refs, fall back to share_pct
                direct_target = alloc.get("target_refs")
                if direct_target is not None:
                    t = max(2, int(direct_target))
                else:
                    pct = float(alloc.get("share_pct", 0))
                    t = max(2, int(total_target * pct / 100.0))
                targets[sp.section_type] = t
                allocated += t
            remainder = total_target - allocated
            for sp in body_sections:
                if sp.section_type not in targets or targets[sp.section_type] <= 2:
                    bonus = max(0, remainder // max(1, len(body_sections)))
                    targets[sp.section_type] = targets.get(sp.section_type, 2) + bonus
        else:
            # Proportional distribution by paragraph count
            total_paras = sum(len(sp.paragraphs) for sp in body_sections) or 1
            for sp in body_sections:
                n_paras = max(1, len(sp.paragraphs))
                share = n_paras / total_paras
                targets[sp.section_type] = max(2, int(total_target * share))

        return targets

    def _rank_references_for_section(
        self,
        papers: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Rank section candidate papers by relevance, quality and recency.
        """
        return sorted(
            papers,
            key=lambda p: (
                float(p.get("relevance_score") or 0.0),
                int(p.get("citation_count") or 0),
                int(p.get("year") or 0),
            ),
            reverse=True,
        )

    def _infer_section_citation_budget(
        self,
        section_type: str,
        paragraph_count: int,
        candidate_refs: List[Dict[str, Any]],
        planner_hint_refs: List[str],
        core_ref_keys: List[str],
        planner_budget: Optional[Dict[str, Any]] = None,
        topdown_target: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Infer per-section citation budget, prioritizing planner-provided budget.

        - **Args**:
          - `topdown_target` (int | None): Target derived from the global
            citation_strategy (top-down distribution).  Used as fallback
            when planner_budget is absent.
        """
        candidate_keys = [p.get("ref_id", "") for p in candidate_refs if p.get("ref_id")]
        allowed_key_set = set(candidate_keys) | set(core_ref_keys)
        planner_hint_refs = [r for r in planner_hint_refs if r in allowed_key_set]
        high_quality = [
            p for p in candidate_refs
            if float(p.get("relevance_score") or 0.0) >= 8.0
            or int(p.get("citation_count") or 0) >= 80
        ]
        candidate_count = len(candidate_keys) + len(core_ref_keys)
        normalized_budget = planner_budget or {}
        budget_min = normalized_budget.get("min_refs")
        budget_target = normalized_budget.get("target_refs")
        budget_max = normalized_budget.get("max_refs")

        if budget_target is not None:
            target_refs = max(0, int(budget_target))
        elif topdown_target is not None and topdown_target > 0:
            target_refs = topdown_target
        else:
            # Last-resort fallback: paragraph count + evidence signals
            complexity_signal = max(1, paragraph_count)
            evidence_signal = len(planner_hint_refs) + max(0, len(high_quality) // 2)
            target_refs = max(3, complexity_signal + evidence_signal)

        if budget_min is not None:
            min_refs = max(0, int(budget_min))
        else:
            min_refs = max(1, min(target_refs, max(paragraph_count, 3)))

        if budget_max is not None:
            max_refs = max(int(budget_max), target_refs, min_refs)
        else:
            max_refs = max(target_refs, min_refs, candidate_count)

        must_use_refs: List[str] = []
        for rid in planner_hint_refs:
            if rid and rid not in must_use_refs:
                must_use_refs.append(rid)
            if len(must_use_refs) >= 3:
                break
        for p in high_quality:
            rid = p.get("ref_id", "")
            if rid and rid in allowed_key_set and rid not in must_use_refs:
                must_use_refs.append(rid)
            if len(must_use_refs) >= 4:
                break

        selected_refs: List[str] = []
        for rid in must_use_refs:
            if rid and rid not in selected_refs:
                selected_refs.append(rid)

        for rid in candidate_keys:
            if len(selected_refs) >= target_refs:
                break
            if rid and rid not in selected_refs:
                selected_refs.append(rid)

        for rid in core_ref_keys:
            if len(selected_refs) >= min_refs:
                break
            if rid and rid not in selected_refs:
                selected_refs.append(rid)

        reserve_refs: List[str] = []
        for rid in candidate_keys:
            if rid and rid not in selected_refs and rid not in reserve_refs:
                reserve_refs.append(rid)
            if len(reserve_refs) >= 8:
                break

        return {
            "section_type": section_type,
            "min_refs": min_refs,
            "target_refs": target_refs,
            "max_refs": max_refs,
            "candidate_count": candidate_count,
            "must_use_refs": must_use_refs,
            "selected_refs": selected_refs[:max_refs] if max_refs > 0 else selected_refs,
            "reserve_refs": reserve_refs,
            "planner_hint_refs": planner_hint_refs[:8],
            "planner_budget_used": bool(budget_target is not None),
        }

    def _build_context_fallback_payload(
        self,
        *,
        plan: "PaperPlan",
        discovered: Dict[str, List[Dict[str, Any]]],
        all_papers: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Build non-empty fallback research context when LLM JSON parsing fails.
        """
        paper_assignments = self._assign_papers_to_sections(plan, discovered)
        claim_evidence_matrix: List[Dict[str, Any]] = []
        for section_type, refs in paper_assignments.items():
            if section_type in {"abstract", "conclusion"}:
                continue
            if not refs:
                continue
            claim_evidence_matrix.append(
                {
                    "section_type": section_type,
                    "claim": f"Key findings and arguments in {section_type} should be supported by assigned evidence.",
                    "support_refs": refs[:4],
                    "reason": "Fallback mapping from section assignment after context parsing failure.",
                    "priority": "P1",
                }
            )

        contribs = list(plan.contributions or [])
        p0 = contribs[:2]
        p1 = contribs[2:4]
        p2 = contribs[4:6]
        contribution_ranking = {
            "P0": [
                {
                    "contribution": c,
                    "why_it_matters": "Core contribution from planner output.",
                    "suggested_sections": ["introduction", "methods", "results"],
                    "suggested_result_focus": "Highlight primary quantitative gains.",
                }
                for c in p0
            ],
            "P1": [
                {
                    "contribution": c,
                    "why_it_matters": "Important but secondary contribution.",
                    "suggested_sections": ["discussion", "results"],
                    "suggested_result_focus": "Position as supporting evidence.",
                }
                for c in p1
            ],
            "P2": [
                {
                    "contribution": c,
                    "why_it_matters": "Optional or auxiliary contribution.",
                    "suggested_sections": ["discussion"],
                    "suggested_result_focus": "Mention briefly if space allows.",
                }
                for c in p2
            ],
        }

        return {
            "research_area": "Research area analysis",
            "summary": f"Found {len(all_papers)} relevant papers across {len(discovered)} sections.",
            "key_papers": [],
            "research_trends": [],
            "gaps": [],
            "claim_evidence_matrix": claim_evidence_matrix,
            "contribution_ranking": contribution_ranking,
            "planning_decision_trace": [
                "Used heuristic fallback context because structured JSON parsing failed."
            ],
            "paper_assignments": paper_assignments,
        }

    async def _generate_search_queries(
        self,
        section_type: str,
        key_points: List[str],
        existing_refs: List[str],
        paper_title: str,
    ) -> List[str]:
        """
        Generate search queries for a section using a lightweight LLM call.
        - **Description**:
            - Asks the LLM to suggest 1-2 search queries based on the
              section's key points and gaps in existing references.

        - **Args**:
            - `section_type` (str): The section type.
            - `key_points` (List[str]): Key points from the plan.
            - `existing_refs` (List[str]): Available citation keys.
            - `paper_title` (str): Paper title for context.

        - **Returns**:
            - `List[str]`: Search queries (1-2 per section).
        """
        kp_text = "; ".join(key_points[:4])
        refs_text = ", ".join(existing_refs[:10]) if existing_refs else "none"
        prompt = (
            f"Paper: {paper_title}\n"
            f"Section: {section_type}\n"
            f"Key points: {kp_text}\n"
            f"Existing references: {refs_text}\n\n"
            "Generate 1-2 academic search queries to find relevant papers "
            "for this section. Output JSON: {\"queries\": [\"...\"]}"
        )
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an academic research assistant. Respond with JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )
            raw = response.choices[0].message.content or ""
            data = self._safe_load_json(raw, expected=dict)
            if data is None:
                raise ValueError("Could not parse JSON object from query generation output")
            queries = data.get("queries", [])
            return [q for q in queries if isinstance(q, str) and len(q.strip()) > 3]
        except Exception as e:
            logger.warning("planner.query_generation_error section=%s: %s", section_type, e)
            return []

    async def _filter_papers_by_relevance(
        self,
        papers: List[Dict[str, Any]],
        section_type: str,
        key_points: List[str],
        paper_title: str,
    ) -> List[Dict[str, Any]]:
        """
        Filter discovered papers by relevance to a specific section.

        - **Description**:
            - Uses LLM to evaluate each paper's relevance, quality, and timeliness.
            - Scores papers on relevance (0-10), quality, and recency.
            - Returns filtered list with relevance scores.

        - **Args**:
            - `papers` (List[Dict]): List of discovered papers.
            - `section_type` (str): The section type.
            - `key_points` (List[str]): Key points from the plan.
            - `paper_title` (str): Paper title for context.

        - **Returns**:
            - `List[Dict]`: Filtered papers with relevance scores.
        """
        if not papers:
            return []

        # Prepare paper information for LLM evaluation
        paper_list = []
        for i, p in enumerate(papers):
            paper_list.append({
                "index": i,
                "title": p.get("title", ""),
                "year": p.get("year", ""),
                "venue": p.get("venue", ""),
                "abstract": p.get("abstract", "")[:300] if p.get("abstract") else "",
            })

        kp_text = "; ".join(key_points[:4])
        papers_json = json.dumps(paper_list, ensure_ascii=False)

        prompt = (
            f"Paper: {paper_title}\n"
            f"Section: {section_type}\n"
            f"Key points: {kp_text}\n\n"
            f"Discovered papers:\n{papers_json}\n\n"
            "Evaluate each paper's relevance to this section on a scale of 0-10. "
            "Consider: (1) relevance to key points, (2) paper quality (venue, citations), "
            "(3) recency (prefer papers from the last 5 years). "
            "Output JSON array with format: "
            "[{\"index\": 0, \"relevance_score\": 8, \"reason\": \"brief justification\"}]"
        )

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
            evaluations = self._safe_load_json(raw, expected=list)
            if evaluations is None:
                raise ValueError("Could not parse JSON array from relevance output")

            # Build score lookup
            score_map: Dict[int, Dict[str, Any]] = {}
            for ev in evaluations:
                idx = ev.get("index")
                if idx is not None and 0 <= idx < len(papers):
                    score_map[idx] = {
                        "relevance_score": ev.get("relevance_score", 0),
                        "reason": ev.get("reason", ""),
                    }

            # Filter papers with score >= 5
            filtered = []
            for i, paper in enumerate(papers):
                score_info = score_map.get(i, {})
                score = score_info.get("relevance_score", 0)
                if score >= 5:
                    paper["relevance_score"] = score
                    paper["relevance_reason"] = score_info.get("reason", "")
                    filtered.append(paper)

            logger.info(
                "planner.filter_papers section=%s input=%d output=%d",
                section_type, len(papers), len(filtered),
            )
            return filtered

        except Exception as e:
            logger.warning("planner.filter_error section=%s: %s", section_type, e)
            # Return all papers if filtering fails
            return papers

    async def _generate_research_context(
        self,
        plan: "PaperPlan",
        discovered: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """
        Generate a research context summary from discovered papers.

        - **Description**:
            - Analyzes all discovered papers to generate a research overview.
            - Identifies key papers, research trends, and gaps.
            - Assigns papers to appropriate sections.
            - Returns a structured context dictionary.

        - **Args**:
            - `plan` (PaperPlan): The paper plan.
            - `discovered` (Dict[str, List[Dict]]): section_type -> papers.

        - **Returns**:
            - `Dict`: Research context with summary, key_papers, paper_assignments, etc.
        """
        # Collect all discovered papers
        all_papers: List[Dict[str, Any]] = []
        for section_papers in discovered.values():
            all_papers.extend(section_papers)

        if not all_papers:
            return {
                "research_area": "",
                "summary": "No papers discovered.",
                "key_papers": [],
                "research_trends": [],
                "gaps": [],
                "paper_assignments": {},
            }

        # Prepare paper information for LLM (cap volume to reduce malformed/truncated outputs)
        ranked_papers = sorted(
            all_papers,
            key=lambda p: (
                int(p.get("citation_count") or 0),
                int(p.get("year") or 0),
            ),
            reverse=True,
        )
        analysis_papers = ranked_papers[:24]
        paper_summaries = []
        for p in analysis_papers:
            paper_summaries.append({
                "title": p.get("title", ""),
                "year": p.get("year"),
                "venue": p.get("venue", ""),
                "citation_count": p.get("citation_count"),
                "abstract": p.get("abstract", "")[:200] if p.get("abstract") else "",
            })

        papers_json = json.dumps(paper_summaries, ensure_ascii=False)

        prompt = (
            f"Paper title: {plan.title}\n\n"
            f"Discovered papers:\n{papers_json}\n\n"
            "Analyze these papers and provide a JSON response with:\n"
            "1. research_area: Main research area/topic (brief)\n"
            "2. summary: Overview of the research landscape (2-3 sentences)\n"
            "3. key_papers: Top 5 most important papers with their contributions\n"
            "4. research_trends: 2-3 key research trends identified\n"
            "5. gaps: 2-3 research gaps or opportunities\n"
            "6. claim_evidence_matrix: 6-10 records with {section_type, claim, support_refs, reason, priority}\n"
            "7. contribution_ranking: object with keys P0/P1/P2, each item has "
            "{contribution, why_it_matters, suggested_sections, suggested_result_focus}\n"
            "8. planning_decision_trace: short list of explicit trade-off decisions\n"
            "Output ONLY JSON with this structure:\n"
            "{\"research_area\": \"...\", \"summary\": \"...\", "
            "\"key_papers\": [{\"title\": \"...\", \"contribution\": \"...\"}], "
            "\"research_trends\": [\"...\"], \"gaps\": [\"...\"], "
            "\"claim_evidence_matrix\": [{\"section_type\": \"method\", \"claim\": \"...\", "
            "\"support_refs\": [\"ref1\"], \"reason\": \"...\", \"priority\": \"P0\"}], "
            "\"contribution_ranking\": {\"P0\": [], \"P1\": [], \"P2\": []}, "
            "\"planning_decision_trace\": [\"...\"]}"
        )

        max_attempts = 3
        llm_raw_outputs: List[str] = []
        context: Optional[Dict[str, Any]] = None

        system_msg = "You are an academic research analyst. Respond with JSON only."
        repair_system_msg = "You are a strict JSON fixer. Return JSON object only."
        repair_prompt_template = (
            "Convert the following model output into a STRICT valid JSON object. "
            "Keep only these keys: research_area, summary, key_papers, research_trends, gaps, "
            "claim_evidence_matrix, contribution_ranking, planning_decision_trace.\n"
            "Rules:\n"
            "- Output ONLY JSON object, no markdown/code fences.\n"
            "- If a field is missing, fill with empty default.\n"
            "- contribution_ranking must be an object with keys P0/P1/P2 (arrays).\n"
            "- claim_evidence_matrix and planning_decision_trace must be arrays.\n\n"
            "Raw output:\n{raw_output}"
        )

        for attempt in range(1, max_attempts + 1):
            try:
                temperature = max(0.1, 0.4 - 0.1 * attempt)
                max_tokens = 1400 + 200 * (attempt - 1)

                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                raw = response.choices[0].message.content or ""
                llm_raw_outputs.append(raw)
                logger.info(
                    "planner.research_context attempt=%d/%d raw_len=%d",
                    attempt, max_attempts, len(raw),
                )

                context = self._safe_load_json(raw, expected=dict)
                if context is not None:
                    break

                # JSON repair pass
                repair_resp = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": repair_system_msg},
                        {
                            "role": "user",
                            "content": repair_prompt_template.format(
                                raw_output=raw[:12000],
                            ),
                        },
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                )
                repaired_raw = repair_resp.choices[0].message.content or ""
                llm_raw_outputs.append(f"[repair_attempt_{attempt}] {repaired_raw}")

                context = self._safe_load_json(repaired_raw, expected=dict)
                if context is not None:
                    logger.info(
                        "planner.research_context attempt=%d/%d parsed via repair",
                        attempt, max_attempts,
                    )
                    break

                logger.warning(
                    "planner.research_context attempt=%d/%d parse_failed",
                    attempt, max_attempts,
                )

            except Exception as e:
                logger.warning(
                    "planner.research_context attempt=%d/%d error: %s",
                    attempt, max_attempts, e,
                )

        # Build paper assignments (always deterministic)
        paper_assignments = self._assign_papers_to_sections(plan, discovered)

        if context is not None:
            fallback_context = self._build_context_fallback_payload(
                plan=plan,
                discovered=discovered,
                all_papers=all_papers,
            )

            parsed_claim_matrix = context.get("claim_evidence_matrix", [])
            parsed_ranking = context.get("contribution_ranking", {"P0": [], "P1": [], "P2": []})
            ranking_empty = (
                not isinstance(parsed_ranking, dict)
                or (
                    not parsed_ranking.get("P0")
                    and not parsed_ranking.get("P1")
                    and not parsed_ranking.get("P2")
                )
            )

            return {
                "research_area": context.get("research_area", ""),
                "summary": context.get("summary", ""),
                "key_papers": context.get("key_papers", [])[:10],
                "research_trends": context.get("research_trends", []),
                "gaps": context.get("gaps", []),
                "claim_evidence_matrix": (
                    parsed_claim_matrix
                    if parsed_claim_matrix
                    else fallback_context.get("claim_evidence_matrix", [])
                ),
                "contribution_ranking": (
                    parsed_ranking
                    if not ranking_empty
                    else fallback_context.get("contribution_ranking", {"P0": [], "P1": [], "P2": []})
                ),
                "planning_decision_trace": context.get("planning_decision_trace", []),
                "paper_assignments": paper_assignments,
            }

        # All attempts failed — build fallback and attach raw LLM outputs
        logger.warning(
            "planner.research_context all %d attempts failed, using fallback",
            max_attempts,
        )
        fallback = self._build_context_fallback_payload(
            plan=plan,
            discovered=discovered,
            all_papers=all_papers,
        )
        fallback["_llm_raw_outputs"] = llm_raw_outputs
        return fallback

    def _assign_papers_to_sections(
        self,
        plan: "PaperPlan",
        discovered: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, List[str]]:
        """
        Assign papers to sections based on where they were discovered.

        - **Args**:
            - `plan` (PaperPlan): The paper plan.
            - `discovered` (Dict[str, List[Dict]]): section_type -> papers.

        - **Returns**:
            - `Dict[str, List[str]]`: section_type -> citation keys.
        """
        assignments: Dict[str, List[str]] = {}

        for section_type, papers in discovered.items():
            citation_keys = [p.get("ref_id", "") for p in papers if p.get("ref_id")]
            assignments[section_type] = citation_keys

        return assignments

    # =====================================================================
    # VLM analysis
    # =====================================================================

    async def _analyze_figures(
        self, figures: List[Any],
    ) -> Dict[str, Any]:
        """Analyze figure images with VLM (returns {fig_id: FigureAnalysis})."""
        results = {}
        if not self.vlm_service:
            return results
        for fig in figures:
            file_path = getattr(fig, "file_path", "") or ""
            if not file_path:
                continue
            try:
                analysis = await self.vlm_service.analyze_figure(file_path)
                results[fig.id] = analysis
                logger.info("planner.vlm_figure id=%s role=%s", fig.id, analysis.semantic_role)
            except Exception as e:
                logger.warning("planner.vlm_figure_error id=%s: %s", fig.id, e)
        return results

    async def _analyze_tables(
        self, tables: List[Any],
    ) -> Dict[str, Any]:
        """Analyze table images with VLM (returns {tbl_id: TableAnalysis})."""
        results = {}
        if not self.vlm_service:
            return results
        for tbl in tables:
            file_path = getattr(tbl, "file_path", "") or ""
            if not file_path:
                continue
            try:
                analysis = await self.vlm_service.analyze_table_image(file_path)
                results[tbl.id] = analysis
                logger.info("planner.vlm_table id=%s role=%s", tbl.id, analysis.semantic_role)
            except Exception as e:
                logger.warning("planner.vlm_table_error id=%s: %s", tbl.id, e)
        return results

    # =====================================================================
    # Building PaperPlan from LLM output
    # =====================================================================

    def _build_paper_plan(
        self,
        plan_data: Dict[str, Any],
        request: PlanRequest,
        total_words: int,
        figure_analyses: Optional[Dict[str, Any]] = None,
        table_analyses: Optional[Dict[str, Any]] = None,
    ) -> PaperPlan:
        """Build PaperPlan from LLM JSON output and VLM analyses."""

        paper_type_str = plan_data.get("paper_type", "empirical").lower()
        try:
            paper_type = PaperType(paper_type_str)
        except ValueError:
            paper_type = PaperType.EMPIRICAL

        style_str = plan_data.get("narrative_style", "technical").lower()
        try:
            narrative_style = NarrativeStyle(style_str)
        except ValueError:
            narrative_style = NarrativeStyle.TECHNICAL

        # Determine section ordering from LLM output
        llm_sections = plan_data.get("sections", [])
        section_map: Dict[str, Dict[str, Any]] = {}
        llm_section_order: List[str] = []
        for s in llm_sections:
            st = self._normalize_section_type_name(str(s.get("section_type", "")))
            if st and st not in section_map:
                if st != s.get("section_type"):
                    s = dict(s)
                    s["section_type"] = st
                section_map[st] = s
                llm_section_order.append(st)

        if len(llm_section_order) >= 3 and "abstract" not in llm_section_order:
            llm_section_order.insert(0, "abstract")

        use_llm_structure = len(llm_section_order) >= 3
        section_type_order = llm_section_order if use_llm_structure else list(DEFAULT_EMPIRICAL_SECTIONS)

        target_paragraphs = estimate_target_paragraphs(total_words)

        sections: List[SectionPlan] = []
        for order, section_type in enumerate(section_type_order):
            llm_section = section_map.get(section_type, {})

            # Parse paragraphs from LLM output
            raw_paragraphs = llm_section.get("paragraphs", [])
            paragraphs = self._parse_paragraph_plans(raw_paragraphs)
            if not paragraphs:
                n_sections = max(1, len(section_type_order))
                default_sents = max(3, (total_words // WORDS_PER_SENTENCE) // n_sections)
                paragraphs = self._generate_default_paragraphs(
                    section_type, default_sents, llm_section,
                )

            # Parse figure/table placements
            raw_figures = llm_section.get("figures", [])
            raw_tables = llm_section.get("tables", [])
            figure_placements = self._build_figure_placements(
                raw_figures, figure_analyses or {},
            )
            table_placements = self._build_table_placements(
                raw_tables, table_analyses or {},
            )

            # Cross-section references
            figs_to_ref = llm_section.get("figures_to_reference", [])
            tbls_to_ref = llm_section.get("tables_to_reference", [])

            sections.append(SectionPlan(
                section_type=section_type,
                section_title=llm_section.get(
                    "section_title", self._get_section_title(section_type),
                ),
                paragraphs=paragraphs,
                figures=figure_placements,
                tables=table_placements,
                figures_to_reference=figs_to_ref,
                tables_to_reference=tbls_to_ref,
                content_sources=llm_section.get(
                    "content_sources", self._get_default_sources(section_type),
                ),
                depends_on=llm_section.get(
                    "depends_on", self._get_dependencies(section_type),
                ),
                citation_budget=llm_section.get("citation_budget", {}),
                topic_clusters=self._normalize_string_list(
                    llm_section.get("topic_clusters", []),
                    max_items=4,
                ),
                transition_intents=self._normalize_string_list(
                    llm_section.get("transition_intents", []),
                    max_items=3,
                ),
                sectioning_recommended=self._coerce_bool(
                    llm_section.get("sectioning_recommended", False),
                ),
                code_focus=self._normalize_code_focus(
                    llm_section.get("code_focus", {}),
                ),
                writing_guidance=llm_section.get("writing_guidance", ""),
                order=order,
            ))

        # Whole-plan paragraph budget validation: if LLM planned far fewer
        # paragraphs than the target, scale up body sections proportionally.
        llm_total_paras = sum(len(sp.paragraphs) for sp in sections)
        if llm_total_paras > 0 and llm_total_paras < target_paragraphs * 0.5:
            scale = target_paragraphs / max(1, llm_total_paras)
            for sp in sections:
                if sp.section_type in ("abstract", "conclusion"):
                    continue
                section_target_sents = int(
                    sum(p.approx_sentences for p in sp.paragraphs) * scale
                )
                sp.paragraphs = self._expand_paragraph_plan(
                    sp.paragraphs, section_target_sents, sp.section_type,
                )
            expanded_total = sum(len(sp.paragraphs) for sp in sections)
            logger.info(
                "planner.plan_budget_expansion llm_paras=%d target=%d expanded=%d",
                llm_total_paras, target_paragraphs, expanded_total,
            )

        # Parse top-level citation strategy
        raw_strategy = plan_data.get("citation_strategy", {})
        if isinstance(raw_strategy, dict):
            citation_strategy = {
                "total_target": int(raw_strategy.get("total_target", 0) or 0),
                "rationale": str(raw_strategy.get("rationale", "")),
                "section_allocation": raw_strategy.get("section_allocation", {}),
            }
        else:
            citation_strategy = {}

        paper_plan = PaperPlan(
            title=request.title,
            paper_type=paper_type,
            sections=sections,
            contributions=plan_data.get("contributions", []),
            narrative_style=narrative_style,
            terminology=plan_data.get("terminology", {}),
            structure_rationale=plan_data.get("structure_rationale", ""),
            abstract_focus=plan_data.get("abstract_focus", ""),
            citation_strategy=citation_strategy,
        )

        # Assign any unassigned figures/tables to sections
        self._assign_figure_table_definitions(paper_plan, request, figure_analyses, table_analyses)

        return paper_plan

    def _parse_paragraph_plans(
        self, raw_paragraphs: List[Dict[str, Any]],
    ) -> List[ParagraphPlan]:
        """Parse paragraph plans from LLM JSON output."""
        paragraphs = []
        for raw in raw_paragraphs:
            if not isinstance(raw, dict):
                continue
            paragraphs.append(ParagraphPlan(
                key_point=raw.get("key_point", ""),
                supporting_points=raw.get("supporting_points", []),
                approx_sentences=raw.get("approx_sentences", 5),
                role=raw.get("role", "evidence"),
                references_to_cite=raw.get("references_to_cite", []),
                figures_to_reference=raw.get("figures_to_reference", []),
                tables_to_reference=raw.get("tables_to_reference", []),
            ))
        return paragraphs

    @staticmethod
    def _normalize_string_list(raw: Any, max_items: int = 5) -> List[str]:
        """
        Normalize mixed list/string into a clean bounded string list.
        """
        if isinstance(raw, str):
            items = [x.strip() for x in raw.split(",") if x.strip()]
        elif isinstance(raw, list):
            items = [str(x).strip() for x in raw if str(x).strip()]
        else:
            items = []
        # De-duplicate while preserving order
        deduped: List[str] = []
        for item in items:
            if item not in deduped:
                deduped.append(item)
            if len(deduped) >= max_items:
                break
        return deduped

    @staticmethod
    def _coerce_bool(raw: Any) -> bool:
        """
        Coerce bool from permissive raw JSON value.
        """
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)):
            return bool(raw)
        text = str(raw).strip().lower()
        return text in {"1", "true", "yes", "y", "recommended"}

    def _generate_default_paragraphs(
        self,
        section_type: str,
        section_sentences: int,
        llm_section: Dict[str, Any],
    ) -> List[ParagraphPlan]:
        """Generate default paragraph structure when LLM doesn't provide one."""
        # Try to use old-style key_points if available
        key_points = llm_section.get("key_points", [])
        refs = llm_section.get("references_to_cite", [])

        default_structures = {
            "abstract": [
                ("Research problem and motivation", "motivation", 2),
                ("Method and key results", "summary", 3),
            ],
            "introduction": [
                ("Research context and motivation", "motivation", 5),
                ("Problem statement and gap", "problem_statement", 4),
                ("Contributions", "summary", 4),
                ("Paper organization", "transition", 2),
            ],
            "related_work": [
                ("Prior work overview", "evidence", 5),
                ("Comparison and gaps", "comparison", 4),
            ],
            "method": [
                ("Overview of approach", "definition", 4),
                ("Technical details", "evidence", 6),
                ("Implementation specifics", "evidence", 4),
            ],
            "experiment": [
                ("Experimental setup", "definition", 4),
                ("Datasets and baselines", "evidence", 4),
            ],
            "result": [
                ("Main results", "evidence", 5),
                ("Analysis and discussion", "comparison", 4),
            ],
            "conclusion": [
                ("Summary of contributions", "summary", 4),
                ("Future work", "transition", 3),
            ],
        }

        if key_points:
            n_paragraphs = len(key_points)
            sentences_per = max(3, section_sentences // n_paragraphs)
            return [
                ParagraphPlan(
                    key_point=kp,
                    approx_sentences=sentences_per,
                    role="evidence",
                    references_to_cite=refs[:2] if i == 0 else [],
                )
                for i, kp in enumerate(key_points)
            ]

        structure = default_structures.get(section_type, [
            ("Main content", "evidence", max(3, section_sentences)),
        ])

        return [
            ParagraphPlan(
                key_point=kp, role=role, approx_sentences=sents,
            )
            for kp, role, sents in structure
        ]

    @staticmethod
    def _expand_paragraph_plan(
        existing: List[ParagraphPlan],
        target_sentences: int,
        section_type: str,
    ) -> List[ParagraphPlan]:
        """
        Expand an under-planned paragraph list to meet the section sentence budget.

        - **Description**:
          - When the Planner LLM generates far fewer paragraphs/sentences than
            the budget requires, this method proportionally scales up the plan.
          - Strategy: first increase approx_sentences on existing paragraphs
            (up to 8 each), then duplicate paragraphs with split sub-topics
            to fill the remaining gap.

        - **Args**:
          - `existing` (List[ParagraphPlan]): LLM-provided paragraphs.
          - `target_sentences` (int): Section sentence budget from word budget.
          - `section_type` (str): Section type for context.

        - **Returns**:
          - `List[ParagraphPlan]`: Expanded paragraph list.
        """
        if not existing:
            return existing

        current_total = sum(p.approx_sentences for p in existing)
        if current_total >= target_sentences * 0.7:
            return existing

        expanded = [
            ParagraphPlan(
                key_point=p.key_point,
                supporting_points=list(p.supporting_points),
                approx_sentences=p.approx_sentences,
                role=p.role,
                references_to_cite=list(p.references_to_cite),
                figures_to_reference=list(p.figures_to_reference),
                tables_to_reference=list(p.tables_to_reference),
            )
            for p in existing
        ]

        # Phase 1: Increase sentence counts on existing paragraphs (up to 8)
        for para in expanded:
            if sum(p.approx_sentences for p in expanded) >= target_sentences:
                break
            para.approx_sentences = min(8, para.approx_sentences + 3)

        # Phase 2: If still under budget, add elaboration paragraphs
        round_idx = 0
        while sum(p.approx_sentences for p in expanded) < target_sentences * 0.75:
            source = existing[round_idx % len(existing)]
            elaboration = ParagraphPlan(
                key_point=f"Further analysis of: {source.key_point}",
                supporting_points=["Additional evidence", "Extended discussion"],
                approx_sentences=min(6, max(3, (target_sentences - sum(p.approx_sentences for p in expanded)) // 3)),
                role=source.role if source.role != "motivation" else "evidence",
                references_to_cite=[],
            )
            expanded.append(elaboration)
            round_idx += 1
            if round_idx > len(existing) * 3:
                break

        return expanded

    def _build_figure_placements(
        self,
        raw_figures: List[Dict[str, Any]],
        figure_analyses: Dict[str, Any],
    ) -> List[FigurePlacement]:
        """Build FigurePlacement objects from LLM output + VLM analysis."""
        placements = []
        for raw in raw_figures:
            if not isinstance(raw, dict):
                continue
            fig_id = raw.get("figure_id", "")
            if not fig_id:
                continue
            vlm = figure_analyses.get(fig_id)
            placements.append(FigurePlacement(
                figure_id=fig_id,
                semantic_role=(
                    getattr(vlm, "semantic_role", "") if vlm
                    else raw.get("semantic_role", "")
                ),
                message=(
                    getattr(vlm, "message", "") if vlm
                    else raw.get("message", "")
                ),
                is_wide=(
                    getattr(vlm, "is_wide", False) if vlm
                    else raw.get("is_wide", False)
                ),
                position_hint=raw.get("position_hint", "mid"),
                caption_guidance=(
                    getattr(vlm, "caption_guidance", "") if vlm
                    else raw.get("caption_guidance", "")
                ),
            ))
        return placements

    def _build_table_placements(
        self,
        raw_tables: List[Dict[str, Any]],
        table_analyses: Dict[str, Any],
    ) -> List[TablePlacement]:
        """Build TablePlacement objects from LLM output + VLM analysis."""
        placements = []
        for raw in raw_tables:
            if not isinstance(raw, dict):
                continue
            tbl_id = raw.get("table_id", "")
            if not tbl_id:
                continue
            vlm = table_analyses.get(tbl_id)
            placements.append(TablePlacement(
                table_id=tbl_id,
                semantic_role=(
                    getattr(vlm, "semantic_role", "") if vlm
                    else raw.get("semantic_role", "")
                ),
                message=(
                    getattr(vlm, "message", "") if vlm
                    else raw.get("message", "")
                ),
                is_wide=(
                    getattr(vlm, "is_wide", False) if vlm
                    else raw.get("is_wide", False)
                ),
                position_hint=raw.get("position_hint", "mid"),
            ))
        return placements

    # =====================================================================
    # Figure/Table assignment
    # =====================================================================

    def _assign_figure_table_definitions(
        self,
        paper_plan: PaperPlan,
        request: PlanRequest,
        figure_analyses: Optional[Dict[str, Any]] = None,
        table_analyses: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Ensure each figure/table is DEFINED in exactly one section."""
        all_figures = {f.id: f for f in (request.figures or [])}
        all_tables = {t.id: t for t in (request.tables or [])}

        if not all_figures and not all_tables:
            return

        figures_defined = set()
        tables_defined = set()
        for section in paper_plan.sections:
            figures_defined.update(f.figure_id for f in section.figures)
            tables_defined.update(t.table_id for t in section.tables)

        fa = figure_analyses or {}
        ta = table_analyses or {}

        # Assign unassigned figures
        for fig_id, fig_info in all_figures.items():
            if self._should_be_wide_figure(fig_info):
                if fig_id not in paper_plan.wide_figures:
                    paper_plan.wide_figures.append(fig_id)

            if fig_id in figures_defined:
                continue

            target = self._find_best_section(
                paper_plan, fig_id, fig_info,
                fa.get(fig_id),
                {
                    "architecture": "method", "overview": "method",
                    "framework": "method", "model": "method",
                    "pipeline": "method", "result": "result",
                    "ablation": "result", "comparison": "experiment",
                    "performance": "experiment",
                },
                fallback="method",
            )
            if target:
                vlm_data = fa.get(fig_id)
                target.figures.append(FigurePlacement(
                    figure_id=fig_id,
                    semantic_role=getattr(vlm_data, "semantic_role", "") if vlm_data else "",
                    message=getattr(vlm_data, "message", "") if vlm_data else "",
                    is_wide=getattr(vlm_data, "is_wide", False) if vlm_data else self._should_be_wide_figure(fig_info),
                    position_hint="mid",
                    caption_guidance=getattr(vlm_data, "caption_guidance", "") if vlm_data else "",
                ))
                figures_defined.add(fig_id)

        # Assign unassigned tables
        for tbl_id, tbl_info in all_tables.items():
            if self._should_be_wide_table(tbl_info):
                if tbl_id not in paper_plan.wide_tables:
                    paper_plan.wide_tables.append(tbl_id)

            if tbl_id in tables_defined:
                continue

            target = self._find_best_section(
                paper_plan, tbl_id, tbl_info,
                ta.get(tbl_id),
                {
                    "main": "experiment", "result": "experiment",
                    "comparison": "experiment", "ablation": "result",
                    "hyperparameter": "experiment", "statistics": "experiment",
                    "dataset": "experiment",
                },
                fallback="experiment",
            )
            if target:
                vlm_data = ta.get(tbl_id)
                target.tables.append(TablePlacement(
                    table_id=tbl_id,
                    semantic_role=getattr(vlm_data, "semantic_role", "") if vlm_data else "",
                    message=getattr(vlm_data, "message", "") if vlm_data else "",
                    is_wide=getattr(vlm_data, "is_wide", False) if vlm_data else self._should_be_wide_table(tbl_info),
                    position_hint="mid",
                ))
                tables_defined.add(tbl_id)

    def _find_best_section(
        self,
        plan: PaperPlan,
        element_id: str,
        element_info: Any,
        vlm_analysis: Optional[Any],
        hint_map: Dict[str, str],
        fallback: str,
    ) -> Optional[SectionPlan]:
        """Find the best section for a figure/table using VLM then keyword fallback."""
        # Try VLM suggestion first
        if vlm_analysis:
            suggested = getattr(vlm_analysis, "suggested_section", "")
            if suggested:
                for section in plan.sections:
                    if section.section_type == suggested:
                        return section

        # User-suggested section
        user_section = getattr(element_info, "section", "")
        if user_section:
            for section in plan.sections:
                if section.section_type == user_section:
                    return section

        # Keyword heuristic fallback
        text = (
            (element_info.id if hasattr(element_info, "id") else "")
            + " " + (element_info.caption if hasattr(element_info, "caption") else "")
            + " " + (element_info.description if hasattr(element_info, "description") else "")
        ).lower()
        for hint, section_type in hint_map.items():
            if hint in text:
                for section in plan.sections:
                    if section.section_type == section_type:
                        return section

        # Fallback
        body = plan.get_body_sections()
        for section in body:
            if section.section_type == fallback:
                return section
        return body[0] if body else None

    # =====================================================================
    # Default plan (fallback)
    # =====================================================================

    def _create_default_plan(
        self, request: PlanRequest, total_words: int,
    ) -> PaperPlan:
        """Create a default plan when LLM fails."""
        total_sentences = total_words // WORDS_PER_SENTENCE
        n_sections = len(DEFAULT_EMPIRICAL_SECTIONS)
        per_section_sents = max(3, total_sentences // max(1, n_sections))

        sections = []
        for order, section_type in enumerate(DEFAULT_EMPIRICAL_SECTIONS):
            paragraphs = self._generate_default_paragraphs(
                section_type, per_section_sents, {},
            )
            sections.append(SectionPlan(
                section_type=section_type,
                section_title=self._get_section_title(section_type),
                paragraphs=paragraphs,
                content_sources=self._get_default_sources(section_type),
                depends_on=self._get_dependencies(section_type),
                order=order,
            ))

        plan = PaperPlan(
            title=request.title,
            paper_type=PaperType.EMPIRICAL,
            sections=sections,
            contributions=[f"We propose {request.title}"],
        )
        self._assign_figure_table_definitions(plan, request)
        return plan

    # =====================================================================
    # Helpers
    # =====================================================================

    @staticmethod
    def _extract_reference_keys(references: List[str]) -> List[str]:
        keys = []
        for ref in references:
            match = re.search(r"@\w+\{([^,]+)", ref)
            if match:
                keys.append(match.group(1).strip())
        return keys

    @staticmethod
    def _parse_plan_json(text: str) -> Dict[str, Any]:
        parsed = PlannerAgent._safe_load_json(text, expected=dict)
        if parsed is None:
            logger.warning("planner.json_parse_error, using defaults")
            return {}
        return parsed

    @staticmethod
    def _assign_figures_to_sections(
        figures: List[Any],
        section_order: List[Dict[str, str]],
    ) -> Dict[str, str]:
        """
        Assign each figure to exactly one section for definition.
        - **Description**:
            - Uses the figure's suggested section if it matches a planned section
            - Falls back to heuristic matching by section_type base name
            - Unmatched figures go to the first body section

        - **Args**:
            - `figures` (list): Figure specs with id, section, caption
            - `section_order` (list): Planned sections from Step 1

        - **Returns**:
            - `assignment` (Dict[str, str]): figure_id -> section_type
        """
        body_sections = [
            s["section_type"] for s in section_order
            if s["section_type"] not in ("abstract", "conclusion")
        ]
        first_body = body_sections[0] if body_sections else "introduction"

        # Build a lookup: base_type -> list of actual section_types
        # e.g. "result" -> ["result", "result_2", "result_3"]
        base_to_sections: Dict[str, List[str]] = {}
        for st in body_sections:
            base = re.sub(r'_\d+$', '', st)
            base_to_sections.setdefault(base, []).append(st)

        assignment: Dict[str, str] = {}
        section_fig_count: Dict[str, int] = {s: 0 for s in body_sections}

        for fig in figures:
            fig_id = fig.id
            suggested = getattr(fig, "section", None) or ""

            assigned = False
            # Try exact match with suggested section
            if suggested in body_sections:
                assignment[fig_id] = suggested
                section_fig_count[suggested] += 1
                assigned = True
            elif suggested:
                # Try base-name match
                suggested_base = re.sub(r'_\d+$', '', suggested.lower())
                candidates = base_to_sections.get(suggested_base, [])
                if candidates:
                    # Pick the candidate with fewest assigned figures
                    best = min(candidates, key=lambda s: section_fig_count[s])
                    assignment[fig_id] = best
                    section_fig_count[best] += 1
                    assigned = True

            if not assigned:
                # Assign to the body section with fewest figures
                best = min(body_sections, key=lambda s: section_fig_count[s])
                assignment[fig_id] = best
                section_fig_count[best] += 1

        logger.info(
            "planner.figure_assignment %s",
            {k: v for k, v in assignment.items()},
        )
        return assignment

    @staticmethod
    def _format_section_figure_info(
        figures: List[Any],
        analyses: Dict[str, Any],
        section_type: str,
        figure_assignment: Dict[str, str],
    ) -> str:
        """
        Format figure info for a specific section, distinguishing DEFINE vs REFERENCE.
        - **Description**:
            - Figures assigned to this section are marked "DEFINE HERE"
            - Other figures are listed as "REFERENCE ONLY (defined elsewhere)"

        - **Args**:
            - `figures` (list): All figure specs
            - `analyses` (dict): VLM analyses
            - `section_type` (str): Current section being planned
            - `figure_assignment` (dict): figure_id -> assigned section_type

        - **Returns**:
            - Formatted string for the prompt
        """
        if not figures:
            return "None provided"

        define_lines = []
        reference_lines = []

        for fig in figures:
            line = f"- {fig.id}: {fig.caption}"
            if fig.description:
                line += f" ({fig.description})"
            vlm = analyses.get(fig.id)
            if vlm:
                line += f" [VLM: role={getattr(vlm, 'semantic_role', '')}, message={getattr(vlm, 'message', '')}]"

            assigned_to = figure_assignment.get(fig.id)
            if assigned_to == section_type:
                define_lines.append(line)
            else:
                reference_lines.append(line)

        parts = []
        if define_lines:
            parts.append(
                "**DEFINE in this section** (include \\begin{figure}...\\end{figure}):\n"
                + "\n".join(define_lines)
            )
        if reference_lines:
            parts.append(
                "**REFERENCE ONLY** (use \\ref{fig:...}, do NOT create \\begin{figure}):\n"
                + "\n".join(reference_lines)
            )
        if not parts:
            return "None assigned to this section"
        return "\n\n".join(parts)

    @staticmethod
    def _format_figure_info(
        figures: List[Any], analyses: Dict[str, Any],
    ) -> str:
        if not figures:
            return "None provided"
        lines = []
        for fig in figures:
            line = f"- {fig.id}: {fig.caption}"
            if fig.description:
                line += f" ({fig.description})"
            if fig.section:
                line += f" [suggested: {fig.section}]"
            vlm = analyses.get(fig.id)
            if vlm:
                line += f" [VLM: role={getattr(vlm, 'semantic_role', '')}, message={getattr(vlm, 'message', '')}]"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _format_table_info(
        tables: List[Any], analyses: Dict[str, Any],
    ) -> str:
        if not tables:
            return "None provided"
        lines = []
        for tbl in tables:
            line = f"- {tbl.id}: {tbl.caption}"
            if tbl.description:
                line += f" ({tbl.description})"
            if tbl.section:
                line += f" [suggested: {tbl.section}]"
            vlm = analyses.get(tbl.id)
            if vlm:
                line += f" [VLM: role={getattr(vlm, 'semantic_role', '')}, message={getattr(vlm, 'message', '')}]"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _get_section_title(section_type: str) -> str:
        titles = {
            "abstract": "Abstract",
            "introduction": "Introduction",
            "related_work": "Related Work",
            "method": "Method",
            "experiment": "Experiments",
            "result": "Results",
            "discussion": "Discussion",
            "conclusion": "Conclusion",
        }
        return titles.get(section_type, section_type.replace("_", " ").title())

    @staticmethod
    def _get_default_sources(section_type: str) -> List[str]:
        mapping = {
            "introduction": ["idea_hypothesis", "method"],
            "related_work": ["idea_hypothesis"],
            "method": ["method"],
            "experiment": ["experiments", "data"],
            "result": ["experiments"],
            "discussion": ["experiments", "method"],
            "conclusion": ["idea_hypothesis", "experiments"],
            "abstract": ["idea_hypothesis", "method", "experiments"],
        }
        return mapping.get(section_type, [])

    @staticmethod
    def _get_dependencies(section_type: str) -> List[str]:
        deps = {
            "related_work": ["introduction"],
            "method": ["introduction"],
            "experiment": ["method"],
            "result": ["experiment"],
            "discussion": ["result"],
            "conclusion": ["introduction", "result"],
            "abstract": ["introduction", "conclusion"],
        }
        return deps.get(section_type, [])

    @staticmethod
    def _should_be_wide_figure(fig_info: Any) -> bool:
        if getattr(fig_info, "wide", False):
            return True
        wide_keywords = [
            "comparison", "overview", "architecture", "pipeline",
            "framework", "full", "complete", "main", "overall",
            "workflow", "system",
        ]
        text = (
            (getattr(fig_info, "id", "") or "")
            + " " + (getattr(fig_info, "caption", "") or "")
            + " " + (getattr(fig_info, "description", "") or "")
        ).lower()
        return any(kw in text for kw in wide_keywords)

    @staticmethod
    def _should_be_wide_table(tbl_info: Any) -> bool:
        if getattr(tbl_info, "wide", False):
            return True
        wide_keywords = [
            "main", "comparison", "full", "complete", "all",
            "overall", "summary", "comprehensive",
        ]
        text = (
            (getattr(tbl_info, "id", "") or "")
            + " " + (getattr(tbl_info, "caption", "") or "")
            + " " + (getattr(tbl_info, "description", "") or "")
        ).lower()
        if any(kw in text for kw in wide_keywords):
            return True
        content = getattr(tbl_info, "content", None)
        if content:
            first_line = content.strip().split("\n")[0] if content.strip() else ""
            for sep, adj in [
                ("|", -1), (",", 1), ("\t", 1),
            ]:
                if sep in first_line:
                    col_count = first_line.count(sep) + adj
                    if col_count > 5:
                        return True
        return False

    async def create_plan_from_metadata(
        self,
        title: str,
        idea_hypothesis: str,
        method: str,
        data: str,
        experiments: str,
        references: List[str],
        target_pages: Optional[int] = None,
        style_guide: Optional[str] = None,
    ) -> PaperPlan:
        """Convenience method to create plan from individual fields."""
        request = PlanRequest(
            title=title,
            idea_hypothesis=idea_hypothesis,
            method=method,
            data=data,
            experiments=experiments,
            references=references,
            target_pages=target_pages,
            style_guide=style_guide,
        )
        return await self.create_plan(request)
