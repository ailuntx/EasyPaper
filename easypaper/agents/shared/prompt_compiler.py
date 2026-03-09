"""
Prompt Compiler - Shared prompt generation utilities
- **Description**:
    - Compiles section plans into LLM prompts
    - Uses paragraph-level structure from PaperPlan
    - Provides section-specific prompt templates
    - Used by both Commander Agent and MetaData Agent
"""
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import json
import os
import re

if TYPE_CHECKING:
    from ...skills.models import WritingSkill


PROMPT_BUDGETS: Dict[str, int] = {
    "metadata_content_chars": 2800,
    "intro_context_chars": 1600,
    "memory_context_chars": 1400,
    "code_context_chars": 2200,
    "research_context_chars": 2200,
    "evidence_abstract_chars": 180,
    "refs_list_limit": 16,
    "evidence_keys_limit": 10,
    "table_latex_chars": 2200,
}


def _truncate_text(text: Optional[str], limit: int) -> str:
    """
    Truncate text to the nearest sentence/newline boundary.
    """
    if not text:
        return ""
    if len(text) <= limit:
        return text
    window = text[:limit]
    boundary = max(window.rfind("\n"), window.rfind(". "), window.rfind("; "))
    if boundary < int(limit * 0.6):
        boundary = limit
    clipped = window[:boundary].rstrip()
    return clipped + " ..."


def _format_code_guidance_block(code_context: Optional[str]) -> str:
    """
    Build a structured code-guidance block for writer prompts.
    """
    if not code_context:
        return ""
    trimmed = _truncate_text(code_context, PROMPT_BUDGETS["code_context_chars"])
    return (
        "## Repository-Derived Writing Guidance\n"
        "Use this as grounding evidence for implementation claims.\n\n"
        f"{trimmed}"
    )


def _normalize_reference_entry(ref: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize dict/object reference to a unified schema.
    """
    if ref is None:
        return None

    if isinstance(ref, dict):
        ref_id = (
            ref.get("ref_id")
            or ref.get("id")
            or ref.get("key")
            or ref.get("citation_key")
            or ""
        )
        if not ref_id:
            return None
        return {
            "id": str(ref_id).strip(),
            "title": str(ref.get("title", "")).strip(),
            "authors": str(ref.get("authors", "")).strip(),
            "year": ref.get("year"),
            "venue": str(ref.get("venue", "")).strip(),
            "abstract": str(ref.get("abstract", "")).strip(),
        }

    ref_id = getattr(ref, "ref_id", None) or getattr(ref, "id", None)
    if not ref_id:
        return None
    return {
        "id": str(ref_id).strip(),
        "title": str(getattr(ref, "title", "")).strip(),
        "authors": str(getattr(ref, "authors", "")).strip(),
        "year": getattr(ref, "year", None),
        "venue": str(getattr(ref, "venue", "")).strip(),
        "abstract": str(getattr(ref, "abstract", "")).strip(),
    }


def _build_reference_blocks(
    references: List[Any],
    assigned_refs: Optional[List[str]] = None,
    ref_limit: int = 16,
    evidence_limit: int = 10,
) -> Dict[str, Any]:
    """
    Build normalized citation blocks for prompt injection.
    """
    normalized: List[Dict[str, Any]] = []
    ref_lookup: Dict[str, Dict[str, Any]] = {}
    for ref in references or []:
        item = _normalize_reference_entry(ref)
        if not item:
            continue
        rid = item["id"]
        if rid in ref_lookup:
            continue
        normalized.append(item)
        ref_lookup[rid] = item

    limited = normalized[: max(1, ref_limit)]
    valid_keys = [r["id"] for r in limited]

    refs_info = []
    for r in limited:
        title = r.get("title", "")
        refs_info.append(
            f"- \\cite{{{r['id']}}}: {title[:90]}" if title else f"- \\cite{{{r['id']}}}"
        )

    assigned = [str(x).strip() for x in (assigned_refs or []) if str(x).strip()]
    coverage_keys = assigned[:evidence_limit] if assigned else valid_keys[:evidence_limit]
    missing_assigned = [k for k in assigned if k not in ref_lookup]

    evidence_lines = []
    for key in coverage_keys:
        ref = ref_lookup.get(key)
        if not ref:
            continue
        evidence_lines.append(f"- {key}: {ref.get('title', '')[:120]}")
        meta_bits = []
        if ref.get("year"):
            meta_bits.append(f"year={ref.get('year')}")
        if ref.get("venue"):
            meta_bits.append(f"venue={ref.get('venue', '')[:80]}")
        if meta_bits:
            evidence_lines.append(f"  - Meta: {', '.join(meta_bits)}")
        abstract = ref.get("abstract", "")
        if abstract:
            evidence_lines.append(
                "  - Abstract gist: "
                + _truncate_text(abstract, PROMPT_BUDGETS["evidence_abstract_chars"])
            )

    return {
        "valid_keys": valid_keys,
        "refs_info": refs_info,
        "evidence_lines": evidence_lines,
        "missing_assigned": missing_assigned,
    }


def _inject_skill_constraints(
    prompt_parts: list,
    active_skills: Optional[List["WritingSkill"]],
    section_type: str,
) -> None:
    """
    Inject writing-style constraints from active skills into prompt_parts (in-place).

    - **Args**:
        - `prompt_parts` (list): Mutable list of prompt segments to append to
        - `active_skills` (List[WritingSkill] | None): Skills loaded from the registry
        - `section_type` (str): Current section being written
    """
    if not active_skills:
        return
    matched = [
        s for s in active_skills
        if "*" in s.target_sections or section_type in s.target_sections
    ]
    matched.sort(key=lambda s: s.priority)
    if matched:
        constraints = "\n\n".join(
            s.system_prompt_append for s in matched if s.system_prompt_append
        )
        if constraints:
            prompt_parts.append(f"\n## Writing Style Constraints\n{constraints}")


# =============================================================================
# Section Prompt Templates
# =============================================================================

SECTION_PROMPTS: Dict[str, str] = {
    "abstract": """You are writing the Abstract section of a research paper.
The abstract should:
- Summarize the research problem and motivation (1-2 sentences)
- Describe the methodology briefly (1-2 sentences)
- Present key results and findings (1-2 sentences)
- State the main conclusions and implications (1-2 sentences)
Keep it concise, typically 150-250 words.""",

    "introduction": """You are writing the Introduction section of a research paper.
The introduction should:
- Establish the research context and background
- Identify the problem or gap in current knowledge
- State the research objectives and contributions
- Outline the paper structure
Use a clear narrative flow from general to specific.""",

    "related_work": """You are writing the Related Work section of a research paper.
This section should:
- Survey relevant prior work systematically
- Group related works by theme or approach
- Identify gaps that your work addresses
- Clearly differentiate your contribution from existing work
Use proper citations throughout.""",

    "method": """You are writing the Method/Methodology section of a research paper.
This section should:
- Describe your approach in sufficient detail for reproduction
- Explain the rationale behind methodological choices
- Include formal definitions, algorithms, or models as needed
- Use clear notation and terminology consistently.""",

    "experiment": """You are writing the Experiment section of a research paper.
This section should:
- Describe the experimental setup and configuration
- Specify datasets, metrics, and baselines used
- Explain evaluation protocols and procedures
- Provide implementation details as necessary.""",

    "result": """You are writing the Results section of a research paper.
This section should:
- Present experimental results clearly and objectively
- Use tables and figures to support key findings
- Compare against baselines and prior work
- Highlight statistically significant results.""",

    "discussion": """You are writing the Discussion section of a research paper.
This section should:
- Interpret the results in context of research questions
- Discuss implications and significance of findings
- Address limitations and potential threats to validity
- Suggest directions for future work.""",

    "conclusion": """You are writing the Conclusion section of a research paper.
This section should:
- Summarize the main contributions concisely
- Restate key findings and their significance
- Discuss broader impact and applications
- End with forward-looking perspective.""",
}


# =============================================================================
# Paragraph-level writing structure
# =============================================================================

def _format_paragraph_guidance(section_plan: Any) -> str:
    """
    Format paragraph-level writing guidance from a SectionPlan.

    - **Args**:
        - `section_plan`: SectionPlan object with paragraphs list

    - **Returns**:
        - `str`: Formatted paragraph guidance for the LLM prompt
    """
    paragraphs = getattr(section_plan, "paragraphs", None)
    if not paragraphs:
        # Backward-compat: fall back to key_points + target_words
        parts = []
        key_points = getattr(section_plan, "key_points", None)
        if callable(key_points):
            key_points = None
        if key_points:
            points_str = "\n".join(f"- {p}" for p in key_points)
            parts.append(f"**Key Points to Cover**:\n{points_str}")
        refs = getattr(section_plan, "references_to_cite", None)
        if callable(refs):
            refs = None
        if refs:
            parts.append(f"**References to Cite**: {', '.join(refs)}")
        guidance = getattr(section_plan, "writing_guidance", "")
        if guidance:
            parts.append(f"**Writing Guidance**: {guidance}")
        return "\n".join(parts) if parts else ""

    n = len(paragraphs)
    total_sentences = sum(getattr(p, "approx_sentences", 5) for p in paragraphs)

    lines = [f"Write this section with **{n} paragraphs** (~{total_sentences} sentences total):\n"]

    for i, para in enumerate(paragraphs, 1):
        role = getattr(para, "role", "evidence")
        sents = getattr(para, "approx_sentences", 5)
        kp = getattr(para, "key_point", "")
        supporting = getattr(para, "supporting_points", [])
        refs = getattr(para, "references_to_cite", [])
        fig_refs = getattr(para, "figures_to_reference", [])
        tbl_refs = getattr(para, "tables_to_reference", [])

        lines.append(f"**Paragraph {i}** (role: {role}, ~{sents} sentences):")
        if kp:
            lines.append(f"  - Key point: {kp}")
        for sp in supporting:
            lines.append(f"  - Supporting: {sp}")
        if refs:
            lines.append(f"  - Cite: {', '.join(refs)}")
        if fig_refs:
            lines.append(f"  - Reference figures: {', '.join(fig_refs)}")
        if tbl_refs:
            lines.append(f"  - Reference tables: {', '.join(tbl_refs)}")
        lines.append("")

    guidance = getattr(section_plan, "writing_guidance", "")
    if guidance:
        lines.append(f"**Writing Guidance**: {guidance}")

    return "\n".join(lines)


def _format_structure_quality_contract(section_type: str, section_plan: Any) -> str:
    """
    Build a soft structure contract for writer quality control.

    - **Description**:
        - Encourages clear thematic block organization without forcing fixed subsection titles.
        - Allows explicit (`\\subsection`) or implicit (strong block transitions) structure.
    """
    if not section_plan:
        return ""

    paragraphs = getattr(section_plan, "paragraphs", None) or []
    paragraph_count = len(paragraphs)
    topic_clusters = getattr(section_plan, "topic_clusters", None) or []
    transition_intents = getattr(section_plan, "transition_intents", None) or []
    sectioning_recommended = bool(
        getattr(section_plan, "sectioning_recommended", False)
    )

    # For short sections, avoid over-constraining structure.
    if paragraph_count < 3 and not sectioning_recommended:
        return ""

    lines: List[str] = ["## Structure Quality Contract"]
    lines.append(
        "- Organize this section into clear thematic blocks with explicit transitions."
    )
    lines.append(
        "- Every major claim cluster must map to at least one paragraph block."
    )

    if topic_clusters:
        lines.append("- Suggested thematic blocks:")
        for cluster in topic_clusters[:4]:
            lines.append(f"  - {cluster}")

    if transition_intents:
        lines.append("- Suggested transitions:")
        for intent in transition_intents[:3]:
            lines.append(f"  - {intent}")

    if sectioning_recommended:
        lines.append(
            "- This section is structurally dense; explicit `\\subsection{}` headings are recommended."
        )
        lines.append(
            "- Ensure block boundaries are unmistakable with clear transition language."
        )
    else:
        lines.append(
            "- **DO NOT use `\\subsection{}` commands in this section.**"
        )
        lines.append(
            "- Use continuous prose with strong paragraph-level transitions instead."
        )
        lines.append(
            "- Organize content through thematic paragraph blocks, topic sentences, "
            "and transition phrases — not through explicit heading commands."
        )
        if section_type in {"introduction", "discussion"}:
            lines.append(
                "- Introduction and Discussion sections in top venues use flowing "
                "narrative prose without subsection headings. This is mandatory."
            )

    if section_type in {"abstract", "conclusion"}:
        lines.append(
            "- Keep synthesis sections compact; avoid unnecessary explicit subsection commands."
        )

    return "\n".join(lines)


def _format_figure_placement_guidance(section_plan: Any, figures: List[Any]) -> str:
    """
    Format figure placement guidance using FigurePlacement semantics.

    - **Args**:
        - `section_plan`: SectionPlan with figures (FigurePlacement list)
        - `figures`: Available FigureSpec objects

    - **Returns**:
        - `str`: Formatted figure guidance for the prompt
    """
    placements = getattr(section_plan, "figures", None)
    if not placements:
        return ""

    figure_map = {}
    for fig in (figures or []):
        fig_id = fig.id if hasattr(fig, "id") else fig.get("id", "")
        if fig_id:
            figure_map[fig_id] = fig

    figures_to_reference = set(getattr(section_plan, "figures_to_reference", []) or [])
    parts = ["\n## Figures to DEFINE in this section"]
    parts.append("**CREATE the complete figure environment for each figure below.**\n")
    overlaps = []

    for fp in placements:
        fig_id = fp.figure_id
        if fig_id in figures_to_reference:
            overlaps.append(fig_id)
            continue
        fig = figure_map.get(fig_id)
        if not fig:
            continue

        caption = fig.caption if hasattr(fig, "caption") else fig.get("caption", "")
        desc = fig.description if hasattr(fig, "description") else fig.get("description", "")
        file_path = fig.file_path if hasattr(fig, "file_path") else fig.get("file_path", "")
        wide = fp.is_wide

        filename = os.path.basename(file_path) if file_path else f"{fig_id.replace('fig:', '')}.pdf"
        env_name = "figure*" if wide else "figure"
        width = "\\\\textwidth" if wide else "0.9\\\\linewidth"

        parts.append(f"- **{fig_id}**: {caption}")
        if desc:
            parts.append(f"  Description: {desc}")
        if fp.message:
            parts.append(f"  Message: {fp.message}")
        if fp.caption_guidance:
            parts.append(f"  Caption guidance: {fp.caption_guidance}")
        if wide:
            parts.append(f"  **Note: WIDE figure - use {env_name} to span both columns.**")
        parts.append(f"  Position: {fp.position_hint} in the section")
        parts.append(f"  **Required LaTeX:**")
        parts.append(f"  ```latex")
        parts.append(f"  \\\\begin{{{env_name}}}[htbp]")
        parts.append(f"  \\\\centering")
        parts.append(f"  \\\\includegraphics[width={width}]{{figures/{filename}}}")
        parts.append(f"  \\\\caption{{{caption}}}\\\\label{{{fig_id}}}")
        parts.append(f"  \\\\end{{{env_name}}}")
        parts.append(f"  ```\n")

    if overlaps:
        parts.append(
            "Conflict notice: these figure IDs were marked as both define and reference; "
            "treat them as REFERENCE-only to avoid duplicate definitions: "
            + ", ".join(overlaps)
        )
    return "\n".join(parts)


def _format_table_placement_guidance(
    section_plan: Any,
    tables: List[Any],
    converted_tables: Optional[Dict[str, str]] = None,
) -> str:
    """Format table placement guidance using TablePlacement semantics."""
    placements = getattr(section_plan, "tables", None)
    if not placements:
        return ""

    table_map = {}
    for tbl in (tables or []):
        tbl_id = tbl.id if hasattr(tbl, "id") else tbl.get("id", "")
        if tbl_id:
            table_map[tbl_id] = tbl

    _converted = converted_tables or {}
    tables_to_reference = set(getattr(section_plan, "tables_to_reference", []) or [])
    parts = ["\n## Tables to DEFINE in this section"]
    parts.append("**Include the complete table environment for each table below.**\n")
    overlaps = []

    for tp in placements:
        tbl_id = tp.table_id
        if tbl_id in tables_to_reference:
            overlaps.append(tbl_id)
            continue
        tbl = table_map.get(tbl_id)
        if not tbl:
            continue

        caption = tbl.caption if hasattr(tbl, "caption") else tbl.get("caption", "")
        desc = tbl.description if hasattr(tbl, "description") else tbl.get("description", "")
        wide = tp.is_wide
        env_name = "table*" if wide else "table"

        parts.append(f"- **{tbl_id}**: {caption}")
        if desc:
            parts.append(f"  Description: {desc}")
        if tp.message:
            parts.append(f"  Message: {tp.message}")
        if wide:
            parts.append(f"  **Note: WIDE table - use {env_name} to span both columns.**")
        parts.append(f"  Position: {tp.position_hint} in the section")

        if tbl_id in _converted:
            parts.append(f"  **Required LaTeX (include this exact table):**")
            parts.append(f"  ```latex")
            rendered = _converted[tbl_id]
            if len(rendered) > PROMPT_BUDGETS["table_latex_chars"]:
                rendered = _truncate_text(rendered, PROMPT_BUDGETS["table_latex_chars"])
                parts.append("  % Table LaTeX truncated for prompt budget; preserve label/caption semantics.")
            parts.append(f"  {rendered}")
            parts.append(f"  ```\n")
        else:
            content = tbl.content if hasattr(tbl, "content") else tbl.get("content", "")
            if content:
                parts.append(f"  Data:\n  {content[:500]}")
            parts.append(f"  **Required: Create \\\\begin{{{env_name}}}...\\\\end{{{env_name}}} with \\\\label{{{tbl_id}}}**\n")

    if overlaps:
        parts.append(
            "Conflict notice: these table IDs were marked as both define and reference; "
            "treat them as REFERENCE-only to avoid duplicate definitions: "
            + ", ".join(overlaps)
        )
    return "\n".join(parts)


# =============================================================================
# Prompt Compilation Functions
# =============================================================================

def compile_section_prompt(
    section_type: str,
    thesis: str = "",
    content_points: List[str] = None,
    references: List[Any] = None,
    figures: List[Any] = None,
    tables: List[Any] = None,
    word_limit: Optional[int] = None,
    style_guide: Optional[str] = None,
    intro_context: Optional[str] = None,
    active_skills: Optional[List["WritingSkill"]] = None,
) -> str:
    """
    Compile a prompt for section generation (generic fallback).

    - **Args**:
        - `section_type` (str): Type of section
        - `thesis` (str): Core thesis/theme
        - `content_points` (List[str]): Key points to express
        - `references` (List[Any]): Available references
        - `figures` (List[Any]): Available figures
        - `tables` (List[Any]): Available tables
        - `word_limit` (Optional[int]): Word limit
        - `style_guide` (Optional[str]): Target venue style
        - `intro_context` (Optional[str]): Introduction content for context
        - `active_skills` (Optional[List[WritingSkill]]): Active writing skills

    - **Returns**:
        - `str`: Compiled prompt string for LLM
    """
    content_points = content_points or []
    references = references or []
    figures = figures or []
    tables = tables or []

    base_prompt = SECTION_PROMPTS.get(section_type, SECTION_PROMPTS.get("method", ""))
    prompt_parts = [base_prompt]

    if thesis:
        prompt_parts.append(f"\n## Core Theme\n{thesis}")

    if content_points:
        points_str = "\n".join(f"- {p}" for p in content_points)
        prompt_parts.append(f"\n## Key Points to Address\n{points_str}")

    if intro_context and section_type not in ["introduction", "abstract"]:
        context = intro_context[:1500] + "..." if len(intro_context) > 1500 else intro_context
        prompt_parts.append(f"\n## Paper Introduction (for context)\n{context}")

    if references:
        refs_info = []
        for ref in references[:20]:
            if hasattr(ref, "ref_id"):
                ref_str = f"- [{ref.ref_id}]"
                if hasattr(ref, "title") and ref.title:
                    ref_str += f": {ref.title}"
                if hasattr(ref, "authors") and ref.authors:
                    ref_str += f" ({ref.authors})"
                refs_info.append(ref_str)
            elif isinstance(ref, dict):
                ref_id = ref.get("ref_id", ref.get("id", "unknown"))
                ref_str = f"- [{ref_id}]"
                if ref.get("title"):
                    ref_str += f": {ref.get('title')}"
                refs_info.append(ref_str)
        if refs_info:
            prompt_parts.append(f"\n## Available References\n" + "\n".join(refs_info))

    if figures:
        figs_info = []
        for fig in figures:
            if hasattr(fig, "figure_id"):
                figs_info.append(f"- {fig.figure_id}")
            elif hasattr(fig, "id"):
                fig_str = f"- {fig.id}"
                if hasattr(fig, "caption") and fig.caption:
                    fig_str += f": {fig.caption}"
                figs_info.append(fig_str)
            elif isinstance(fig, dict):
                fig_id = fig.get("figure_id", fig.get("id", "unknown"))
                figs_info.append(f"- {fig_id}")
        if figs_info:
            prompt_parts.append(f"\n## Available Figures\n" + "\n".join(figs_info))

    if tables:
        tables_info = []
        for tbl in tables:
            if hasattr(tbl, "table_id"):
                tables_info.append(f"- {tbl.table_id}")
            elif hasattr(tbl, "id"):
                tbl_str = f"- {tbl.id}"
                if hasattr(tbl, "caption") and tbl.caption:
                    tbl_str += f": {tbl.caption}"
                tables_info.append(tbl_str)
            elif isinstance(tbl, dict):
                tbl_id = tbl.get("table_id", tbl.get("id", "unknown"))
                tables_info.append(f"- {tbl_id}")
        if tables_info:
            prompt_parts.append(f"\n## Available Tables\n" + "\n".join(tables_info))

    constraints = []
    if style_guide:
        constraints.append(f"- Style guide: {style_guide}")
    if constraints:
        prompt_parts.append(f"\n## Constraints\n" + "\n".join(constraints))

    _inject_skill_constraints(prompt_parts, active_skills, section_type)

    prompt_parts.append("""
## Output Instructions
- Generate LaTeX content for the section body only
- Do NOT include \\section{} command - just the content
- Use \\cite{key} for citations
- Use \\ref{fig:id} for figure references
- Use \\ref{tab:id} for table references
- Write in academic English with clear, precise language
""")

    return "\n".join(prompt_parts)


def compile_introduction_prompt(
    paper_title: str,
    idea_hypothesis: str,
    method_summary: str,
    data_summary: str,
    experiments_summary: str,
    references: List[Any] = None,
    style_guide: Optional[str] = None,
    section_plan: Any = None,
    figures: List[Any] = None,
    tables: List[Any] = None,
    active_skills: Optional[List["WritingSkill"]] = None,
    code_context: Optional[str] = None,
    research_context: Optional[str] = None,
    enable_structure_contract: bool = True,
) -> str:
    """
    Compile prompt for Introduction generation (Phase 1 - Leader section).

    - **Args**:
        - `section_plan`: SectionPlan with paragraph-level structure
        - `figures`: Available FigureSpec list
        - `tables`: Available TableSpec list
    """
    references = references or []
    figures = figures or []
    tables = tables or []

    prompt = f"""You are writing the Introduction section for a research paper titled: "{paper_title}"

## Role of Introduction
The Introduction is the LEADER section that:
1. Establishes the research context and motivation
2. Identifies the problem or gap being addressed
3. States the key contributions (typically 3-4 bullet points)
4. Outlines the paper structure

## Research Content

### Idea/Hypothesis
{idea_hypothesis}

### Method Overview
{method_summary}

### Data/Validation
{data_summary}

### Experiments Overview
{experiments_summary}
"""

    # Paragraph-level planning guidance
    if section_plan:
        guidance = _format_paragraph_guidance(section_plan)
        if guidance:
            prompt += f"\n## Writing Structure\n{guidance}\n"
        if enable_structure_contract:
            structure_contract = _format_structure_quality_contract("introduction", section_plan)
            if structure_contract:
                prompt += f"\n{structure_contract}\n"

    # References with citation rules
    if references:
        assigned_refs = getattr(section_plan, "assigned_refs", []) if section_plan else []
        ref_blocks = _build_reference_blocks(
            references=references,
            assigned_refs=assigned_refs,
            ref_limit=PROMPT_BUDGETS["refs_list_limit"],
            evidence_limit=PROMPT_BUDGETS["evidence_keys_limit"],
        )
        refs_info = ref_blocks["refs_info"]
        valid_keys = ref_blocks["valid_keys"]
        if refs_info and valid_keys:
            prompt += f"\n### CRITICAL: Citation Rules\n"
            prompt += f"**ONLY use these citation keys. DO NOT invent or hallucinate citations.**\n"
            prompt += f"**Valid keys**: {', '.join(valid_keys)}\n\n"
            prompt += "Available references:\n" + "\n".join(refs_info)
            prompt += "\n\n**WARNING**: Any citation not in the above list will be automatically removed.\n"
            if assigned_refs:
                prompt += (
                    "\n\n**Coverage priority for this section**:\n"
                    "Prioritize integrating these assigned citation keys in this section where relevant:\n"
                    + ", ".join(assigned_refs[:8])
                    + "\nDo not force unrelated citations; integrate naturally with matching claims.\n"
                )

            evidence_lines = ref_blocks["evidence_lines"]
            if evidence_lines:
                prompt += (
                    "\n\n## Reference Evidence Map\n"
                    "Use these key-to-paper mappings when selecting citations:\n"
                    + "\n".join(evidence_lines)
                    + "\n"
                )
            missing_assigned = ref_blocks["missing_assigned"]
            if missing_assigned:
                prompt += (
                    "\nUnavailable assigned refs (do NOT invent): "
                    + ", ".join(missing_assigned[:8])
                    + "\n"
                )
            citation_budget = getattr(section_plan, "citation_budget", {}) if section_plan else {}
            budget_selected_refs = getattr(section_plan, "budget_selected_refs", []) if section_plan else []
            budget_reserve_refs = getattr(section_plan, "budget_reserve_refs", []) if section_plan else []
            if citation_budget and citation_budget.get("enabled"):
                prompt += (
                    "\n\n## Citation Budget Guidance\n"
                    f"- Target refs for this section: {citation_budget.get('target_refs', 0)} "
                    f"(min={citation_budget.get('min_refs', 0)}, max={citation_budget.get('max_refs', 0)})\n"
                )
                if budget_selected_refs:
                    prompt += (
                        "- Budget-selected keys (use these first):\n"
                        + ", ".join(budget_selected_refs[:10])
                        + "\n"
                    )
                if budget_reserve_refs:
                    prompt += (
                        "- Reserve keys (only if needed for strong claim support):\n"
                        + ", ".join(budget_reserve_refs[:8])
                        + "\n"
                    )

    # Figure placement guidance
    if section_plan:
        fig_guidance = _format_figure_placement_guidance(section_plan, figures)
        if fig_guidance:
            prompt += fig_guidance

        # Cross-section figure references
        figs_to_ref = getattr(section_plan, "figures_to_reference", [])
        if figs_to_ref:
            prompt += f"\n## Figures to REFERENCE (already defined elsewhere)\n"
            prompt += "**DO NOT create \\\\begin{{figure}} - just reference with Figure~\\\\ref{{fig:id}}.**\n"
            for fig_id in figs_to_ref:
                prompt += f"- {fig_id}\n"
    elif figures:
        figs_info = []
        for fig in figures:
            fig_id = fig.id if hasattr(fig, "id") else fig.get("id", "")
            caption = fig.caption if hasattr(fig, "caption") else fig.get("caption", "")
            if fig_id:
                figs_info.append(f"- \\ref{{{fig_id}}}: {caption}")
        if figs_info:
            prompt += f"\n### Available Figures\n" + "\n".join(figs_info)

    # Table guidance
    if section_plan:
        tbl_guidance = _format_table_placement_guidance(section_plan, tables)
        if tbl_guidance:
            prompt += tbl_guidance
    elif tables:
        tables_info = []
        for tbl in tables:
            tbl_id = tbl.id if hasattr(tbl, "id") else tbl.get("id", "")
            caption = tbl.caption if hasattr(tbl, "caption") else tbl.get("caption", "")
            if tbl_id:
                tables_info.append(f"- \\ref{{{tbl_id}}}: {caption}")
        if tables_info:
            prompt += f"\n### Available Tables\n" + "\n".join(tables_info)

    if code_context:
        prompt += f"\n\n{_format_code_guidance_block(code_context)}"

    if research_context:
        prompt += f"\n\n{_truncate_text(research_context, PROMPT_BUDGETS['research_context_chars'])}"

    if style_guide:
        prompt += f"\n\n## Target Venue: {style_guide}"

    if active_skills:
        intro_parts: list = []
        _inject_skill_constraints(intro_parts, active_skills, "introduction")
        if intro_parts:
            prompt += "\n" + "\n".join(intro_parts)

    prompt += """

## Output Requirements
1. Generate LaTeX content for the Introduction section body
2. Do NOT include \\section{Introduction} - just the content
3. Structure the introduction with clear paragraphs as specified above
4. Use \\cite{key} for citations
5. Use \\ref{fig:id} for figure references and \\ref{tab:id} for table references
6. Write in formal academic English

## Subsection Policy
- Unless the plan explicitly recommends sectioning for Introduction, prefer implicit structure.
- Do NOT add multiple \\subsection{} blocks by default.
- Do NOT create a single placeholder subsection mirroring the parent title (e.g., \\subsection{Introduction}).

## Important
At the end, clearly state the contributions using:
\\begin{itemize}
\\item Contribution 1...
\\item Contribution 2...
\\end{itemize}

This helps maintain consistency across the paper.
"""

    return prompt


def compile_body_section_prompt(
    section_type: str,
    metadata_content: str,
    intro_context: str,
    contributions: List[str] = None,
    references: List[Any] = None,
    style_guide: Optional[str] = None,
    section_plan: Any = None,
    figures: List[Any] = None,
    tables: List[Any] = None,
    converted_tables: Optional[Dict[str, str]] = None,
    active_skills: Optional[List["WritingSkill"]] = None,
    memory_context: Optional[str] = None,
    code_context: Optional[str] = None,
    research_context: Optional[str] = None,
    enable_structure_contract: bool = True,
) -> str:
    """
    Compile prompt for Body section generation (Phase 2).

    - **Args**:
        - `section_plan`: SectionPlan with paragraph-level structure and FigurePlacement
        - `figures`: Available FigureSpec list
        - `tables`: Available TableSpec list
        - `converted_tables`: table_id -> LaTeX code mapping
        - `memory_context` (str, optional): Cross-section context from SessionMemory
    """
    contributions = contributions or []
    references = references or []
    figures = figures or []
    tables = tables or []

    base_prompt = SECTION_PROMPTS.get(section_type, "")

    prompt = f"""{base_prompt}

## Section Content Source
{_truncate_text(metadata_content, PROMPT_BUDGETS["metadata_content_chars"])}

## Introduction Context (maintain consistency)
{_truncate_text(intro_context, PROMPT_BUDGETS["intro_context_chars"])}

## Key Contributions to Support
"""
    for i, contrib in enumerate(contributions, 1):
        prompt += f"{i}. {contrib}\n"

    # Memory-provided cross-section coordination context
    if memory_context:
        prompt += (
            "\n## Coordination Context (from Session Memory)\n"
            + _truncate_text(memory_context, PROMPT_BUDGETS["memory_context_chars"])
            + "\n"
        )

    if code_context:
        prompt += f"\n{_format_code_guidance_block(code_context)}\n"

    if research_context:
        prompt += f"\n{_truncate_text(research_context, PROMPT_BUDGETS['research_context_chars'])}\n"

    # Paragraph-level planning guidance
    if section_plan:
        guidance = _format_paragraph_guidance(section_plan)
        if guidance:
            prompt += f"\n## Writing Structure\n{guidance}\n"
        if enable_structure_contract:
            structure_contract = _format_structure_quality_contract(section_type, section_plan)
            if structure_contract:
                prompt += f"\n{structure_contract}\n"

    # References
    if references:
        assigned_refs = getattr(section_plan, "assigned_refs", []) if section_plan else []
        ref_blocks = _build_reference_blocks(
            references=references,
            assigned_refs=assigned_refs,
            ref_limit=PROMPT_BUDGETS["refs_list_limit"],
            evidence_limit=PROMPT_BUDGETS["evidence_keys_limit"],
        )
        refs_info = ref_blocks["refs_info"]
        valid_keys = ref_blocks["valid_keys"]
        if refs_info and valid_keys:
            prompt += f"\n## CRITICAL: Citation Rules\n"
            prompt += f"**ONLY use these citation keys. DO NOT invent citations.**\n"
            prompt += f"**Valid keys**: {', '.join(valid_keys)}\n\n"
            prompt += "\n".join(refs_info)
            if assigned_refs:
                prompt += (
                    "\n\n## Citation Coverage Priority\n"
                    "To improve reference coverage, prioritize citing these assigned keys in this section when relevant:\n"
                    + ", ".join(assigned_refs[:10])
                    + "\nUse each key only if it supports an actual statement in the text.\n"
                )
            evidence_lines = ref_blocks["evidence_lines"]
            if evidence_lines:
                prompt += (
                    "\n\n## Reference Evidence Map\n"
                    "Use these key-to-paper mappings when selecting citations:\n"
                    + "\n".join(evidence_lines)
                    + "\n"
                )
            missing_assigned = ref_blocks["missing_assigned"]
            if missing_assigned:
                prompt += (
                    "\nUnavailable assigned refs (do NOT invent): "
                    + ", ".join(missing_assigned[:10])
                    + "\n"
                )
            citation_budget = getattr(section_plan, "citation_budget", {}) if section_plan else {}
            budget_selected_refs = getattr(section_plan, "budget_selected_refs", []) if section_plan else []
            budget_reserve_refs = getattr(section_plan, "budget_reserve_refs", []) if section_plan else []
            if citation_budget and citation_budget.get("enabled"):
                prompt += (
                    "\n\n## Citation Budget Guidance\n"
                    f"- Target refs for this section: {citation_budget.get('target_refs', 0)} "
                    f"(min={citation_budget.get('min_refs', 0)}, max={citation_budget.get('max_refs', 0)})\n"
                )
                if budget_selected_refs:
                    prompt += (
                        "- Budget-selected keys (use these first):\n"
                        + ", ".join(budget_selected_refs[:12])
                        + "\n"
                    )
                if budget_reserve_refs:
                    prompt += (
                        "- Reserve keys (only if needed for strong claim support):\n"
                        + ", ".join(budget_reserve_refs[:10])
                        + "\n"
                    )

    # Figure placement guidance (using FigurePlacement semantics)
    if section_plan:
        fig_guidance = _format_figure_placement_guidance(section_plan, figures)
        if fig_guidance:
            prompt += fig_guidance

        figs_to_ref = getattr(section_plan, "figures_to_reference", [])
        if figs_to_ref:
            prompt += f"\n## Figures to REFERENCE (already defined elsewhere)\n"
            prompt += "**DO NOT create \\\\begin{{figure}} for these - just reference them with Figure~\\\\ref{{fig:id}}.**\n"
            figure_map = {}
            for fig in figures:
                fid = fig.id if hasattr(fig, "id") else fig.get("id", "")
                if fid:
                    figure_map[fid] = fig
            for fig_id in figs_to_ref:
                fig = figure_map.get(fig_id)
                caption = ""
                if fig:
                    caption = fig.caption if hasattr(fig, "caption") else fig.get("caption", "")
                prompt += f"- {fig_id}: {caption} -> use `Figure~\\\\ref{{{fig_id}}}`\n"

        # Table placement guidance
        tbl_guidance = _format_table_placement_guidance(section_plan, tables, converted_tables)
        if tbl_guidance:
            prompt += tbl_guidance

        tbls_to_ref = getattr(section_plan, "tables_to_reference", [])
        if tbls_to_ref:
            prompt += f"\n## Tables to REFERENCE (already defined elsewhere)\n"
            prompt += "**DO NOT create \\\\begin{{table}} for these - just reference them with Table~\\\\ref{{tab:id}}.**\n"
            table_map = {}
            for tbl in tables:
                tid = tbl.id if hasattr(tbl, "id") else tbl.get("id", "")
                if tid:
                    table_map[tid] = tbl
            for tbl_id in tbls_to_ref:
                tbl = table_map.get(tbl_id)
                caption = ""
                if tbl:
                    caption = tbl.caption if hasattr(tbl, "caption") else tbl.get("caption", "")
                prompt += f"- {tbl_id}: {caption} -> use `Table~\\\\ref{{{tbl_id}}}`\n"

    else:
        # Legacy fallback: no section_plan, show all figures/tables as available
        if figures:
            figs_info = []
            for fig in figures:
                fig_id = fig.id if hasattr(fig, "id") else fig.get("id", "")
                caption = fig.caption if hasattr(fig, "caption") else fig.get("caption", "")
                if fig_id:
                    figs_info.append(f"- {fig_id}: {caption}")
            if figs_info:
                prompt += f"\n## Available Figures (reference only with \\\\ref{{}})\n" + "\n".join(figs_info)

        if tables:
            tables_info = []
            for tbl in tables:
                tbl_id = tbl.id if hasattr(tbl, "id") else tbl.get("id", "")
                caption = tbl.caption if hasattr(tbl, "caption") else tbl.get("caption", "")
                if tbl_id:
                    tables_info.append(f"- {tbl_id}: {caption}")
            if tables_info:
                prompt += f"\n## Available Tables (reference only with \\\\ref{{}})\n" + "\n".join(tables_info)

    if style_guide:
        prompt += f"\n\n## Target Venue: {style_guide}"

    if active_skills:
        body_parts: list = []
        _inject_skill_constraints(body_parts, active_skills, section_type)
        if body_parts:
            prompt += "\n" + "\n".join(body_parts)

    prompt += """

## Output Requirements
1. Generate LaTeX content for the section body only
2. Do NOT include \\section{} command
3. Follow the paragraph structure specified above
4. Maintain consistency with the Introduction's framing
5. Support the stated contributions where relevant
6. Use \\cite{key} for citations
7. Use \\ref{fig:id} for figure references and \\ref{tab:id} for table references
8. Use clear academic writing style

## Subsection Policy
- Use explicit \\subsection{} headings only when section-level structure signals recommend it or when block separation would otherwise be unclear.
- Avoid boilerplate subsection proliferation in every section.
- If only one subsection would be created, do not use subsection; keep a clean paragraph-only structure.
- Never use a subsection title identical to the parent section title.
"""

    return prompt


def compile_synthesis_prompt(
    section_type: str,
    paper_title: str,
    prior_sections: Dict[str, str],
    key_contributions: List[str] = None,
    word_limit: Optional[int] = None,
    style_guide: Optional[str] = None,
    section_plan: Any = None,
    active_skills: Optional[List["WritingSkill"]] = None,
    memory_context: Optional[str] = None,
) -> str:
    """
    Compile prompt for Synthesis sections (Abstract/Conclusion - Phase 3).

    - **Args**:
        - `section_plan`: SectionPlan with paragraph-level structure
        - `memory_context` (str, optional): Cross-section summary from SessionMemory
    """
    key_contributions = key_contributions or []

    # Extract plan guidance
    plan_guidance = ""
    plan_writing_guidance = ""
    if section_plan:
        plan_guidance = _format_paragraph_guidance(section_plan)
        plan_writing_guidance = getattr(section_plan, "writing_guidance", "")

    if section_type == "abstract":
        prompt = f"""You are writing the Abstract for a research paper titled: "{paper_title}"

## Task
Synthesize a concise abstract (150-250 words) from the following paper sections.

## Introduction
{prior_sections.get('introduction', '')[:1500]}{"..." if len(prior_sections.get('introduction', '')) > 1500 else ""}

## Method (summary)
{prior_sections.get('method', '')[:800]}{"..." if len(prior_sections.get('method', '')) > 800 else ""}

## Key Results
{prior_sections.get('result', prior_sections.get('experiment', ''))[:800]}{"..." if len(prior_sections.get('result', prior_sections.get('experiment', ''))) > 800 else ""}

## Key Contributions
"""
        for contrib in key_contributions:
            prompt += f"- {contrib}\n"

        if plan_guidance:
            prompt += f"\n## Writing Structure (from Planner)\n{plan_guidance}\n"

        prompt += """
## Abstract Structure
1. Problem/Motivation (1-2 sentences)
2. Method/Approach (1-2 sentences)
3. Key Results (1-2 sentences)
4. Conclusions/Impact (1 sentence)

## Hard Constraints (Highest Priority)
- If any instruction conflicts with this block, follow this block.
- Do NOT include any citations (\\cite{{...}}) in the abstract.
- Do NOT include any cross-references: NO \\ref{{...}}, NO Figure~\\ref{{...}},
  NO Table~\\ref{{...}}, NO Section~\\ref{{...}}. The abstract must be fully
  self-contained — a reader should understand it without seeing any figures,
  tables, or section numbers.
- Keep the abstract self-contained and concise.

## Output Requirements
- Generate ONLY the abstract text
- Do NOT include \\begin{abstract} or any LaTeX commands
- Do NOT include any citations (\\cite{...}) — abstracts must be self-contained
- Do NOT reference any figures, tables, or sections by number or label
- Write in third person, present/past tense
- Be specific about results (include numbers if available)
"""
        if plan_writing_guidance:
            prompt += f"\n## Writing Guidance (IMPORTANT - follow strictly)\n{plan_writing_guidance}\n"

    elif section_type == "conclusion":
        prompt = f"""You are writing the Conclusion for a research paper titled: "{paper_title}"

## Task
Write a conclusion that synthesizes the paper's contributions and findings.

## Paper Sections for Reference

### Introduction
{_truncate_text(prior_sections.get('introduction', ''), 1000)}

### Method
{_truncate_text(prior_sections.get('method', ''), 800)}

### Results
{_truncate_text(prior_sections.get('result', prior_sections.get('experiment', '')), 1000)}

## Key Contributions
"""
        for contrib in key_contributions:
            prompt += f"- {contrib}\n"

        if plan_guidance:
            prompt += f"\n## Writing Structure (from Planner)\n{plan_guidance}\n"

        prompt += """
## Conclusion Structure
1. Summary of contributions (1 paragraph)
2. Key findings and their significance (1 paragraph)
3. Limitations (brief, 2-3 sentences)
4. Future work (2-3 sentences)

## Hard Constraints (Highest Priority)
- If any instruction conflicts with this block, follow this block.
- Do NOT include any citations (\\cite{{...}}) in the conclusion.
- Do NOT include any cross-references: NO \\ref{{...}}, NO Figure~\\ref{{...}},
  NO Table~\\ref{{...}}, NO Section~\\ref{{...}}. The conclusion must stand
  on its own without referencing specific figures, tables, or sections.

## Output Requirements
- Generate LaTeX content for the Conclusion section body
- Do NOT include \\section{Conclusion}
- Do NOT include any citations (\\cite{...}) — conclusions must stand alone
- Do NOT reference any figures, tables, or sections by number or label
- Be concise but comprehensive
- End on a forward-looking note
"""
        if plan_writing_guidance:
            prompt += f"\n## Writing Guidance (IMPORTANT - follow strictly)\n{plan_writing_guidance}\n"
    else:
        prompt = f"""Synthesize content for the {section_type} section based on:

{json.dumps(prior_sections, indent=2)[:3000]}

Key contributions: {key_contributions}
"""

    # Memory-provided global context for synthesis
    if memory_context:
        prompt += f"\n## Section Overview (from Session Memory)\n{memory_context}\n"

    if active_skills:
        synth_parts: list = []
        _inject_skill_constraints(synth_parts, active_skills, section_type)
        if synth_parts:
            prompt += "\n" + "\n".join(synth_parts)

    if style_guide:
        prompt += f"\n- Style guide: {style_guide}"

    return prompt


def extract_contributions_from_intro(intro_content: str) -> List[str]:
    """
    Extract contribution statements from Introduction content.
    Looks for itemize environments or numbered contributions.
    """
    contributions = []
    import re

    item_pattern = r"\\item\s*(.+?)(?=\\item|\\end{itemize}|$)"
    itemize_pattern = r"\\begin{itemize}(.*?)\\end{itemize}"
    itemize_matches = re.findall(itemize_pattern, intro_content, re.DOTALL)

    for block in itemize_matches:
        items = re.findall(item_pattern, block, re.DOTALL)
        for item in items:
            clean_item = item.strip()
            clean_item = re.sub(r"\\[a-zA-Z]+{([^}]*)}", r"\1", clean_item)
            clean_item = re.sub(r"\s+", " ", clean_item)
            if clean_item and len(clean_item) > 10:
                contributions.append(clean_item[:200])

    if not contributions:
        contrib_pattern = r"(?:contribution|we propose|we introduce|our approach)\s*[:\-]?\s*(.+?)(?:\.|$)"
        matches = re.findall(contrib_pattern, intro_content.lower(), re.IGNORECASE)
        for match in matches[:5]:
            if len(match) > 10:
                contributions.append(match.strip()[:200])

    return contributions[:5]
