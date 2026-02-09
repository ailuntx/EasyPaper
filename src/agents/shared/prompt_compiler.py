"""
Prompt Compiler - Shared prompt generation utilities
- **Description**:
    - Compiles SimpleSectionInput/SynthesisSectionInput into LLM prompts
    - Provides section-specific prompt templates
    - Used by both Commander Agent and MetaData Agent
"""
from typing import Dict, List, Optional, Any, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from ...skills.models import WritingSkill


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
        - `section_type` (str): Current section being written (used for matching)
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

# Import models (will be available after section_models.py is updated)
# Using forward references to avoid circular imports


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
    Compile a prompt for section generation
    
    Args:
        section_type: Type of section (introduction, method, etc.)
        thesis: Core thesis/theme of the section
        content_points: Key points to express
        references: Available references
        figures: Available figures
        tables: Available tables
        word_limit: Optional word limit
        style_guide: Target venue style
        intro_context: Introduction content for context (body sections)
    
    Returns:
        Compiled prompt string for LLM
    """
    content_points = content_points or []
    references = references or []
    figures = figures or []
    tables = tables or []
    
    # Get base prompt for section type
    base_prompt = SECTION_PROMPTS.get(section_type, SECTION_PROMPTS.get("method", ""))
    
    # Build the prompt
    prompt_parts = [base_prompt]
    
    # Add thesis if provided
    if thesis:
        prompt_parts.append(f"\n## Core Theme\n{thesis}")
    
    # Add content points
    if content_points:
        points_str = "\n".join(f"- {p}" for p in content_points)
        prompt_parts.append(f"\n## Key Points to Address\n{points_str}")
    
    # Add introduction context for body sections
    if intro_context and section_type not in ["introduction", "abstract"]:
        # Truncate if too long
        context = intro_context[:1500] + "..." if len(intro_context) > 1500 else intro_context
        prompt_parts.append(f"\n## Paper Introduction (for context)\n{context}")
    
    # Add references info
    if references:
        refs_info = []
        for ref in references[:20]:  # Limit to 20 refs
            if hasattr(ref, 'ref_id'):
                ref_str = f"- [{ref.ref_id}]"
                if hasattr(ref, 'title') and ref.title:
                    ref_str += f": {ref.title}"
                if hasattr(ref, 'authors') and ref.authors:
                    ref_str += f" ({ref.authors})"
                refs_info.append(ref_str)
            elif isinstance(ref, dict):
                ref_id = ref.get('ref_id', ref.get('id', 'unknown'))
                ref_str = f"- [{ref_id}]"
                if ref.get('title'):
                    ref_str += f": {ref.get('title')}"
                refs_info.append(ref_str)
        if refs_info:
            prompt_parts.append(f"\n## Available References\n" + "\n".join(refs_info))
    
    # Add figures info
    if figures:
        figs_info = []
        for fig in figures:
            if hasattr(fig, 'figure_id'):
                fig_str = f"- {fig.figure_id}"
                if hasattr(fig, 'caption') and fig.caption:
                    fig_str += f": {fig.caption}"
                figs_info.append(fig_str)
            elif isinstance(fig, dict):
                fig_id = fig.get('figure_id', fig.get('id', 'unknown'))
                fig_str = f"- {fig_id}"
                if fig.get('caption'):
                    fig_str += f": {fig.get('caption')}"
                figs_info.append(fig_str)
        if figs_info:
            prompt_parts.append(f"\n## Available Figures\n" + "\n".join(figs_info))
    
    # Add tables info
    if tables:
        tables_info = []
        for tbl in tables:
            if hasattr(tbl, 'table_id'):
                tbl_str = f"- {tbl.table_id}"
                if hasattr(tbl, 'caption') and tbl.caption:
                    tbl_str += f": {tbl.caption}"
                tables_info.append(tbl_str)
            elif isinstance(tbl, dict):
                tbl_id = tbl.get('table_id', tbl.get('id', 'unknown'))
                tbl_str = f"- {tbl_id}"
                if tbl.get('caption'):
                    tbl_str += f": {tbl.get('caption')}"
                tables_info.append(tbl_str)
        if tables_info:
            prompt_parts.append(f"\n## Available Tables\n" + "\n".join(tables_info))
    
    # Add constraints
    constraints = []
    if word_limit:
        constraints.append(f"- Word limit: approximately {word_limit} words")
    if style_guide:
        constraints.append(f"- Style guide: {style_guide}")
    if constraints:
        prompt_parts.append(f"\n## Constraints\n" + "\n".join(constraints))
    
    # Inject skill constraints before output instructions
    _inject_skill_constraints(prompt_parts, active_skills, section_type)

    # Add output instructions
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
    section_plan: Any = None,  # SectionPlan from planner
    figures: List[Any] = None,  # FigureSpec list
    tables: List[Any] = None,   # TableSpec list
    active_skills: Optional[List["WritingSkill"]] = None,
) -> str:
    """
    Compile prompt for Introduction generation (Phase 1 - Leader section)
    
    The Introduction sets the tone for the entire paper and extracts
    key contributions that will be used by subsequent sections.
    
    Args:
        section_plan: Optional SectionPlan with target_words, key_points, writing_guidance
        figures: Optional list of FigureSpec for available figures
        tables: Optional list of TableSpec for available tables
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
    
    # Add plan guidance if available
    if section_plan:
        prompt += "\n## Planning Guidance\n"
        if hasattr(section_plan, 'target_words') and section_plan.target_words:
            prompt += f"**Target Length**: Approximately {section_plan.target_words} words\n"
        if hasattr(section_plan, 'key_points') and section_plan.key_points:
            points_str = "\n".join(f"- {p}" for p in section_plan.key_points)
            prompt += f"**Key Points to Cover**:\n{points_str}\n"
        if hasattr(section_plan, 'references_to_cite') and section_plan.references_to_cite:
            prompt += f"**References to Cite**: {', '.join(section_plan.references_to_cite)}\n"
        if hasattr(section_plan, 'writing_guidance') and section_plan.writing_guidance:
            prompt += f"**Writing Guidance**: {section_plan.writing_guidance}\n"
    
    # Add references with strong constraint on valid keys
    if references:
        refs_info = []
        valid_keys = []
        for ref in references[:15]:
            if isinstance(ref, dict):
                ref_id = ref.get('ref_id', ref.get('id', ''))
                title = ref.get('title', '')
                if ref_id:
                    valid_keys.append(ref_id)
                    refs_info.append(f"- \\cite{{{ref_id}}}: {title[:80]}" if title else f"- \\cite{{{ref_id}}}")
        if refs_info:
            prompt += f"\n### CRITICAL: Citation Rules\n"
            prompt += f"**ONLY use these citation keys. DO NOT invent or hallucinate citations.**\n"
            prompt += f"**Valid keys**: {', '.join(valid_keys)}\n\n"
            prompt += "Available references:\n" + "\n".join(refs_info)
            prompt += "\n\n**WARNING**: Any citation not in the above list will be automatically removed.\n"
    
    # Add figures info
    if figures:
        figs_info = []
        for fig in figures:
            fig_id = fig.id if hasattr(fig, 'id') else fig.get('id', '')
            caption = fig.caption if hasattr(fig, 'caption') else fig.get('caption', '')
            desc = fig.description if hasattr(fig, 'description') else fig.get('description', '')
            if fig_id:
                info = f"- \\ref{{{fig_id}}}: {caption}"
                if desc:
                    info += f" ({desc})"
                figs_info.append(info)
        if figs_info:
            prompt += f"\n### Available Figures\n" + "\n".join(figs_info)
    
    # Add tables info
    if tables:
        tables_info = []
        for tbl in tables:
            tbl_id = tbl.id if hasattr(tbl, 'id') else tbl.get('id', '')
            caption = tbl.caption if hasattr(tbl, 'caption') else tbl.get('caption', '')
            desc = tbl.description if hasattr(tbl, 'description') else tbl.get('description', '')
            if tbl_id:
                info = f"- \\ref{{{tbl_id}}}: {caption}"
                if desc:
                    info += f" ({desc})"
                tables_info.append(info)
        if tables_info:
            prompt += f"\n### Available Tables\n" + "\n".join(tables_info)
    
    # Add style guide
    if style_guide:
        prompt += f"\n\n## Target Venue: {style_guide}"
    
    # Inject skill constraints
    if active_skills:
        intro_parts: list = []
        _inject_skill_constraints(intro_parts, active_skills, "introduction")
        if intro_parts:
            prompt += "\n" + "\n".join(intro_parts)
    
    prompt += """

## Output Requirements
1. Generate LaTeX content for the Introduction section body
2. Do NOT include \\section{Introduction} - just the content
3. Structure the introduction with clear paragraphs:
   - Opening: Context and motivation
   - Problem statement and gap
   - Contributions (use itemize environment)
   - Paper organization (optional)
4. Use \\cite{key} for citations
5. Use \\ref{fig:id} for figure references and \\ref{tab:id} for table references
6. Write in formal academic English

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
    section_plan: Any = None,  # SectionPlan from planner
    figures: List[Any] = None,  # FigureSpec list
    tables: List[Any] = None,   # TableSpec list
    converted_tables: Optional[Dict[str, str]] = None,  # table_id -> LaTeX code
    active_skills: Optional[List["WritingSkill"]] = None,
) -> str:
    """
    Compile prompt for Body section generation (Phase 2)
    
    Body sections receive context from Introduction to maintain consistency.
    
    Args:
        section_plan: Optional SectionPlan with target_words, key_points, writing_guidance
        figures: Optional list of FigureSpec for available figures
        tables: Optional list of TableSpec for available tables
    """
    contributions = contributions or []
    references = references or []
    figures = figures or []
    tables = tables or []
    
    base_prompt = SECTION_PROMPTS.get(section_type, "")
    
    prompt = f"""{base_prompt}

## Section Content Source
{metadata_content}

## Introduction Context (maintain consistency)
{intro_context[:2000]}{"..." if len(intro_context) > 2000 else ""}

## Key Contributions to Support
"""
    for i, contrib in enumerate(contributions, 1):
        prompt += f"{i}. {contrib}\n"
    
    # Add plan guidance if available
    if section_plan:
        prompt += "\n## Planning Guidance\n"
        if hasattr(section_plan, 'target_words') and section_plan.target_words:
            prompt += f"**Target Length**: Approximately {section_plan.target_words} words\n"
        if hasattr(section_plan, 'key_points') and section_plan.key_points:
            points_str = "\n".join(f"- {p}" for p in section_plan.key_points)
            prompt += f"**Key Points to Cover**:\n{points_str}\n"
        if hasattr(section_plan, 'references_to_cite') and section_plan.references_to_cite:
            prompt += f"**References to Cite**: {', '.join(section_plan.references_to_cite)}\n"
        if hasattr(section_plan, 'writing_guidance') and section_plan.writing_guidance:
            prompt += f"**Writing Guidance**: {section_plan.writing_guidance}\n"
    
    if references:
        refs_info = []
        valid_keys = []
        for ref in references[:10]:
            if isinstance(ref, dict):
                ref_id = ref.get('ref_id', ref.get('id', ''))
                title = ref.get('title', '')
                if ref_id:
                    valid_keys.append(ref_id)
                    refs_info.append(f"- \\cite{{{ref_id}}}: {title[:60]}" if title else f"- \\cite{{{ref_id}}}")
        if refs_info:
            prompt += f"\n## CRITICAL: Citation Rules\n"
            prompt += f"**ONLY use these citation keys. DO NOT invent citations.**\n"
            prompt += f"**Valid keys**: {', '.join(valid_keys)}\n\n"
            prompt += "\n".join(refs_info)
    
    # Determine which figures to DEFINE vs REFERENCE
    figures_to_define = []
    figures_to_reference = []
    
    if section_plan:
        figures_to_define = getattr(section_plan, 'figures_to_define', []) or []
        figures_to_reference = getattr(section_plan, 'figures_to_reference', []) or []
    
    # Build figure lookup
    figure_map = {}
    if figures:
        for fig in figures:
            fig_id = fig.id if hasattr(fig, 'id') else fig.get('id', '')
            if fig_id:
                figure_map[fig_id] = fig
    
    # Add figures to DEFINE (create \begin{figure} environment)
    if figures_to_define:
        prompt += f"\n## Figures to DEFINE in this section\n"
        prompt += "**CREATE the complete figure environment for each figure below.**\n\n"
        for fig_id in figures_to_define:
            fig = figure_map.get(fig_id)
            if fig:
                caption = fig.caption if hasattr(fig, 'caption') else fig.get('caption', '')
                desc = fig.description if hasattr(fig, 'description') else fig.get('description', '')
                file_path = fig.file_path if hasattr(fig, 'file_path') else fig.get('file_path', '')
                wide = fig.wide if hasattr(fig, 'wide') else fig.get('wide', False)
                import os
                filename = os.path.basename(file_path) if file_path else f"{fig_id.replace('fig:', '')}.pdf"
                
                # Use figure* for wide figures (double-column spanning)
                env_name = "figure*" if wide else "figure"
                width = "\\\\textwidth" if wide else "0.9\\\\linewidth"
                
                prompt += f"- **{fig_id}**: {caption}\n"
                if desc:
                    prompt += f"  Description: {desc}\n"
                if wide:
                    prompt += f"  **Note: This is a WIDE figure - use {env_name} to span both columns.**\n"
                prompt += f"  **Required LaTeX:**\n"
                prompt += f"  ```latex\n"
                prompt += f"  \\\\begin{{{env_name}}}[t]\n"
                prompt += f"  \\\\centering\n"
                prompt += f"  \\\\includegraphics[width={width}]{{figures/{filename}}}\n"
                prompt += f"  \\\\caption{{{caption}}}\\\\label{{{fig_id}}}\n"
                prompt += f"  \\\\end{{{env_name}}}\n"
                prompt += f"  ```\n\n"
    
    # Add figures to REFERENCE only (use \ref{})
    if figures_to_reference:
        prompt += f"\n## Figures to REFERENCE (already defined elsewhere)\n"
        prompt += "**DO NOT create \\\\begin{{figure}} for these - just reference them with Figure~\\\\ref{{fig:id}}.**\n"
        for fig_id in figures_to_reference:
            fig = figure_map.get(fig_id)
            if fig:
                caption = fig.caption if hasattr(fig, 'caption') else fig.get('caption', '')
                prompt += f"- {fig_id}: {caption} → use `Figure~\\\\ref{{{fig_id}}}`\n"
    
    # Determine which tables to DEFINE vs REFERENCE
    tables_to_define = []
    tables_to_reference = []
    
    if section_plan:
        tables_to_define = getattr(section_plan, 'tables_to_define', []) or []
        tables_to_reference = getattr(section_plan, 'tables_to_reference', []) or []
    
    # Build table lookup
    table_map = {}
    if tables:
        for tbl in tables:
            tbl_id = tbl.id if hasattr(tbl, 'id') else tbl.get('id', '')
            if tbl_id:
                table_map[tbl_id] = tbl
    
    # Add tables to DEFINE (create \begin{table} environment)
    _converted = converted_tables or {}
    if tables_to_define:
        prompt += f"\n## Tables to DEFINE in this section\n"
        prompt += "**Include the complete table environment for each table below.**\n\n"
        for tbl_id in tables_to_define:
            tbl = table_map.get(tbl_id)
            if tbl:
                caption = tbl.caption if hasattr(tbl, 'caption') else tbl.get('caption', '')
                desc = tbl.description if hasattr(tbl, 'description') else tbl.get('description', '')
                wide = tbl.wide if hasattr(tbl, 'wide') else tbl.get('wide', False)
                env_name = "table*" if wide else "table"

                # If pre-converted LaTeX exists, give it directly (like figures)
                if tbl_id in _converted:
                    prompt += f"- **{tbl_id}**: {caption}\n"
                    if desc:
                        prompt += f"  Description: {desc}\n"
                    prompt += f"  **Required LaTeX (include this exact table in your output):**\n"
                    prompt += f"  ```latex\n"
                    prompt += f"  {_converted[tbl_id]}\n"
                    prompt += f"  ```\n\n"
                else:
                    # Fallback: ask Writer to create from raw content
                    content = tbl.content if hasattr(tbl, 'content') else tbl.get('content', '')
                    prompt += f"- **{tbl_id}**: {caption}\n"
                    if desc:
                        prompt += f"  Description: {desc}\n"
                    if content:
                        prompt += f"  Data:\n  {content[:500]}\n"
                    if wide:
                        prompt += f"  **Note: This is a WIDE table - use {env_name} to span both columns.**\n"
                    prompt += f"  **Required: Create \\\\begin{{{env_name}}}...\\\\end{{{env_name}}} with \\\\label{{{tbl_id}}}**\n\n"
    
    # Add tables to REFERENCE only (use \ref{})
    if tables_to_reference:
        prompt += f"\n## Tables to REFERENCE (already defined elsewhere)\n"
        prompt += "**DO NOT create \\\\begin{{table}} for these - just reference them with Table~\\\\ref{{tab:id}}.**\n"
        for tbl_id in tables_to_reference:
            tbl = table_map.get(tbl_id)
            if tbl:
                caption = tbl.caption if hasattr(tbl, 'caption') else tbl.get('caption', '')
                prompt += f"- {tbl_id}: {caption} → use `Table~\\\\ref{{{tbl_id}}}`\n"
    
    # Legacy fallback: if no define/reference lists, show all as available (backward compat)
    if not figures_to_define and not figures_to_reference and figures:
        figs_info = []
        for fig in figures:
            fig_id = fig.id if hasattr(fig, 'id') else fig.get('id', '')
            caption = fig.caption if hasattr(fig, 'caption') else fig.get('caption', '')
            if fig_id:
                figs_info.append(f"- {fig_id}: {caption}")
        if figs_info:
            prompt += f"\n## Available Figures (reference only with \\\\ref{{}})\n" + "\n".join(figs_info)
    
    if not tables_to_define and not tables_to_reference and tables:
        tables_info = []
        for tbl in tables:
            tbl_id = tbl.id if hasattr(tbl, 'id') else tbl.get('id', '')
            caption = tbl.caption if hasattr(tbl, 'caption') else tbl.get('caption', '')
            if tbl_id:
                tables_info.append(f"- {tbl_id}: {caption}")
        if tables_info:
            prompt += f"\n## Available Tables (reference only with \\\\ref{{}})\n" + "\n".join(tables_info)
    
    if style_guide:
        prompt += f"\n\n## Target Venue: {style_guide}"
    
    # Inject skill constraints
    if active_skills:
        body_parts: list = []
        _inject_skill_constraints(body_parts, active_skills, section_type)
        if body_parts:
            prompt += "\n" + "\n".join(body_parts)
    
    prompt += """

## Output Requirements
1. Generate LaTeX content for the section body only
2. Do NOT include \\section{} command
3. Maintain consistency with the Introduction's framing
4. Support the stated contributions where relevant
5. Use \\cite{key} for citations
6. Use \\ref{fig:id} for figure references and \\ref{tab:id} for table references
7. Use clear academic writing style
"""
    
    return prompt


def compile_synthesis_prompt(
    section_type: str,
    paper_title: str,
    prior_sections: Dict[str, str],
    key_contributions: List[str] = None,
    word_limit: Optional[int] = None,
    style_guide: Optional[str] = None,
    section_plan: Any = None,  # SectionPlan from planner
    active_skills: Optional[List["WritingSkill"]] = None,
) -> str:
    """
    Compile prompt for Synthesis sections (Abstract/Conclusion - Phase 3)
    
    These sections synthesize content from already-generated sections
    rather than generating from scratch.
    
    Args:
        section_plan: Optional SectionPlan with target_words, writing_guidance
    """
    key_contributions = key_contributions or []
    
    # Extract plan guidance
    plan_key_points = []
    plan_writing_guidance = ""
    if section_plan:
        if hasattr(section_plan, 'target_words') and section_plan.target_words:
            word_limit = section_plan.target_words
        if hasattr(section_plan, 'key_points') and section_plan.key_points:
            plan_key_points = section_plan.key_points
        if hasattr(section_plan, 'writing_guidance') and section_plan.writing_guidance:
            plan_writing_guidance = section_plan.writing_guidance
    
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
        
        # Add plan key points if available
        if plan_key_points:
            prompt += "\n## Key Points to Cover (from Planner - MUST include all)\n"
            for point in plan_key_points:
                prompt += f"- {point}\n"
        
        prompt += """
## Abstract Structure
1. Problem/Motivation (1-2 sentences)
2. Method/Approach (1-2 sentences)
3. Key Results (1-2 sentences)
4. Conclusions/Impact (1 sentence)

## Output Requirements
- Generate ONLY the abstract text
- Do NOT include \\begin{abstract} or any LaTeX commands
- Write in third person, present/past tense
- Be specific about results (include numbers if available)
"""
        
        # Add plan writing guidance if available
        if plan_writing_guidance:
            prompt += f"\n## Writing Guidance (IMPORTANT - follow strictly)\n{plan_writing_guidance}\n"
        
    elif section_type == "conclusion":
        prompt = f"""You are writing the Conclusion for a research paper titled: "{paper_title}"

## Task
Write a conclusion that synthesizes the paper's contributions and findings.

## Paper Sections for Reference

### Introduction
{prior_sections.get('introduction', '')[:1000]}...

### Method
{prior_sections.get('method', '')[:800]}...

### Results
{prior_sections.get('result', prior_sections.get('experiment', ''))[:1000]}...

## Key Contributions
"""
        for contrib in key_contributions:
            prompt += f"- {contrib}\n"
        
        # Add plan key points if available
        if plan_key_points:
            prompt += "\n## Key Points to Cover (from Planner - MUST include all)\n"
            for point in plan_key_points:
                prompt += f"- {point}\n"
        
        prompt += """
## Conclusion Structure
1. Summary of contributions (1 paragraph)
2. Key findings and their significance (1 paragraph)
3. Limitations (brief, 2-3 sentences)
4. Future work (2-3 sentences)

## Output Requirements
- Generate LaTeX content for the Conclusion section body
- Do NOT include \\section{Conclusion}
- Be concise but comprehensive
- End on a forward-looking note
"""
        
        # Add plan writing guidance if available
        if plan_writing_guidance:
            prompt += f"\n## Writing Guidance (IMPORTANT - follow strictly)\n{plan_writing_guidance}\n"
    else:
        # Generic synthesis prompt
        prompt = f"""Synthesize content for the {section_type} section based on:

{json.dumps(prior_sections, indent=2)[:3000]}

Key contributions: {key_contributions}
"""
    
    # Inject skill constraints
    if active_skills:
        synth_parts: list = []
        _inject_skill_constraints(synth_parts, active_skills, section_type)
        if synth_parts:
            prompt += "\n" + "\n".join(synth_parts)

    # Add constraints with strong emphasis on word limit
    if word_limit:
        prompt += f"\n\n## STRICT LENGTH CONSTRAINT\n"
        prompt += f"**MAXIMUM {word_limit} words** - This is a HARD limit. Do NOT exceed this word count.\n"
        prompt += f"Count your words before finalizing. If over {word_limit}, cut content ruthlessly.\n"
    if style_guide:
        prompt += f"\n- Style guide: {style_guide}"
    
    return prompt


def extract_contributions_from_intro(intro_content: str) -> List[str]:
    """
    Extract contribution statements from Introduction content
    
    Looks for itemize environments or numbered contributions.
    """
    contributions = []
    
    # Look for itemize content
    import re
    
    # Pattern for \item content
    item_pattern = r'\\item\s*(.+?)(?=\\item|\\end{itemize}|$)'
    
    # Find itemize blocks
    itemize_pattern = r'\\begin{itemize}(.*?)\\end{itemize}'
    itemize_matches = re.findall(itemize_pattern, intro_content, re.DOTALL)
    
    for block in itemize_matches:
        items = re.findall(item_pattern, block, re.DOTALL)
        for item in items:
            # Clean up the item text
            clean_item = item.strip()
            clean_item = re.sub(r'\\[a-zA-Z]+{([^}]*)}', r'\1', clean_item)  # Remove LaTeX commands
            clean_item = re.sub(r'\s+', ' ', clean_item)  # Normalize whitespace
            if clean_item and len(clean_item) > 10:
                contributions.append(clean_item[:200])  # Limit length
    
    # If no itemize found, look for "contribution" mentions
    if not contributions:
        contrib_pattern = r'(?:contribution|we propose|we introduce|our approach)\s*[:\-]?\s*(.+?)(?:\.|$)'
        matches = re.findall(contrib_pattern, intro_content.lower(), re.IGNORECASE)
        for match in matches[:5]:
            if len(match) > 10:
                contributions.append(match.strip()[:200])
    
    return contributions[:5]  # Return at most 5 contributions
