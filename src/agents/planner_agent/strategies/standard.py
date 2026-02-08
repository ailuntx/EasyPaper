"""
Standard Planning Strategy
- **Description**:
    - Default planning strategy for empirical papers
    - Uses LLM to analyze metadata and create detailed plan
"""
import json
import logging
from typing import List, Dict, Any, Optional

from .base import PlanningStrategy
from ..models import (
    PaperPlan,
    SectionPlan,
    PlanRequest,
    PaperType,
    NarrativeStyle,
    SECTION_RATIOS_BY_TYPE,
    DEFAULT_EMPIRICAL_SECTIONS,
    calculate_total_words,
)


logger = logging.getLogger("uvicorn.error")


PLANNING_SYSTEM_PROMPT = """You are an expert academic paper planner. Your job is to analyze paper metadata and create a detailed writing plan.

Given the paper's idea/hypothesis, method, data, experiments, references, figures, and tables, you must:
1. Determine the paper type (empirical, theoretical, survey, position, system, benchmark)
2. Identify the key contributions (usually 2-4)
3. Plan what each section should cover
4. Suggest which references to cite in each section
5. Assign figures and tables to appropriate sections
6. Provide specific writing guidance

Output a JSON object with this structure:
{
    "paper_type": "empirical",
    "contributions": [
        "Contribution 1 statement",
        "Contribution 2 statement"
    ],
    "narrative_style": "technical",
    "terminology": {
        "key_term": "definition to use consistently"
    },
    "structure_rationale": "Why this structure works for this paper",
    "abstract_focus": "What the abstract should emphasize",
    "sections": [
        {
            "section_type": "introduction",
            "key_points": ["Point 1", "Point 2"],
            "content_sources": ["idea_hypothesis", "method"],
            "references_to_cite": ["ref_key1", "ref_key2"],
            "figures_to_use": ["fig:architecture"],
            "tables_to_use": ["tab:results"],
            "writing_guidance": "Specific guidance for this section"
        }
    ]
}

Be specific and actionable. The plan will guide an AI writer."""


PLANNING_USER_PROMPT_TEMPLATE = """Create a detailed paper plan for:

**Title**: {title}

**Idea/Hypothesis**:
{idea_hypothesis}

**Method**:
{method}

**Data**:
{data}

**Experiments**:
{experiments}

**Available References** (BibTeX keys):
{reference_keys}

**Available Figures**:
{figure_info}

**Available Tables**:
{table_info}

**Target**: {target_pages} pages ({total_words} words) for {style_guide}

Analyze this content and create a comprehensive writing plan. Focus on:
1. What are the 2-4 key contributions?
2. How should content be distributed across sections?
3. Which references should be cited where?
4. Which figures and tables should appear in which sections?
5. What specific guidance helps each section?

Output valid JSON only."""


class StandardPlanningStrategy(PlanningStrategy):
    """
    Standard planning strategy for empirical papers
    - **Description**:
        - Uses LLM to analyze metadata
        - Creates detailed section plans
        - Allocates word budgets based on paper type
    """
    
    @property
    def name(self) -> str:
        return "standard"
    
    @property
    def description(self) -> str:
        return "Standard planning for empirical research papers"
    
    async def create_plan(
        self,
        request: PlanRequest,
        llm_client: Any,
        model_name: str,
    ) -> PaperPlan:
        """Create a paper plan using LLM analysis"""
        
        # Calculate total word budget
        total_words = calculate_total_words(request.target_pages, request.style_guide)
        target_pages = request.target_pages or 8
        style_guide = request.style_guide or "DEFAULT"
        
        # Extract reference keys from BibTeX
        reference_keys = self._extract_reference_keys(request.references)
        
        # Format figure info
        figure_info = "None provided"
        if request.figures:
            fig_lines = []
            for fig in request.figures:
                line = f"- {fig.id}: {fig.caption}"
                if fig.description:
                    line += f" ({fig.description})"
                if fig.section:
                    line += f" [suggested: {fig.section}]"
                fig_lines.append(line)
            figure_info = "\n".join(fig_lines)
        
        # Format table info
        table_info = "None provided"
        if request.tables:
            tbl_lines = []
            for tbl in request.tables:
                line = f"- {tbl.id}: {tbl.caption}"
                if tbl.description:
                    line += f" ({tbl.description})"
                if tbl.section:
                    line += f" [suggested: {tbl.section}]"
                tbl_lines.append(line)
            table_info = "\n".join(tbl_lines)
        
        # Build prompt
        user_prompt = PLANNING_USER_PROMPT_TEMPLATE.format(
            title=request.title,
            idea_hypothesis=request.idea_hypothesis[:2000],
            method=request.method[:2000],
            data=request.data[:1500],
            experiments=request.experiments[:2000],
            reference_keys=", ".join(reference_keys) if reference_keys else "None provided",
            figure_info=figure_info,
            table_info=table_info,
            target_pages=target_pages,
            total_words=total_words,
            style_guide=style_guide,
        )
        
        try:
            # Call LLM for planning
            response = await llm_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": PLANNING_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,  # Lower temperature for consistent planning
            )
            
            plan_text = response.choices[0].message.content.strip()
            
            # Parse JSON response
            plan_data = self._parse_plan_json(plan_text)
            
            # Build PaperPlan from LLM output
            paper_plan = self._build_paper_plan(
                plan_data=plan_data,
                request=request,
                total_words=total_words,
            )
            
            logger.info(
                "planner.plan_created title=%s sections=%d words=%d",
                request.title[:30],
                len(paper_plan.sections),
                paper_plan.total_target_words,
            )
            
            return paper_plan
            
        except Exception as e:
            logger.error("planner.llm_error: %s", str(e))
            # Fall back to default plan
            return self._create_default_plan(request, total_words)
    
    def _extract_reference_keys(self, references: List[str]) -> List[str]:
        """Extract BibTeX keys from reference entries"""
        import re
        keys = []
        for ref in references:
            match = re.search(r'@\w+\{([^,]+)', ref)
            if match:
                keys.append(match.group(1).strip())
        return keys
    
    def _parse_plan_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response"""
        # Try to extract JSON from markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            text = text[start:end].strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("planner.json_parse_error, using defaults")
            return {}
    
    def _build_paper_plan(
        self,
        plan_data: Dict[str, Any],
        request: PlanRequest,
        total_words: int,
    ) -> PaperPlan:
        """Build PaperPlan from LLM output"""
        
        # Determine paper type
        paper_type_str = plan_data.get("paper_type", "empirical").lower()
        try:
            paper_type = PaperType(paper_type_str)
        except ValueError:
            paper_type = PaperType.EMPIRICAL
        
        # Get word ratios for this paper type
        ratios = SECTION_RATIOS_BY_TYPE.get(paper_type, SECTION_RATIOS_BY_TYPE[PaperType.EMPIRICAL])
        
        # Determine narrative style
        style_str = plan_data.get("narrative_style", "technical").lower()
        try:
            narrative_style = NarrativeStyle(style_str)
        except ValueError:
            narrative_style = NarrativeStyle.TECHNICAL
        
        # Build section plans
        llm_sections = plan_data.get("sections", [])
        section_map = {s.get("section_type"): s for s in llm_sections if s.get("section_type")}
        
        sections = []
        for order, section_type in enumerate(DEFAULT_EMPIRICAL_SECTIONS):
            # Get ratio and calculate target words
            ratio = ratios.get(section_type, 0.1)
            target_words_section = int(total_words * ratio)
            
            # Get LLM-provided details if available
            llm_section = section_map.get(section_type, {})
            
            # Determine default content sources
            default_sources = self._get_default_sources(section_type)
            
            # Handle figures: distinguish define vs reference
            figures_to_define = llm_section.get("figures_to_define", [])
            figures_to_reference = llm_section.get("figures_to_reference", [])
            # Backward compat: if only figures_to_use provided, treat as reference
            if not figures_to_define and not figures_to_reference:
                figures_to_reference = llm_section.get("figures_to_use", [])
            
            # Handle tables: distinguish define vs reference
            tables_to_define = llm_section.get("tables_to_define", [])
            tables_to_reference = llm_section.get("tables_to_reference", [])
            # Backward compat: if only tables_to_use provided, treat as reference
            if not tables_to_define and not tables_to_reference:
                tables_to_reference = llm_section.get("tables_to_use", [])
            
            section_plan = SectionPlan(
                section_type=section_type,
                section_title=self._get_section_title(section_type),
                target_words=target_words_section,
                key_points=llm_section.get("key_points", []),
                content_sources=llm_section.get("content_sources", default_sources),
                references_to_cite=llm_section.get("references_to_cite", []),
                figures_to_use=llm_section.get("figures_to_use", []),  # Backward compat
                figures_to_define=figures_to_define,
                figures_to_reference=figures_to_reference,
                tables_to_use=llm_section.get("tables_to_use", []),  # Backward compat
                tables_to_define=tables_to_define,
                tables_to_reference=tables_to_reference,
                depends_on=self._get_dependencies(section_type),
                writing_guidance=llm_section.get("writing_guidance", ""),
                order=order,
            )
            sections.append(section_plan)
        
        paper_plan = PaperPlan(
            title=request.title,
            paper_type=paper_type,
            total_target_words=total_words,
            sections=sections,
            contributions=plan_data.get("contributions", []),
            narrative_style=narrative_style,
            terminology=plan_data.get("terminology", {}),
            structure_rationale=plan_data.get("structure_rationale", ""),
            abstract_focus=plan_data.get("abstract_focus", ""),
        )
        
        # Ensure each figure/table is defined in exactly one section
        self._assign_figure_table_definitions(paper_plan, request)
        
        return paper_plan
    
    def _create_default_plan(self, request: PlanRequest, total_words: int) -> PaperPlan:
        """Create a default plan when LLM fails"""
        ratios = SECTION_RATIOS_BY_TYPE[PaperType.EMPIRICAL]
        
        sections = []
        for order, section_type in enumerate(DEFAULT_EMPIRICAL_SECTIONS):
            ratio = ratios.get(section_type, 0.1)
            section_plan = SectionPlan(
                section_type=section_type,
                section_title=self._get_section_title(section_type),
                target_words=int(total_words * ratio),
                content_sources=self._get_default_sources(section_type),
                depends_on=self._get_dependencies(section_type),
                order=order,
            )
            sections.append(section_plan)
        
        return PaperPlan(
            title=request.title,
            paper_type=PaperType.EMPIRICAL,
            total_target_words=total_words,
            sections=sections,
            narrative_style=NarrativeStyle.TECHNICAL,
        )
    
    def _get_section_title(self, section_type: str) -> str:
        """Get display title for section type"""
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
    
    def _get_default_sources(self, section_type: str) -> List[str]:
        """Get default metadata fields for each section"""
        sources = {
            "abstract": ["idea_hypothesis", "method", "experiments"],
            "introduction": ["idea_hypothesis", "method", "data", "experiments"],
            "related_work": ["references"],
            "method": ["method"],
            "experiment": ["data", "experiments"],
            "result": ["experiments"],
            "discussion": ["experiments"],
            "conclusion": ["idea_hypothesis", "experiments"],
        }
        return sources.get(section_type, [])
    
    def _get_dependencies(self, section_type: str) -> List[str]:
        """Get section dependencies"""
        deps = {
            "abstract": ["introduction", "method", "experiment", "result", "conclusion"],
            "introduction": [],
            "related_work": ["introduction"],
            "method": ["introduction"],
            "experiment": ["method"],
            "result": ["experiment"],
            "discussion": ["result"],
            "conclusion": ["introduction", "result"],
        }
        return deps.get(section_type, [])
    
    def _should_be_wide_figure(self, fig_info) -> bool:
        """
        Determine if a figure should use figure* (double-column spanning).
        
        Rules:
        - User-specified wide=True always takes precedence
        - Keywords in caption/description suggesting wide layout
        - "overview", "comparison", "architecture", "pipeline" figures often need wide
        """
        # User explicitly set wide
        if getattr(fig_info, 'wide', False):
            return True
        
        # Check for keywords suggesting wide figure
        wide_keywords = [
            "overview", "comparison", "architecture", "pipeline", 
            "framework", "full", "complete", "main", "overall",
            "workflow", "system"
        ]
        
        text = (
            (fig_info.id if hasattr(fig_info, 'id') else "") + " " +
            (fig_info.caption if hasattr(fig_info, 'caption') else "") + " " +
            (fig_info.description if hasattr(fig_info, 'description') else "")
        ).lower()
        
        for keyword in wide_keywords:
            if keyword in text:
                return True
        
        return False
    
    def _should_be_wide_table(self, tbl_info) -> bool:
        """
        Determine if a table should use table* (double-column spanning).
        
        Rules:
        - User-specified wide=True always takes precedence
        - Tables with many columns (>5) should be wide
        - "main", "comparison", "full" tables often need wide
        - Check content for column count if available
        """
        # User explicitly set wide
        if getattr(tbl_info, 'wide', False):
            return True
        
        # Check for keywords suggesting wide table
        wide_keywords = [
            "main", "comparison", "full", "complete", "all",
            "overall", "summary", "comprehensive"
        ]
        
        text = (
            (tbl_info.id if hasattr(tbl_info, 'id') else "") + " " +
            (tbl_info.caption if hasattr(tbl_info, 'caption') else "") + " " +
            (tbl_info.description if hasattr(tbl_info, 'description') else "")
        ).lower()
        
        for keyword in wide_keywords:
            if keyword in text:
                return True
        
        # Try to estimate column count from content if available
        content = getattr(tbl_info, 'content', None)
        if content:
            # Check first line for column separators
            first_line = content.strip().split('\n')[0] if content.strip() else ""
            # Count pipes for markdown tables
            if '|' in first_line:
                col_count = first_line.count('|') - 1
                if col_count > 5:
                    return True
            # Count commas for CSV
            elif ',' in first_line:
                col_count = first_line.count(',') + 1
                if col_count > 5:
                    return True
            # Count tabs
            elif '\t' in first_line:
                col_count = first_line.count('\t') + 1
                if col_count > 5:
                    return True
        
        return False
    
    def _assign_figure_table_definitions(
        self, 
        paper_plan: PaperPlan, 
        request: PlanRequest
    ) -> None:
        """
        Ensure each figure/table is DEFINED in exactly one section.
        
        This method:
        1. Collects all figures/tables from the request
        2. Checks which sections already have them assigned for definition
        3. For any unassigned, assigns to the most appropriate section
        4. Ensures other sections only REFERENCE them
        5. Auto-detects and sets 'wide' flag for double-column spanning
        
        Default assignment rules:
        - Architecture/overview figures -> method
        - Result/ablation figures -> result or experiment
        - Main results tables -> experiment or result
        - Ablation tables -> result
        
        Wide detection rules:
        - Figures: "comparison", "overview", "architecture" keywords
        - Tables: > 5 columns or "main", "comparison", "full" keywords
        """
        # Collect all available figures and tables
        all_figures = {f.id: f for f in request.figures}
        all_tables = {t.id: t for t in request.tables}
        
        if not all_figures and not all_tables:
            return
        
        # Track which are already assigned for definition
        figures_defined = set()
        tables_defined = set()
        
        for section in paper_plan.sections:
            figures_defined.update(section.figures_to_define)
            tables_defined.update(section.tables_to_define)
        
        # Default section assignments for figures
        figure_section_hints = {
            "architecture": "method",
            "overview": "method",
            "framework": "method",
            "model": "method",
            "pipeline": "method",
            "result": "result",
            "ablation": "result",
            "comparison": "experiment",
            "performance": "experiment",
        }
        
        # Default section assignments for tables
        table_section_hints = {
            "main": "experiment",
            "result": "experiment",
            "comparison": "experiment",
            "ablation": "result",
            "hyperparameter": "experiment",
            "statistics": "experiment",
            "dataset": "experiment",
        }
        
        # Assign unassigned figures and auto-detect wide
        for fig_id, fig_info in all_figures.items():
            # Auto-detect wide for all figures
            if self._should_be_wide_figure(fig_info):
                if fig_id not in paper_plan.wide_figures:
                    paper_plan.wide_figures.append(fig_id)
                    print(f"[Planner] Auto-detected figure '{fig_id}' as WIDE (double-column)")
            
            if fig_id in figures_defined:
                continue
            
            # Try to find the best section
            target_section = None
            
            # First, check user-suggested section
            if fig_info.section:
                for section in paper_plan.sections:
                    if section.section_type == fig_info.section:
                        target_section = section
                        break
            
            # Second, infer from figure ID/caption
            if not target_section:
                fig_lower = (fig_id + " " + fig_info.caption + " " + fig_info.description).lower()
                for hint, section_type in figure_section_hints.items():
                    if hint in fig_lower:
                        for section in paper_plan.sections:
                            if section.section_type == section_type:
                                target_section = section
                                break
                    if target_section:
                        break
            
            # Default: assign to method section
            if not target_section:
                for section in paper_plan.sections:
                    if section.section_type == "method":
                        target_section = section
                        break
            
            # Assign to definition
            if target_section:
                target_section.figures_to_define.append(fig_id)
                figures_defined.add(fig_id)
                print(f"[Planner] Assigned figure '{fig_id}' to be DEFINED in '{target_section.section_type}'")
        
        # Assign unassigned tables and auto-detect wide
        for tbl_id, tbl_info in all_tables.items():
            # Auto-detect wide for all tables
            if self._should_be_wide_table(tbl_info):
                if tbl_id not in paper_plan.wide_tables:
                    paper_plan.wide_tables.append(tbl_id)
                    print(f"[Planner] Auto-detected table '{tbl_id}' as WIDE (double-column)")
            
            if tbl_id in tables_defined:
                continue
            
            target_section = None
            
            # First, check user-suggested section
            if tbl_info.section:
                for section in paper_plan.sections:
                    if section.section_type == tbl_info.section:
                        target_section = section
                        break
            
            # Second, infer from table ID/caption
            if not target_section:
                tbl_lower = (tbl_id + " " + tbl_info.caption + " " + tbl_info.description).lower()
                for hint, section_type in table_section_hints.items():
                    if hint in tbl_lower:
                        for section in paper_plan.sections:
                            if section.section_type == section_type:
                                target_section = section
                                break
                    if target_section:
                        break
            
            # Default: assign to experiment section
            if not target_section:
                for section in paper_plan.sections:
                    if section.section_type == "experiment":
                        target_section = section
                        break
            
            if target_section:
                target_section.tables_to_define.append(tbl_id)
                tables_defined.add(tbl_id)
                print(f"[Planner] Assigned table '{tbl_id}' to be DEFINED in '{target_section.section_type}'")
        
        # Now, for sections that reference but don't define, move to reference list
        for section in paper_plan.sections:
            # Handle figures_to_use (legacy)
            for fig_id in list(section.figures_to_use):
                if fig_id not in section.figures_to_define:
                    if fig_id not in section.figures_to_reference:
                        section.figures_to_reference.append(fig_id)
            
            # Handle tables_to_use (legacy)
            for tbl_id in list(section.tables_to_use):
                if tbl_id not in section.tables_to_define:
                    if tbl_id not in section.tables_to_reference:
                        section.tables_to_reference.append(tbl_id)
