"""
MetaData Agent - Simple Mode Paper Generation
- **Description**:
    - Generates complete papers from simplified MetaData input
    - Five-phase generation:
        0. Planning - creates detailed paper plan
        1. Introduction (Leader) - sets tone and extracts contributions
        2. Body Sections (parallel) - Method, Experiment, Results, Related Work
        3. Synthesis Sections - Abstract and Conclusion from prior sections
        3.5. Review Loop - iterative feedback and revision
        4. PDF Compilation - via Typesetter Agent
    - Independent API, no frontend dependency
"""
import asyncio
import json
import re
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import httpx
from openai import AsyncOpenAI
from pydantic import BaseModel
from fastapi import APIRouter

from ..base import BaseAgent
from ...config.schema import ModelConfig
from .models import (
    PaperMetaData,
    PaperGenerationRequest,
    PaperGenerationResult,
    SectionResult,
    SectionGenerationRequest,
    BODY_SECTION_SOURCES,
    SYNTHESIS_SECTIONS,
    DEFAULT_SECTION_ORDER,
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
from ..reviewer_agent.models import ReviewResult, FeedbackResult, Severity, SectionFeedback
from .models import FigureSpec, TableSpec


class MetaDataAgent(BaseAgent):
    """
    MetaData Agent for simple-mode paper generation
    
    - **Description**:
        - Accepts 5 natural language fields + BibTeX references
        - Generates complete paper through three-phase process
        - Independent API, can be called directly via curl/Postman
    """
    
    def __init__(self, config: ModelConfig):
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.model_name = config.model_name
        self.results_dir = Path(__file__).parent.parent.parent.parent / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self._router = self._create_router()
    
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
        
        # Parse references first
        parsed_refs = self._parse_references(metadata.references)
        
        # Extract valid citation keys for validation
        valid_citation_keys = self._extract_valid_citation_keys(parsed_refs)
        print(f"[MetaDataAgent] Valid citation keys: {valid_citation_keys}")
        
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
            # Phase 0: Planning (if enabled)
            # =================================================================
            if enable_planning:
                print(f"[MetaDataAgent] Phase 0: Creating Paper Plan...")
                paper_plan = await self._create_paper_plan(
                    metadata=metadata,
                    target_pages=target_pages,
                    style_guide=metadata.style_guide,
                )
                if paper_plan:
                    print(f"[MetaDataAgent] Plan created: {len(paper_plan.sections)} sections, {paper_plan.total_target_words} words")
                    
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
                        plan_path = paper_dir / "paper_plan.json"
                        plan_path.write_text(
                            paper_plan.model_dump_json(indent=2),
                            encoding="utf-8",
                        )
                else:
                    print(f"[MetaDataAgent] Planning skipped or failed, using defaults")
            
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
                metadata, parsed_refs, section_plan=intro_plan,
                figures=metadata.figures, tables=metadata.tables,
            )
            sections_results.append(intro_result)
            
            if intro_result.status == "ok":
                # Mini-review: validate citations and check quality
                review_result = self._mini_review_section(
                    section_type="introduction",
                    content=intro_result.latex_content,
                    section_plan=intro_plan,
                    valid_citation_keys=valid_citation_keys,
                )
                # Use fixed content (with invalid citations removed)
                intro_result.latex_content = review_result["fixed_content"]
                intro_result.word_count = review_result["fixed_word_count"]
                
                generated_sections["introduction"] = intro_result.latex_content
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
            
            # Use contributions from plan if available
            if paper_plan and paper_plan.contributions:
                contributions = paper_plan.contributions
            
            # =================================================================
            # Phase 2: Body Sections (can be parallel)
            # =================================================================
            print(f"[MetaDataAgent] Phase 2: Generating Body Sections...")
            body_section_types = ["related_work", "method", "experiment", "result"]
            
            # Generate body sections (can be parallelized)
            body_tasks = []
            for section_type in body_section_types:
                section_plan = paper_plan.get_section(section_type) if paper_plan else None
                # Filter figures/tables for this section
                section_figures = [f for f in metadata.figures if f.section == section_type or not f.section]
                section_tables = [t for t in metadata.tables if t.section == section_type or not t.section]
                task = self._generate_body_section(
                    section_type=section_type,
                    metadata=metadata,
                    intro_context=generated_sections.get("introduction", ""),
                    contributions=contributions,
                    parsed_refs=parsed_refs,
                    section_plan=section_plan,
                    figures=section_figures,
                    tables=section_tables,
                )
                body_tasks.append(task)
            
            body_results = await asyncio.gather(*body_tasks, return_exceptions=True)
            
            for idx, (section_type, result) in enumerate(zip(body_section_types, body_results)):
                if isinstance(result, Exception):
                    error_result = SectionResult(
                        section_type=section_type,
                        status="error",
                        error=str(result),
                    )
                    sections_results.append(error_result)
                    errors.append(f"{section_type} generation failed: {result}")
                else:
                    sections_results.append(result)
                    if result.status == "ok":
                        # Mini-review: validate citations and check quality
                        section_plan = paper_plan.get_section(section_type) if paper_plan else None
                        review_result = self._mini_review_section(
                            section_type=section_type,
                            content=result.latex_content,
                            section_plan=section_plan,
                            valid_citation_keys=valid_citation_keys,
                        )
                        # Use fixed content (with invalid citations removed)
                        result.latex_content = review_result["fixed_content"]
                        result.word_count = review_result["fixed_word_count"]
                        
                        generated_sections[section_type] = result.latex_content
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
            )
            sections_results.insert(0, abstract_result)  # Abstract goes first
            if abstract_result.status == "ok":
                # Mini-review for abstract (mostly citation check)
                abstract_plan = paper_plan.get_section("abstract") if paper_plan else None
                review_result = self._mini_review_section(
                    section_type="abstract",
                    content=abstract_result.latex_content,
                    section_plan=abstract_plan,
                    valid_citation_keys=valid_citation_keys,
                )
                abstract_result.latex_content = review_result["fixed_content"]
                abstract_result.word_count = review_result["fixed_word_count"]
                
                generated_sections["abstract"] = abstract_result.latex_content
            else:
                errors.append(f"Abstract generation failed: {abstract_result.error}")
            
            # Generate Conclusion
            conclusion_result = await self._generate_synthesis_section(
                section_type="conclusion",
                paper_title=metadata.title,
                prior_sections=generated_sections,
                contributions=contributions,
                style_guide=metadata.style_guide,
                section_plan=paper_plan.get_section("conclusion") if paper_plan else None,
            )
            sections_results.append(conclusion_result)
            if conclusion_result.status == "ok":
                # Mini-review for conclusion
                conclusion_plan = paper_plan.get_section("conclusion") if paper_plan else None
                review_result = self._mini_review_section(
                    section_type="conclusion",
                    content=conclusion_result.latex_content,
                    section_plan=conclusion_plan,
                    valid_citation_keys=valid_citation_keys,
                )
                conclusion_result.latex_content = review_result["fixed_content"]
                conclusion_result.word_count = review_result["fixed_word_count"]
                
                generated_sections["conclusion"] = conclusion_result.latex_content
            else:
                errors.append(f"Conclusion generation failed: {conclusion_result.error}")
            
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
                parsed_refs=parsed_refs,
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
            )
            if orchestration_errors:
                errors.extend(orchestration_errors)
            
            # =================================================================
            # Assemble Paper
            # =================================================================
            print(f"[MetaDataAgent] Assembling paper...")
            latex_content = self._assemble_paper(
                title=metadata.title,
                sections=generated_sections,
                references=parsed_refs,
                valid_citation_keys=valid_citation_keys,
            )
            
            # Calculate total word count
            total_words = sum(r.word_count for r in sections_results if r.word_count)
            
            # Save output if requested
            output_path = None
            if save_output and paper_dir:
                output_path = str(paper_dir)
                
                # Save main.tex
                tex_path = paper_dir / "main.tex"
                tex_path.write_text(latex_content, encoding="utf-8")
                
                # Save references.bib
                bib_content = self._generate_bib_file(parsed_refs)
                bib_path = paper_dir / "references.bib"
                bib_path.write_text(bib_content, encoding="utf-8")
                
                # Save metadata.json
                meta_path = paper_dir / "metadata.json"
                meta_path.write_text(
                    json.dumps(metadata.model_dump(), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                
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
        parsed_refs = self._parse_references(metadata.references)
        
        if request.section_type == "introduction":
            return await self._generate_introduction(metadata, parsed_refs)
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
                parsed_refs=parsed_refs,
            )
    
    # =========================================================================
    # Phase 1: Introduction Generation
    # =========================================================================
    
    async def _generate_introduction(
        self,
        metadata: PaperMetaData,
        parsed_refs: List[Dict[str, Any]],
        section_plan: Optional[SectionPlan] = None,
        figures: Optional[List[FigureSpec]] = None,
        tables: Optional[List[TableSpec]] = None,
    ) -> SectionResult:
        """Generate Introduction section (Leader)"""
        try:
            # Build prompt with plan guidance if available
            prompt = compile_introduction_prompt(
                paper_title=metadata.title,
                idea_hypothesis=metadata.idea_hypothesis,
                method_summary=metadata.method,
                data_summary=metadata.data,
                experiments_summary=metadata.experiments,
                references=parsed_refs,
                style_guide=metadata.style_guide,
                section_plan=section_plan,
                figures=figures,
                tables=tables,
            )
            
            # Adjust max_tokens based on target words
            max_tokens = 2000
            if section_plan and section_plan.target_words:
                # Approximate: 1 token ≈ 0.75 words
                max_tokens = max(1500, int(section_plan.target_words * 1.5))
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert academic writer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            
            content = response.choices[0].message.content.strip()
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
        parsed_refs: List[Dict[str, Any]],
        section_plan: Optional[SectionPlan] = None,
        figures: Optional[List[FigureSpec]] = None,
        tables: Optional[List[TableSpec]] = None,
    ) -> SectionResult:
        """Generate a body section (Method, Experiment, Results, Related Work)"""
        try:
            # Get relevant content from metadata based on section type
            # Use plan's content_sources if available, otherwise fall back to defaults
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
            
            prompt = compile_body_section_prompt(
                section_type=section_type,
                metadata_content=metadata_content,
                intro_context=intro_context,
                contributions=contributions,
                references=parsed_refs,
                style_guide=metadata.style_guide,
                section_plan=section_plan,
                figures=figures,
                tables=tables,
            )
            
            # Adjust max_tokens based on target words
            max_tokens = 2500
            if section_plan and section_plan.target_words:
                max_tokens = max(1500, int(section_plan.target_words * 1.5))
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert academic writer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            
            content = response.choices[0].message.content.strip()
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
    ) -> SectionResult:
        """Generate synthesis section (Abstract or Conclusion)"""
        try:
            prompt = compile_synthesis_prompt(
                section_type=section_type,
                paper_title=paper_title,
                prior_sections=prior_sections,
                key_contributions=contributions,
                style_guide=style_guide,
                section_plan=section_plan,  # Pass plan for guidance
            )
            
            # Adjust max_tokens based on target words
            if section_plan and section_plan.target_words:
                max_tokens = max(400, int(section_plan.target_words * 1.5))
            else:
                max_tokens = 1500 if section_type == "conclusion" else 500
            
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert academic writer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.6,
                max_tokens=max_tokens,
            )
            
            content = response.choices[0].message.content.strip()
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
        
        # Add sections in order
        section_order = ["introduction", "related_work", "method", "experiment", "result", "discussion", "conclusion"]
        section_titles = {
            "introduction": "Introduction",
            "related_work": "Related Work",
            "method": "Methodology",
            "experiment": "Experiments",
            "result": "Results",
            "discussion": "Discussion",
            "conclusion": "Conclusion",
        }
        
        for section_type in section_order:
            if section_type in sections and sections[section_type]:
                title = section_titles.get(section_type, section_type.title())
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
        # Handle \cite{key1, key2} and \cite{key} formats
        cite_pattern = r'\\cite\{([^}]+)\}'
        
        invalid_keys = []
        valid_keys_used = []
        
        def process_cite(match):
            cite_content = match.group(1)
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
                    return f'\\cite{{{", ".join(valid_in_cite)}}}'
                else:
                    # All keys invalid, remove entire citation
                    return ''
            else:
                return match.group(0)
        
        fixed_content = re.sub(cite_pattern, process_cite, content)
        
        # Clean up empty citations and dangling text
        # Remove patterns like "text \cite{} more" -> "text more"
        fixed_content = re.sub(r'\\cite\{\s*\}', '', fixed_content)
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
        import os
        
        if not paper_plan or not figures:
            return generated_sections
        
        # Build figure lookup
        figure_map = {fig.id: fig for fig in figures}
        
        for section in paper_plan.sections:
            section_type = section.section_type
            figures_to_define = getattr(section, 'figures_to_define', []) or []
            
            if not figures_to_define or section_type not in generated_sections:
                continue
            
            content = generated_sections[section_type]
            
            for fig_id in figures_to_define:
                fig = figure_map.get(fig_id)
                if not fig:
                    continue
                
                # Check if figure environment exists with this label
                # Look for \begin{figure} or \begin{figure*} with \label{fig_id}
                fig_pattern = rf'\\begin{{figure\*?}}.*?\\label{{{re.escape(fig_id)}}}.*?\\end{{figure\*?}}'
                if re.search(fig_pattern, content, re.DOTALL):
                    # Figure already defined
                    continue
                
                # Figure not defined - inject it
                print(f"[EnsureFigures] Injecting missing figure '{fig_id}' in '{section_type}'")
                
                # Determine environment and width based on wide flag
                env_name = "figure*" if fig.wide else "figure"
                width = "\\textwidth" if fig.wide else "0.9\\linewidth"
                
                # Get file path
                file_path = fig.file_path or ""
                filename = os.path.basename(file_path) if file_path else f"{fig_id.replace('fig:', '')}.pdf"
                
                # Build figure LaTeX
                figure_latex = f"""
\\begin{{{env_name}}}[t]
\\centering
\\includegraphics[width={width}]{{figures/{filename}}}
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
    
    def _mini_review_section(
        self,
        section_type: str,
        content: str,
        section_plan: Optional[SectionPlan],
        valid_citation_keys: set,
    ) -> Dict[str, Any]:
        """
        Perform mini-review on a generated section.
        
        Checks:
        - Word count vs target
        - Citation validity
        - Key points coverage (basic check)
        
        Args:
            section_type: Type of section
            content: Generated LaTeX content
            section_plan: Plan for this section (if available)
            valid_citation_keys: Set of valid citation keys
            
        Returns:
            Dict with review results and fixed content
        """
        print(f"[MiniReview] Starting review for '{section_type}'...")
        
        issues = []
        warnings = []
        
        # 1. Word count check
        print(f"[MiniReview:count_words] Checking word count...")
        word_count = len(content.split())
        target_words = section_plan.target_words if section_plan else None
        
        if target_words:
            min_words = int(target_words * 0.7)
            max_words = int(target_words * 1.3)
            
            if word_count < min_words:
                issues.append(f"Word count too low: {word_count} < {min_words} (target: {target_words})")
                print(f"[MiniReview:count_words] UNDER: {word_count} < {min_words} (target: {target_words})")
            elif word_count > max_words:
                warnings.append(f"Word count high: {word_count} > {max_words} (target: {target_words})")
                print(f"[MiniReview:count_words] OVER: {word_count} > {max_words} (target: {target_words})")
            else:
                print(f"[MiniReview:count_words] OK: {word_count} words (target: {target_words})")
        else:
            print(f"[MiniReview:count_words] Result: {word_count} words (no target)")
        
        # 2. Citation validation and fix
        print(f"[MiniReview:validate_citations] Checking citations against {len(valid_citation_keys)} valid keys...")
        fixed_content, invalid_citations, valid_citations = self._validate_and_fix_citations(
            content, valid_citation_keys, remove_invalid=True
        )
        
        if invalid_citations:
            issues.append(f"Removed {len(invalid_citations)} invalid citations: {invalid_citations[:5]}{'...' if len(invalid_citations) > 5 else ''}")
            print(f"[MiniReview:validate_citations] REMOVED: {invalid_citations}")
        else:
            print(f"[MiniReview:validate_citations] OK: {len(valid_citations)} valid citations")
        
        # 3. Key points coverage (basic keyword check)
        if section_plan and section_plan.key_points:
            print(f"[MiniReview:check_key_points] Checking {len(section_plan.key_points)} key points...")
            content_lower = content.lower()
            covered_points = 0
            for point in section_plan.key_points:
                # Extract key words from the point
                key_words = [w for w in point.lower().split() if len(w) > 4][:3]
                if any(kw in content_lower for kw in key_words):
                    covered_points += 1
            
            coverage_ratio = covered_points / len(section_plan.key_points) if section_plan.key_points else 1.0
            if coverage_ratio < 0.5:
                warnings.append(f"Low key point coverage: {covered_points}/{len(section_plan.key_points)}")
                print(f"[MiniReview:check_key_points] LOW: {covered_points}/{len(section_plan.key_points)} ({coverage_ratio:.0%})")
            else:
                print(f"[MiniReview:check_key_points] OK: {covered_points}/{len(section_plan.key_points)} ({coverage_ratio:.0%})")
        
        # 4. Figure/Table reference check
        if section_plan:
            if section_plan.figures_to_use:
                print(f"[MiniReview:check_figures] Checking {len(section_plan.figures_to_use)} expected figures...")
                missing_figs = []
                for fig_id in section_plan.figures_to_use:
                    if fig_id not in content and f'ref{{{fig_id}}}' not in content:
                        warnings.append(f"Missing expected figure reference: {fig_id}")
                        missing_figs.append(fig_id)
                if missing_figs:
                    print(f"[MiniReview:check_figures] MISSING: {missing_figs}")
                else:
                    print(f"[MiniReview:check_figures] OK: all figures referenced")
            
            if section_plan.tables_to_use:
                print(f"[MiniReview:check_tables] Checking {len(section_plan.tables_to_use)} expected tables...")
                missing_tbls = []
                for tbl_id in section_plan.tables_to_use:
                    if tbl_id not in content and f'ref{{{tbl_id}}}' not in content:
                        warnings.append(f"Missing expected table reference: {tbl_id}")
                        missing_tbls.append(tbl_id)
                if missing_tbls:
                    print(f"[MiniReview:check_tables] MISSING: {missing_tbls}")
                else:
                    print(f"[MiniReview:check_tables] OK: all tables referenced")
        
        # Log summary
        passed = len(issues) == 0
        print(f"[MiniReview] {section_type} - {'PASSED' if passed else 'FAILED'} (issues: {len(issues)}, warnings: {len(warnings)})")
        if issues:
            print(f"[MiniReview] {section_type} - Issues: {issues}")
        if warnings:
            print(f"[MiniReview] {section_type} - Warnings: {warnings}")
        
        return {
            "section_type": section_type,
            "original_word_count": word_count,
            "fixed_content": fixed_content,
            "fixed_word_count": len(fixed_content.split()),
            "issues": issues,
            "warnings": warnings,
            "invalid_citations_removed": invalid_citations,
            "valid_citations_used": valid_citations,
            "passed": len(issues) == 0,
        }
    
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
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Compile PDF using Typesetter Agent
        
        Args:
            generated_sections: Dict of section_type -> latex_content
            template_path: Path to .zip template file
            references: Parsed reference list
            output_dir: Output directory
            paper_title: Paper title
            figures_source_dir: Optional directory with figure files (legacy)
            figure_paths: Dict of figure_id -> file_path
            converted_tables: Dict of table_id -> LaTeX table code
            paper_plan: Optional paper plan with figure/table assignments
            figures: Optional list of figure specifications
        
        Returns:
            Tuple of (pdf_path, latex_path) or (None, None) on failure
        """
        print(f"[MetaDataAgent] Phase 4: Compiling PDF with template: {template_path}")
        
        try:
            # Build the sections content for Typesetter
            # The Typesetter expects content without \section{} commands
            section_order = ["introduction", "related_work", "method", "experiment", "result", "conclusion"]
            section_titles = {
                "introduction": "Introduction",
                "related_work": "Related Work", 
                "method": "Methodology",
                "experiment": "Experiments",
                "result": "Results",
                "conclusion": "Conclusion",
            }
            
            # Final citation validation pass (safety net)
            # This ensures no invalid citations slip through after Review Loop revisions
            valid_citation_keys = self._extract_valid_citation_keys(references)
            total_invalid_removed = 0
            for section_type in list(generated_sections.keys()):
                content = generated_sections[section_type]
                # Also apply LaTeX fixes (escaped comments, etc.)
                content = self._fix_latex_references(content)
                fixed_content, invalid, valid = self._validate_and_fix_citations(
                    content, valid_citation_keys, remove_invalid=True
                )
                if invalid:
                    print(f"[CompilePDF] Removed {len(invalid)} invalid citations from {section_type}: {invalid[:3]}{'...' if len(invalid) > 3 else ''}")
                    total_invalid_removed += len(invalid)
                generated_sections[section_type] = fixed_content
            
            if total_invalid_removed > 0:
                print(f"[CompilePDF] Total invalid citations removed: {total_invalid_removed}")
            
            # Ensure all assigned figures have their environments created
            if paper_plan and figures:
                generated_sections = self._ensure_figures_defined(
                    generated_sections=generated_sections,
                    paper_plan=paper_plan,
                    figures=figures,
                )
            
            # Build full latex content with section markers
            # The Typesetter expects content with markers like "% === Section: xxx ==="
            latex_parts = []
            
            # Add abstract with marker (Typesetter parses this)
            abstract_content = generated_sections.get("abstract", "")
            if abstract_content:
                latex_parts.append("% === Section: abstract ===")
                latex_parts.append(abstract_content)
            
            # Add body sections
            for section_type in section_order:
                if section_type in generated_sections and generated_sections[section_type]:
                    title = section_titles.get(section_type, section_type.title())
                    latex_parts.append(f"% === Section: {section_type} ===")
                    latex_parts.append(f"\\section{{{title}}}")
                    latex_parts.append(generated_sections[section_type])
            
            latex_body = "\n\n".join(latex_parts)
            
            # Prepare references for Typesetter
            typesetter_refs = []
            for ref in references:
                if ref.get("bibtex"):
                    typesetter_refs.append({
                        "ref_id": ref.get("ref_id", ""),
                        "bibtex": ref.get("bibtex"),
                    })
            
            # Call Typesetter Agent API
            # TypesetterPayload expects: request_id, user_id, payload (dict with actual data)
            # Note: abstract is embedded in latex_content with "% === Section: abstract ===" marker
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    "http://localhost:8000/agent/typesetter/compile",
                    json={
                        "request_id": str(uuid.uuid4()),
                        "payload": {
                            "latex_content": latex_body,
                            "template_path": template_path,
                            "template_config": {
                                "paper_title": paper_title,  # Use paper_title to match TemplateConfig
                            },
                            "references": typesetter_refs,
                            "output_dir": str(output_dir),
                            "figures_source_dir": figures_source_dir,
                            "figure_paths": figure_paths or {},  # Structured figure paths
                            "converted_tables": converted_tables or {},  # Pre-converted table LaTeX
                        }
                    }
                )
                
                if response.status_code != 200:
                    print(f"[MetaDataAgent] Typesetter error: {response.status_code} - {response.text}")
                    return None, None
                
                result = response.json()
                
                # TypesetterResult structure: {status, result: {pdf_path, source_path}, error}
                if result.get("status") == "ok" and result.get("result"):
                    compilation_result = result["result"]
                    pdf_path = compilation_result.get("pdf_path")
                    latex_path = compilation_result.get("source_path")
                    
                    if pdf_path:
                        print(f"[MetaDataAgent] PDF compiled successfully: {pdf_path}")
                    else:
                        print(f"[MetaDataAgent] PDF compilation failed: compilation result has no pdf_path")
                    
                    return pdf_path, latex_path
                else:
                    error_msg = result.get("error", "Unknown error")
                    print(f"[MetaDataAgent] PDF compilation failed: {error_msg}")
                    return None, None
                
        except httpx.ConnectError:
            print("[MetaDataAgent] Error: Could not connect to Typesetter Agent")
            return None, None
        except Exception as e:
            print(f"[MetaDataAgent] PDF compilation error: {e}")
            return None, None
    
    # =========================================================================
    # Phase 5: VLM Review
    # =========================================================================
    
    async def _call_vlm_review(
        self,
        pdf_path: str,
        page_limit: int = 8,
        template_type: str = "ICML",
        sections_info: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Call VLM Review Agent to check PDF for overflow/underfill/layout issues
        
        Args:
            pdf_path: Path to compiled PDF
            page_limit: Maximum allowed pages
            template_type: Template type for context
            sections_info: Optional section word counts for recommendations
            
        Returns:
            VLM review result dict or None on failure
        """
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    "http://localhost:8000/agent/vlm_review/review",
                    json={
                        "pdf_path": pdf_path,
                        "page_limit": page_limit,
                        "template_type": template_type,
                        "check_overflow": True,
                        "check_underfill": True,
                        "check_layout": False,  # Disable layout checks for now
                        "sections_info": sections_info or {},
                    }
                )
                
                if response.status_code != 200:
                    print(f"[MetaDataAgent] VLM Review error: {response.status_code}")
                    return None
                
                result = response.json()
                return result
                
        except httpx.ConnectError:
            print("[MetaDataAgent] VLM Review Agent not available, skipping")
            return None
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
                })
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "http://localhost:8000/agent/planner/plan",
                    json={
                        "title": metadata.title,
                        "idea_hypothesis": metadata.idea_hypothesis,
                        "method": metadata.method,
                        "data": metadata.data,
                        "experiments": metadata.experiments,
                        "references": metadata.references,
                        "figures": figures_info,
                        "tables": tables_info,
                        "target_pages": target_pages,
                        "style_guide": style_guide,
                    }
                )
                
                if response.status_code != 200:
                    print(f"[MetaDataAgent] Planner error: {response.status_code}")
                    return None
                
                result = response.json()
                
                if result.get("status") == "ok" and result.get("plan"):
                    plan_data = result["plan"]
                    return PaperPlan(**plan_data)
                else:
                    error_msg = result.get("error", "Unknown error")
                    print(f"[MetaDataAgent] Planning failed: {error_msg}")
                    return None
                
        except httpx.ConnectError:
            print("[MetaDataAgent] Planner Agent not available, using default structure")
            return None
        except Exception as e:
            print(f"[MetaDataAgent] Planning error: {e}")
            return None
    
    # =========================================================================
    # Phase 3.5: Review Loop Methods
    # =========================================================================
    
    def _build_vlm_feedback(
        self,
        vlm_result: Dict[str, Any],
    ) -> Tuple[List[FeedbackResult], List[SectionFeedback]]:
        """
        Build feedback and section revisions from VLM result.
        - **Description**:
            - Converts VLM review output into Reviewer-compatible feedback
            - Maps overflow/underfill to FeedbackResult and section advice to SectionFeedback
        
        - **Args**:
            - `vlm_result` (Dict[str, Any]): Raw VLM review result dict
        
        - **Returns**:
            - `feedbacks` (List[FeedbackResult]): Aggregated feedback results
            - `section_feedbacks` (List[SectionFeedback]): Per-section revision guidance
        """
        feedbacks: List[FeedbackResult] = []
        section_feedbacks: List[SectionFeedback] = []
        
        if not vlm_result:
            return feedbacks, section_feedbacks
        
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
        
        section_recommendations = vlm_result.get("section_recommendations", {}) or {}
        for section_type, advice in section_recommendations.items():
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
            
            revision_prompt = self._build_vlm_revision_prompt(
                section_type=section_type,
                action=action,
                delta_words=delta_words,
                guidance=guidance,
            )
            
            section_feedbacks.append(SectionFeedback(
                section_type=section_type,
                current_word_count=0,
                target_word_count=0,
                action=action,
                delta_words=delta_words,
                revision_prompt=revision_prompt,
            ))
        
        return feedbacks, section_feedbacks
    
    def _build_vlm_revision_prompt(
        self,
        section_type: str,
        action: str,
        delta_words: int,
        guidance: Optional[str] = None,
    ) -> str:
        """
        Build revision prompt from VLM guidance.
        - **Description**:
            - Produces a concise revision instruction for the writer agent
            - Ensures prompt is compatible with _revise_section()
        
        - **Args**:
            - `section_type` (str): Section name to revise
            - `action` (str): "reduce" or "expand"
            - `delta_words` (int): Target word change (+/-)
            - `guidance` (Optional[str]): Extra VLM guidance
        
        - **Returns**:
            - `prompt` (str): Revision instruction prompt
        """
        action_text = "reduce" if action == "reduce" else "expand"
        delta = abs(delta_words) if delta_words else 0
        guidance_text = f"Guidance: {guidance}" if guidance else ""
        
        return (
            f"Revise the {section_type} section to {action_text} by approximately {delta} words. "
            "Preserve factual consistency, citations, and LaTeX formatting. "
            "Prioritize trimming low-impact details or expanding evidence/details. "
            f"{guidance_text}"
        ).strip()
    
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
            
            if existing.action != fb.action and prefer_vlm:
                merged[fb.section_type] = fb
            elif existing.action == fb.action and abs(fb.delta_words) > abs(existing.delta_words):
                merged[fb.section_type] = fb
        
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
    ) -> set:
        """
        Apply revisions based on a unified review result.
        - **Description**:
            - Uses review_result.section_feedbacks to revise sections
            - Updates generated_sections and sections_results in place
        
        - **Args**:
            - `review_result` (ReviewResult): Unified review result
            - `generated_sections` (Dict[str, str]): Section contents
            - `sections_results` (List[SectionResult]): Section results to update
            - `valid_citation_keys` (set): Valid citation keys
            - `metadata` (PaperMetaData): Original metadata for context
        
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
            
            revised_content = await self._revise_section(
                section_type=sf.section_type,
                current_content=generated_sections[sf.section_type],
                revision_prompt=revision_prompt,
                metadata=metadata,
            )
            
            if revised_content:
                revised_content = self._fix_latex_references(revised_content)
                revised_content, invalid_citations, _ = self._validate_and_fix_citations(
                    revised_content, valid_citation_keys, remove_invalid=True
                )
                if invalid_citations:
                    print(f"[ReviewLoop] Removed {len(invalid_citations)} invalid citations from {sf.section_type}: {invalid_citations[:3]}{'...' if len(invalid_citations) > 3 else ''}")
                
                generated_sections[sf.section_type] = revised_content
                new_word_count = len(revised_content.split())
                
                for sr in sections_results:
                    if sr.section_type == sf.section_type:
                        sr.latex_content = revised_content
                        sr.word_count = new_word_count
                        break
                
                revised_sections.add(sf.section_type)
                print(f"[MetaDataAgent] Revised {sf.section_type}: {new_word_count} words")
        
        return revised_sections
    
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
        last_vlm_result = None
        
        if enable_review:
            print(f"[MetaDataAgent] Unified Review Loop (max {max_review_iterations} iterations)...")
        
        for iteration in range(max_review_iterations):
            review_iterations = iteration + 1
            print(f"[MetaDataAgent] Review iteration {review_iterations}/{max_review_iterations}")
            
            word_counts = {
                sr.section_type: sr.word_count
                for sr in sections_results
                if sr.status == "ok"
            }
            
            review_result = ReviewResult(iteration=iteration)
            if enable_review:
                reviewer_result, target_word_count = await self._call_reviewer(
                    sections=generated_sections,
                    word_counts=word_counts,
                    target_pages=target_pages,
                    style_guide=metadata.style_guide,
                    template_path=template_path,
                    iteration=iteration,
                )
                if reviewer_result is None:
                    print("[MetaDataAgent] Reviewer not available, skipping content review")
                else:
                    review_result = ReviewResult(**reviewer_result)
            
            reviewer_revised_sections = await self._apply_revisions(
                review_result=review_result,
                generated_sections=generated_sections,
                sections_results=sections_results,
                valid_citation_keys=self._extract_valid_citation_keys(parsed_refs),
                metadata=metadata,
            )
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
            
            # Compile PDF and run VLM review if enabled
            if compile_pdf and template_path and paper_dir:
                figure_base_path = os.getcwd()
                figure_paths = self._collect_figure_paths(metadata.figures, base_path=figure_base_path)
                pdf_result_path, _ = await self._compile_pdf(
                    generated_sections=generated_sections,
                    template_path=template_path,
                    references=parsed_refs,
                    output_dir=paper_dir,
                    paper_title=metadata.title,
                    figures_source_dir=figures_source_dir,
                    figure_paths=figure_paths,
                    converted_tables=converted_tables,
                    paper_plan=paper_plan,
                    figures=metadata.figures,
                )
                if pdf_result_path:
                    pdf_path = pdf_result_path
                else:
                    errors.append("PDF compilation failed")
                    break
                
                if enable_vlm_review and pdf_path:
                    print(f"[MetaDataAgent] VLM Review...")
                    last_vlm_result = await self._call_vlm_review(
                        pdf_path=pdf_path,
                        page_limit=target_pages or 8,
                        template_type=metadata.style_guide or "ICML",
                        sections_info={
                            sr.section_type: {"word_count": sr.word_count}
                            for sr in sections_results if sr.word_count
                        },
                    )
                    if last_vlm_result:
                        vlm_feedbacks, vlm_section_feedbacks = self._build_vlm_feedback(last_vlm_result)
                        for fb in vlm_feedbacks:
                            review_result.add_feedback(fb)
                        
                        prefer_vlm = bool(last_vlm_result.get("needs_trim") or last_vlm_result.get("needs_expand"))
                        merged_section_feedbacks = self._merge_section_feedbacks(
                            review_result.section_feedbacks,
                            vlm_section_feedbacks,
                            prefer_vlm=prefer_vlm,
                        )
                        review_result.section_feedbacks = merged_section_feedbacks
                        for sf in review_result.section_feedbacks:
                            if sf.section_type in word_counts:
                                sf.current_word_count = word_counts.get(sf.section_type, 0)
                            if paper_plan:
                                section_plan = paper_plan.get_section(sf.section_type)
                                if section_plan and section_plan.target_words:
                                    sf.target_word_count = section_plan.target_words
                        
                        for sf in merged_section_feedbacks:
                            if sf.action != "ok":
                                review_result.add_section_revision(sf.section_type, "VLM adjustment")
                    else:
                        print("[MetaDataAgent] VLM review unavailable, skipping")
            elif enable_vlm_review:
                errors.append("VLM review skipped: PDF not compiled (missing template or output path)")
            
            vlm_revised_sections = await self._apply_revisions(
                review_result=review_result,
                generated_sections=generated_sections,
                sections_results=sections_results,
                valid_citation_keys=self._extract_valid_citation_keys(parsed_refs),
                metadata=metadata,
            )
            self._resolve_section_feedbacks(
                section_feedbacks=review_result.section_feedbacks,
                revised_sections=vlm_revised_sections,
                review_result=review_result,
            )
            
            current_fingerprint = self._get_sections_fingerprint(generated_sections)
            if current_fingerprint == last_fingerprint:
                if review_result.passed and (not last_vlm_result or last_vlm_result.get("passed", True)):
                    print("[MetaDataAgent] Review passed with no further changes")
                else:
                    if last_vlm_result and not last_vlm_result.get("passed", True):
                        errors.append(last_vlm_result.get("summary", "VLM review failed"))
                break
            
            last_fingerprint = current_fingerprint
            if not reviewer_revised_sections and not vlm_revised_sections and review_result.passed and (not last_vlm_result or last_vlm_result.get("passed", True)):
                break
        
        return generated_sections, sections_results, review_iterations, target_word_count, pdf_path, errors
    
    async def _call_reviewer(
        self,
        sections: Dict[str, str],
        word_counts: Dict[str, int],
        target_pages: Optional[int],
        style_guide: Optional[str],
        template_path: Optional[str],
        iteration: int,
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int]]:
        """
        Call Reviewer Agent to check the paper
        
        Returns:
            Tuple of (review_result_dict, target_word_count) or (None, None) on failure
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "http://localhost:8000/agent/reviewer/review",
                    json={
                        "sections": sections,
                        "word_counts": word_counts,
                        "target_pages": target_pages,
                        "style_guide": style_guide,
                        "template_path": template_path,
                        "metadata": {},
                        "iteration": iteration,
                    }
                )
                
                if response.status_code != 200:
                    print(f"[MetaDataAgent] Reviewer error: {response.status_code}")
                    return None, None
                
                result = response.json()
                
                # Extract target word count from feedback details
                target_word_count = None
                for fb in result.get("feedbacks", []):
                    details = fb.get("details", {})
                    if "target_words" in details:
                        target_word_count = details["target_words"]
                        break
                
                return result, target_word_count
                
        except httpx.ConnectError:
            print("[MetaDataAgent] Reviewer Agent not available")
            return None, None
        except Exception as e:
            print(f"[MetaDataAgent] Review error: {e}")
            return None, None
    
    async def _revise_section(
        self,
        section_type: str,
        current_content: str,
        revision_prompt: str,
        metadata: PaperMetaData,
    ) -> Optional[str]:
        """
        Revise a section based on feedback
        
        Args:
            section_type: Type of section to revise
            current_content: Current section content
            revision_prompt: Prompt with revision instructions
            metadata: Paper metadata for context
            
        Returns:
            Revised content or None on failure
        """
        try:
            system_prompt = """You are an expert academic writer revising a paper section.
Follow the revision instructions carefully to improve the content.
Maintain academic writing quality and the original structure.
Output ONLY the revised LaTeX content, no explanations."""

            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": revision_prompt},
                ],
                temperature=0.7,
            )
            
            revised_content = response.choices[0].message.content.strip()
            
            # Clean up any markdown code blocks
            if revised_content.startswith("```"):
                lines = revised_content.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                revised_content = "\n".join(lines)
            
            return revised_content
            
        except Exception as e:
            print(f"[MetaDataAgent] Revision error for {section_type}: {e}")
            return None
