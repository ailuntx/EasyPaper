"""
Router for Writer Agent endpoints
"""
from fastapi import APIRouter, HTTPException, status
from typing import Optional, Any, Dict
from enum import Enum
import time
import logging
import os
import json
from datetime import datetime
from .models import WriterPayload, WriterResult, GeneratedContent
from typing import List
from .section_models import (
    SectionWritePayload,
    SectionWriteResult,
    SectionResources,
    validate_section_payload,
    get_section_requirements,
    # Paper assembly models
    SectionChainItem,
    PaperChainConfig,
    PaperAssemblyResult,
    SectionGenerationStatus,
)


class SaveType(str, Enum):
    """Output save format types"""
    JSON = "json"       # Full result as JSON
    TXT = "txt"         # Plain text content only
    LATEX = "latex"     # LaTeX content with .tex extension
    TEX = "tex"         # Alias for latex
    MD = "md"           # Markdown format


# Default output directory for saved files
DEFAULT_OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "outputs")


def _save_result(
    result: SectionWriteResult,
    save_type: SaveType,
    save_path: Optional[str] = None,
    logger: logging.Logger = None
) -> str:
    """
    Save the result to a file
    - **Args**:
        - `result` (SectionWriteResult): The result to save
        - `save_type` (SaveType): Output format
        - `save_path` (str, optional): Custom save path

    - **Returns**:
        - `str`: The path where the file was saved
    """
    # Determine file extension
    ext_map = {
        SaveType.JSON: ".json",
        SaveType.TXT: ".txt",
        SaveType.LATEX: ".tex",
        SaveType.TEX: ".tex",
        SaveType.MD: ".md",
    }
    ext = ext_map.get(save_type, ".txt")
    
    # Generate default filename if path not provided
    if not save_path:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.section_type}_{timestamp}{ext}"
        save_path = os.path.join(DEFAULT_OUTPUT_DIR, filename)
    else:
        # Replace extension based on save_type
        base_path, _ = os.path.splitext(save_path)
        save_path = base_path + ext
        
        # Ensure directory exists
        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
    
    # Format content based on save type
    if save_type == SaveType.JSON:
        content = result.model_dump_json(indent=2)
    elif save_type in (SaveType.LATEX, SaveType.TEX):
        # LaTeX format with section header
        content = f"% Section: {result.section_title}\n"
        content += f"% Generated: {datetime.now().isoformat()}\n"
        content += f"% Word count: {result.word_count}\n\n"
        content += result.latex_content
    elif save_type == SaveType.MD:
        # Markdown format
        content = f"# {result.section_title}\n\n"
        content += f"**Section Type:** {result.section_type}  \n"
        content += f"**Word Count:** {result.word_count}  \n"
        content += f"**Generated:** {datetime.now().isoformat()}  \n\n"
        content += "---\n\n"
        content += "```latex\n"
        content += result.latex_content
        content += "\n```\n"
    else:  # TXT
        content = result.latex_content
    
    # Write to file
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    if logger:
        logger.info("writer.save_result path=%s type=%s", save_path, save_type.value)
    
    return save_path


def create_writer_router(agent_instance):
    """
    Create router for writer agent endpoints
    - **Args**:
        - `agent_instance`: The WriterAgent instance

    - **Returns**:
        - `APIRouter`: FastAPI router with writer endpoints
    """
    router = APIRouter()
    logger = logging.getLogger("uvicorn.error")

    @router.post("/agent/writer/generate", response_model=WriterResult, status_code=status.HTTP_200_OK)
    async def generate_section(payload: WriterPayload):
        """
        Generate LaTeX content for a paper section
        - **Description**:
            - Takes compiled prompt and generates LaTeX content
            - Returns content with extracted citation and figure references

        - **Args**:
            - `payload` (WriterPayload): Request payload with prompt info

        - **Returns**:
            - `WriterResult`: Generated content or error
        """
        start = time.time()
        logger.info("writer.generate.request %s user=%s", payload.request_id, payload.user_id)

        try:
            # Extract parameters from payload
            system_prompt = payload.payload.get("system_prompt", "")
            user_prompt = payload.payload.get("user_prompt", "")
            section_type = payload.payload.get("section_type", "introduction")
            citation_format = payload.payload.get("citation_format", "cite")
            constraints = payload.payload.get("constraints", [])

            if not user_prompt:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="user_prompt must be provided"
                )

            # Run the agent with new iterative review fields
            agent_result = await agent_instance.run(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                section_type=section_type,
                citation_format=citation_format,
                constraints=constraints,
                # New iterative review parameters
                valid_citation_keys=payload.valid_citation_keys,
                target_words=payload.target_words,
                key_points=payload.key_points,
                max_iterations=payload.max_iterations,
                enable_review=payload.enable_review,
            )

            # Extract results
            generated_content = agent_result.get("generated_content", "")
            citation_ids = agent_result.get("citation_ids", [])
            figure_ids = agent_result.get("figure_ids", [])
            table_ids = agent_result.get("table_ids", [])
            invalid_citations_removed = agent_result.get("invalid_citations_removed", [])
            review_history = agent_result.get("review_history", [])

            # Calculate word count (rough estimate)
            word_count = len(generated_content.split())

            # Build result with review info
            result = GeneratedContent(
                latex_content=generated_content,
                section_type=section_type,
                word_count=word_count,
                citation_ids=citation_ids,
                figure_ids=figure_ids,
                table_ids=table_ids,
                iterations_used=agent_result.get("iteration", 1),
                review_passed=agent_result.get("review_result", {}).get("passed", True),
                invalid_citations_removed=invalid_citations_removed,
            )

            latency = time.time() - start
            logger.info("writer.generate.complete %s latency=%.3f words=%d iterations=%d", 
                       payload.request_id, latency, word_count, agent_result.get("iteration", 1))

            # Convert review history to ReviewResult models
            from .models import ReviewResult as ReviewResultModel
            review_history_models = []
            for rh in review_history:
                review_history_models.append(ReviewResultModel(
                    passed=rh.get("passed", True),
                    issues=rh.get("issues", []),
                    warnings=rh.get("warnings", []),
                    invalid_citations=rh.get("invalid_citations", []),
                    word_count=rh.get("word_count", 0),
                    target_words=rh.get("target_words"),
                    key_point_coverage=rh.get("key_point_coverage", 1.0),
                ))

            return WriterResult(
                request_id=payload.request_id,
                status="ok",
                result=result,
                review_history=review_history_models,
            )

        except Exception as e:
            latency = time.time() - start
            logger.error("writer.generate.error %s latency=%.3f error=%s", 
                        payload.request_id, latency, str(e))
            return WriterResult(
                request_id=payload.request_id,
                status="error",
                error=str(e)
            )

    @router.post("/agent/writer/write-section", response_model=SectionWriteResult, status_code=status.HTTP_200_OK)
    async def write_section(
        payload: SectionWritePayload,
        validate: bool = True,
        save: bool = False,
        save_type: SaveType = SaveType.JSON,
        save_path: Optional[str] = None,
    ):
        """
        Direct section writing API
        - **Description**:
            - Accepts complete SectionWritePayload with context
            - Validates required resources before writing
            - Generates LaTeX content for standalone API usage
            - Independent of FlowGram.ai - any system can call this endpoint
            - Optionally saves output to file

        - **Args**:
            - `payload` (SectionWritePayload): Complete writing payload with context
            - `validate` (bool): Whether to validate required resources (default: True)
            - `save` (bool): Whether to save output to file (default: False)
            - `save_type` (SaveType): Output format - json, txt, latex/tex, md (default: json)
            - `save_path` (str, optional): Custom save path (default: auto-generated in outputs/)

        - **Returns**:
            - `SectionWriteResult`: Generated LaTeX content and metadata
        """
        start = time.time()
        request_id = payload.request_id or "unknown"
        logger.info("writer.write_section.request %s section=%s save=%s", 
                   request_id, payload.section_type, save)

        try:
            # Validate payload if requested
            if validate:
                validation = validate_section_payload(payload)
                if not validation.is_valid:
                    error_msg = "; ".join(validation.errors)
                    logger.warning("writer.write_section.validation_failed %s errors=%s", 
                                  request_id, error_msg)
                    return SectionWriteResult(
                        request_id=request_id,
                        status="error",
                        section_type=payload.section_type,
                        error=f"Validation failed: {error_msg}"
                    )
                
                # Log warnings but continue
                if validation.warnings:
                    for warning in validation.warnings:
                        logger.info("writer.write_section.warning %s %s", request_id, warning)

            # Build system prompt from provided context
            system_prompt = _build_system_prompt_from_context(payload)
            
            # Expand template syntax in user_prompt
            user_prompt = payload.user_prompt or f"Write the {payload.section_type} section."
            user_prompt = _expand_template_syntax(user_prompt, payload)
            
            # Run the agent
            agent_result = await agent_instance.run(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                section_type=payload.section_type,
                citation_format=payload.constraints.citation_format,
                constraints=payload.constraints.additional_instructions,
            )

            # Extract results
            generated_content = agent_result.get("generated_content", "")
            citation_ids = agent_result.get("citation_ids", [])
            figure_ids = agent_result.get("figure_ids", [])
            table_ids = agent_result.get("table_ids", [])

            # Calculate word count
            word_count = len(generated_content.split())

            latency = time.time() - start
            logger.info("writer.write_section.complete %s latency=%.3f words=%d", 
                       request_id, latency, word_count)

            # Build result
            result = SectionWriteResult(
                request_id=request_id,
                status="ok",
                section_type=payload.section_type,
                section_title=payload.section_title or payload.section_type.replace("_", " ").title(),
                latex_content=generated_content,
                word_count=word_count,
                citation_ids=citation_ids,
                figure_ids=figure_ids,
                table_ids=table_ids,
            )

            # Save to file if requested
            if save:
                saved_path = _save_result(result, save_type, save_path, logger)
                result.saved_path = saved_path

            return result

        except Exception as e:
            latency = time.time() - start
            logger.error("writer.write_section.error %s latency=%.3f error=%s", 
                        request_id, latency, str(e))
            return SectionWriteResult(
                request_id=request_id,
                status="error",
                section_type=payload.section_type,
                error=str(e)
            )

    @router.get("/agent/writer/requirements/{section_type}")
    async def get_requirements(section_type: str):
        """
        Get resource requirements for a section type
        - **Description**:
            - Returns the required and optional resources for a section type
            - Clients can use this to understand what context to provide

        - **Args**:
            - `section_type` (str): The section type (abstract, introduction, etc.)

        - **Returns**:
            - `ResourceRequirement`: Requirements for this section type
        """
        requirements = get_section_requirements(section_type)
        return {
            "section_type": section_type,
            "requirements": requirements.model_dump(),
        }

    @router.post("/agent/writer/validate")
    async def validate_payload(payload: SectionWritePayload):
        """
        Validate a SectionWritePayload without writing
        - **Description**:
            - Pre-validates payload before submitting for writing
            - Returns detailed validation result

        - **Args**:
            - `payload` (SectionWritePayload): The payload to validate

        - **Returns**:
            - `ValidationResult`: Validation result with errors and warnings
        """
        validation = validate_section_payload(payload)
        return {
            "section_type": payload.section_type,
            "validation": validation.model_dump(),
        }

    # Add paper assembly endpoint
    create_paper_assembly_endpoint(router, agent_instance, logger)

    return router


def _expand_template_syntax(text: str, payload: SectionWritePayload) -> str:
    """
    Expand template syntax in user_prompt
    - **Description**:
        - Supports {{point:id}}, {{ref:id}}, {{fig:id}}, {{eq:id}} syntax
        - Expands references to their content or citation commands

    - **Args**:
        - `text` (str): Text with template syntax
        - `payload` (SectionWritePayload): Payload with points and resources

    - **Returns**:
        - `str`: Expanded text
    """
    import re
    
    # Build lookup dictionaries
    points_dict = {}
    def collect_points(points, prefix=""):
        for point in points:
            points_dict[point.id] = point.statement
            collect_points(point.sub_points)
    collect_points(payload.argument.main_points)
    
    refs_dict = {r.ref_id: r.title for r in payload.resources.references}
    figs_dict = {f.figure_id: f.title for f in payload.resources.figures}
    eqs_dict = {e.equation_id: e.title for e in payload.resources.equations}
    tables_dict = {t.table_id: t.title for t in payload.resources.tables}
    
    # Expand {{point:id}} -> "Point statement"
    def replace_point(match):
        point_id = match.group(1)
        return points_dict.get(point_id, f"[POINT:{point_id}]")
    text = re.sub(r'\{\{point:([^}]+)\}\}', replace_point, text)
    
    # Expand {{ref:id}} -> \cite{id} (or just mention)
    def replace_ref(match):
        ref_id = match.group(1)
        title = refs_dict.get(ref_id, "")
        return f"\\cite{{{ref_id}}}" if ref_id in refs_dict else f"[REF:{ref_id}]"
    text = re.sub(r'\{\{ref:([^}]+)\}\}', replace_ref, text)
    
    # Expand {{fig:id}} -> Figure~\ref{id}
    def replace_fig(match):
        fig_id = match.group(1)
        return f"Figure~\\ref{{{fig_id}}}" if fig_id in figs_dict else f"[FIG:{fig_id}]"
    text = re.sub(r'\{\{fig:([^}]+)\}\}', replace_fig, text)
    
    # Expand {{eq:id}} -> Equation~\ref{id}
    def replace_eq(match):
        eq_id = match.group(1)
        return f"Equation~\\ref{{{eq_id}}}" if eq_id in eqs_dict else f"[EQ:{eq_id}]"
    text = re.sub(r'\{\{eq:([^}]+)\}\}', replace_eq, text)
    
    # Expand {{table:id}} -> Table~\ref{id}
    def replace_table(match):
        table_id = match.group(1)
        return f"Table~\\ref{{{table_id}}}" if table_id in tables_dict else f"[TABLE:{table_id}]"
    text = re.sub(r'\{\{table:([^}]+)\}\}', replace_table, text)
    
    return text


def _format_material_with_links(mat, resources, citation_format: str = "cite") -> str:
    """
    Format a material with its linked resources
    - **Description**:
        - Includes linked_refs, linked_figures, etc. in the output

    - **Args**:
        - `mat`: Material object
        - `resources`: SectionResources object
        - `citation_format`: Citation command to use

    - **Returns**:
        - `str`: Formatted material string with links
    """
    parts = []
    mat_str = f"[{mat.material_type.upper()}] {mat.title or mat.id}"
    
    # Add linked references
    if mat.linked_refs:
        refs_str = ", ".join(f"\\{citation_format}{{{r}}}" for r in mat.linked_refs)
        mat_str += f" (Cite: {refs_str})"
    
    # Add linked figures
    if mat.linked_figures:
        figs_str = ", ".join(f"Figure~\\ref{{{f}}}" for f in mat.linked_figures)
        mat_str += f" (See: {figs_str})"
    
    # Add linked tables
    if mat.linked_tables:
        tbls_str = ", ".join(f"Table~\\ref{{{t}}}" for t in mat.linked_tables)
        mat_str += f" (See: {tbls_str})"
    
    # Add linked equations
    if mat.linked_equations:
        eqs_str = ", ".join(f"Eq.~\\ref{{{e}}}" for e in mat.linked_equations)
        mat_str += f" (See: {eqs_str})"
    
    return mat_str


def _build_system_prompt_from_context(payload: SectionWritePayload) -> str:
    """
    Build a system prompt from SectionWritePayload with argument structure
    - **Description**:
        - Converts argument tree structure into a formatted system prompt
        - Expresses points and their supporting materials clearly
        - Includes explicit resource links from materials/points
        - Expands template syntax in user_prompt
        - Used for standalone Writer API calls

    - **Args**:
        - `payload` (SectionWritePayload): The writing payload with argument structure

    - **Returns**:
        - `str`: Formatted system prompt
    """
    from .section_models import Point, Material
    
    # Section-specific base prompts
    section_prompts = {
        "abstract": "You are writing the Abstract section of a research paper. Keep it concise (150-250 words), summarizing problem, methods, results, and conclusions.",
        "introduction": "You are writing the Introduction section. Establish context, identify the problem, state objectives, and outline the paper structure.",
        "related_work": "You are writing the Related Work section. Survey prior work systematically, identify gaps, and differentiate your contribution.",
        "background": "You are writing the Background section. Provide necessary context and definitions for understanding the work.",
        "method": "You are writing the Method section. Describe your approach in detail for reproduction, include formal definitions and algorithms.",
        "experiment": "You are writing the Experiment section. Describe setup, datasets, metrics, baselines, and evaluation protocols.",
        "result": "You are writing the Results section. Present results objectively, use tables/figures, compare with baselines.",
        "analysis": "You are writing the Analysis section. Interpret results, perform ablations, analyze patterns.",
        "discussion": "You are writing the Discussion section. Interpret results, discuss implications, address limitations, suggest future work.",
        "conclusion": "You are writing the Conclusion section. Summarize contributions, restate key findings, discuss broader impact.",
    }
    
    citation_format = payload.constraints.citation_format
    base_prompt = section_prompts.get(payload.section_type, section_prompts.get("introduction", ""))
    
    # Build argument structure representation
    argument_parts = []
    argument = payload.argument
    resources = payload.resources
    
    # Thesis statement
    if argument.thesis:
        argument_parts.append(f"=== THESIS ===\n{argument.thesis}")
    
    # Format a point recursively
    def format_point(point: Point, level: int = 0) -> str:
        indent = "  " * level
        parts = []
        
        # Point statement
        point_label = point.point_type.upper()
        point_header = f"{indent}[{point_label} POINT: {point.id}] {point.statement}"
        
        # Add linked refs at point level
        if point.linked_refs:
            refs_str = ", ".join(f"\\{citation_format}{{{r}}}" for r in point.linked_refs)
            point_header += f" (Cite: {refs_str})"
        
        parts.append(point_header)
        
        # Supporting materials
        if point.supporting_materials:
            parts.append(f"{indent}  Supporting Materials:")
            for mat in point.supporting_materials:
                mat_str = _format_material_with_links(mat, resources, citation_format)
                parts.append(f"{indent}    - {mat_str}")
                if mat.content:
                    parts.append(f"{indent}      Content: {mat.content[:400]}")
        
        # Counter materials
        if point.counter_materials:
            parts.append(f"{indent}  Counter/Challenge Materials:")
            for mat in point.counter_materials:
                mat_str = _format_material_with_links(mat, resources, citation_format)
                parts.append(f"{indent}    - {mat_str}")
                if mat.content:
                    parts.append(f"{indent}      Content: {mat.content[:300]}")
        
        # Sub-points
        for sub_point in point.sub_points:
            relation = sub_point.relation_to_parent or "supports"
            parts.append(f"{indent}  [{relation.upper()}]:")
            parts.append(format_point(sub_point, level + 1))
        
        return "\n".join(parts)
    
    # Main points
    if argument.main_points:
        argument_parts.append("\n=== ARGUMENT STRUCTURE ===")
        for i, point in enumerate(argument.main_points, 1):
            argument_parts.append(f"\n--- Point {i} ---")
            argument_parts.append(format_point(point))
    
    # Background context
    if argument.background_context:
        argument_parts.append("\n=== BACKGROUND CONTEXT ===")
        for mat in argument.background_context:
            mat_str = _format_material_with_links(mat, resources, citation_format)
            argument_parts.append(mat_str)
            if mat.content:
                argument_parts.append(f"  Content: {mat.content[:400]}")
    
    argument_summary = "\n".join(argument_parts)
    
    # Build resources section
    resource_parts = []
    
    # Figures
    if resources.figures:
        resource_parts.append("\n=== AVAILABLE FIGURES ===")
        for fig in resources.figures:
            fig_str = f"- [FIGURE:{fig.figure_id}] {fig.title}"
            if fig.caption:
                fig_str += f"\n    Caption: {fig.caption[:200]}"
            resource_parts.append(fig_str)
    
    # Tables
    if resources.tables:
        resource_parts.append("\n=== AVAILABLE TABLES ===")
        for tbl in resources.tables:
            tbl_str = f"- [TABLE:{tbl.table_id}] {tbl.title}"
            if tbl.caption:
                tbl_str += f"\n    Caption: {tbl.caption[:200]}"
            resource_parts.append(tbl_str)
    
    # Equations
    if resources.equations:
        resource_parts.append("\n=== AVAILABLE EQUATIONS ===")
        for eq in resources.equations:
            eq_str = f"- [EQUATION:{eq.equation_id}] {eq.title}"
            if eq.latex:
                eq_str += f"\n    LaTeX: {eq.latex[:100]}"
            resource_parts.append(eq_str)
    
    # References - with explicit list of allowed citation keys
    allowed_citation_keys = []
    if resources.references:
        resource_parts.append("\n=== AVAILABLE REFERENCES (ONLY USE THESE CITATION KEYS) ===")
        for ref in resources.references[:20]:
            allowed_citation_keys.append(ref.ref_id)
            ref_str = f"- [{ref.ref_id}] {ref.title or '(BibTeX provided)'}"
            if ref.authors:
                ref_str += f" by {ref.authors}"
            if ref.year:
                ref_str += f" ({ref.year})"
            resource_parts.append(ref_str)
    
    resources_summary = "\n".join(resource_parts)
    
    # Build constraints
    constraints = [
        "Generate only LaTeX body content, no document preamble",
        f"Use \\{citation_format}{{ref_id}} for citations",
        "Use \\includegraphics{figure_id} for figures",
        "Use \\ref{table_id} for table references",
        "Maintain academic writing style and rigor",
        "Follow the argument structure provided - address each point systematically",
        "When a material or point has linked resources, use those citations/references",
    ]
    
    # Add explicit citation constraint with allowed keys
    if allowed_citation_keys:
        constraints.append(
            f"**CRITICAL**: You may ONLY use these citation keys: {', '.join(allowed_citation_keys)}. "
            "DO NOT use any other citation keys. DO NOT use placeholders like \\cite{{need_citation}}. "
            "If you need to cite work not in this list, simply omit the \\cite command entirely."
        )
    
    if payload.constraints.word_count_limit:
        constraints.append(f"Target word count: approximately {payload.constraints.word_count_limit} words")
    
    if payload.constraints.style_guide:
        constraints.append(f"Follow {payload.constraints.style_guide} style guidelines")
    
    constraints.extend(payload.constraints.additional_instructions)
    
    # Compile full prompt
    system_prompt = f"""{base_prompt}

IMPORTANT CONSTRAINTS:
{chr(10).join(f'- {c}' for c in constraints)}

{argument_summary}
{resources_summary}
"""
    
    return system_prompt


# =============================================================================
# Paper Assembly Endpoint
# =============================================================================

def _load_section_payload(
    item: SectionChainItem,
    base_path: Optional[str] = None,
    shared_resources: Optional[SectionResources] = None,
    logger: logging.Logger = None,
) -> SectionWritePayload:
    """
    Load section payload from file or use inline payload
    - **Description**:
        - Loads payload from JSON file if payload_file is specified
        - Uses inline payload if provided
        - Merges shared_resources into the payload

    - **Args**:
        - `item` (SectionChainItem): Section chain item
        - `base_path` (str, optional): Base path for resolving relative paths
        - `shared_resources` (SectionResources, optional): Shared resources to merge

    - **Returns**:
        - `SectionWritePayload`: Loaded and merged payload
    """
    payload = None
    
    if item.payload:
        # Use inline payload
        payload = item.payload
    elif item.payload_file:
        # Load from file
        file_path = item.payload_file
        if base_path and not os.path.isabs(file_path):
            file_path = os.path.join(base_path, file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Payload file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        payload = SectionWritePayload(**data)
        if logger:
            logger.info("paper_assembly.load_payload file=%s", file_path)
    else:
        raise ValueError(f"Section '{item.section_type}' has neither payload nor payload_file")
    
    # Merge shared resources if provided
    if shared_resources and payload:
        existing_refs = {r.ref_id for r in payload.resources.references}
        for ref in shared_resources.references:
            if ref.ref_id not in existing_refs:
                payload.resources.references.append(ref)
        
        existing_figs = {f.figure_id for f in payload.resources.figures}
        for fig in shared_resources.figures:
            if fig.figure_id not in existing_figs:
                payload.resources.figures.append(fig)
        
        existing_tables = {t.table_id for t in payload.resources.tables}
        for table in shared_resources.tables:
            if table.table_id not in existing_tables:
                payload.resources.tables.append(table)
        
        existing_eqs = {e.equation_id for e in payload.resources.equations}
        for eq in shared_resources.equations:
            if eq.equation_id not in existing_eqs:
                payload.resources.equations.append(eq)
    
    return payload


def _assemble_latex_document(
    sections: List[SectionGenerationStatus],
    paper_title: str,
) -> str:
    """
    Assemble complete LaTeX document from sections
    - **Description**:
        - Combines all section LaTeX content into a complete document
        - Adds document structure if needed

    - **Args**:
        - `sections` (List[SectionGenerationStatus]): Generated section results
        - `paper_title` (str): Paper title

    - **Returns**:
        - `str`: Complete LaTeX document body
    """
    parts = []
    
    # Add title (optional, template may handle this)
    parts.append(f"% Paper: {paper_title}")
    parts.append(f"% Generated: {datetime.now().isoformat()}")
    parts.append("")
    
    for section in sections:
        if section.status == 'ok' and section.latex_content:
            parts.append(f"% === Section: {section.section_type} ===")
            parts.append(section.latex_content)
            parts.append("")
    
    return "\n".join(parts)


async def _generate_section_internal(
    agent_instance,
    payload: SectionWritePayload,
    logger: logging.Logger,
) -> SectionGenerationStatus:
    """
    Internal function to generate a single section
    """
    section_type = payload.section_type
    
    try:
        # Build prompts
        system_prompt = _build_system_prompt_from_context(payload)
        user_prompt = payload.user_prompt or f"Write the {section_type} section."
        user_prompt = _expand_template_syntax(user_prompt, payload)
        
        # Run agent
        agent_result = await agent_instance.run(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            section_type=section_type,
            citation_format=payload.constraints.citation_format,
            constraints=payload.constraints.additional_instructions,
        )
        
        generated_content = agent_result.get("generated_content", "")
        word_count = len(generated_content.split())
        
        return SectionGenerationStatus(
            section_type=section_type,
            status="ok",
            word_count=word_count,
            latex_content=generated_content,
        )
    
    except Exception as e:
        logger.error("paper_assembly.section_error section=%s error=%s", section_type, str(e))
        return SectionGenerationStatus(
            section_type=section_type,
            status="error",
            error=str(e),
        )


def create_paper_assembly_endpoint(router: APIRouter, agent_instance, logger: logging.Logger):
    """
    Add paper assembly endpoint to router
    """
    
    @router.post("/agent/writer/assemble-paper", response_model=PaperAssemblyResult, status_code=status.HTTP_200_OK)
    async def assemble_paper(config: PaperChainConfig):
        """
        Assemble complete paper from chain configuration
        - **Description**:
            - Generates each section specified in the chain config
            - Assembles all sections into a complete LaTeX document
            - Optionally compiles to PDF via Typesetter agent
            - Supports file-based or inline payload specification

        - **Args**:
            - `config` (PaperChainConfig): Paper assembly configuration

        - **Returns**:
            - `PaperAssemblyResult`: Assembly result with content and paths
        """
        start = time.time()
        logger.info("paper_assembly.start title=%s sections=%d", 
                   config.paper_title, len(config.sections))
        
        sections_status: List[SectionGenerationStatus] = []
        errors: List[str] = []
        all_references: List[Dict[str, Any]] = []  # Collect references from all sections
        
        # Ensure output directory exists
        output_dir = config.output_dir
        if config.base_path and not os.path.isabs(output_dir):
            output_dir = os.path.join(config.base_path, output_dir)
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate each section
        for item in config.sections:
            if not item.enabled:
                sections_status.append(SectionGenerationStatus(
                    section_type=item.section_type,
                    status="skipped",
                ))
                continue
            
            try:
                # Load payload
                payload = _load_section_payload(
                    item,
                    base_path=config.base_path,
                    shared_resources=config.shared_resources,
                    logger=logger,
                )
                
                # Collect references from this section's resources
                if payload.resources and payload.resources.references:
                    for ref in payload.resources.references:
                        # Convert to dict if it's a Pydantic model
                        ref_dict = ref.model_dump() if hasattr(ref, 'model_dump') else ref
                        all_references.append(ref_dict)
                
                # Generate section
                logger.info("paper_assembly.generating section=%s", item.section_type)
                section_result = await _generate_section_internal(
                    agent_instance,
                    payload,
                    logger,
                )
                sections_status.append(section_result)
                
                if section_result.status == "error":
                    errors.append(f"{item.section_type}: {section_result.error}")
                
            except Exception as e:
                error_msg = f"{item.section_type}: {str(e)}"
                errors.append(error_msg)
                sections_status.append(SectionGenerationStatus(
                    section_type=item.section_type,
                    status="error",
                    error=str(e),
                ))
                logger.error("paper_assembly.section_failed section=%s error=%s", 
                           item.section_type, str(e))
        
        # Deduplicate references by ref_id
        seen_refs = set()
        unique_references = []
        for ref in all_references:
            ref_id = ref.get("ref_id") or ref.get("id")
            if ref_id and ref_id not in seen_refs:
                seen_refs.add(ref_id)
                unique_references.append(ref)
        
        logger.info("paper_assembly.collected_references count=%d unique=%d", 
                   len(all_references), len(unique_references))
        
        # Assemble LaTeX content
        latex_content = _assemble_latex_document(sections_status, config.paper_title)
        total_word_count = sum(s.word_count for s in sections_status)
        
        # Save LaTeX file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        latex_filename = f"{config.paper_title.replace(' ', '_')}_{timestamp}.tex"
        latex_path = os.path.join(output_dir, latex_filename)
        
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        logger.info("paper_assembly.saved_latex path=%s", latex_path)
        
        # Compile to PDF if requested
        pdf_path = None
        if config.compile_pdf:
            try:
                # Import httpx client
                import httpx
                
                # Resolve template_path if relative
                template_path = config.template_path
                if template_path and config.base_path and not os.path.isabs(template_path):
                    template_path = os.path.join(config.base_path, template_path)
                
                # Resolve figures_source_dir if relative
                figures_source_dir = config.figures_source_dir
                logger.info("paper_assembly.figures_source_dir raw=%s base_path=%s", 
                           figures_source_dir, config.base_path)
                if figures_source_dir and config.base_path and not os.path.isabs(figures_source_dir):
                    figures_source_dir = os.path.join(config.base_path, figures_source_dir)
                    logger.info("paper_assembly.figures_source_dir resolved=%s exists=%s", 
                               figures_source_dir, os.path.isdir(figures_source_dir) if figures_source_dir else False)
                
                # Build template_config for Typesetter
                # Step 1: If template_path provided, parse the zip to get base config
                typesetter_template_config = {}
                
                if template_path and os.path.exists(template_path):
                    logger.info("paper_assembly.parsing_template path=%s", template_path)
                    try:
                        async with httpx.AsyncClient() as client:
                            parse_response = await client.post(
                                "http://127.0.0.1:8000/agent/template/parse",
                                json={
                                    "user_id": "paper_assembly",
                                    "payload": {
                                        "file_path": template_path,
                                        "template_id": f"assembly_{timestamp}",
                                    }
                                },
                                timeout=120.0
                            )
                            
                            if parse_response.status_code == 200:
                                parse_result = parse_response.json()
                                if parse_result.get("status") == "ok" and parse_result.get("result"):
                                    parsed_info = parse_result["result"]
                                    # Extract relevant fields for TemplateConfig
                                    typesetter_template_config = {
                                        "template_id": parsed_info.get("template_id"),
                                        "document_class": parsed_info.get("document_class", "article"),
                                        "citation_style": parsed_info.get("citation_style", "cite"),
                                        "column_format": parsed_info.get("column_format", "single"),
                                        "raw_preamble": parsed_info.get("raw_preamble"),
                                        "bib_style": parsed_info.get("bib_style", "plain"),
                                        "required_packages": parsed_info.get("required_packages", []),
                                        "figure_placement": parsed_info.get("figure_placement", "htbp"),
                                        "has_abstract": parsed_info.get("has_abstract", True),
                                        "has_acknowledgment": parsed_info.get("has_acknowledgment", False),
                                    }
                                    logger.info("paper_assembly.template_parsed doc_class=%s", 
                                               typesetter_template_config.get("document_class"))
                                else:
                                    logger.warning("paper_assembly.template_parse_failed error=%s", 
                                                  parse_result.get("error"))
                            else:
                                logger.warning("paper_assembly.template_parse_http_error status=%d", 
                                              parse_response.status_code)
                    except Exception as parse_error:
                        logger.warning("paper_assembly.template_parse_exception error=%s", str(parse_error))
                
                # Step 2: Apply user's template_config as override
                if config.template_config:
                    for key, value in config.template_config.items():
                        if value is not None:  # Only override if value is provided
                            typesetter_template_config[key] = value
                
                # Step 3: Ensure paper_title is set
                if "paper_title" not in typesetter_template_config or not typesetter_template_config["paper_title"]:
                    typesetter_template_config["paper_title"] = config.paper_title
                
                # Call Typesetter to compile
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        "http://127.0.0.1:8000/agent/typesetter/compile",
                        json={
                            "request_id": f"assembly_{timestamp}",
                            "user_id": "paper_assembly",
                            "payload": {
                                "latex_content": latex_content,
                                "template_path": template_path,
                                "template_config": typesetter_template_config,
                                "output_dir": output_dir,
                                "references": unique_references,  # Pass collected references
                                "figures_source_dir": figures_source_dir,  # Local figures directory
                                "work_id": None,
                            }
                        },
                        timeout=300.0
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if result.get("status") == "ok":
                            pdf_path = result.get("result", {}).get("pdf_path")
                            logger.info("paper_assembly.compiled_pdf path=%s", pdf_path)
                        else:
                            errors.append(f"PDF compilation: {result.get('error')}")
                    else:
                        errors.append(f"PDF compilation failed: {response.status_code}")
                        
            except Exception as e:
                errors.append(f"PDF compilation error: {str(e)}")
                logger.error("paper_assembly.pdf_error error=%s", str(e))
        
        # Determine overall status
        successful_sections = sum(1 for s in sections_status if s.status == "ok")
        total_enabled = sum(1 for s in sections_status if s.status != "skipped")
        
        if successful_sections == total_enabled:
            overall_status = "ok"
        elif successful_sections > 0:
            overall_status = "partial"
        else:
            overall_status = "error"
        
        latency = time.time() - start
        logger.info("paper_assembly.complete status=%s sections=%d/%d latency=%.3f", 
                   overall_status, successful_sections, total_enabled, latency)
        
        return PaperAssemblyResult(
            status=overall_status,
            paper_title=config.paper_title,
            sections_status=sections_status,
            latex_content=latex_content,
            latex_path=latex_path,
            pdf_path=pdf_path,
            total_word_count=total_word_count,
            errors=errors,
        )
