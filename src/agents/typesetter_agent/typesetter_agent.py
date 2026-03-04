"""
Typesetter Agent
- **Description**:
    - Handles resource fetching, template injection, and LaTeX compilation
    - Implements self-healing compilation with error recovery
"""
from langchain.messages import AnyMessage
from ..shared.llm_client import LLMClient
from typing_extensions import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
import operator
import os
import re
import shutil
import tempfile
import subprocess
import zipfile
import httpx
import logging
from pathlib import Path
from typing import List, Dict, Any
from fastapi import APIRouter
from jinja2 import Template
from ...config.schema import ModelConfig
from ..base import BaseAgent
from .router import create_typesetter_router
from .models import ResourceInfo, BibEntry, CompilationResult, TemplateConfig


# Backend API URL for fetching resources
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://127.0.0.1:9001/api")

# Logger for typesetter agent
logger = logging.getLogger("uvicorn.error")

# Maximum compilation attempts for self-healing
MAX_COMPILE_ATTEMPTS = 3


# Jinja2 template for injecting content into main.tex
MAIN_TEX_TEMPLATE = """{{ preamble }}

\\begin{document}

\\maketitle

{{ content }}

{% if has_bibliography %}
\\bibliographystyle{ {{- bib_style -}} }
\\bibliography{references}
{% endif %}

\\end{document}
"""


class TypesetterAgentState(TypedDict):
    """
    State for Typesetter Agent workflow
    """
    messages: Annotated[list[AnyMessage], operator.add]
    latex_content: Optional[str]
    sections: Optional[Dict[str, str]]  # Multi-file mode: section_type -> content
    section_order: Optional[List[str]]  # Multi-file mode: body section ordering
    section_titles: Optional[Dict[str, str]]  # Multi-file mode: section_type -> display title
    template_path: Optional[str]
    template_config: Optional[TemplateConfig]  # Template configuration with constraints
    figure_ids: Optional[List[str]]
    citation_ids: Optional[List[str]]
    references: Optional[List[Dict[str, Any]]]
    work_id: Optional[str]
    output_dir: Optional[str]  # User-specified output directory for final files
    figures_source_dir: Optional[str]  # Local directory with figure files to copy
    figure_paths: Optional[Dict[str, str]]  # Structured figure paths: id -> file_path
    converted_tables: Optional[Dict[str, str]]  # Pre-converted table LaTeX: id -> latex_code
    work_dir: Optional[str]  # Temporary working directory for compilation
    main_tex_path: Optional[str]  # Path to detected main tex file
    resources: Optional[List[ResourceInfo]]
    bib_entries: Optional[List[BibEntry]]
    compiled_tex: Optional[str]
    section_file_map: Optional[Dict[str, str]]  # Multi-file mode: section_type -> rel path
    compilation_result: Optional[CompilationResult]
    llm_calls: int


class TypesetterAgent(BaseAgent):
    """
    Typesetter Agent for LaTeX compilation
    - **Description**:
        - Fetches resources from backend
        - Generates BibTeX file
        - Injects content into template
        - Compiles LaTeX with self-healing
    """
    
    def __init__(self, config: ModelConfig):
        self.client = LLMClient(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.model_name = config.model_name
        self.agent = self.init_agent()

    def init_agent(self):
        """
        Initialize the agent workflow graph
        """
        agent_builder = StateGraph(TypesetterAgentState)
        agent_builder.add_node("setup_workspace", self.setup_workspace)
        agent_builder.add_node("fetch_resources", self.fetch_resources)
        agent_builder.add_node("generate_bibtex", self.generate_bibtex)
        agent_builder.add_node("inject_template", self.inject_template)
        agent_builder.add_node("compile_latex", self.compile_latex)
        
        agent_builder.add_edge(START, "setup_workspace")
        agent_builder.add_edge("setup_workspace", "fetch_resources")
        agent_builder.add_edge("fetch_resources", "generate_bibtex")
        agent_builder.add_edge("generate_bibtex", "inject_template")
        agent_builder.add_edge("inject_template", "compile_latex")
        agent_builder.add_edge("compile_latex", END)
        
        return agent_builder.compile()

    def _find_main_tex(self, work_dir: str) -> Optional[str]:
        """
        Find the main tex file containing documentclass
        - **Description**:
            - Searches for .tex files with both \\documentclass and \\begin{document}
            - Prefers files named main.tex or containing 'paper' in name

        - **Args**:
            - `work_dir` (str): Directory to search

        - **Returns**:
            - `str`: Path to main tex file, or None if not found
        """
        candidates = []
        
        for root, dirs, files in os.walk(work_dir):
            for f in files:
                if f.endswith('.tex'):
                    path = os.path.join(root, f)
                    try:
                        with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                            content = file.read()
                            if '\\documentclass' in content and '\\begin{document}' in content:
                                # Calculate priority: lower is better
                                priority = 10
                                fname_lower = f.lower()
                                if fname_lower == 'main.tex':
                                    priority = 0
                                elif 'paper' in fname_lower:
                                    priority = 1
                                elif 'example' in fname_lower:
                                    priority = 2
                                candidates.append((priority, path))
                    except Exception:
                        continue
        
        if candidates:
            # Sort by priority and return best match
            candidates.sort(key=lambda x: x[0])
            return candidates[0][1]
        
        return None

    @staticmethod
    def _flatten_support_files(work_dir: str) -> None:
        """
        Copy .bst, .cls, .sty files from subdirectories to work_dir root.
        - **Description**:
            - Template zips often nest support files in subdirectories
              (e.g. ``bst/``, ``styles/``). BibTeX and pdflatex only search
              the current directory by default, so these must be at the root.

        - **Args**:
            - `work_dir` (str): The compilation working directory.
        """
        extensions = ('.bst', '.cls', '.sty')
        for root, _dirs, files in os.walk(work_dir):
            if root == work_dir:
                continue
            for fname in files:
                if fname.endswith(extensions):
                    src = os.path.join(root, fname)
                    dst = os.path.join(work_dir, fname)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        logger.info("typesetter.flattened_support_file %s", fname)

    async def setup_workspace(self, state: TypesetterAgentState) -> Dict[str, Any]:
        """
        Set up temporary workspace for compilation
        - **Description**:
            - Creates temporary directory structure
            - Extracts template if provided
            - Detects main tex file automatically

        - **Args**:
            - `state` (TypesetterAgentState): Current workflow state

        - **Returns**:
            - `dict`: Updated state with work_dir and main_tex_path
        """
        logger.info("typesetter.setup_workspace start")
        
        # Create temporary directory
        work_dir = tempfile.mkdtemp(prefix="typesetter_")
        figures_dir = os.path.join(work_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        logger.info("typesetter.setup_workspace work_dir=%s", work_dir)
        
        # Extract template if provided
        template_path = state.get("template_path")
        main_tex_path = None
        
        if template_path and os.path.exists(template_path):
            if template_path.endswith('.zip'):
                logger.info("typesetter.extracting_template path=%s", template_path)
                with zipfile.ZipFile(template_path, 'r') as zip_ref:
                    zip_ref.extractall(work_dir)
                
                # Detect main tex file
                main_tex_path = self._find_main_tex(work_dir)
                if main_tex_path:
                    logger.info("typesetter.detected_main_tex file=%s", os.path.basename(main_tex_path))
                    # Flatten .bst/.cls/.sty files to the work_dir root so
                    # pdflatex/bibtex can find them regardless of zip structure
                    self._flatten_support_files(work_dir)
                else:
                    logger.warning("typesetter.no_main_tex_found")
        
        # Copy figures from local source directory if provided (legacy)
        figures_source_dir = state.get("figures_source_dir")
        if figures_source_dir and os.path.isdir(figures_source_dir):
            logger.info("typesetter.copying_local_figures source=%s", figures_source_dir)
            copied_count = 0
            for item in os.listdir(figures_source_dir):
                src_path = os.path.join(figures_source_dir, item)
                if os.path.isfile(src_path):
                    # Copy figure file to work_dir/figures/
                    dst_path = os.path.join(figures_dir, item)
                    try:
                        shutil.copy2(src_path, dst_path)
                        copied_count += 1
                    except Exception as e:
                        logger.warning("typesetter.figure_copy_failed file=%s error=%s", item, str(e))
            logger.info("typesetter.local_figures_copied count=%d", copied_count)
        
        # Copy figures from structured figure_paths (new method)
        figure_paths = state.get("figure_paths", {})
        if figure_paths:
            logger.info("typesetter.copying_structured_figures count=%d", len(figure_paths))
            for fig_id, file_path in figure_paths.items():
                if file_path and os.path.exists(file_path):
                    # Keep original filename for reference
                    filename = os.path.basename(file_path)
                    dst_path = os.path.join(figures_dir, filename)
                    try:
                        shutil.copy2(file_path, dst_path)
                        logger.info("typesetter.figure_copied id=%s path=%s", fig_id, dst_path)
                    except Exception as e:
                        logger.warning("typesetter.figure_copy_failed id=%s error=%s", fig_id, str(e))
                else:
                    logger.warning("typesetter.figure_not_found id=%s path=%s", fig_id, file_path)
        
        return {"work_dir": work_dir, "main_tex_path": main_tex_path}

    async def fetch_resources(self, state: TypesetterAgentState) -> Dict[str, Any]:
        """
        Fetch figure resources from backend
        - **Description**:
            - Downloads figures referenced in the LaTeX content

        - **Args**:
            - `state` (TypesetterAgentState): Current workflow state

        - **Returns**:
            - `dict`: Updated state with fetched resources
        """
        print(f"INPUT STATE [fetch_resources]: figure_ids={state.get('figure_ids')}")
        
        work_dir = state.get("work_dir")
        figure_ids = self._resolve_figure_ids(state)
        figures_dir = os.path.join(work_dir, "figures")
        
        resources = []
        
        async with httpx.AsyncClient() as client:
            for fig_id in figure_ids:
                resource = ResourceInfo(
                    resource_id=fig_id,
                    resource_type="figure",
                    status="pending"
                )
                
                try:
                    # Try to fetch figure from backend
                    # Figure path format: figures/{filename}
                    response = await client.get(
                        f"{BACKEND_API_URL}/files/figures/{fig_id}",
                        timeout=30.0
                    )
                    
                    if response.status_code == 200:
                        # Determine file extension from content-type
                        content_type = response.headers.get("content-type", "")
                        ext = ".png"
                        if "pdf" in content_type:
                            ext = ".pdf"
                        elif "jpeg" in content_type or "jpg" in content_type:
                            ext = ".jpg"
                        elif "svg" in content_type:
                            ext = ".svg"
                        
                        local_path = os.path.join(figures_dir, f"{fig_id}{ext}")
                        with open(local_path, "wb") as f:
                            f.write(response.content)
                        
                        resource.local_path = local_path
                        resource.status = "downloaded"
                    else:
                        resource.status = "failed"
                        
                except Exception as e:
                    print(f"Failed to fetch figure {fig_id}: {e}")
                    resource.status = "failed"
                
                resources.append(resource)
        
        return {"resources": resources, "figure_ids": figure_ids}

    def _resolve_figure_ids(self, state: TypesetterAgentState) -> List[str]:
        """
        Resolve figure IDs from payload and LaTeX content.
        - **Description**:
            - Prioritizes explicit `figure_ids` from caller.
            - Falls back to parsing includegraphics targets and figure_paths keys.

        - **Args**:
            - `state` (TypesetterAgentState): Current workflow state.

        - **Returns**:
            - `figure_ids` (List[str]): Deduplicated figure IDs.
        """
        ids: set[str] = set()
        for item in (state.get("figure_ids") or []):
            if item:
                ids.add(str(item))
        for key in (state.get("figure_paths") or {}).keys():
            if key:
                ids.add(str(key))

        if not ids:
            latex_content = state.get("latex_content", "") or ""
            sections = state.get("sections") or {}
            combined = latex_content + "\n" + "\n".join(
                content for content in sections.values() if isinstance(content, str)
            )
            for token in self._extract_includegraphics_targets(combined):
                ids.add(token)
        return sorted(ids)

    @staticmethod
    def _extract_includegraphics_targets(content: str) -> List[str]:
        """
        Extract likely figure IDs from includegraphics targets.
        - **Description**:
            - Skips explicit path-like targets and common file extensions.

        - **Args**:
            - `content` (str): LaTeX content.

        - **Returns**:
            - `targets` (List[str]): Candidate figure IDs.
        """
        pattern = r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
        targets: List[str] = []
        seen = set()
        for raw in re.findall(pattern, content or ""):
            token = str(raw).strip()
            if not token:
                continue
            if "/" in token or token.startswith(".") or token.endswith(
                (".png", ".jpg", ".jpeg", ".pdf", ".svg")
            ):
                continue
            if token not in seen:
                seen.add(token)
                targets.append(token)
        return targets

    @staticmethod
    def _strip_graphics_extension(path: str) -> str:
        """
        Strip known graphics extensions for LaTeX includegraphics.
        - **Args**:
            - `path` (str): Input graphics path.
        - **Returns**:
            - `str`: Path without extension when extension is image/pdf.
        """
        root, ext = os.path.splitext(path)
        if ext.lower() in {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".eps"}:
            return root
        return path

    def _rewrite_includegraphics_targets(
        self,
        content: str,
        work_dir: str,
        id_to_rel_path: Dict[str, str],
    ) -> str:
        """
        Rewrite includegraphics targets using resource-aware resolution.
        - **Description**:
            - Resolution order:
              1) exact ID mapping from resources/figure_paths
              2) existing valid relative path under work_dir
              3) basename fallback under figures/ when file exists
            - Adds a safe default width when includegraphics has no option block.
            - Leaves unresolved targets unchanged.
        """
        if not content:
            return content

        def _resolve_target(raw_target: str) -> str:
            target = raw_target.strip()
            if not target:
                return target
            if target in id_to_rel_path:
                return id_to_rel_path[target]

            # Keep explicit, already-valid relative path.
            abs_existing = os.path.join(work_dir, target)
            if os.path.exists(abs_existing):
                return self._strip_graphics_extension(target)

            # Fallback: try basename in figures directory.
            basename = os.path.basename(target)
            candidate = os.path.join("figures", basename)
            if os.path.exists(os.path.join(work_dir, candidate)):
                return self._strip_graphics_extension(candidate)
            # If model emits bare token without extension (e.g. {fig1}),
            # resolve against common graphics extensions under figures/.
            for ext in (".pdf", ".png", ".jpg", ".jpeg", ".svg", ".eps"):
                ext_candidate = os.path.join("figures", f"{basename}{ext}")
                if os.path.exists(os.path.join(work_dir, ext_candidate)):
                    return self._strip_graphics_extension(ext_candidate)
            return target

        pattern = r'(\\includegraphics)(?:\[([^\]]*)\])?\{([^}]+)\}'

        def _rewrite(match: re.Match) -> str:
            cmd = match.group(1)
            opts = match.group(2)
            target = match.group(3)
            resolved_target = _resolve_target(target)
            if opts is None or not opts.strip():
                # Prevent oversized figures when model omits explicit width.
                return f"{cmd}[width=0.9\\linewidth]{{{resolved_target}}}"
            return f"{cmd}[{opts}]{{{resolved_target}}}"

        return re.sub(pattern, _rewrite, content)

    def _extract_citations_from_content(self, latex_content: str) -> List[str]:
        """
        Extract citation keys from LaTeX content
        - **Description**:
            - Finds all \cite{key}, \citet{key}, \citep{key} etc. commands
            - Handles multiple keys in single cite command
            - Filters out invalid keys

        - **Args**:
            - `latex_content` (str): LaTeX content to scan

        - **Returns**:
            - `list`: Unique citation keys found
        """
        # Match various cite commands: \cite{}, \citet{}, \citep{}, \citeauthor{}, etc.
        pattern = r'\\cite[tp]?\{([^}]+)\}'
        matches = re.findall(pattern, latex_content)
        
        # Split multiple keys (e.g., \cite{key1, key2})
        all_keys = []
        for match in matches:
            keys = [k.strip() for k in match.split(',')]
            all_keys.extend(keys)
        
        # Filter out invalid keys (must look like actual citation keys)
        # Valid keys typically contain letters, numbers, underscores, hyphens
        # and should not be generic field names like "ref_id"
        invalid_keys = {'ref_id', 'id', 'key', 'citation', 'reference'}
        
        # Return unique keys preserving order
        seen = set()
        unique_keys = []
        for key in all_keys:
            # Skip empty, already seen, or invalid keys
            if not key or key in seen or key.lower() in invalid_keys:
                continue
            # Basic validation: key should contain at least one letter and one digit (typical citation format)
            # or be a reasonable identifier
            if re.match(r'^[a-zA-Z][a-zA-Z0-9_\-:]+$', key):
                seen.add(key)
                unique_keys.append(key)
        
        return unique_keys

    async def generate_bibtex(self, state: TypesetterAgentState) -> Dict[str, Any]:
        """
        Generate BibTeX file from references
        - **Description**:
            - Creates references.bib from provided reference data
            - Prioritizes raw bibtex string if provided
            - Auto-extracts citation keys from latex_content to find needed refs

        - **Args**:
            - `state` (TypesetterAgentState): Current workflow state

        - **Returns**:
            - `dict`: Updated state with bib_entries
        """
        work_dir = state.get("work_dir")
        latex_content = state.get("latex_content", "")
        references = state.get("references", [])
        sections_dict = state.get("sections") or {}
        
        # Extract citation keys from ALL content (single-file + multi-file sections)
        all_content = latex_content or ""
        for sec_content in sections_dict.values():
            all_content += "\n" + sec_content
        citation_ids = self._extract_citations_from_content(all_content) if all_content.strip() else []
        logger.info("typesetter.extracted_citations count=%d keys=%s (sections=%d)", 
                   len(citation_ids), citation_ids[:5] if len(citation_ids) > 5 else citation_ids,
                   len(sections_dict))
        
        bib_entries = []
        bib_content_parts = []
        
        # Build reference map from provided references
        # Support both "id" and "ref_id" keys for flexibility
        ref_map = {}
        for ref in references:
            ref_id = ref.get("ref_id") or ref.get("id")
            if ref_id:
                ref_map[ref_id] = ref
        
        logger.info("typesetter.provided_references count=%d keys=%s", 
                   len(ref_map), list(ref_map.keys())[:5])
        
        # Generate entries for references we have data for
        for cid in citation_ids:
            ref = ref_map.get(cid)
            
            if ref:
                # Check if raw bibtex is provided - use it directly
                if ref.get("bibtex"):
                    bib_content_parts.append(ref["bibtex"].strip())
                    # Still create a BibEntry for tracking
                    entry = BibEntry(
                        key=cid,
                        entry_type="article",
                        title=ref.get("title", ""),
                        raw_bibtex=ref["bibtex"],
                    )
                    bib_entries.append(entry)
                else:
                    # Generate from structured data
                    entry = BibEntry(
                        key=cid,
                        entry_type=ref.get("entry_type", "article"),
                        title=ref.get("title", "Untitled"),
                        authors=ref.get("authors"),
                        year=ref.get("year"),
                        doi=ref.get("doi"),
                        url=ref.get("url"),
                        venue=ref.get("venue"),
                        journal=ref.get("journal"),
                        booktitle=ref.get("booktitle"),
                    )
                    bib_entries.append(entry)
                    bib_str = self._generate_bibtex_entry(entry)
                    bib_content_parts.append(bib_str)
            else:
                # Skip - reference data should be provided by upstream
                logger.warning("typesetter.missing_reference key=%s (should be provided by upstream)", cid)
        
        # Safety fallback: if no citation keys extracted but references exist,
        # write ALL provided references to .bib (ensures bibtex is never empty
        # when references are available)
        if not bib_content_parts and references:
            logger.warning("typesetter.no_citations_extracted_but_refs_exist count=%d — writing all refs as fallback", len(references))
            for ref in references:
                if ref.get("bibtex"):
                    bib_content_parts.append(ref["bibtex"].strip())
                    ref_id = ref.get("ref_id") or ref.get("id") or "unknown"
                    bib_entries.append(BibEntry(
                        key=ref_id,
                        entry_type="article",
                        title=ref.get("title", ""),
                        raw_bibtex=ref["bibtex"],
                    ))
        
        # Write references.bib
        bib_path = os.path.join(work_dir, "references.bib")
        with open(bib_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(bib_content_parts))
        
        logger.info("typesetter.written_bibtex entries=%d path=%s", len(bib_entries), bib_path)
        
        return {"bib_entries": bib_entries}

    def _generate_bibtex_entry(self, entry: BibEntry) -> str:
        """Generate a BibTeX entry string"""
        lines = [f"@{entry.entry_type}{{{entry.key},"]
        
        if entry.title:
            lines.append(f"  title = {{{entry.title}}},")
        if entry.authors:
            lines.append(f"  author = {{{entry.authors}}},")
        if entry.year:
            lines.append(f"  year = {{{entry.year}}},")
        
        # Handle venue: use booktitle for conferences, journal for journals
        if entry.booktitle:
            lines.append(f"  booktitle = {{{entry.booktitle}}},")
        elif entry.journal:
            lines.append(f"  journal = {{{entry.journal}}},")
        elif entry.venue:
            # Generic venue - decide based on entry_type
            if entry.entry_type in ["inproceedings", "conference"]:
                lines.append(f"  booktitle = {{{entry.venue}}},")
            else:
                lines.append(f"  journal = {{{entry.venue}}},")
        
        if entry.doi:
            lines.append(f"  doi = {{{entry.doi}}},")
        if entry.url:
            lines.append(f"  url = {{{entry.url}}},")
        
        lines.append("}")
        return "\n".join(lines)

    def _build_preamble_from_config(self, config: TemplateConfig) -> str:
        """
        Build LaTeX preamble from TemplateConfig
        - **Description**:
            - Constructs a complete preamble based on template configuration
            - Handles document class, packages, and formatting options

        - **Args**:
            - `config` (TemplateConfig): Template configuration

        - **Returns**:
            - `str`: Complete LaTeX preamble
        """
        # If raw_preamble is provided, use it directly
        if config.raw_preamble:
            return config.raw_preamble
        
        # Build document class line with options
        doc_options = list(config.document_class_options)
        
        # Add column format option
        if config.column_format == "double" and "twocolumn" not in doc_options:
            doc_options.append("twocolumn")
        
        # Build options string
        options_str = ",".join(doc_options) if doc_options else ""
        if options_str:
            doc_class_line = f"\\documentclass[{options_str}]{{{config.document_class}}}"
        else:
            doc_class_line = f"\\documentclass{{{config.document_class}}}"
        
        # Build package list
        packages = [
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage[T1]{fontenc}",
            "\\usepackage{graphicx}",
            "\\usepackage{amsmath}",
            "\\usepackage{amssymb}",
            "\\usepackage{hyperref}",
            "\\usepackage[margin=1in]{geometry}",
        ]
        
        # Add citation package based on style
        if config.citation_style in ("citep", "citet"):
            packages.append("\\usepackage{natbib}")
        else:
            packages.append("\\usepackage{cite}")
        
        # Add additional required packages
        for pkg in config.required_packages:
            pkg_line = f"\\usepackage{{{pkg}}}"
            if pkg_line not in packages:
                packages.append(pkg_line)
        
        # Build title/author section
        title_section = []
        if config.paper_title:
            title_section.append(f"\\title{{{config.paper_title}}}")
        else:
            title_section.append("\\title{Generated Paper}")
        
        if config.paper_authors:
            title_section.append(f"\\author{{{config.paper_authors}}}")
        else:
            title_section.append("\\author{}")
        
        title_section.append("\\date{\\today}")
        
        # Assemble preamble
        preamble_parts = [
            doc_class_line,
            "",
            "% Packages",
            "\n".join(packages),
            "",
            "% Title",
            "\n".join(title_section),
        ]
        
        return "\n".join(preamble_parts)

    # Default section ordering and titles for multi-file mode
    DEFAULT_SECTION_ORDER = [
        "introduction", "related_work", "method", "experiment", "result", "conclusion",
    ]
    DEFAULT_SECTION_TITLES = {
        "introduction": "Introduction",
        "related_work": "Related Work",
        "method": "Methodology",
        "experiment": "Experiments",
        "result": "Results",
        "conclusion": "Conclusion",
        "appendix": "Appendix",
    }

    @staticmethod
    def _strip_leading_section_command(content: str) -> str:
        """
        Strip all leading \\section{} and \\section*{} commands from content.
        - **Description**:
            - WriterAgent output may already contain \\section{Title} commands
            - Since _write_section_files always prepends its own canonical
              \\section{Title}, any existing ones must be removed to avoid duplicates
            - Also strips multiple consecutive section commands (e.g. \\section{Results}
              followed by \\section*{Result})

        - **Args**:
            - `content` (str): Raw LaTeX section content

        - **Returns**:
            - `str`: Content with leading section commands removed
        """
        # Repeatedly strip leading \section{...} or \section*{...} commands
        # Also handle optional \label{...} right after the section command
        pattern = r'^\s*\\section\*?\{[^}]*\}\s*(?:\\label\{[^}]*\}\s*)?'
        prev = None
        while prev != content:
            prev = content
            content = re.sub(pattern, '', content, count=1)
        return content.strip()

    def _write_section_files(
        self,
        work_dir: str,
        sections: Dict[str, str],
        section_order: Optional[List[str]] = None,
        section_titles: Optional[Dict[str, str]] = None,
        citation_style: str = "cite",
        use_appendices_env: bool = False,
    ) -> Dict[str, str]:
        """
        Write each section to an independent .tex file under work_dir/sections/.
        - **Description**:
            - Creates a sections/ subdirectory inside the working directory
            - Writes each section as a standalone .tex file
            - Body sections get a \\section{Title} command prepended
            - Abstract does not get a \\section{} command
            - Applies citation style conversion to each section
            - When use_appendices_env is True, wraps appendix in
              \\begin{appendices}...\\end{appendices} (required by templates
              that load the `appendix` package, e.g. Springer Nature).

        - **Args**:
            - `work_dir` (str): The LaTeX compilation working directory
            - `sections` (Dict[str, str]): section_type -> raw LaTeX content
            - `section_order` (List[str], optional): Order of body sections
            - `section_titles` (Dict[str, str], optional): section_type -> display title
            - `citation_style` (str): Citation command style to apply
            - `use_appendices_env` (bool): Use \\begin{appendices} environment format

        - **Returns**:
            - `section_file_map` (Dict[str, str]): section_type -> relative path from work_dir
        """
        order = section_order or self.DEFAULT_SECTION_ORDER
        titles = section_titles or self.DEFAULT_SECTION_TITLES

        sections_dir = os.path.join(work_dir, "sections")
        os.makedirs(sections_dir, exist_ok=True)

        section_file_map: Dict[str, str] = {}

        # Write body sections in order
        for section_type in order:
            if section_type not in sections or not sections[section_type].strip():
                continue
            if section_type in ("abstract", "conclusion"):
                # Abstract and conclusion are inlined into main.tex directly.
                continue

            content = sections[section_type].strip()
            # Strip any leading \section{} or \section*{} already in the content
            # to avoid duplicates (WriterAgent may include them)
            content = self._strip_leading_section_command(content)
            content = self._apply_citation_style(content, citation_style)
            # Escape unescaped % characters
            content = re.sub(r'(?<!\\)%', r'\\%', content)

            title = titles.get(section_type, section_type.replace("_", " ").title())

            if section_type == "appendix":
                if use_appendices_env:
                    file_content = (
                        f"\\begin{{appendices}}\n"
                        f"\\section{{{title}}}\\label{{secA1}}\n\n{content}\n"
                        f"\\end{{appendices}}\n"
                    )
                else:
                    file_content = f"\\appendix\n\\section{{{title}}}\n\n{content}\n"
            else:
                file_content = f"\\section{{{title}}}\n\n{content}\n"

            file_name = f"{section_type}.tex"
            file_path = os.path.join(sections_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_content)
            section_file_map[section_type] = f"sections/{section_type}"
            logger.info("typesetter.write_section file=sections/%s chars=%d", file_name, len(file_content))

        # Write any remaining sections not in the order list (excluding abstract)
        for section_type, content in sections.items():
            if section_type in ("abstract", "conclusion") or section_type in section_file_map:
                continue
            if not content.strip():
                continue

            content = content.strip()
            content = self._strip_leading_section_command(content)
            content = self._apply_citation_style(content, citation_style)
            content = re.sub(r'(?<!\\)%', r'\\%', content)

            title = titles.get(section_type, section_type.replace("_", " ").title())

            if section_type == "appendix":
                if use_appendices_env:
                    file_content = (
                        f"\\begin{{appendices}}\n"
                        f"\\section{{{title}}}\\label{{secA1}}\n\n{content}\n"
                        f"\\end{{appendices}}\n"
                    )
                else:
                    file_content = f"\\appendix\n\\section{{{title}}}\n\n{content}\n"
            else:
                file_content = f"\\section{{{title}}}\n\n{content}\n"

            file_name = f"{section_type}.tex"
            file_path = os.path.join(sections_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(file_content)
            section_file_map[section_type] = f"sections/{section_type}"
            logger.info("typesetter.write_section file=sections/%s chars=%d", file_name, len(file_content))

        logger.info("typesetter.write_sections total=%d files=%s",
                     len(section_file_map), list(section_file_map.keys()))
        return section_file_map

    def _apply_citation_style(self, content: str, citation_style: str) -> str:
        """
        Apply citation style to content
        - **Description**:
            - Converts citation commands to match the target style

        - **Args**:
            - `content` (str): LaTeX content
            - `citation_style` (str): Target citation style

        - **Returns**:
            - `str`: Content with updated citations
        """
        if citation_style == "citep":
            # Convert \cite to \citep for natbib
            content = re.sub(r'\\cite\{', r'\\citep{', content)
        elif citation_style == "citet":
            # Convert \cite to \citet for natbib
            content = re.sub(r'\\cite\{', r'\\citet{', content)
        # If citation_style is "cite", no conversion needed
        return content

    def _parse_sections_from_content(self, latex_content: str) -> Dict[str, str]:
        """
        Parse sections from generated LaTeX content
        - **Description**:
            - Extracts abstract and main sections from content
            - Handles % === Section: xxx === markers
            - Strips \begin{abstract} and \end{abstract} tags from abstract content

        - **Returns**:
            - `dict`: Dictionary with 'abstract' and 'body' keys
        """
        result = {"abstract": "", "body": ""}
        
        lines = latex_content.split('\n')
        current_section = None
        abstract_lines = []
        body_lines = []
        
        for line in lines:
            # Check for section markers
            if '% === Section: abstract ===' in line.lower() or '% === section: abstract ===' in line.lower():
                current_section = 'abstract'
                continue
            elif '% === Section:' in line or '% === section:' in line:
                current_section = 'body'
                body_lines.append(line)
                continue
            
            if current_section == 'abstract':
                # Check if we've hit the next section
                if line.strip().startswith('\\section{') or line.strip().startswith('% ==='):
                    current_section = 'body'
                    body_lines.append(line)
                else:
                    abstract_lines.append(line)
            elif current_section == 'body':
                body_lines.append(line)
            else:
                # Before any section marker, check if it looks like abstract
                body_lines.append(line)
        
        abstract_text = '\n'.join(abstract_lines).strip()
        
        # Remove \begin{abstract} and \end{abstract} tags if present
        # The template will add these, so we only want the content
        abstract_text = re.sub(r'^\\begin\{abstract\}\s*', '', abstract_text)
        abstract_text = re.sub(r'\s*\\end\{abstract\}$', '', abstract_text)
        abstract_text = abstract_text.strip()
        
        result["abstract"] = abstract_text
        result["body"] = '\n'.join(body_lines).strip()
        
        return result

    def _smart_inject_content(self, template_content: str, sections: Dict[str, str], 
                              template_config: TemplateConfig, bib_entries: List[BibEntry]) -> str:
        """
        Smart inject content into template
        - **Description**:
            - Replaces title/author if config provides them
            - Injects abstract in the correct location
            - Injects main content sections
            - Handles different template formats (standard, ICML, etc.)

        - **Args**:
            - `template_content` (str): Original template content
            - `sections` (dict): Parsed sections with 'abstract' and 'body'
            - `template_config` (TemplateConfig): Configuration with title/author
            - `bib_entries` (list): Bibliography entries

        - **Returns**:
            - `str`: Template with injected content
        """
        result = template_content
        
        # Step 1: Replace title if provided
        if template_config.paper_title:
            title = template_config.paper_title
            # Handle \title{...} and \title[short]{full} (Nature/Springer optional arg)
            result = re.sub(
                r'\\title(?:\[[^\]]*\])?\{[^}]*\}',
                lambda m: f'\\title{{{title}}}',
                result
            )
            # Handle ICML-style \icmltitle{...} which may span multiple lines
            # Use a custom function to find matching braces
            def replace_icmltitle(text: str, new_title: str) -> str:
                """Replace \icmltitle{...} handling nested braces and multi-line content"""
                pattern_start = '\\icmltitle{'
                start_idx = text.find(pattern_start)
                if start_idx == -1:
                    return text
                
                # Find matching closing brace
                brace_count = 1
                content_start = start_idx + len(pattern_start)
                i = content_start
                while i < len(text) and brace_count > 0:
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                    i += 1
                
                if brace_count == 0:
                    # Found matching brace
                    return text[:start_idx] + f'\\icmltitle{{{new_title}}}' + text[i:]
                return text
            
            result = replace_icmltitle(result, title)
            
            # Also update \icmltitlerunning if present
            result = re.sub(
                r'\\icmltitlerunning\{[^}]*\}',
                lambda m: f'\\icmltitlerunning{{{title[:50]}...}}' if len(title) > 50 else f'\\icmltitlerunning{{{title}}}',
                result
            )
        
        # Step 2: Replace author
        authors = template_config.paper_authors or "EasyPaper"
        # Remove all \author variants: \author{...}, \author*[...]{...}, etc.
        # Handle nested braces from \fnm{}, \sur{} inside author args
        result = self._replace_all_authors(result, authors)
        
        # Step 2b: Clear affiliation/institute/email/equalcont if present
        result = re.sub(r'\\affil\*?\[[^\]]*\]\{[^}]*\}', '', result)
        result = re.sub(r'\\affiliation\{[^}]*\}', '', result)
        result = re.sub(r'\\institute\{[^}]*\}', '', result)
        result = re.sub(r'\\equalcont\{[^}]*\}', '', result)
        result = re.sub(r'\\email\{[^}]*\}', '', result)
        # Remove orphaned \orgname/\orgaddress lines (Nature/Springer affiliation blocks)
        result = re.sub(r'^[,\s]*\\org(?:name|address|div)\{[^}]*(?:\{[^}]*\}[^}]*)*\}[^\n]*$',
                        '', result, flags=re.MULTILINE)
        # Clean up blank lines left behind
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        # Step 3: Find and replace abstract
        abstract_content = sections.get("abstract", "")
        if abstract_content:
            # Escape unescaped % characters (they start LaTeX comments)
            abstract_content = re.sub(r'(?<!\\)%', r'\\%', abstract_content)
            
            has_env = '\\begin{abstract}' in result
            # Detect \abstract{...} command form (Nature/Springer templates)
            has_cmd = bool(re.search(r'(?<!\\begin\{)\\abstract\{', result))
            
            if has_cmd:
                # Replace \abstract{...} command content (may span multiple lines)
                result = self._replace_abstract_command(result, abstract_content)
            
            if has_env:
                result = re.sub(
                    r'\\begin\{abstract\}.*?\\end\{abstract\}',
                    lambda m: f'\\begin{{abstract}}\n{abstract_content}\n\\end{{abstract}}',
                    result,
                    flags=re.DOTALL
                )
                if has_cmd:
                    # Both forms exist — remove the \abstract{} command to avoid duplication
                    result = self._remove_abstract_command(result)
            
            if not has_env and not has_cmd:
                if '\\maketitle' in result:
                    result = result.replace(
                        '\\maketitle',
                        f'\\maketitle\n\n\\begin{{abstract}}\n{abstract_content}\n\\end{{abstract}}'
                    )
        
        # Step 4: Replace ALL template body content with generated content
        body_content = sections.get("body", "")
        if body_content:
            body_content = re.sub(r'(?<!\\)%', r'\\%', body_content)
            body_content = re.sub(r'\\% === Section: \w+ ===\n?', '', body_content)
            
            # Extract bibliography commands from the template before replacement
            bib_commands = self._extract_bib_commands(result)
            
            if '\\end{abstract}' in result and '\\end{document}' in result:
                result = re.sub(
                    r'(\\end\{abstract\}).*?(\\end\{document\})',
                    lambda m: f'{m.group(1)}\n\n{body_content}\n\n{bib_commands}\n{m.group(2)}',
                    result,
                    flags=re.DOTALL
                )
            elif '\\maketitle' in result and '\\end{document}' in result:
                # Fallback for templates without \end{abstract} (e.g. Nature \abstract{...})
                # Find the anchor: after \abstract{...} if present, else after \maketitle
                abstract_cmd_match = re.search(r'\\abstract\{', result)
                if abstract_cmd_match:
                    # Find end of \abstract{...} using brace matching
                    anchor_end = self._find_brace_end(result, abstract_cmd_match.start() + len('\\abstract'))
                else:
                    anchor_end = result.index('\\maketitle') + len('\\maketitle')
                
                end_doc_match = re.search(r'\\end\{document\}', result)
                if end_doc_match:
                    result = (result[:anchor_end]
                              + f'\n\n{body_content}\n\n{bib_commands}\n'
                              + result[end_doc_match.start():])
            else:
                if '\\maketitle' not in result:
                    result = result.replace(
                        '\\begin{document}',
                        f'\\begin{{document}}\n\n\\maketitle\n\n{body_content}'
                    )
                else:
                    result = result.replace(
                        '\\begin{document}',
                        f'\\begin{{document}}\n\n{body_content}'
                    )
        
        # Step 5: Ensure bibliography is included (fallback if not preserved from template)
        if bib_entries and '\\bibliography' not in result and '\\printbibliography' not in result:
            # Detect which style to use based on template packages
            if 'icml' in result.lower() or 'natbib' in result.lower():
                bib_style = 'icml2026'  # ICML uses natbib with author-year style
            elif 'neurips' in result.lower() or 'nips' in result.lower():
                bib_style = 'plainnat'  # NeurIPS typically uses plainnat
            else:
                bib_style = 'plainnat'  # Default to plainnat for natbib compatibility
            
            bib_command = f'\\bibliographystyle{{{bib_style}}}\n\\bibliography{{references}}\n'
            result = result.replace(
                '\\end{document}',
                f'\n{bib_command}\n\\end{{document}}'
            )

        # Some templates rely on \maketitle to render title/abstract metadata.
        # Ensure it exists even if earlier body replacement paths removed it.
        result = self._ensure_maketitle_present(result)
        
        return result

    @staticmethod
    def _validate_compiled_tex_structure(compiled_tex: str) -> List[str]:
        """
        Validate compiled main.tex has non-empty title and abstract.
        - **Description**:
            - Checks title command content.
            - Checks abstract command/environment content.

        - **Args**:
            - `compiled_tex` (str): Fully injected LaTeX content.

        - **Returns**:
            - `List[str]`: Validation errors; empty list means pass.
        """
        errors: List[str] = []

        title_match = re.search(r'\\title(?:\[[^\]]*\])?\{([^}]*)\}', compiled_tex, flags=re.DOTALL)
        if not title_match or not title_match.group(1).strip():
            errors.append("missing_or_empty_title")

        abstract_cmd = re.search(r'\\abstract\{([^}]*)\}', compiled_tex, flags=re.DOTALL)
        abstract_env = re.search(
            r'\\begin\{abstract\}(.*?)\\end\{abstract\}',
            compiled_tex,
            flags=re.DOTALL,
        )
        abstract_text = ""
        if abstract_cmd:
            abstract_text = abstract_cmd.group(1)
        elif abstract_env:
            abstract_text = abstract_env.group(1)
        if not abstract_text.strip():
            errors.append("missing_or_empty_abstract")

        return errors

    @staticmethod
    def _ensure_maketitle_present(text: str) -> str:
        """
        Ensure \\maketitle exists in the document body.
        - **Description**:
            - Inserts \\maketitle if missing.
            - Prefers insertion after abstract content when available.

        - **Args**:
            - `text` (str): LaTeX source text.

        - **Returns**:
            - `str`: LaTeX text guaranteed to contain \\maketitle.
        """
        if '\\maketitle' in text or '\\begin{document}' not in text:
            return text

        anchor = text.index('\\begin{document}') + len('\\begin{document}')

        abstract_cmd_match = re.search(r'\\abstract\{', text)
        if abstract_cmd_match:
            end = TypesetterAgent._find_brace_end(text, abstract_cmd_match.start() + len('\\abstract'))
            anchor = max(anchor, end)
        else:
            abstract_env_match = re.search(r'\\begin\{abstract\}.*?\\end\{abstract\}', text, flags=re.DOTALL)
            if abstract_env_match:
                anchor = max(anchor, abstract_env_match.end())

        return text[:anchor] + '\n\n\\maketitle' + text[anchor:]

    @staticmethod
    def _replace_all_authors(text: str, new_author: str) -> str:
        """
        Remove all \\author commands and replace with a single clean one.
        - **Description**:
            - Handles simple \\author{...}, \\author*[...]{...\\fnm{}\\sur{}},
              and multi-line author blocks with nested braces.
            - Inserts one clean \\author{new_author} at the position of the
              first original \\author command.

        - **Args**:
            - `text` (str): LaTeX source
            - `new_author` (str): Replacement author string

        - **Returns**:
            - `str`: Text with all author commands replaced
        """
        first_pos = None
        # Find all \author variants with brace-matching for nested content
        i = 0
        regions_to_remove = []
        while i < len(text):
            # Look for \author (optionally followed by * and/or [...])
            match = re.search(r'\\author\*?', text[i:])
            if not match:
                break
            start = i + match.start()
            # Skip commented-out author samples (e.g., "%% \\author...")
            line_start = text.rfind('\n', 0, start) + 1
            if re.match(r'^\s*%', text[line_start:start]):
                i = start + len(match.group(0))
                continue
            pos = start + match.end() - match.start()
            # Skip optional [...]
            if pos < len(text) and text[pos] == '[':
                bracket_depth = 1
                pos += 1
                while pos < len(text) and bracket_depth > 0:
                    if text[pos] == '[':
                        bracket_depth += 1
                    elif text[pos] == ']':
                        bracket_depth -= 1
                    pos += 1
            # Match required {...} with nested brace support
            if pos < len(text) and text[pos] == '{':
                brace_depth = 1
                pos += 1
                while pos < len(text) and brace_depth > 0:
                    if text[pos] == '{':
                        brace_depth += 1
                    elif text[pos] == '}':
                        brace_depth -= 1
                    pos += 1
                if first_pos is None:
                    first_pos = start
                regions_to_remove.append((start, pos))
            i = pos if pos > start else start + 1

        if not regions_to_remove:
            title_match = re.search(r'\\title(?:\[[^\]]*\])?\{[^}]*\}', text, flags=re.DOTALL)
            insertion = f'\n\\author{{{new_author}}}\n'
            if title_match:
                insert_pos = title_match.end()
                return text[:insert_pos] + insertion + text[insert_pos:]
            return text

        # Remove regions in reverse order to preserve indices
        for start, end in reversed(regions_to_remove):
            text = text[:start] + text[end:]

        # Insert the new author at the first original position
        replacement = f'\\author{{{new_author}}}'
        text = text[:first_pos] + replacement + text[first_pos:]
        return text

    @staticmethod
    def _replace_abstract_command(text: str, new_content: str) -> str:
        """
        Replace the content inside \\abstract{...} command (brace-matched).
        - **Description**:
            - Handles nested braces and multi-line content.

        - **Args**:
            - `text` (str): LaTeX source
            - `new_content` (str): New abstract text

        - **Returns**:
            - `str`: Text with abstract command content replaced
        """
        pattern_start = '\\abstract{'
        start_idx = text.find(pattern_start)
        if start_idx == -1:
            return text
        brace_count = 1
        content_start = start_idx + len(pattern_start)
        i = content_start
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        if brace_count == 0:
            return text[:start_idx] + f'\\abstract{{{new_content}}}' + text[i:]
        return text

    @staticmethod
    def _remove_abstract_command(text: str) -> str:
        """
        Remove \\abstract{...} command entirely (brace-matched).
        - **Description**:
            - Used when both \\abstract{} and \\begin{abstract} exist to
              eliminate duplication.

        - **Args**:
            - `text` (str): LaTeX source

        - **Returns**:
            - `str`: Text with abstract command removed
        """
        pattern_start = '\\abstract{'
        start_idx = text.find(pattern_start)
        if start_idx == -1:
            return text
        brace_count = 1
        i = start_idx + len(pattern_start)
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        if brace_count == 0:
            return text[:start_idx] + text[i:]
        return text

    @staticmethod
    def _extract_bib_commands(text: str) -> str:
        """
        Extract and normalise bibliography commands from template text.
        - **Returns**:
            - `str`: Bibliography commands with file name replaced by 'references'.
        """
        bib_commands = ""
        bib_style_match = re.search(r'\\bibliographystyle\{[^}]+\}', text)
        bib_file_match = re.search(r'\\bibliography\{[^}]+\}', text)
        printbib_match = re.search(r'\\printbibliography', text)
        if bib_style_match:
            bib_commands += bib_style_match.group(0) + "\n"
        if bib_file_match:
            bib_commands += "\\bibliography{references}\n"
        elif printbib_match:
            bib_commands += "\\printbibliography\n"
        return bib_commands

    @staticmethod
    def _find_brace_end(text: str, open_brace_pos: int) -> int:
        """
        Find the position just after the closing brace matching the one at open_brace_pos.
        - **Args**:
            - `text` (str): Source text.
            - `open_brace_pos` (int): Index of the opening '{'.
        - **Returns**:
            - `int`: Index just past the matching '}'.  Falls back to open_brace_pos+1.
        """
        depth = 0
        i = open_brace_pos
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return open_brace_pos + 1

    async def inject_template(self, state: TypesetterAgentState) -> Dict[str, Any]:
        """
        Inject content into LaTeX template
        - **Description**:
            - Supports two modes:
              1. Legacy: single latex_content string, parsed and injected inline
              2. Multi-file: sections dict, each section written to its own .tex file,
                 main.tex uses \\input{sections/xxx} commands
            - Smart injection that handles title, abstract, and sections
            - Uses detected main_tex_path from template
            - Falls back to building from TemplateConfig if no template

        - **Args**:
            - `state` (TypesetterAgentState): Current workflow state

        - **Returns**:
            - `dict`: Updated state with compiled_tex and optionally section_file_map
        """
        logger.info("typesetter.inject_template start")
        
        work_dir = state.get("work_dir")
        latex_content = state.get("latex_content", "")
        sections_dict = state.get("sections")  # Multi-file mode
        section_order = state.get("section_order")
        section_titles = state.get("section_titles")
        resources = state.get("resources", [])
        bib_entries = state.get("bib_entries", [])
        template_config = state.get("template_config")
        main_tex_path = state.get("main_tex_path")
        figure_paths = state.get("figure_paths", {}) or {}

        id_to_rel_path: Dict[str, str] = {}
        for resource in resources:
            if resource.status == "downloaded" and resource.local_path:
                rel_path = os.path.relpath(resource.local_path, work_dir)
                id_to_rel_path[str(resource.resource_id)] = self._strip_graphics_extension(rel_path)
        for fig_id, file_path in figure_paths.items():
            if not fig_id:
                continue
            filename = os.path.basename(file_path or "")
            if not filename:
                continue
            candidate = os.path.join("figures", filename)
            if os.path.exists(os.path.join(work_dir, candidate)):
                id_to_rel_path[str(fig_id)] = self._strip_graphics_extension(candidate)
        
        # Create default template_config if not provided
        if template_config is None:
            template_config = TemplateConfig()
        if not (template_config.paper_title or "").strip():
            raise ValueError("typesetter.inject_template requires non-empty template_config.paper_title")
        
        # Determine final main.tex path
        final_main_tex = os.path.join(work_dir, "main.tex")
        
        # =====================================================================
        # Multi-file mode: sections dict provided
        # =====================================================================
        if sections_dict:
            logger.info("typesetter.inject_template mode=multi-file sections=%s",
                        list(sections_dict.keys()))
            
            # Resolve includegraphics targets in each section.
            for sec_type in list(sections_dict.keys()):
                sections_dict[sec_type] = self._rewrite_includegraphics_targets(
                    sections_dict[sec_type],
                    work_dir=work_dir,
                    id_to_rel_path=id_to_rel_path,
                )
            
            # Detect whether the template uses the appendix package's
            # \begin{appendices} environment (e.g. Springer Nature templates).
            _use_appendices_env = False
            if main_tex_path and os.path.exists(main_tex_path):
                with open(main_tex_path, "r", encoding="utf-8", errors="ignore") as _tf:
                    _tpl_src = _tf.read()
                _use_appendices_env = bool(
                    re.search(r'\\usepackage(?:\[.*?\])?\{appendix\}', _tpl_src)
                    or r'\begin{appendices}' in _tpl_src
                )
                if _use_appendices_env:
                    logger.info("typesetter.appendix_format detected=appendices_env")

            # Write individual section files
            section_file_map = self._write_section_files(
                work_dir=work_dir,
                sections=sections_dict,
                section_order=section_order,
                section_titles=section_titles,
                citation_style=template_config.citation_style,
                use_appendices_env=_use_appendices_env,
            )
            
            # Build \input{} commands for body sections
            order = section_order or self.DEFAULT_SECTION_ORDER
            input_commands = []
            for sec_type in order:
                if sec_type in section_file_map:
                    input_commands.append(f"\\input{{{section_file_map[sec_type]}}}")
            # Add any extra sections not in the order
            for sec_type, rel_path in section_file_map.items():
                if sec_type != "abstract" and sec_type not in order:
                    input_commands.append(f"\\input{{{rel_path}}}")
            
            body_input_text = "\n\n".join(input_commands)
            
            # Abstract is inlined into main.tex (do not use sections/abstract.tex)
            abstract_inline = ""
            if sections_dict.get("abstract", "").strip():
                abstract_inline = sections_dict["abstract"].strip()
                abstract_inline = re.sub(r'^\\begin\{abstract\}\s*', '', abstract_inline)
                abstract_inline = re.sub(r'\s*\\end\{abstract\}$', '', abstract_inline)
                abstract_inline = self._apply_citation_style(abstract_inline, template_config.citation_style)
                abstract_inline = re.sub(r'(?<!\\)%', r'\\%', abstract_inline).strip()
            if not abstract_inline:
                raise ValueError("typesetter.inject_template missing abstract content in multi-file mode")

            # Conclusion is also inlined into main.tex (do not use sections/conclusion.tex)
            conclusion_inline = ""
            if sections_dict.get("conclusion", "").strip():
                conclusion_text = sections_dict["conclusion"].strip()
                conclusion_text = self._strip_leading_section_command(conclusion_text)
                conclusion_text = self._apply_citation_style(conclusion_text, template_config.citation_style)
                conclusion_text = re.sub(r'(?<!\\)%', r'\\%', conclusion_text)
                conclusion_title = section_titles.get("conclusion", "Conclusion") if section_titles else "Conclusion"
                conclusion_inline = f"\\section{{{conclusion_title}}}\n\n{conclusion_text}"
            
            # Inject into template
            if main_tex_path and os.path.exists(main_tex_path):
                logger.info("typesetter.using_template file=%s", os.path.basename(main_tex_path))
                with open(main_tex_path, "r", encoding="utf-8", errors="ignore") as f:
                    template_content = f.read()
                
                # Build a sections dict compatible with _smart_inject_content
                # but using \input commands instead of inline content
                input_sections = {
                    "abstract": abstract_inline,
                    "body": "\n\n".join([p for p in [body_input_text, conclusion_inline] if p.strip()]),
                }
                compiled_tex = self._smart_inject_content(
                    template_content, input_sections, template_config, bib_entries
                )
                structure_errors = self._validate_compiled_tex_structure(compiled_tex)
                if structure_errors:
                    raise ValueError(
                        f"typesetter.inject_template structure validation failed: {structure_errors}"
                    )
                
                # Copy template directory files if needed
                if main_tex_path != final_main_tex:
                    template_dir = os.path.dirname(main_tex_path)
                    if template_dir != work_dir:
                        for item in os.listdir(template_dir):
                            src = os.path.join(template_dir, item)
                            dst = os.path.join(work_dir, item)
                            if os.path.isfile(src) and not os.path.exists(dst):
                                shutil.copy2(src, dst)
            else:
                # No template - build from config with \input commands
                logger.info("typesetter.no_template building_from_config mode=multi-file")
                preamble = self._build_preamble_from_config(template_config)
                
                full_content = ""
                if abstract_inline:
                    full_content = f"\\begin{{abstract}}\n{abstract_inline}\n\\end{{abstract}}\n\n"
                full_content += "\n\n".join([p for p in [body_input_text, conclusion_inline] if p.strip()])
                
                bib_style = template_config.bib_style or "plain"
                template = Template(MAIN_TEX_TEMPLATE)
                compiled_tex = template.render(
                    preamble=preamble,
                    content=full_content,
                    has_bibliography=len(bib_entries) > 0,
                    bib_style=bib_style,
                )
                structure_errors = self._validate_compiled_tex_structure(compiled_tex)
                if structure_errors:
                    raise ValueError(
                        f"typesetter.inject_template structure validation failed: {structure_errors}"
                    )
            
            # Write main.tex
            with open(final_main_tex, "w", encoding="utf-8") as f:
                f.write(compiled_tex)
            logger.info("typesetter.written_main_tex chars=%d mode=multi-file", len(compiled_tex))
            
            return {"compiled_tex": compiled_tex, "section_file_map": section_file_map}
        
        # =====================================================================
        # Legacy mode: single latex_content string
        # =====================================================================
        logger.info("typesetter.inject_template mode=legacy")
        
        # Apply citation style to content
        latex_content = self._apply_citation_style(latex_content, template_config.citation_style)
        
        # Resolve includegraphics targets in legacy content.
        latex_content = self._rewrite_includegraphics_targets(
            latex_content,
            work_dir=work_dir,
            id_to_rel_path=id_to_rel_path,
        )
        
        # Parse sections from content
        sections = self._parse_sections_from_content(latex_content)
        logger.info("typesetter.parsed_sections abstract_chars=%d body_chars=%d", 
                    len(sections.get('abstract', '')), len(sections.get('body', '')))
        
        if main_tex_path and os.path.exists(main_tex_path):
            # Use detected template main tex
            logger.info("typesetter.using_template file=%s", os.path.basename(main_tex_path))
            with open(main_tex_path, "r", encoding="utf-8", errors="ignore") as f:
                template_content = f.read()
            
            # Smart inject content
            compiled_tex = self._smart_inject_content(
                template_content, 
                sections, 
                template_config, 
                bib_entries
            )
            
            # If the original file wasn't main.tex, we still write to main.tex
            # but also copy other template files
            if main_tex_path != final_main_tex:
                template_dir = os.path.dirname(main_tex_path)
                if template_dir != work_dir:
                    for item in os.listdir(template_dir):
                        src = os.path.join(template_dir, item)
                        dst = os.path.join(work_dir, item)
                        if os.path.isfile(src) and not os.path.exists(dst):
                            shutil.copy2(src, dst)
        else:
            # No template file - build from TemplateConfig
            logger.info("typesetter.no_template building_from_config")
            preamble = self._build_preamble_from_config(template_config)
            
            # Combine abstract and body for content
            full_content = ""
            if sections.get("abstract"):
                full_content = f"\\begin{{abstract}}\n{sections['abstract']}\n\\end{{abstract}}\n\n"
            full_content += sections.get("body", latex_content)
            
            bib_style = template_config.bib_style or "plain"
            
            template = Template(MAIN_TEX_TEMPLATE)
            compiled_tex = template.render(
                preamble=preamble,
                content=full_content,
                has_bibliography=len(bib_entries) > 0,
                bib_style=bib_style,
            )
        
        # Write main.tex
        with open(final_main_tex, "w", encoding="utf-8") as f:
            f.write(compiled_tex)
        logger.info("typesetter.written_main_tex chars=%d mode=legacy", len(compiled_tex))
        
        return {"compiled_tex": compiled_tex}

    def _copy_to_output_dir(self, work_dir: str, output_dir: str) -> Dict[str, str]:
        """
        Copy compilation results to output directory
        - **Description**:
            - Creates standard output structure with tex, bib, figures, sty, and pdf
            - Returns paths to copied files

        - **Args**:
            - `work_dir` (str): Temporary working directory
            - `output_dir` (str): Target output directory

        - **Returns**:
            - `dict`: Paths to pdf_path and source_path in output_dir
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create figures subdirectory
        output_figures = os.path.join(output_dir, "figures")
        os.makedirs(output_figures, exist_ok=True)
        
        result_paths = {"pdf_path": None, "source_path": output_dir}
        
        # Files to copy (with their target names/locations)
        files_to_copy = []
        
        for item in os.listdir(work_dir):
            src_path = os.path.join(work_dir, item)
            
            if os.path.isfile(src_path):
                # Determine destination based on file type
                if item.endswith('.pdf') and item == 'main.pdf':
                    dst_path = os.path.join(output_dir, 'main.pdf')
                    result_paths["pdf_path"] = dst_path
                    files_to_copy.append((src_path, dst_path))
                elif item.endswith(('.tex', '.bib', '.bst', '.sty', '.cls', '.bbl', '.blg', '.aux', '.log')):
                    dst_path = os.path.join(output_dir, item)
                    files_to_copy.append((src_path, dst_path))
                elif item.endswith(('.png', '.jpg', '.jpeg', '.pdf', '.eps')) and item != 'main.pdf':
                    # Image files go to figures/
                    dst_path = os.path.join(output_figures, item)
                    files_to_copy.append((src_path, dst_path))
            elif os.path.isdir(src_path) and item == 'figures':
                # Copy entire figures directory
                for fig_file in os.listdir(src_path):
                    fig_src = os.path.join(src_path, fig_file)
                    if os.path.isfile(fig_src):
                        fig_dst = os.path.join(output_figures, fig_file)
                        files_to_copy.append((fig_src, fig_dst))
            elif os.path.isdir(src_path) and item == 'sections':
                # Copy sections directory (multi-file mode)
                output_sections = os.path.join(output_dir, "sections")
                os.makedirs(output_sections, exist_ok=True)
                for sec_file in os.listdir(src_path):
                    sec_src = os.path.join(src_path, sec_file)
                    if os.path.isfile(sec_src):
                        sec_dst = os.path.join(output_sections, sec_file)
                        files_to_copy.append((sec_src, sec_dst))
        
        # Perform the copy
        for src, dst in files_to_copy:
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                logger.warning("typesetter.copy_failed src=%s dst=%s error=%s", src, dst, str(e))
        
        return result_paths

    async def compile_latex(self, state: TypesetterAgentState) -> Dict[str, Any]:
        """
        Compile LaTeX with self-healing
        - **Description**:
            - Runs pdflatex -> bibtex -> pdflatex compilation
            - Copies results to output_dir if specified
            - Attempts to fix common errors automatically

        - **Args**:
            - `state` (TypesetterAgentState): Current workflow state

        - **Returns**:
            - `dict`: Updated state with compilation_result
        """
        logger.info("typesetter.compile_latex start")
        
        work_dir = state.get("work_dir")
        output_dir = state.get("output_dir")
        section_file_map = state.get("section_file_map")  # From inject_template multi-file mode
        main_tex = os.path.join(work_dir, "main.tex")
        
        result = CompilationResult(
            success=False,
            attempts=0,
            errors=[],
            warnings=[],
        )
        
        # Populate section_files if we have the mapping
        if section_file_map:
            result.section_files = {
                sec_type: os.path.join(work_dir, rel_path + ".tex")
                for sec_type, rel_path in section_file_map.items()
            }
        
        for attempt in range(MAX_COMPILE_ATTEMPTS):
            result.attempts = attempt + 1
            logger.info("typesetter.compile attempt=%d/%d", attempt + 1, MAX_COMPILE_ATTEMPTS)
            
            try:
                # First pdflatex pass
                logger.info("typesetter.pdflatex pass=1")
                proc1 = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-output-directory", work_dir, main_tex],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",  # Handle non-UTF-8 chars in pdflatex output
                    timeout=60,
                    cwd=work_dir,
                )
                
                # Run bibtex if references exist
                bib_file = os.path.join(work_dir, "references.bib")
                if os.path.exists(bib_file):
                    logger.info("typesetter.bibtex")
                    # NOTE: bibtex expects the AUX *basename* (without extension) in cwd.
                    # Passing an absolute path can cause bibtex to fail to locate/write files.
                    aux_name = "main"
                    bibtex_proc = subprocess.run(
                        ["bibtex", aux_name],
                        capture_output=True,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        timeout=30,
                        cwd=work_dir,
                    )
                    if bibtex_proc.returncode != 0:
                        logger.warning(
                            "typesetter.bibtex_failed code=%s stderr=%s",
                            bibtex_proc.returncode,
                            (bibtex_proc.stderr or "").strip()[:2000],
                        )
                    else:
                        logger.info("typesetter.bibtex_ok")
                    
                    bbl_path = os.path.join(work_dir, "main.bbl")
                    if not os.path.exists(bbl_path):
                        logger.warning("typesetter.bbl_missing path=%s", bbl_path)
                
                # Second pdflatex pass
                logger.info("typesetter.pdflatex pass=2")
                proc2 = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-output-directory", work_dir, main_tex],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                    cwd=work_dir,
                )
                
                # Third pass for references
                logger.info("typesetter.pdflatex pass=3")
                proc3 = subprocess.run(
                    ["pdflatex", "-interaction=nonstopmode", "-output-directory", work_dir, main_tex],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=60,
                    cwd=work_dir,
                )
                
                # Check for PDF
                pdf_path = os.path.join(work_dir, "main.pdf")
                if os.path.exists(pdf_path):
                    result.success = True
                    result.pdf_path = pdf_path
                    result.source_path = work_dir
                    
                    # Parse log for warnings
                    log_path = os.path.join(work_dir, "main.log")
                    if os.path.exists(log_path):
                        with open(log_path, "r", errors="ignore") as f:
                            log_content = f.read()
                            result.log_content = log_content[-5000:]
                            result.warnings = self._extract_warnings(log_content)
                            # Also extract section-level errors (may exist even on success)
                            if section_file_map:
                                result.errors = self._extract_errors(log_content)
                                result.section_errors = self._extract_section_errors(
                                    log_content, section_file_map
                                )
                    
                    logger.info("typesetter.compile_success")
                    
                    # Copy to output_dir if specified
                    if output_dir:
                        logger.info("typesetter.copying_to_output output_dir=%s", output_dir)
                        output_paths = self._copy_to_output_dir(work_dir, output_dir)
                        result.pdf_path = output_paths["pdf_path"]
                        result.source_path = output_paths["source_path"]
                        logger.info("typesetter.output_complete pdf=%s", result.pdf_path)
                    
                    break
                else:
                    # Parse errors from log
                    log_path = os.path.join(work_dir, "main.log")
                    if os.path.exists(log_path):
                        with open(log_path, "r", errors="ignore") as f:
                            log_content = f.read()
                            result.log_content = log_content[-5000:]
                            result.errors = self._extract_errors(log_content)
                            # Extract per-section errors if multi-file mode
                            if section_file_map:
                                result.section_errors = self._extract_section_errors(
                                    log_content, section_file_map
                                )
                    
                    logger.warning("typesetter.compile_failed errors=%s section_errors=%s",
                                   result.errors[:2],
                                   {k: v[:1] for k, v in result.section_errors.items()} if result.section_errors else {})
                    
                    # Try to fix errors
                    if attempt < MAX_COMPILE_ATTEMPTS - 1:
                        fixed = await self._try_fix_errors(work_dir, main_tex, result.errors)
                        if not fixed:
                            break
                    
            except subprocess.TimeoutExpired:
                result.errors.append("Compilation timed out")
                logger.error("typesetter.compile_timeout")
                break
            except FileNotFoundError:
                result.errors.append("pdflatex not found. Please install TeX distribution.")
                logger.error("typesetter.pdflatex_not_found")
                break
            except Exception as e:
                result.errors.append(f"Compilation error: {str(e)}")
                logger.error("typesetter.compile_error error=%s", str(e))
                break
        
        return {"compilation_result": result}

    def _extract_errors(self, log_content: str) -> List[str]:
        """Extract error messages from LaTeX log"""
        errors = []
        
        # Pattern for LaTeX errors
        error_patterns = [
            r'! (.*?)(?:\n|$)',
            r'Error: (.*?)(?:\n|$)',
            r'Fatal error occurred, (.*?)(?:\n|$)',
        ]
        
        for pattern in error_patterns:
            matches = re.findall(pattern, log_content)
            errors.extend(matches[:5])  # Limit to 5 errors
        
        return list(set(errors))[:10]

    def _extract_warnings(self, log_content: str) -> List[str]:
        """Extract warning messages from LaTeX log"""
        warnings = []
        
        warning_patterns = [
            r'Warning: (.*?)(?:\n|$)',
            r'LaTeX Warning: (.*?)(?:\n|$)',
        ]
        
        for pattern in warning_patterns:
            matches = re.findall(pattern, log_content)
            warnings.extend(matches[:5])
        
        return list(set(warnings))[:10]

    def _extract_section_errors(
        self,
        log_content: str,
        section_file_map: Dict[str, str],
    ) -> Dict[str, List[str]]:
        """
        Extract per-section errors from LaTeX log using file tracking.
        - **Description**:
            - pdflatex logs file switches with parentheses: (./sections/intro.tex ... )
            - Tracks the current file context as the log is parsed line by line
            - Maps errors to their source section based on the active file

        - **Args**:
            - `log_content` (str): Full LaTeX compilation log
            - `section_file_map` (Dict[str, str]): section_type -> relative path (no .tex)

        - **Returns**:
            - `section_errors` (Dict[str, List[str]]): section_type -> list of error messages
        """
        # Build reverse mapping: filename -> section_type
        # section_file_map is like {"introduction": "sections/introduction"}
        file_to_section: Dict[str, str] = {}
        for section_type, rel_path in section_file_map.items():
            # Match both "sections/introduction.tex" and "./sections/introduction.tex"
            fname = rel_path + ".tex"
            file_to_section[fname] = section_type
            file_to_section["./" + fname] = section_type

        section_errors: Dict[str, List[str]] = {}
        
        # Track the file stack (pdflatex uses nested parentheses for file context)
        file_stack: List[str] = []
        current_section: Optional[str] = None
        
        lines = log_content.split('\n')
        for line in lines:
            # Track file context changes via parentheses
            # Opening: (./sections/introduction.tex
            # Closing: )
            # These can appear multiple times on a single line
            i = 0
            while i < len(line):
                if line[i] == '(' and i + 1 < len(line):
                    # Find the file path after '('
                    rest = line[i + 1:]
                    # Extract path: up to space, newline, or another '('
                    match = re.match(r'([^\s()]+)', rest)
                    if match:
                        fpath = match.group(1)
                        file_stack.append(fpath)
                        # Check if this file is one of our sections
                        sec = file_to_section.get(fpath)
                        if sec:
                            current_section = sec
                        i += 1 + len(match.group(0))
                        continue
                elif line[i] == ')':
                    if file_stack:
                        popped = file_stack.pop()
                        popped_sec = file_to_section.get(popped)
                        if popped_sec and popped_sec == current_section:
                            # Exiting the current section file
                            # Recompute current_section from remaining stack
                            current_section = None
                            for f in reversed(file_stack):
                                sec = file_to_section.get(f)
                                if sec:
                                    current_section = sec
                                    break
                    i += 1
                    continue
                i += 1
            
            # Check if this line contains an error
            if line.startswith('!'):
                error_msg = line[2:].strip() if len(line) > 2 else line.strip()
                if current_section and error_msg:
                    if current_section not in section_errors:
                        section_errors[current_section] = []
                    section_errors[current_section].append(error_msg)
            
            # Also check for line-number references: "l.42 ..."
            # These confirm the error location within the current file
            line_match = re.match(r'^l\.(\d+)\s+(.*)', line)
            if line_match and current_section:
                line_num = line_match.group(1)
                context = line_match.group(2).strip()
                if context and section_errors.get(current_section):
                    # Append line context to the last error for this section
                    last_err = section_errors[current_section][-1]
                    if f" (line {line_num})" not in last_err:
                        section_errors[current_section][-1] = f"{last_err} (line {line_num})"

        # Limit errors per section
        for sec_type in section_errors:
            section_errors[sec_type] = section_errors[sec_type][:10]
        
        if section_errors:
            logger.info("typesetter.section_errors %s",
                        {k: len(v) for k, v in section_errors.items()})
        
        return section_errors

    async def _try_fix_errors(self, work_dir: str, main_tex: str, errors: List[str]) -> bool:
        """
        Try to fix common LaTeX errors
        - **Description**:
            - Uses LLM to suggest fixes for compilation errors

        - **Returns**:
            - `bool`: True if fix was attempted
        """
        if not errors:
            return False
        
        try:
            with open(main_tex, "r", encoding="utf-8") as f:
                tex_content = f.read()
            
            # Try simple fixes first
            fixed_content = tex_content
            
            # Fix common issues
            for error in errors:
                error_lower = error.lower()
                
                if "undefined control sequence" in error_lower:
                    # Try to comment out problematic command
                    pass
                elif "missing $ inserted" in error_lower:
                    # Math mode issue - hard to auto-fix
                    pass
                elif "file not found" in error_lower:
                    # Missing file - replace with placeholder
                    match = re.search(r"file `([^']+)'", error, re.IGNORECASE)
                    if match:
                        missing_file = match.group(1)
                        fixed_content = fixed_content.replace(
                            f"\\includegraphics{{{missing_file}}}",
                            "% [Figure not available]"
                        )
            
            if fixed_content != tex_content:
                with open(main_tex, "w", encoding="utf-8") as f:
                    f.write(fixed_content)
                return True
                
        except Exception as e:
            print(f"Error fixing LaTeX: {e}")
        
        return False

    async def run(self,
                  latex_content: str = "",
                  sections: Optional[Dict[str, str]] = None,
                  section_order: Optional[List[str]] = None,
                  section_titles: Optional[Dict[str, str]] = None,
                  template_path: Optional[str] = None,
                  template_config: Optional[TemplateConfig] = None,
                  figure_ids: Optional[List[str]] = None,
                  citation_ids: Optional[List[str]] = None,
                  references: Optional[List[Dict[str, Any]]] = None,
                  work_id: Optional[str] = None,
                  output_dir: Optional[str] = None,
                  figures_source_dir: Optional[str] = None,
                  figure_paths: Optional[Dict[str, str]] = None,
                  converted_tables: Optional[Dict[str, str]] = None):
        """
        Run the Typesetter Agent
        - **Description**:
            - Supports two content modes (mutually exclusive):
              1. latex_content: Single concatenated LaTeX body (legacy)
              2. sections: Per-section content dict for multi-file output

        - **Args**:
            - `latex_content` (str): LaTeX content to compile (legacy single-string mode)
            - `sections` (Dict[str, str], optional): Per-section content (multi-file mode)
            - `section_order` (List[str], optional): Body section ordering for multi-file mode
            - `section_titles` (Dict[str, str], optional): section_type -> display title
            - `template_path` (str, optional): Path to template zip
            - `template_config` (TemplateConfig, optional): Template configuration with constraints
            - `figure_ids` (List[str], optional): Figure IDs to fetch
            - `citation_ids` (List[str], optional): Citation IDs
            - `references` (List[Dict], optional): Reference metadata
            - `work_id` (str, optional): Work ID for resource lookup
            - `output_dir` (str, optional): Directory to save final output files
            - `figures_source_dir` (str, optional): Local directory with figure files
            - `figure_paths` (Dict[str, str], optional): Structured figure paths (id -> file_path)
            - `converted_tables` (Dict[str, str], optional): Pre-converted table LaTeX (id -> code)

        - **Returns**:
            - `dict`: Compilation result with PDF path
        """
        return await self.agent.ainvoke({
            "latex_content": latex_content,
            "sections": sections,
            "section_order": section_order,
            "section_titles": section_titles,
            "template_path": template_path,
            "template_config": template_config,
            "figure_ids": figure_ids or [],
            "citation_ids": citation_ids or [],
            "references": references or [],
            "work_id": work_id,
            "output_dir": output_dir,
            "figures_source_dir": figures_source_dir,
            "figure_paths": figure_paths or {},
            "converted_tables": converted_tables or {},
            "messages": [],
            "llm_calls": 0,
        })

    @property
    def name(self) -> str:
        """Agent name identifier"""
        return "typesetter"

    @property
    def description(self) -> str:
        """Agent description"""
        return "Handles resource fetching, template injection, and LaTeX compilation with self-healing"

    @property
    def router(self) -> APIRouter:
        """Return the FastAPI router for this agent"""
        return create_typesetter_router(self)

    @property
    def endpoints_info(self) -> List[Dict[str, Any]]:
        """Return endpoint metadata for list_agents"""
        return [
            {
                "path": "/agent/typesetter/compile",
                "method": "POST",
                "description": "Compile LaTeX content into PDF with resource handling",
                "input_model": "TypesetterPayload",
                "output_model": "TypesetterResult"
            }
        ]
