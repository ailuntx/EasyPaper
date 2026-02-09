"""
VLM Review Agent
- **Description**:
    - Vision Language Model based PDF review agent
    - Detects page overflow, underfill, and layout issues
    - Provides recommendations for content adjustment
"""
import json
import logging
import os
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import APIRouter
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

from ..base import BaseAgent
from ...config.schema import ModelConfig
from .models import (
    VLMReviewRequest,
    VLMReviewResult,
    PageAnalysis,
    LayoutIssue,
    SectionAdvice,
    BlankSpace,
    IssueSeverity,
    IssueType,
    SECTION_TRIM_PRIORITY,
    SECTION_EXPAND_PRIORITY,
    WORDS_PER_PAGE,
)
from .providers.base import VLMProvider, VLMFactory
from .utils.pdf_renderer import PDFRenderer
from .utils.page_counter import PageCounter


logger = logging.getLogger("uvicorn.error")


# Prompts for VLM analysis
PAGE_ANALYSIS_PROMPT = """Analyze this academic paper page image. Respond with JSON only.

Identify:
1. **Page Fill**: Estimate what percentage (0-100) of the page has content vs blank space
2. **Content Boundary**: Does any text or figure extend beyond the normal margins?
3. **Blank Spaces**: List any significant empty areas (location and size)
4. **Layout Issues**: Identify widows (single line at top), orphans (single line at bottom), 
   badly placed figures/tables, or equations that overflow margins
5. **Page Type**: Determine if this page contains bibliography/references or appendix content
6. **Body Content Ratio**: Estimate what percentage (0-100) of this page is MAIN BODY content
   (title, abstract, introduction, methods, experiments, results, discussion, conclusion).
   Exclude any references/bibliography section and appendix content from this percentage.
   For example, if the top 70% of the page is the end of a section and the bottom 30% is
   the start of "References", body_content_percentage should be 70.

Return ONLY this JSON structure:
{
    "fill_percentage": <0-100>,
    "is_overflow": <true if content beyond margins>,
    "blank_spaces": [{"location": "<top/bottom/left/right/center>", "size": "<small/medium/large>"}],
    "layout_issues": [{"type": "<widow/orphan/bad_figure/equation_overflow>", "description": "<details>", "severity": "<low/medium/high>"}],
    "is_references_page": <true if this page contains bibliography/references>,
    "is_appendix_page": <true if this page is part of an appendix section (look for "Appendix" heading or "A.", "B." style section numbering)>,
    "body_content_percentage": <0-100, percentage of page area occupied by main body content (not references, not appendix)>
}"""

LAST_PAGE_PROMPT = """Analyze this LAST page of an academic paper. Respond with JSON only.

Identify:
1. **Page Fill**: Estimate what percentage (0-100) of the page has content vs blank space
2. **Content Boundary**: Does any text or figure extend beyond the normal margins?
3. **Blank Spaces**: List any significant empty areas (location and size)
4. **Layout Issues**: Identify widows, orphans, badly placed figures/tables, or equation overflows
5. **Page Type**: Is this a references/bibliography page, appendix page, or main body content page?
6. **Body Content Ratio**: What percentage (0-100) of this page is MAIN BODY content?
   Exclude references/bibliography and appendix from this count.
7. **Empty Space**: How many more lines of text could fit in the empty space?

Return ONLY this JSON:
{
    "fill_percentage": <0-100>,
    "is_overflow": <true if content beyond margins>,
    "blank_spaces": [{"location": "<top/bottom/left/right/center>", "size": "<small/medium/large>"}],
    "layout_issues": [{"type": "<widow/orphan/bad_figure/equation_overflow>", "description": "<details>", "severity": "<low/medium/high>"}],
    "is_references_page": <true if this page contains bibliography/references>,
    "is_appendix_page": <true if this page is part of an appendix section>,
    "body_content_percentage": <0-100, percentage of page area occupied by main body content>,
    "estimated_empty_lines": <number>,
    "recommendation": "<can_add_content/well_filled/too_empty>"
}"""


class VLMReviewState(TypedDict):
    """State for VLM Review workflow"""
    request: VLMReviewRequest
    pdf_path: str
    total_pages: int
    page_images: List[bytes]
    page_analyses: List[PageAnalysis]
    issues: List[LayoutIssue]
    overflow_detected: bool
    underfill_detected: bool
    result: Optional[VLMReviewResult]
    error: Optional[str]


class VLMReviewAgent(BaseAgent):
    """
    VLM Review Agent for PDF analysis
    
    - **Description**:
        - Uses Vision Language Models to analyze PDF layout
        - Detects page overflow, underfill, and layout issues
        - Provides section-level recommendations for content adjustment
    """
    
    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        vlm_review_config: Optional[Any] = None,
        vlm_provider: str = "openai",
        vlm_api_key: Optional[str] = None,
        vlm_model: Optional[str] = None,
        vlm_base_url: Optional[str] = None,
        render_dpi: int = 150,
        **kwargs
    ):
        """
        Initialize VLM Review Agent
        
        Args:
            model_config: Model configuration (from config file)
            vlm_review_config: VLM Review specific config (from config file)
            vlm_provider: VLM provider name (openai, claude, qwen)
            vlm_api_key: API key for VLM provider
            vlm_model: Model name to use
            vlm_base_url: Custom base URL (for OpenRouter, etc.)
            render_dpi: DPI for PDF rendering
            **kwargs: Additional options
        """
        self.model_config = model_config
        self.vlm_review_config = vlm_review_config
        
        # Read from vlm_review_config if available, otherwise use defaults
        if vlm_review_config:
            self.vlm_provider_name = vlm_review_config.provider or vlm_provider
            self.vlm_api_key = (
                vlm_review_config.vlm_api_key or 
                (model_config.api_key if model_config else None) or
                vlm_api_key or 
                os.environ.get("OPENAI_API_KEY")
            )
            self.vlm_model = (
                vlm_review_config.vlm_model or 
                (model_config.model_name if model_config else None) or
                vlm_model or 
                self._default_model(self.vlm_provider_name)
            )
            self.vlm_base_url = (
                vlm_review_config.vlm_base_url or
                (model_config.base_url if model_config else None) or
                vlm_base_url
            )
            self.render_dpi = vlm_review_config.render_dpi or render_dpi
        else:
            # Fallback to model_config or direct args
            self.vlm_provider_name = vlm_provider
            self.vlm_api_key = (
                (model_config.api_key if model_config else None) or
                vlm_api_key or 
                os.environ.get("OPENAI_API_KEY")
            )
            self.vlm_model = (
                (model_config.model_name if model_config else None) or
                vlm_model or 
                self._default_model(vlm_provider)
            )
            self.vlm_base_url = (
                (model_config.base_url if model_config else None) or
                vlm_base_url
            )
            self.render_dpi = render_dpi
        
        self.kwargs = kwargs
        
        # Initialize components
        self.pdf_renderer = PDFRenderer(dpi=self.render_dpi)
        self.page_counter = PageCounter()
        
        # VLM provider (lazy init)
        self._vlm: Optional[VLMProvider] = None
        
        # Build LangGraph workflow
        self.graph = self._build_graph()
    
    def _default_model(self, provider: str) -> str:
        """Get default model for provider"""
        defaults = {
            "openai": "gpt-4o",
            "claude": "claude-3-5-sonnet-20241022",
            "qwen": "qwen-vl-max",
        }
        return defaults.get(provider, "gpt-4o")
    
    @property
    def vlm(self) -> VLMProvider:
        """Get or create VLM provider"""
        if self._vlm is None:
            if not self.vlm_api_key:
                raise ValueError(f"API key required for VLM provider: {self.vlm_provider_name}")
            
            # Build kwargs with base_url if provided
            create_kwargs = dict(self.kwargs)
            if self.vlm_base_url:
                create_kwargs["base_url"] = self.vlm_base_url
            
            self._vlm = VLMFactory.create(
                provider=self.vlm_provider_name,
                api_key=self.vlm_api_key,
                model=self.vlm_model,
                **create_kwargs
            )
        return self._vlm
    
    def _build_graph(self) -> StateGraph:
        """Build LangGraph workflow for VLM review"""
        
        workflow = StateGraph(VLMReviewState)
        
        # Add nodes
        workflow.add_node("count_pages", self._count_pages)
        workflow.add_node("check_overflow", self._check_overflow_quick)
        workflow.add_node("render_pages", self._render_pages)
        workflow.add_node("analyze_pages", self._analyze_pages)
        workflow.add_node("generate_result", self._generate_result)
        
        # Add edges
        workflow.set_entry_point("count_pages")
        workflow.add_edge("count_pages", "check_overflow")
        workflow.add_conditional_edges(
            "check_overflow",
            self._should_analyze_with_vlm,
            {
                "analyze": "render_pages",
                "skip": "generate_result",
            }
        )
        workflow.add_edge("render_pages", "analyze_pages")
        workflow.add_edge("analyze_pages", "generate_result")
        workflow.add_edge("generate_result", END)
        
        return workflow.compile()
    
    # =========================================================================
    # Workflow Nodes
    # =========================================================================
    
    async def _count_pages(self, state: VLMReviewState) -> Dict[str, Any]:
        """Count PDF pages"""
        pdf_path = state["request"].pdf_path
        
        try:
            total_pages = self.page_counter.count_pages(pdf_path)
            print(f"[VLMReview] PDF has {total_pages} pages")
            return {"total_pages": total_pages, "pdf_path": pdf_path}
        except Exception as e:
            logger.error(f"Failed to count pages: {e}")
            return {"total_pages": 0, "error": str(e)}
    
    # References and appendix have NO page limit, so the quick check must
    # be generous.  A typical paper may have 2-5 pages of references alone.
    _QUICK_CHECK_BUFFER = 5

    async def _check_overflow_quick(self, state: VLMReviewState) -> Dict[str, Any]:
        """
        Quick check for page overflow using page count.
        - **Description**:
            - Page limits apply to the MAIN BODY only; references and appendix
              are unlimited.  Since this quick check has no VLM data to
              distinguish page types, we add a generous buffer to avoid
              false positives that waste VLM API calls.
            - The precise check happens later in _generate_result() after
              VLM classifies each page.
        """
        total_pages = state["total_pages"]
        page_limit = state["request"].page_limit
        buffer = self._QUICK_CHECK_BUFFER
        
        # Allow extra pages for references + appendix (no VLM data yet)
        quick_limit = page_limit + buffer
        is_overflow = total_pages > quick_limit
        overflow_pages = max(0, total_pages - quick_limit)
        
        if is_overflow:
            print(f"[VLMReview] QUICK OVERFLOW: {total_pages} pages > limit {page_limit}+{buffer} buffer")
            issue = LayoutIssue(
                issue_type=IssueType.OVERFLOW,
                severity=IssueSeverity.CRITICAL,
                description=f"PDF has {total_pages} pages, likely exceeds body limit of {page_limit}",
                page_number=page_limit + 1,
            )
            return {
                "overflow_detected": True,
                "issues": [issue],
            }
        else:
            print(f"[VLMReview] Quick page count OK: {total_pages} <= {page_limit}+{buffer} buffer")
            return {"overflow_detected": False}
    
    def _should_analyze_with_vlm(self, state: VLMReviewState) -> str:
        """Decide whether to do VLM analysis"""
        request = state["request"]
        
        # Always analyze if overflow detected (to see what to cut)
        if state.get("overflow_detected"):
            return "analyze"
        
        # Analyze for underfill or layout checks
        if request.check_underfill or request.check_layout:
            return "analyze"
        
        # Skip VLM analysis if only checking overflow and it passed
        return "skip"
    
    async def _render_pages(self, state: VLMReviewState) -> Dict[str, Any]:
        """Render PDF pages to images"""
        pdf_path = state["pdf_path"]
        total_pages = state["total_pages"]
        
        try:
            # Render ALL pages for accurate content/references/appendix classification
            max_pages = total_pages
            
            print(f"[VLMReview] Rendering {max_pages} pages...")
            images = self.pdf_renderer.render_pages(
                pdf_path,
                first_page=1,
                last_page=max_pages,
            )
            
            print(f"[VLMReview] Rendered {len(images)} page images")
            return {"page_images": images}
        
        except Exception as e:
            logger.error(f"Failed to render pages: {e}")
            return {"page_images": [], "error": str(e)}
    
    async def _analyze_pages(self, state: VLMReviewState) -> Dict[str, Any]:
        """Analyze pages with VLM"""
        images = state.get("page_images", [])
        total_pages = state["total_pages"]
        request = state["request"]
        
        if not images:
            return {"page_analyses": [], "issues": state.get("issues", [])}
        
        page_analyses = []
        issues = list(state.get("issues", []))
        underfill_detected = False
        
        for i, image in enumerate(images):
            page_num = i + 1
            is_last = (page_num == len(images) and page_num == total_pages)
            
            # Choose appropriate prompt
            prompt = LAST_PAGE_PROMPT if is_last else PAGE_ANALYSIS_PROMPT
            
            print(f"[VLMReview] Analyzing page {page_num}/{len(images)}...")
            
            try:
                response = await self.vlm.analyze_page(image, prompt)
                
                if response.success and response.content:
                    analysis = self._parse_page_analysis(
                        response.content, 
                        page_num,
                        is_last,
                        response.raw_response,
                    )
                    page_analyses.append(analysis)
                    
                    # Collect issues
                    issues.extend(analysis.layout_issues)
                    
                    # Check for underfill on last page (only if it has meaningful body content)
                    if is_last and analysis.body_content_percentage > 20:
                        if analysis.fill_percentage < request.min_fill_percentage * 100:
                            underfill_detected = True
                            issues.append(LayoutIssue(
                                issue_type=IssueType.UNDERFILL,
                                severity=IssueSeverity.MEDIUM,
                                description=f"Last page only {analysis.fill_percentage:.0f}% filled",
                                page_number=page_num,
                            ))
                else:
                    logger.warning(f"VLM analysis failed for page {page_num}: {response.error}")
                    page_analyses.append(PageAnalysis(
                        page_number=page_num,
                        fill_percentage=0,
                        raw_vlm_response=response.error,
                    ))
            
            except Exception as e:
                logger.error(f"Error analyzing page {page_num}: {e}")
                page_analyses.append(PageAnalysis(
                    page_number=page_num,
                    fill_percentage=0,
                ))
        
        return {
            "page_analyses": page_analyses,
            "issues": issues,
            "underfill_detected": underfill_detected,
        }
    
    async def _generate_result(self, state: VLMReviewState) -> Dict[str, Any]:
        """
        Generate final review result.
        - **Description**:
            - Page limit applies to main body only (excludes references
              and appendix pages).  Uses content_pages for overflow check.
        """
        request = state["request"]
        total_pages = state["total_pages"]
        page_analyses = state.get("page_analyses", [])
        issues = state.get("issues", [])
        overflow_detected_quick = state.get("overflow_detected", False)
        underfill_detected = state.get("underfill_detected", False)
        
        # Estimate content pages using weighted body_content_percentage
        # This handles mixed pages (e.g., 70% body + 30% references) accurately
        content_pages = 0.0
        for analysis in page_analyses:
            content_pages += analysis.body_content_percentage / 100.0
        # For any unanalyzed pages (shouldn't happen now), count as full content
        content_pages += max(0, total_pages - len(page_analyses))
        # Round to 1 decimal for readable output
        content_pages = round(content_pages, 1)

        # Precise overflow based on content pages (body only)
        overflow_pages = round(max(0, content_pages - request.page_limit), 1)
        overflow_detected = content_pages > request.page_limit

        print(
            f"[VLMReview] Page breakdown: total={total_pages} "
            f"content(body)={content_pages} non-body={round(total_pages - content_pages, 1)} "
            f"limit={request.page_limit} overflow={overflow_pages}"
        )

        # Remove quick-check overflow issues if precise check says OK
        if overflow_detected_quick and not overflow_detected:
            issues = [i for i in issues if i.issue_type != IssueType.OVERFLOW]
            print("[VLMReview] Quick overflow was false positive (refs/appendix pages)")
        # Add precise overflow issue if needed and not already present
        if overflow_detected and not any(i.issue_type == IssueType.OVERFLOW for i in issues):
            issues.append(LayoutIssue(
                issue_type=IssueType.OVERFLOW,
                severity=IssueSeverity.CRITICAL,
                description=(
                    f"Body content is ~{content_pages} pages, exceeds limit of "
                    f"{request.page_limit} (total PDF: {total_pages} pages, "
                    f"overflow ~{overflow_pages} pages)"
                ),
                page_number=request.page_limit + 1,
            ))
        
        # Generate section recommendations
        section_recommendations = self._generate_section_advice(
            request=request,
            overflow_detected=overflow_detected,
            underfill_detected=underfill_detected,
            overflow_pages=overflow_pages,
            page_analyses=page_analyses,
        )
        
        # Calculate trim/expand targets
        words_per_page = WORDS_PER_PAGE.get(request.template_type, 800)
        trim_target = int(overflow_pages * words_per_page) if overflow_detected else 0
        expand_target = 0
        if underfill_detected and page_analyses:
            # Find last page with meaningful body content for underfill calculation
            last_content = None
            for a in reversed(page_analyses):
                if a.body_content_percentage > 20:
                    last_content = a
                    break
            if last_content:
                last_fill = last_content.fill_percentage / 100
                expand_target = int((1 - last_fill) * words_per_page * 0.9)
        
        # Determine if passed
        has_critical = any(i.severity == IssueSeverity.CRITICAL for i in issues)
        passed = not overflow_detected and not has_critical
        
        # Generate summary
        summary = self._generate_summary(
            total_pages=total_pages,
            page_limit=request.page_limit,
            overflow_detected=overflow_detected,
            underfill_detected=underfill_detected,
            issues=issues,
        )
        
        result = VLMReviewResult(
            passed=passed,
            total_pages=total_pages,
            content_pages=content_pages,
            overflow_pages=overflow_pages,
            underfill_detected=underfill_detected,
            issues=issues,
            page_analyses=page_analyses,
            section_recommendations=section_recommendations,
            summary=summary,
            needs_trim=overflow_detected,
            needs_expand=underfill_detected and not overflow_detected,
            trim_target_words=trim_target,
            expand_target_words=expand_target,
        )
        
        return {"result": result}
    
    # =========================================================================
    # Helper Methods
    # =========================================================================
    
    def _parse_page_analysis(
        self,
        content: str,
        page_number: int,
        is_last: bool,
        raw_response: Optional[str],
    ) -> PageAnalysis:
        """Parse VLM response into PageAnalysis"""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return PageAnalysis(
                page_number=page_number,
                fill_percentage=50,  # Default
                raw_vlm_response=raw_response,
            )
        
        # Parse blank spaces
        blank_spaces = []
        for bs in data.get("blank_spaces", []):
            blank_spaces.append(BlankSpace(
                location=bs.get("location", "unknown"),
                size=bs.get("size", "small"),
                page_number=page_number,
            ))
        
        # Parse layout issues
        layout_issues = []
        for issue in data.get("layout_issues", []):
            issue_type = issue.get("type", "bad_break")
            try:
                issue_type_enum = IssueType(issue_type)
            except ValueError:
                issue_type_enum = IssueType.BAD_BREAK
            
            severity = issue.get("severity", "low")
            try:
                severity_enum = IssueSeverity(severity)
            except ValueError:
                severity_enum = IssueSeverity.LOW
            
            layout_issues.append(LayoutIssue(
                issue_type=issue_type_enum,
                severity=severity_enum,
                description=issue.get("description", ""),
                page_number=page_number,
            ))
        
        # Use `or` instead of relying on dict.get() default — VLM may return
        # explicit null values for these fields, and get() won't fall back
        # to the default when the key exists with value None.
        fill_pct = data.get("fill_percentage")
        if fill_pct is None:
            fill_pct = 50.0
        is_overflow = data.get("is_overflow")
        if is_overflow is None:
            is_overflow = False
        is_refs_page = data.get("is_references_page")
        if is_refs_page is None:
            is_refs_page = False
        is_appendix = data.get("is_appendix_page")
        if is_appendix is None:
            is_appendix = False

        # Parse body_content_percentage (0-100)
        body_pct = data.get("body_content_percentage")
        if body_pct is None:
            # Infer from page type if VLM didn't return it
            if is_refs_page and not is_appendix:
                body_pct = 0.0
            elif is_appendix and not is_refs_page:
                body_pct = 0.0
            else:
                body_pct = 100.0

        return PageAnalysis(
            page_number=page_number,
            fill_percentage=float(fill_pct),
            is_overflow=bool(is_overflow),
            has_significant_blank=len(blank_spaces) > 0,
            blank_spaces=blank_spaces,
            layout_issues=layout_issues,
            is_last_content_page=is_last and not is_refs_page and not is_appendix,
            is_references_page=bool(is_refs_page),
            is_appendix_page=bool(is_appendix),
            body_content_percentage=float(body_pct),
            raw_vlm_response=raw_response,
        )
    
    def _generate_section_advice(
        self,
        request: VLMReviewRequest,
        overflow_detected: bool,
        underfill_detected: bool,
        overflow_pages: float,
        page_analyses: List[PageAnalysis],
    ) -> Dict[str, SectionAdvice]:
        """Generate section-level advice based on analysis"""
        advice = {}
        
        if not request.sections_info:
            return advice
        
        sections_info = request.sections_info
        words_per_page = WORDS_PER_PAGE.get(request.template_type, 800)
        
        if overflow_detected:
            # Need to trim — use multi-pass allocation to ensure target is met
            target_reduction = int(overflow_pages * words_per_page)
            remaining = target_reduction
            
            # Sort by trim priority (higher = trim first)
            sorted_sections = sorted(
                sections_info.items(),
                key=lambda x: SECTION_TRIM_PRIORITY.get(x[0], 5),
                reverse=True
            )
            
            # Dynamic cap based on overflow severity:
            # <=1 page overflow -> 30% cap
            # <=2 pages -> 50% cap
            # >2 pages -> 60% cap
            if overflow_pages <= 1:
                trim_cap = 0.35
            elif overflow_pages <= 2:
                trim_cap = 0.50
            else:
                trim_cap = 0.60
            
            # Pass 1: allocate with initial cap
            section_trims = {}
            for section_type, info in sorted_sections:
                if remaining <= 0:
                    break
                word_count = info.get("word_count", 0)
                if word_count <= 0:
                    continue
                max_trim = int(word_count * trim_cap)
                trim_amount = min(remaining, max_trim)
                section_trims[section_type] = trim_amount
                remaining -= trim_amount
            
            # Pass 2: if remaining > 0, increase allocation with higher cap
            if remaining > 0:
                higher_cap = min(trim_cap + 0.20, 0.70)
                for section_type, info in sorted_sections:
                    if remaining <= 0:
                        break
                    word_count = info.get("word_count", 0)
                    if word_count <= 0:
                        continue
                    already = section_trims.get(section_type, 0)
                    max_trim = int(word_count * higher_cap)
                    extra = max(0, max_trim - already)
                    extra = min(remaining, extra)
                    section_trims[section_type] = already + extra
                    remaining -= extra
            
            # Build advice
            for section_type, info in sorted_sections:
                trim_amount = section_trims.get(section_type, 0)
                if trim_amount > 0:
                    advice[section_type] = SectionAdvice(
                        section_type=section_type,
                        current_length="too_long",
                        recommended_action="trim",
                        target_change=-trim_amount,
                        priority=SECTION_TRIM_PRIORITY.get(section_type, 5),
                        specific_guidance=f"Reduce by ~{trim_amount} words",
                    )
                else:
                    advice[section_type] = SectionAdvice(
                        section_type=section_type,
                        current_length="appropriate",
                        recommended_action="keep",
                        priority=1,
                    )
        
        elif underfill_detected:
            # Can expand — find last body content page
            last_content = None
            for a in reversed(page_analyses):
                if a.body_content_percentage > 20:
                    last_content = a
                    break
            last_fill = last_content.fill_percentage / 100 if last_content else 1.0
            target_expansion = int((1 - last_fill) * words_per_page * 0.9)
            remaining = target_expansion
            
            # Sort by expand priority
            sorted_sections = sorted(
                sections_info.items(),
                key=lambda x: SECTION_EXPAND_PRIORITY.get(x[0], 5),
                reverse=True
            )
            
            for section_type, info in sorted_sections:
                if remaining <= 0:
                    break
                
                word_count = info.get("word_count", 0)
                max_expand = int(word_count * 0.40)  # Up to 40% expansion
                expand_amount = min(remaining, max_expand)
                
                if expand_amount > 20:  # Lower threshold for meaningful expansion
                    advice[section_type] = SectionAdvice(
                        section_type=section_type,
                        current_length="too_short",
                        recommended_action="expand",
                        target_change=expand_amount,
                        priority=SECTION_EXPAND_PRIORITY.get(section_type, 5),
                        specific_guidance=f"Expand by ~{expand_amount} words",
                    )
                    remaining -= expand_amount
        
        return advice
    
    def _generate_summary(
        self,
        total_pages: int,
        page_limit: int,
        overflow_detected: bool,
        underfill_detected: bool,
        issues: List[LayoutIssue],
    ) -> str:
        """Generate human-readable summary"""
        parts = []
        
        parts.append(f"PDF has {total_pages} total pages (body limit: {page_limit}).")
        
        if overflow_detected:
            parts.append(f"CRITICAL: Body content exceeds page limit. Content must be trimmed.")
        elif underfill_detected:
            parts.append("Last page has significant blank space. Consider expanding content.")
        else:
            parts.append("Page count is within limits.")
        
        # Summarize issues by severity
        critical = sum(1 for i in issues if i.severity == IssueSeverity.CRITICAL)
        high = sum(1 for i in issues if i.severity == IssueSeverity.HIGH)
        medium = sum(1 for i in issues if i.severity == IssueSeverity.MEDIUM)
        
        if critical or high or medium:
            parts.append(f"Issues found: {critical} critical, {high} high, {medium} medium.")
        
        return " ".join(parts)
    
    # =========================================================================
    # Public API
    # =========================================================================
    
    async def review(self, request: VLMReviewRequest) -> VLMReviewResult:
        """
        Review a PDF file
        
        Args:
            request: VLMReviewRequest with PDF path and options
            
        Returns:
            VLMReviewResult with analysis and recommendations
        """
        print(f"[VLMReview] Starting review: {request.pdf_path}")
        print(f"[VLMReview] Page limit: {request.page_limit}, Template: {request.template_type}")
        
        # Initialize state
        initial_state: VLMReviewState = {
            "request": request,
            "pdf_path": request.pdf_path,
            "total_pages": 0,
            "page_images": [],
            "page_analyses": [],
            "issues": [],
            "overflow_detected": False,
            "underfill_detected": False,
            "result": None,
            "error": None,
        }
        
        # Run workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        result = final_state.get("result")
        if result:
            print(f"[VLMReview] Review complete: {'PASSED' if result.passed else 'FAILED'}")
            return result
        else:
            # Return error result
            return VLMReviewResult(
                passed=False,
                total_pages=final_state.get("total_pages", 0),
                summary=f"Review failed: {final_state.get('error', 'Unknown error')}",
            )
    
    async def quick_check(
        self, 
        pdf_path: str, 
        page_limit: int
    ) -> tuple:
        """
        Quick overflow check without VLM (system-level only)
        
        Args:
            pdf_path: Path to PDF
            page_limit: Maximum allowed pages
            
        Returns:
            Tuple of (is_overflow, total_pages, overflow_count)
        """
        total_pages = self.page_counter.count_pages(pdf_path)
        is_overflow = total_pages > page_limit
        overflow_count = max(0, total_pages - page_limit)
        
        return is_overflow, total_pages, overflow_count
    
    # =========================================================================
    # BaseAgent Interface
    # =========================================================================
    
    @property
    def name(self) -> str:
        """Agent name identifier"""
        return "vlm_review"
    
    @property
    def description(self) -> str:
        """Agent description"""
        return "VLM-based PDF review agent for page overflow, underfill, and layout detection"
    
    @property
    def router(self) -> APIRouter:
        """Return the FastAPI router for this agent"""
        from .router import create_vlm_review_router
        return create_vlm_review_router(self)
    
    @property
    def endpoints_info(self) -> List[Dict[str, Any]]:
        """Return endpoint metadata for list_agents"""
        return [
            {
                "path": "/agent/vlm_review/review",
                "method": "POST",
                "description": "Full VLM-based PDF review for overflow, underfill, and layout issues",
            },
            {
                "path": "/agent/vlm_review/quick_check",
                "method": "POST", 
                "description": "Quick page count check without VLM",
            },
            {
                "path": "/agent/vlm_review/health",
                "method": "GET",
                "description": "Health check",
            },
        ]
