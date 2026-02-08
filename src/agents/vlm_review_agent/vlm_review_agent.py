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

Return ONLY this JSON structure:
{
    "fill_percentage": <0-100>,
    "is_overflow": <true if content beyond margins>,
    "blank_spaces": [{"location": "<top/bottom/left/right/center>", "size": "<small/medium/large>"}],
    "layout_issues": [{"type": "<widow/orphan/bad_figure/equation_overflow>", "description": "<details>", "severity": "<low/medium/high>"}],
    "is_references_page": <true if this page contains bibliography/references>
}"""

LAST_PAGE_PROMPT = """Analyze this LAST page of an academic paper. Respond with JSON only.

Determine:
1. How much of the page is filled with content (percentage)?
2. Is this the references/bibliography page, or does it have main content?
3. How many more lines of text could fit in the empty space?

Return ONLY this JSON:
{
    "fill_percentage": <0-100>,
    "is_references_page": <true/false>,
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
    
    async def _check_overflow_quick(self, state: VLMReviewState) -> Dict[str, Any]:
        """Quick check for page overflow using page count"""
        total_pages = state["total_pages"]
        page_limit = state["request"].page_limit
        
        is_overflow = total_pages > page_limit
        overflow_pages = max(0, total_pages - page_limit)
        
        if is_overflow:
            print(f"[VLMReview] OVERFLOW: {total_pages} pages > limit {page_limit}")
            issue = LayoutIssue(
                issue_type=IssueType.OVERFLOW,
                severity=IssueSeverity.CRITICAL,
                description=f"PDF has {total_pages} pages, exceeds limit of {page_limit}",
                page_number=page_limit + 1,
            )
            return {
                "overflow_detected": True,
                "issues": [issue],
            }
        else:
            print(f"[VLMReview] Page count OK: {total_pages} <= {page_limit}")
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
            # Render pages (limit to prevent excessive API calls)
            max_pages = min(total_pages, 12)  # Max 12 pages
            
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
                    
                    # Check for underfill on last page
                    if is_last and not analysis.is_references_page:
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
        """Generate final review result"""
        request = state["request"]
        total_pages = state["total_pages"]
        page_analyses = state.get("page_analyses", [])
        issues = state.get("issues", [])
        overflow_detected = state.get("overflow_detected", False)
        underfill_detected = state.get("underfill_detected", False)
        
        # Calculate overflow
        overflow_pages = max(0, total_pages - request.page_limit)
        
        # Estimate content pages (exclude references)
        content_pages = total_pages
        for analysis in reversed(page_analyses):
            if analysis.is_references_page:
                content_pages -= 1
            else:
                break
        
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
        trim_target = overflow_pages * words_per_page if overflow_detected else 0
        expand_target = 0
        if underfill_detected and page_analyses:
            last_fill = page_analyses[-1].fill_percentage / 100
            expand_target = int((1 - last_fill) * words_per_page * 0.8)  # 80% of available
        
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
        
        return PageAnalysis(
            page_number=page_number,
            fill_percentage=data.get("fill_percentage", 50),
            is_overflow=data.get("is_overflow", False),
            has_significant_blank=len(blank_spaces) > 0,
            blank_spaces=blank_spaces,
            layout_issues=layout_issues,
            is_last_content_page=is_last and not data.get("is_references_page", False),
            is_references_page=data.get("is_references_page", False),
            raw_vlm_response=raw_response,
        )
    
    def _generate_section_advice(
        self,
        request: VLMReviewRequest,
        overflow_detected: bool,
        underfill_detected: bool,
        overflow_pages: int,
        page_analyses: List[PageAnalysis],
    ) -> Dict[str, SectionAdvice]:
        """Generate section-level advice based on analysis"""
        advice = {}
        
        if not request.sections_info:
            return advice
        
        sections_info = request.sections_info
        words_per_page = WORDS_PER_PAGE.get(request.template_type, 800)
        
        if overflow_detected:
            # Need to trim
            target_reduction = overflow_pages * words_per_page
            remaining = target_reduction
            
            # Sort by trim priority
            sorted_sections = sorted(
                sections_info.items(),
                key=lambda x: SECTION_TRIM_PRIORITY.get(x[0], 5),
                reverse=True
            )
            
            for section_type, info in sorted_sections:
                if remaining <= 0:
                    advice[section_type] = SectionAdvice(
                        section_type=section_type,
                        current_length="appropriate",
                        recommended_action="keep",
                        priority=1,
                    )
                else:
                    word_count = info.get("word_count", 0)
                    # Suggest trimming up to 30% of section
                    max_trim = int(word_count * 0.3)
                    trim_amount = min(remaining, max_trim)
                    
                    advice[section_type] = SectionAdvice(
                        section_type=section_type,
                        current_length="too_long",
                        recommended_action="trim",
                        target_change=-trim_amount,
                        priority=SECTION_TRIM_PRIORITY.get(section_type, 5),
                        specific_guidance=f"Reduce by ~{trim_amount} words",
                    )
                    remaining -= trim_amount
        
        elif underfill_detected:
            # Can expand
            last_fill = page_analyses[-1].fill_percentage / 100 if page_analyses else 1.0
            target_expansion = int((1 - last_fill) * words_per_page * 0.8)
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
                max_expand = int(word_count * 0.2)  # Up to 20% expansion
                expand_amount = min(remaining, max_expand)
                
                if expand_amount > 50:  # Only suggest if meaningful
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
        
        parts.append(f"PDF has {total_pages} pages (limit: {page_limit}).")
        
        if overflow_detected:
            parts.append(f"CRITICAL: Exceeds page limit by {total_pages - page_limit} page(s). Content must be trimmed.")
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
