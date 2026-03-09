"""
Router for VLM Review Agent endpoints
- **Description**:
    - Defines HTTP API for VLM-based PDF review
    - Provides quick check and full review endpoints
"""
from fastapi import APIRouter, HTTPException
from typing import TYPE_CHECKING, Optional
from pydantic import BaseModel
import logging
import os

from .models import (
    VLMReviewRequest,
    VLMReviewResult,
)

if TYPE_CHECKING:
    from .vlm_review_agent import VLMReviewAgent

logger = logging.getLogger("uvicorn.error")


class QuickCheckRequest(BaseModel):
    """Request for quick page count check"""
    pdf_path: str
    page_limit: int = 8


class QuickCheckResult(BaseModel):
    """Result of quick page count check"""
    is_overflow: bool
    total_pages: int
    overflow_count: int
    page_limit: int


def create_vlm_review_router(agent: "VLMReviewAgent") -> APIRouter:
    """
    Create FastAPI router for VLM Review Agent
    
    Args:
        agent: VLMReviewAgent instance
        
    Returns:
        APIRouter with endpoints
    """
    router = APIRouter()
    
    @router.post("/agent/vlm_review/review", response_model=VLMReviewResult)
    async def review_pdf(request: VLMReviewRequest) -> VLMReviewResult:
        """
        Full VLM-based PDF review
        
        Analyzes PDF for:
        - Page overflow (exceeds limit)
        - Page underfill (too much blank space)
        - Layout issues (widows, orphans, bad placements)
        
        Returns detailed analysis and section recommendations.
        
        ## Request Body
        
        ```json
        {
            "pdf_path": "/path/to/paper.pdf",
            "page_limit": 8,
            "template_type": "ICML",
            "check_overflow": true,
            "check_underfill": true,
            "check_layout": true,
            "sections_info": {
                "introduction": {"word_count": 800},
                "method": {"word_count": 1200}
            }
        }
        ```
        
        ## Response
        
        VLMReviewResult with:
        - passed: Whether PDF passes all checks
        - total_pages: Actual page count
        - issues: List of detected issues
        - section_recommendations: Advice for each section
        """
        # Validate PDF exists
        if not os.path.exists(request.pdf_path):
            raise HTTPException(
                status_code=404,
                detail=f"PDF not found: {request.pdf_path}"
            )
        
        try:
            result = await agent.review(request)
            return result
        except Exception as e:
            logger.error(f"VLM review failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    @router.post("/agent/vlm_review/quick_check", response_model=QuickCheckResult)
    async def quick_check(request: QuickCheckRequest) -> QuickCheckResult:
        """
        Quick page count check (no VLM, system-level only)
        
        Fast check for page overflow using PDF metadata.
        
        ## Request Body
        
        ```json
        {
            "pdf_path": "/path/to/paper.pdf",
            "page_limit": 8
        }
        ```
        
        ## Response
        
        ```json
        {
            "is_overflow": true,
            "total_pages": 9,
            "overflow_count": 1,
            "page_limit": 8
        }
        ```
        """
        if not os.path.exists(request.pdf_path):
            raise HTTPException(
                status_code=404,
                detail=f"PDF not found: {request.pdf_path}"
            )
        
        try:
            is_overflow, total_pages, overflow_count = await agent.quick_check(
                pdf_path=request.pdf_path,
                page_limit=request.page_limit,
            )
            
            return QuickCheckResult(
                is_overflow=is_overflow,
                total_pages=total_pages,
                overflow_count=overflow_count,
                page_limit=request.page_limit,
            )
        except Exception as e:
            logger.error(f"Quick check failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=str(e)
            )
    
    @router.get("/agent/vlm_review/health")
    async def health_check():
        """
        Health check endpoint
        
        Returns:
            Agent status and configuration
        """
        info = {
            "status": "ok",
            "agent": "vlm_review",
            "render_dpi": agent.render_dpi,
            "shared_vlm_service": agent._vlm_service is not None,
        }
        if agent._vlm_service is not None:
            info["vlm_provider"] = agent._vlm_service._provider_name
            info["vlm_model"] = agent._vlm_service._model or "default"
        else:
            info["vlm_provider"] = agent.vlm_provider_name
            info["vlm_model"] = agent.vlm_model
        return info
    
    return router
