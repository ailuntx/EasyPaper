"""
Router for Reviewer Agent endpoints
- **Description**:
    - Defines HTTP API for paper review
    - Provides checker management endpoints
"""
from fastapi import APIRouter, HTTPException, status
from typing import TYPE_CHECKING
import logging

from .models import (
    ReviewRequest,
    ReviewResult,
    ReviewContext,
)

if TYPE_CHECKING:
    from .reviewer_agent import ReviewerAgent

logger = logging.getLogger("uvicorn.error")


def create_reviewer_router(agent: "ReviewerAgent") -> APIRouter:
    """
    Create FastAPI router for Reviewer Agent
    
    - **Args**:
        - `agent`: ReviewerAgent instance
        
    - **Returns**:
        - `APIRouter`: FastAPI router with endpoints
    """
    router = APIRouter()
    
    @router.post("/agent/reviewer/review", response_model=ReviewResult)
    async def review_paper(request: ReviewRequest) -> ReviewResult:
        """
        Review paper content and provide feedback
        
        ## Request Body
        
        ```json
        {
            "sections": {
                "introduction": "...",
                "method": "...",
                ...
            },
            "word_counts": {
                "introduction": 500,
                "method": 800,
                ...
            },
            "target_pages": 8,
            "style_guide": "ICML",
            "template_path": "path/to/template.zip",
            "metadata": {},
            "iteration": 0
        }
        ```
        
        ## Response
        
        ```json
        {
            "passed": false,
            "feedbacks": [...],
            "iteration": 0,
            "requires_revision": {
                "method": ["Expand by ~200 words"],
                ...
            },
            "section_feedbacks": [...]
        }
        ```
        """
        try:
            # Build review context
            context = ReviewContext(
                sections=request.sections,
                word_counts=request.word_counts,
                target_pages=request.target_pages or 8,
                section_targets=request.section_targets,
                template_path=request.template_path,
                style_guide=request.style_guide,
                metadata=request.metadata,
                memory_context=request.memory_context,
            )
            
            # Run review
            result = await agent.review(context, iteration=request.iteration)
            return result
            
        except Exception as e:
            logger.error("reviewer.review.error: %s", str(e))
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(e),
            )
    
    @router.get("/agent/reviewer/checkers")
    async def list_checkers():
        """
        List all registered feedback checkers
        
        ## Response
        
        ```json
        {
            "checkers": [
                {
                    "name": "word_count",
                    "priority": 1,
                    "enabled": true,
                    "class": "WordCountChecker"
                },
                ...
            ]
        }
        ```
        """
        return {"checkers": agent.get_checkers()}
    
    @router.get("/agent/reviewer/health")
    async def health_check():
        """
        Health check endpoint
        
        ## Response
        
        ```json
        {
            "status": "ok",
            "agent": "reviewer",
            "checkers_count": 1
        }
        ```
        """
        return {
            "status": "ok",
            "agent": "reviewer",
            "checkers_count": len(agent.get_checkers()),
            "checkers": [c["name"] for c in agent.get_checkers()],
        }
    
    return router
