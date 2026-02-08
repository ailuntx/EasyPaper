"""
Router for Planner Agent endpoints
- **Description**:
    - Defines HTTP API for paper planning
    - Provides strategy management endpoints
"""
from fastapi import APIRouter, HTTPException, status
from typing import TYPE_CHECKING, Optional
from pydantic import BaseModel
import logging

from .models import (
    PlanRequest,
    PlanResult,
    PaperPlan,
    FigureInfo,
    TableInfo,
)

if TYPE_CHECKING:
    from .planner_agent import PlannerAgent

logger = logging.getLogger("uvicorn.error")


class CreatePlanRequest(BaseModel):
    """Extended request with strategy option"""
    title: str = "Untitled Paper"
    idea_hypothesis: str
    method: str
    data: str
    experiments: str
    references: list = []
    figures: list = []   # List of figure info dicts
    tables: list = []    # List of table info dicts
    target_pages: Optional[int] = None
    style_guide: Optional[str] = None
    strategy: Optional[str] = None  # Planning strategy to use


def create_planner_router(agent: "PlannerAgent") -> APIRouter:
    """
    Create FastAPI router for Planner Agent
    
    - **Args**:
        - `agent`: PlannerAgent instance
        
    - **Returns**:
        - `APIRouter`: FastAPI router with endpoints
    """
    router = APIRouter()
    
    @router.post("/agent/planner/plan", response_model=PlanResult)
    async def create_paper_plan(request: CreatePlanRequest) -> PlanResult:
        """
        Create a paper plan from metadata
        
        ## Request Body
        
        ```json
        {
            "title": "Paper Title",
            "idea_hypothesis": "Research hypothesis...",
            "method": "Method description...",
            "data": "Data description...",
            "experiments": "Experiments and results...",
            "references": ["@article{key, ...}", ...],
            "target_pages": 8,
            "style_guide": "ICML",
            "strategy": "standard"
        }
        ```
        
        ## Response
        
        ```json
        {
            "status": "ok",
            "plan": {
                "title": "Paper Title",
                "paper_type": "empirical",
                "total_target_words": 6800,
                "sections": [...],
                "contributions": [...],
                "narrative_style": "technical",
                ...
            },
            "error": null
        }
        ```
        """
        try:
            # Convert figure/table dicts to model instances
            figures = []
            for fig in request.figures:
                if isinstance(fig, dict):
                    figures.append(FigureInfo(
                        id=fig.get("id", ""),
                        caption=fig.get("caption", ""),
                        description=fig.get("description", ""),
                        section=fig.get("section", ""),
                        wide=fig.get("wide", False),
                    ))
                else:
                    figures.append(fig)
            
            tables = []
            for tbl in request.tables:
                if isinstance(tbl, dict):
                    tables.append(TableInfo(
                        id=tbl.get("id", ""),
                        caption=tbl.get("caption", ""),
                        description=tbl.get("description", ""),
                        section=tbl.get("section", ""),
                        wide=tbl.get("wide", False),
                    ))
                else:
                    tables.append(tbl)
            
            plan_request = PlanRequest(
                title=request.title,
                idea_hypothesis=request.idea_hypothesis,
                method=request.method,
                data=request.data,
                experiments=request.experiments,
                references=request.references,
                figures=figures,
                tables=tables,
                target_pages=request.target_pages,
                style_guide=request.style_guide,
            )
            
            plan = await agent.create_plan(
                request=plan_request,
                strategy_name=request.strategy,
            )
            
            return PlanResult(status="ok", plan=plan)
            
        except Exception as e:
            logger.error("planner.plan.error: %s", str(e))
            return PlanResult(status="error", error=str(e))
    
    @router.get("/agent/planner/strategies")
    async def list_strategies():
        """
        List available planning strategies
        
        ## Response
        
        ```json
        {
            "strategies": [
                {
                    "name": "standard",
                    "description": "Standard planning for empirical papers",
                    "class": "StandardPlanningStrategy"
                }
            ]
        }
        ```
        """
        return {"strategies": agent.get_strategies()}
    
    @router.get("/agent/planner/health")
    async def health_check():
        """
        Health check endpoint
        
        ## Response
        
        ```json
        {
            "status": "ok",
            "agent": "planner",
            "strategies_count": 1
        }
        ```
        """
        return {
            "status": "ok",
            "agent": "planner",
            "strategies_count": len(agent.get_strategies()),
            "strategies": [s["name"] for s in agent.get_strategies()],
        }
    
    return router
