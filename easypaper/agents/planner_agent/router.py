"""
Router for Planner Agent endpoints
- **Description**:
    - Defines HTTP API for paper planning
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
    """Request model for the planning endpoint."""
    title: str = "Untitled Paper"
    idea_hypothesis: str
    method: str
    data: str
    experiments: str
    references: list = []
    figures: list = []
    tables: list = []
    target_pages: Optional[int] = None
    style_guide: Optional[str] = None


def create_planner_router(agent: "PlannerAgent") -> APIRouter:
    """
    Create FastAPI router for Planner Agent.

    - **Args**:
        - `agent`: PlannerAgent instance

    - **Returns**:
        - `APIRouter`: FastAPI router with endpoints
    """
    router = APIRouter()

    @router.post("/agent/planner/plan", response_model=PlanResult)
    async def create_paper_plan(request: CreatePlanRequest) -> PlanResult:
        """Create a paper plan from metadata."""
        try:
            figures = []
            for fig in request.figures:
                if isinstance(fig, dict):
                    figures.append(FigureInfo(
                        id=fig.get("id", ""),
                        caption=fig.get("caption", ""),
                        description=fig.get("description", ""),
                        section=fig.get("section", ""),
                        wide=fig.get("wide", False),
                        file_path=fig.get("file_path", ""),
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
                        file_path=tbl.get("file_path", ""),
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

            plan = await agent.create_plan(request=plan_request)
            return PlanResult(status="ok", plan=plan)

        except Exception as e:
            logger.error("planner.plan.error: %s", str(e))
            return PlanResult(status="error", error=str(e))

    @router.get("/agent/planner/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "ok",
            "agent": "planner",
        }

    return router
