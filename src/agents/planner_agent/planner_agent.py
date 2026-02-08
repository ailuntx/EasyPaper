"""
Planner Agent
- **Description**:
    - Creates detailed paper plans before generation
    - Analyzes metadata to determine structure, content, and style
    - Outputs PaperPlan to guide Writers and Reviewers
"""
import logging
from typing import List, Dict, Any, Optional, Type
from fastapi import APIRouter
from openai import AsyncOpenAI

from ..base import BaseAgent
from ...config.schema import ModelConfig
from .models import (
    PaperPlan,
    SectionPlan,
    PlanRequest,
    PlanResult,
    calculate_total_words,
)
from .strategies.base import PlanningStrategy
from .strategies.standard import StandardPlanningStrategy


logger = logging.getLogger("uvicorn.error")


class PlannerAgent(BaseAgent):
    """
    Planner Agent for paper planning
    - **Description**:
        - Creates comprehensive plans before paper generation
        - Supports multiple planning strategies
        - Provides word budgets and content guidance
    """
    
    # Default strategy
    DEFAULT_STRATEGY: Type[PlanningStrategy] = StandardPlanningStrategy
    
    def __init__(self, config: ModelConfig):
        """Initialize the Planner Agent"""
        self.config = config
        self.model_name = config.model_name
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self._strategies: Dict[str, PlanningStrategy] = {}
        self._router = self._create_router()
        
        # Register default strategy
        self.register_strategy(self.DEFAULT_STRATEGY())
        
        logger.info(
            "PlannerAgent initialized with strategies: %s",
            list(self._strategies.keys())
        )
    
    @property
    def name(self) -> str:
        return "planner"
    
    @property
    def description(self) -> str:
        return "Creates detailed paper plans for structure, content, and style guidance"
    
    @property
    def router(self) -> APIRouter:
        return self._router
    
    @property
    def endpoints_info(self) -> List[Dict[str, Any]]:
        return [
            {
                "path": "/agent/planner/plan",
                "method": "POST",
                "description": "Create a paper plan from metadata",
            },
            {
                "path": "/agent/planner/strategies",
                "method": "GET",
                "description": "List available planning strategies",
            },
            {
                "path": "/agent/planner/health",
                "method": "GET",
                "description": "Health check",
            },
        ]
    
    def _create_router(self) -> APIRouter:
        """Create FastAPI router"""
        from .router import create_planner_router
        return create_planner_router(self)
    
    def register_strategy(self, strategy: PlanningStrategy) -> None:
        """
        Register a planning strategy
        - **Args**:
            - `strategy`: PlanningStrategy instance to register
        """
        if strategy.name in self._strategies:
            logger.warning("Strategy '%s' already registered, replacing", strategy.name)
        self._strategies[strategy.name] = strategy
        logger.info("Registered strategy: %s", strategy.name)
    
    def get_strategies(self) -> List[Dict[str, Any]]:
        """Get list of registered strategies"""
        return [
            {
                "name": s.name,
                "description": s.description,
                "class": s.__class__.__name__,
            }
            for s in self._strategies.values()
        ]
    
    async def create_plan(
        self,
        request: PlanRequest,
        strategy_name: Optional[str] = None,
    ) -> PaperPlan:
        """
        Create a paper plan from metadata
        
        - **Args**:
            - `request` (PlanRequest): Planning request with metadata
            - `strategy_name` (str, optional): Strategy to use (default: 'standard')
            
        - **Returns**:
            - `PaperPlan`: Complete paper plan
        """
        # Select strategy
        strategy_name = strategy_name or "standard"
        strategy = self._strategies.get(strategy_name)
        
        if not strategy:
            logger.warning(
                "Strategy '%s' not found, using 'standard'",
                strategy_name
            )
            strategy = self._strategies.get("standard")
        
        if not strategy:
            raise ValueError("No planning strategy available")
        
        logger.info(
            "planner.create_plan title=%s strategy=%s",
            request.title[:30],
            strategy.name,
        )
        
        # Create plan using strategy
        plan = await strategy.create_plan(
            request=request,
            llm_client=self.client,
            model_name=self.model_name,
        )
        
        # Validate plan
        if not plan.validate_word_budget():
            logger.warning(
                "planner.word_budget_mismatch total=%d section_sum=%d",
                plan.total_target_words,
                sum(s.target_words for s in plan.sections),
            )
        
        logger.info(
            "planner.plan_complete sections=%d contributions=%d words=%d",
            len(plan.sections),
            len(plan.contributions),
            plan.total_target_words,
        )
        
        return plan
    
    async def create_plan_from_metadata(
        self,
        title: str,
        idea_hypothesis: str,
        method: str,
        data: str,
        experiments: str,
        references: List[str],
        target_pages: Optional[int] = None,
        style_guide: Optional[str] = None,
        strategy_name: Optional[str] = None,
    ) -> PaperPlan:
        """
        Convenience method to create plan from individual fields
        
        - **Args**:
            - Individual metadata fields
            
        - **Returns**:
            - `PaperPlan`: Complete paper plan
        """
        request = PlanRequest(
            title=title,
            idea_hypothesis=idea_hypothesis,
            method=method,
            data=data,
            experiments=experiments,
            references=references,
            target_pages=target_pages,
            style_guide=style_guide,
        )
        return await self.create_plan(request, strategy_name)
