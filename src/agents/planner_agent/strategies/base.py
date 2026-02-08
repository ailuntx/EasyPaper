"""
Planning Strategy Base Class
- **Description**:
    - Abstract base class for planning strategies
    - Defines the interface for creating paper plans
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import PaperPlan, PlanRequest


class PlanningStrategy(ABC):
    """
    Abstract base class for planning strategies
    - **Description**:
        - Different strategies can be used for different paper types
        - Strategies determine section structure and content allocation
        
    - **Required Methods**:
        - `create_plan()`: Create a complete paper plan
        - `name`: Strategy identifier
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier"""
        pass
    
    @property
    def description(self) -> str:
        """Strategy description"""
        return ""
    
    @abstractmethod
    async def create_plan(
        self,
        request: "PlanRequest",
        llm_client: Any,
        model_name: str,
    ) -> "PaperPlan":
        """
        Create a paper plan from the request
        
        - **Args**:
            - `request` (PlanRequest): Planning request with metadata
            - `llm_client`: OpenAI-compatible async client
            - `model_name` (str): Model to use for planning
            
        - **Returns**:
            - `PaperPlan`: Complete paper plan
        """
        pass
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name='{self.name}'>"
