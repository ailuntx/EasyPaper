from abc import ABC, abstractmethod
from typing import Any, List, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter


class BaseAgent(ABC):
    """Base class for all agents providing common interface"""

    @property
    @abstractmethod
    def name(self) -> str:
        """Agent name identifier"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Agent description"""
        pass

    @property
    @abstractmethod
    def router(self) -> "APIRouter | None":
        """Return the FastAPI router for this agent (None when FastAPI is not installed)"""
        pass

    @property
    @abstractmethod
    def endpoints_info(self) -> List[Dict[str, Any]]:
        """Return endpoint metadata for list_agents"""
        pass