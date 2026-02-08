"""
Base classes for Writer Agent tools.

This module defines the abstract base class for tools that can be used
by the Writer Agent during content generation and review.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    """
    Result of a tool execution.
    
    Attributes:
        success: Whether the tool executed successfully
        data: The result data from the tool
        message: Human-readable message about the result
        errors: List of error messages if any
    """
    success: bool
    data: Any = None
    message: str = ""
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "data": self.data,
            "message": self.message,
            "errors": self.errors,
        }


class WriterTool(ABC):
    """
    Abstract base class for Writer Agent tools.
    
    Tools are operations that can be invoked during content generation
    or review to perform specific tasks like citation validation,
    word counting, literature search, etc.
    
    Subclasses must implement:
        - name: Unique identifier for the tool
        - description: Human-readable description for LLM prompts
        - execute: The actual tool logic
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this tool."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of what this tool does.
        This is used in LLM prompts to help the model understand
        when and how to use the tool.
        """
        pass
    
    @property
    def parameters_schema(self) -> Dict[str, Any]:
        """
        JSON Schema describing the tool's parameters.
        Override this to provide structured parameter definitions.
        """
        return {}
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with the given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with success status and data
        """
        pass
    
    def get_prompt_description(self) -> str:
        """
        Get a formatted description for inclusion in LLM prompts.
        """
        desc = f"**{self.name}**: {self.description}"
        if self.parameters_schema:
            params = self.parameters_schema.get("properties", {})
            if params:
                param_strs = []
                for name, schema in params.items():
                    param_type = schema.get("type", "any")
                    param_desc = schema.get("description", "")
                    param_strs.append(f"  - {name} ({param_type}): {param_desc}")
                desc += "\n  Parameters:\n" + "\n".join(param_strs)
        return desc
