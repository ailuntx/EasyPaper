"""
Tool Registry for Writer Agent.

Manages registration and execution of tools available to the Writer Agent.
"""

from typing import Dict, List, Optional, Type
from .base import WriterTool, ToolResult


class ToolRegistry:
    """
    Registry for Writer Agent tools.
    
    Provides a centralized way to register, discover, and execute tools.
    Tools can be registered by instance or by class.
    
    Example:
        registry = ToolRegistry()
        registry.register(CitationValidatorTool(valid_keys))
        registry.register(WordCountTool())
        
        # Get tool descriptions for LLM prompt
        descriptions = registry.get_tool_descriptions()
        
        # Execute a tool
        result = await registry.execute("validate_citations", content="...")
    """
    
    def __init__(self):
        self._tools: Dict[str, WriterTool] = {}
    
    def register(self, tool: WriterTool) -> None:
        """
        Register a tool instance.
        
        Args:
            tool: The tool instance to register
            
        Raises:
            ValueError: If a tool with the same name is already registered
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool
    
    def unregister(self, tool_name: str) -> bool:
        """
        Unregister a tool by name.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False
    
    def get(self, tool_name: str) -> Optional[WriterTool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            The tool instance or None if not found
        """
        return self._tools.get(tool_name)
    
    def list_tools(self) -> List[str]:
        """
        List all registered tool names.
        
        Returns:
            List of tool names
        """
        return list(self._tools.keys())
    
    def get_tool_descriptions(self) -> str:
        """
        Get formatted descriptions of all tools for LLM prompts.
        
        Returns:
            Formatted string with all tool descriptions
        """
        if not self._tools:
            return "No tools available."
        
        descriptions = []
        for tool in self._tools.values():
            descriptions.append(tool.get_prompt_description())
        
        return "\n\n".join(descriptions)
    
    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool
            
        Returns:
            ToolResult from the tool execution
        """
        tool = self._tools.get(tool_name)
        if tool is None:
            print(f"[ToolRegistry] ERROR: Tool '{tool_name}' not found")
            return ToolResult(
                success=False,
                message=f"Tool '{tool_name}' not found",
                errors=[f"Available tools: {', '.join(self._tools.keys())}"]
            )
        
        try:
            print(f"[ToolRegistry] Executing tool: {tool_name}")
            result = await tool.execute(**kwargs)
            print(f"[ToolRegistry] Tool '{tool_name}' completed: {result.message}")
            return result
        except Exception as e:
            print(f"[ToolRegistry] ERROR: Tool '{tool_name}' failed: {e}")
            return ToolResult(
                success=False,
                message=f"Tool execution failed: {str(e)}",
                errors=[str(e)]
            )
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools


# Global default registry
_default_registry: Optional[ToolRegistry] = None


def get_default_registry() -> ToolRegistry:
    """
    Get the default tool registry singleton.
    
    Returns:
        The default ToolRegistry instance
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


def register_default_tools(registry: ToolRegistry, valid_citation_keys: set = None) -> None:
    """
    Register the default set of tools.
    
    Args:
        registry: The registry to register tools with
        valid_citation_keys: Optional set of valid citation keys for validation
    """
    from .citation_tools import CitationValidatorTool, WordCountTool
    
    # Register citation validator if keys provided
    if valid_citation_keys:
        registry.register(CitationValidatorTool(valid_citation_keys))
    
    # Always register word count tool
    registry.register(WordCountTool())
