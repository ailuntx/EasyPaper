"""
Tool Registry for Writer Agent.

Manages registration and execution of tools available to the Writer Agent.
Supports both direct execution and OpenAI Function Calling format for ReAct loops.
"""

import json
from typing import Any, Dict, List, Optional, Type
from .base import WriterTool, ToolResult


class ToolRegistry:
    """
    Registry for Writer Agent tools.

    - **Description**:
        - Provides a centralized way to register, discover, and execute tools.
        - Supports OpenAI Function Calling format via get_openai_tools()
          and execute_tool_calls() for ReAct integration.
        - Tools can be registered by instance, replaced, or cleared.

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

        - **Args**:
            - `tool` (WriterTool): The tool instance to register.

        - **Raises**:
            - ValueError: If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def register_or_replace(self, tool: WriterTool) -> None:
        """
        Register a tool instance, replacing any existing tool with the same name.

        - **Args**:
            - `tool` (WriterTool): The tool instance to register.
        """
        self._tools[tool.name] = tool

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()

    def unregister(self, tool_name: str) -> bool:
        """
        Unregister a tool by name.

        - **Args**:
            - `tool_name` (str): Name of the tool to unregister.

        - **Returns**:
            - `bool`: True if tool was unregistered, False if not found.
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False

    def get(self, tool_name: str) -> Optional[WriterTool]:
        """
        Get a tool by name.

        - **Args**:
            - `tool_name` (str): Name of the tool.

        - **Returns**:
            - `WriterTool` or None if not found.
        """
        return self._tools.get(tool_name)

    def list_tools(self) -> List[str]:
        """
        List all registered tool names.

        - **Returns**:
            - `List[str]`: List of tool names.
        """
        return list(self._tools.keys())

    def get_tool_descriptions(self) -> str:
        """
        Get formatted descriptions of all tools for LLM prompts.

        - **Returns**:
            - `str`: Formatted string with all tool descriptions.
        """
        if not self._tools:
            return "No tools available."

        descriptions = []
        for tool in self._tools.values():
            descriptions.append(tool.get_prompt_description())

        return "\n\n".join(descriptions)

    def get_openai_tools(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get tool definitions in OpenAI Function Calling format.

        - **Description**:
            - Converts registered tools into the format expected by
              OpenAI's `tools` parameter in chat completions.

        - **Args**:
            - `tool_names` (List[str], optional): Subset of tools to include.
              If None, includes all registered tools.

        - **Returns**:
            - `List[dict]`: OpenAI-format tool definitions.
        """
        tools_to_export = {}
        if tool_names:
            for name in tool_names:
                if name in self._tools:
                    tools_to_export[name] = self._tools[name]
        else:
            tools_to_export = self._tools

        openai_tools = []
        for tool in tools_to_export.values():
            schema = tool.parameters_schema or {
                "type": "object",
                "properties": {},
            }
            # Ensure schema has required top-level fields
            if "type" not in schema:
                schema["type"] = "object"
            if "properties" not in schema:
                schema["properties"] = {}

            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema,
                },
            })
        return openai_tools

    async def execute_tool_calls(self, tool_calls: list) -> List[Dict[str, Any]]:
        """
        Execute tool calls from an OpenAI assistant message.

        - **Description**:
            - Takes the tool_calls list from an assistant message and
              executes each tool via the registry.
            - Returns a list of tool-role messages ready to be appended
              to the conversation history.

        - **Args**:
            - `tool_calls` (list): tool_calls from assistant message
              (each has .id, .function.name, .function.arguments).

        - **Returns**:
            - `List[dict]`: List of {"role": "tool", ...} messages.
        """
        import time as _time

        tool_messages = []
        for tc in tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
            except json.JSONDecodeError:
                fn_args = {}

            t0 = _time.time()
            result = await self.execute(fn_name, **fn_args)
            elapsed = _time.time() - t0
            print(f"[ToolRegistry] {fn_name} finished in {elapsed:.2f}s "
                  f"(success={result.success})")

            tool_messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result.to_dict(), ensure_ascii=False),
            })
        return tool_messages

    async def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """
        Execute a tool by name.

        - **Args**:
            - `tool_name` (str): Name of the tool to execute.
            - `**kwargs`: Parameters to pass to the tool.

        - **Returns**:
            - `ToolResult`: Result from the tool execution.
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

    - **Returns**:
        - `ToolRegistry`: The default ToolRegistry instance.
    """
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


def register_default_tools(registry: ToolRegistry, valid_citation_keys: set = None) -> None:
    """
    Register the default set of tools.

    - **Args**:
        - `registry` (ToolRegistry): The registry to register tools with.
        - `valid_citation_keys` (set, optional): Valid citation keys for validation.
    """
    from .citation_tools import CitationValidatorTool, WordCountTool

    # Register citation validator if keys provided
    if valid_citation_keys:
        registry.register(CitationValidatorTool(valid_citation_keys))

    # Always register word count tool
    registry.register(WordCountTool())
