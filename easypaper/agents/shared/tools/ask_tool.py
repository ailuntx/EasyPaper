"""
AskTool - Unified routing consultation tool for ReAct agents.
- **Description**:
    - Routes questions to the appropriate agent or resource based on
      a ``target`` parameter.
    - Handlers are injected at construction time; the set of available
      targets is determined dynamically.
    - Designed to be the *only* consultation tool an agent needs —
      adding a new consultable target is just a matter of registering
      another handler.
"""

import logging
from typing import Any, Callable, Dict, Optional

from .base import WriterTool, ToolResult

logger = logging.getLogger(__name__)


class AskTool(WriterTool):
    """
    Unified consultation tool that dispatches questions to registered
    handlers (agents or resources) by target name.

    - **Description**:
        - Each handler is an async callable: ``(question: str) -> str``
        - The ``target`` enum is built dynamically from registered handlers.
        - Logs inter-agent consultations to SessionMemory when available.
    """

    def __init__(self, handlers: Dict[str, Callable], memory: Optional[Any] = None) -> None:
        """
        Initialize with a mapping of target names to handler callables.

        - **Args**:
            - `handlers` (Dict[str, Callable]): Mapping from target name
              to an async callable ``(question: str) -> str``.
            - `memory` (SessionMemory, optional): For logging consultations.
        """
        self._handlers = handlers
        self._memory = memory

    @property
    def name(self) -> str:
        return "ask"

    @property
    def description(self) -> str:
        targets = ", ".join(sorted(self._handlers.keys()))
        return (
            f"Consult an agent or resource for information. "
            f"Available targets: {targets}. "
            "Use 'memory' to look up plan details, prior section content, "
            "review feedback, or contributions. "
            "Use 'planner' to ask about section structure, paragraph "
            "allocation, or figure/table placement. "
            "Use 'reviewer' to check consistency or quality of a specific "
            "claim or paragraph."
        )

    @property
    def parameters_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": sorted(self._handlers.keys()),
                    "description": "Who to consult",
                },
                "question": {
                    "type": "string",
                    "description": "What you want to know or check",
                },
            },
            "required": ["target", "question"],
        }

    async def execute(self, target: str = "", question: str = "", **kwargs) -> ToolResult:
        """
        Route the question to the appropriate handler.

        - **Args**:
            - `target` (str): Registered handler name
            - `question` (str): The question to forward

        - **Returns**:
            - `ToolResult`: Handler response wrapped in a ToolResult
        """
        handler = self._handlers.get(target)
        if handler is None:
            available = sorted(self._handlers.keys())
            return ToolResult(
                success=False,
                message=f"Unknown target '{target}'. Available: {available}",
            )
        try:
            result = await handler(question)
            msg = result if result else "No relevant information found."
            # Log the inter-agent consultation
            if self._memory is not None and hasattr(self._memory, 'log'):
                snippet = (msg[:120] + "...") if len(msg) > 120 else msg
                self._memory.log(
                    "ask_tool", "consultation", f"consulted_{target}",
                    narrative=f"Writer consulted {target}: '{question[:80]}' → {snippet}",
                    communication={
                        "from": "writer",
                        "to": target,
                        "question": question,
                        "answer_preview": snippet,
                    },
                )
            return ToolResult(success=True, message=msg)
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error consulting '{target}': {e}",
            )
