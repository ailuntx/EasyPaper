"""
Extensible event system for streaming paper generation progress.

- **Description**:
    - Defines typed events that are yielded during generate_stream().
    - EventEmitter is injected into MetaDataAgent.generate_paper() so that
      any phase boundary can emit events without changing the generator signature.
    - To add a new event point, simply call ``await emitter.emit(...)`` at
      the desired location — no changes to the public API are required.
"""
import inspect
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field


class EventType(str, Enum):
    """Categories of generation events."""
    PHASE_START = "phase_start"
    PHASE_COMPLETE = "phase_complete"
    SECTION_COMPLETE = "section_complete"
    PROGRESS = "progress"
    WARNING = "warning"
    ERROR = "error"
    COMPLETE = "complete"


class GenerationEvent(BaseModel):
    """
    A single event emitted during paper generation.

    - **Args**:
        - `event_type` (EventType): The category of this event.
        - `phase` (str): Logical phase name, e.g. "planning", "introduction".
        - `message` (str): Human-readable description of what happened.
        - `data` (dict, optional): Extensible payload for structured data.
        - `timestamp` (datetime): When this event was created.
    """
    event_type: EventType
    phase: str
    message: str
    data: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.now)

# Callback type accepted by EventEmitter.on()
EventCallback = Callable[[GenerationEvent], Any]


class EventEmitter:
    """
    Async event emitter that dispatches GenerationEvent instances to
    registered callbacks.

    - **Description**:
        - Register listeners via ``on(callback)``.
        - Emit events via ``await emit(event)``; all listeners are called
          sequentially in registration order.
        - Accepts both sync and async callbacks.
        - Designed to be injected into long-running generation methods so
          that callers can observe progress without polling.
    """

    def __init__(self) -> None:
        self._listeners: List[EventCallback] = []

    def on(self, callback: EventCallback) -> None:
        """
        Register a listener (sync or async).

        - **Args**:
            - `callback`: Callable that receives a GenerationEvent.
        """
        self._listeners.append(callback)

    async def emit(self, event: GenerationEvent) -> None:
        """
        Dispatch *event* to every registered listener.

        - **Args**:
            - `event` (GenerationEvent): The event to broadcast.
        """
        for listener in self._listeners:
            result = listener(event)
            if inspect.isawaitable(result):
                await result
