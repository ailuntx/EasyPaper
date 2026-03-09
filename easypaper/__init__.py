"""
EasyPaper — AI-powered academic paper generation SDK.

Public API::

    from easypaper import EasyPaper, PaperMetaData, EventType

    ep = EasyPaper(config_path="configs/my.yaml")

    # one-shot
    result = await ep.generate(metadata)

    # streaming
    async for event in ep.generate_stream(metadata):
        print(event.phase, event.message)
"""
from .client import EasyPaper
from .events import EventEmitter, EventType, GenerationEvent
from .agents.metadata_agent.models import (
    PaperMetaData,
    PaperGenerationResult,
    PaperGenerationRequest,
    SectionResult,
)

__all__ = [
    "EasyPaper",
    "EventEmitter",
    "EventType",
    "GenerationEvent",
    "PaperMetaData",
    "PaperGenerationResult",
    "PaperGenerationRequest",
    "SectionResult",
]
