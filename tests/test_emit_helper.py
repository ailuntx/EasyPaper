"""Tests for MetaDataAgent._emit static helper."""
import pytest

from src.events import EventEmitter, EventType, GenerationEvent
from src.agents.metadata_agent.metadata_agent import MetaDataAgent


class TestEmitHelper:
    async def test_emit_with_none_emitter(self):
        """_emit must silently no-op when emitter is None."""
        await MetaDataAgent._emit(None, EventType.PHASE_START, "test", "msg")

    async def test_emit_fires_event(self):
        collected = []
        emitter = EventEmitter()
        emitter.on(lambda e: collected.append(e))

        await MetaDataAgent._emit(
            emitter, EventType.PHASE_START, "planning", "Starting plan"
        )

        assert len(collected) == 1
        ev = collected[0]
        assert ev.event_type == EventType.PHASE_START
        assert ev.phase == "planning"
        assert ev.message == "Starting plan"
        assert ev.data is None

    async def test_emit_with_extra_data(self):
        collected = []
        emitter = EventEmitter()
        emitter.on(lambda e: collected.append(e))

        await MetaDataAgent._emit(
            emitter,
            EventType.SECTION_COMPLETE,
            "introduction",
            "Intro done",
            section_type="introduction",
            word_count=500,
            latex_content="\\section{Intro}",
        )

        ev = collected[0]
        assert ev.data["section_type"] == "introduction"
        assert ev.data["word_count"] == 500
        assert ev.data["latex_content"] == "\\section{Intro}"
