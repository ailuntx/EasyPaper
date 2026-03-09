"""Tests for the extensible event system (src/events.py)."""
import asyncio
from datetime import datetime

import pytest

from easypaper.events import EventEmitter, EventType, GenerationEvent


# ---------------------------------------------------------------------------
# GenerationEvent model
# ---------------------------------------------------------------------------

class TestGenerationEvent:
    def test_minimal_construction(self):
        ev = GenerationEvent(
            event_type=EventType.PHASE_START,
            phase="planning",
            message="hello",
        )
        assert ev.event_type == EventType.PHASE_START
        assert ev.phase == "planning"
        assert ev.message == "hello"
        assert ev.data is None
        assert isinstance(ev.timestamp, datetime)

    def test_with_data_payload(self):
        ev = GenerationEvent(
            event_type=EventType.SECTION_COMPLETE,
            phase="introduction",
            message="done",
            data={"latex_content": "\\section{Intro}", "word_count": 42},
        )
        assert ev.data["word_count"] == 42

    def test_event_type_values(self):
        expected = {
            "phase_start", "phase_complete", "section_complete",
            "progress", "warning", "error", "complete",
        }
        assert {e.value for e in EventType} == expected

    def test_serialization_roundtrip(self):
        ev = GenerationEvent(
            event_type=EventType.COMPLETE,
            phase="complete",
            message="fin",
            data={"result": {"status": "ok"}},
        )
        payload = ev.model_dump()
        restored = GenerationEvent(**payload)
        assert restored.event_type == ev.event_type
        assert restored.data == ev.data


# ---------------------------------------------------------------------------
# EventEmitter
# ---------------------------------------------------------------------------

class TestEventEmitter:
    @pytest.fixture
    def emitter(self):
        return EventEmitter()

    @pytest.fixture
    def sample_event(self):
        return GenerationEvent(
            event_type=EventType.PROGRESS,
            phase="test",
            message="ping",
        )

    async def test_sync_callback(self, emitter, sample_event):
        collected = []
        emitter.on(lambda e: collected.append(e))
        await emitter.emit(sample_event)
        assert len(collected) == 1
        assert collected[0] is sample_event

    async def test_async_callback(self, emitter, sample_event):
        collected = []

        async def cb(e):
            collected.append(e)

        emitter.on(cb)
        await emitter.emit(sample_event)
        assert len(collected) == 1

    async def test_multiple_listeners(self, emitter, sample_event):
        a, b = [], []
        emitter.on(lambda e: a.append(e))
        emitter.on(lambda e: b.append(e))
        await emitter.emit(sample_event)
        assert len(a) == 1
        assert len(b) == 1

    async def test_no_listeners(self, emitter, sample_event):
        """Emit with zero listeners must not raise."""
        await emitter.emit(sample_event)

    async def test_mixed_sync_async_callbacks(self, emitter, sample_event):
        results = []
        emitter.on(lambda e: results.append("sync"))

        async def async_cb(e):
            results.append("async")

        emitter.on(async_cb)
        await emitter.emit(sample_event)
        assert results == ["sync", "async"]

    async def test_queue_put_as_callback(self, emitter, sample_event):
        """asyncio.Queue.put is the pattern used by generate_stream."""
        q: asyncio.Queue = asyncio.Queue()
        emitter.on(q.put)
        await emitter.emit(sample_event)
        assert not q.empty()
        assert (await q.get()) is sample_event
