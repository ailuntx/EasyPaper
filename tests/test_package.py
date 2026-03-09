"""Tests for public package API surface (easypaper/__init__.py)."""
import inspect


class TestPublicExports:
    def test_all_symbols_importable(self):
        import easypaper
        for name in easypaper.__all__:
            assert hasattr(easypaper, name), f"{name} listed in __all__ but not importable"

    def test_easypaper_class(self):
        from easypaper.client import EasyPaper
        assert hasattr(EasyPaper, "generate")
        assert hasattr(EasyPaper, "generate_stream")
        assert inspect.iscoroutinefunction(EasyPaper.generate)
        assert inspect.isasyncgenfunction(EasyPaper.generate_stream)

    def test_paper_metadata_model(self):
        from easypaper.agents.metadata_agent.models import PaperMetaData
        meta = PaperMetaData(
            idea_hypothesis="test hypothesis",
            method="test method",
            data="test data",
            experiments="test experiments",
        )
        assert meta.title == "Untitled Paper"
        assert meta.idea_hypothesis == "test hypothesis"

    def test_event_types_available(self):
        from easypaper.events import EventType
        assert EventType.PHASE_START.value == "phase_start"
        assert EventType.COMPLETE.value == "complete"

    def test_generation_event_model(self):
        from easypaper.events import GenerationEvent, EventType
        ev = GenerationEvent(
            event_type=EventType.PROGRESS,
            phase="test",
            message="hello",
        )
        assert ev.phase == "test"

    def test_metadata_agent_has_event_emitter_param(self):
        from easypaper.agents.metadata_agent.metadata_agent import MetaDataAgent
        sig = inspect.signature(MetaDataAgent.generate_paper)
        assert "event_emitter" in sig.parameters

    def test_metadata_agent_has_emit_helper(self):
        from easypaper.agents.metadata_agent.metadata_agent import MetaDataAgent
        assert hasattr(MetaDataAgent, "_emit")
