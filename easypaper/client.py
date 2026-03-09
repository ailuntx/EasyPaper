"""
EasyPaper SDK client — public entry point for programmatic paper generation.

- **Description**:
    - Wraps MetaDataAgent.generate_paper() behind a simple interface.
    - Supports one-shot ``generate()`` and streaming ``generate_stream()``.
    - Loads configuration from a YAML file and wires up internal agents
      automatically — callers never touch agent internals.

Usage::

    from easypaper import EasyPaper, PaperMetaData

    ep = EasyPaper(config_path="configs/my.yaml")
    result = await ep.generate(PaperMetaData(
        title="My Paper",
        idea_hypothesis="...",
        method="...",
        data="...",
        experiments="...",
    ))
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, Optional

from .config.loader import load_config
from .config.schema import (
    AgentConfig,
    AppConfig,
    ModelConfig,
    ToolsConfig,
    VLMServiceConfig,
)
from .events import EventEmitter, EventType, GenerationEvent


_SENTINEL = object()


class EasyPaper:
    """
    High-level SDK client for EasyPaper paper generation.

    - **Args**:
        - `config_path` (str | Path, optional): Path to a YAML config file.
            If omitted, falls back to ``AGENT_CONFIG_PATH`` env var /
            ``./configs/dev.yaml`` (same logic as the server).
        - `config` (AppConfig, optional): Pre-built config object.  Takes
            precedence over *config_path* when both are given.
    """

    def __init__(
        self,
        config_path: Optional[str | Path] = None,
        config: Optional[AppConfig] = None,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            if config_path is not None:
                import os
                os.environ["AGENT_CONFIG_PATH"] = str(config_path)
            self._config = load_config()

        self._metadata_agent = self._build_metadata_agent(self._config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        metadata: "PaperMetaData",
        **options: Any,
    ) -> "PaperGenerationResult":
        """
        One-shot paper generation.

        - **Args**:
            - `metadata` (PaperMetaData): Paper input (title, idea, method, …).
            - `**options`: Forwarded to ``MetaDataAgent.generate_paper()``
                (e.g. ``compile_pdf``, ``enable_review``, ``output_dir``).

        - **Returns**:
            - `PaperGenerationResult`: The final generation result.
        """
        return await self._metadata_agent.generate_paper(
            metadata=metadata,
            **options,
        )

    async def generate_stream(
        self,
        metadata: "PaperMetaData",
        **options: Any,
    ) -> AsyncGenerator[GenerationEvent, None]:
        """
        Streaming paper generation via async generator.

        Yields ``GenerationEvent`` instances at each phase boundary.  The
        last event has ``event_type == EventType.COMPLETE`` (or ``ERROR``)
        and carries the full ``PaperGenerationResult`` in ``event.data["result"]``.

        - **Args**:
            - `metadata` (PaperMetaData): Paper input.
            - `**options`: Forwarded to ``MetaDataAgent.generate_paper()``.

        - **Yields**:
            - `GenerationEvent`
        """
        queue: asyncio.Queue[GenerationEvent | object] = asyncio.Queue()
        emitter = EventEmitter()
        emitter.on(queue.put)

        async def _run() -> None:
            try:
                await self._metadata_agent.generate_paper(
                    metadata=metadata,
                    event_emitter=emitter,
                    **options,
                )
            finally:
                await queue.put(_SENTINEL)

        task = asyncio.create_task(_run())

        try:
            while True:
                item = await queue.get()
                if item is _SENTINEL:
                    break
                yield item  # type: ignore[misc]
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    # ------------------------------------------------------------------
    # Internal wiring
    # ------------------------------------------------------------------

    @staticmethod
    def _build_metadata_agent(config: AppConfig) -> Any:
        """
        Instantiate MetaDataAgent and its peer agents from *config*.

        - **Args**:
            - `config` (AppConfig): Parsed application config.

        - **Returns**:
            - Fully wired ``MetaDataAgent`` instance.
        """
        from .agents.metadata_agent.metadata_agent import MetaDataAgent
        from .agents.writer_agent.writer_agent import WriterAgent
        from .agents.reviewer_agent.reviewer_agent import ReviewerAgent
        from .agents.planner_agent.planner_agent import PlannerAgent

        agent_map: Dict[str, AgentConfig] = {
            a.name: a for a in config.agents
        }

        global_tools = config.tools or ToolsConfig()

        def _model(name: str) -> ModelConfig:
            cfg = agent_map.get(name)
            if cfg and cfg.model:
                return cfg.model
            raise ValueError(
                f"Agent '{name}' not found or has no model in config. "
                f"Available agents: {list(agent_map.keys())}"
            )

        # Build VLM service (shared between Planner and VLMReviewAgent)
        vlm_service = None
        if config.vlm_service and config.vlm_service.enabled:
            try:
                from .agents.vlm_review_agent.vlm_review_agent import VLMService
                vlm_service = VLMService(
                    provider=config.vlm_service.provider,
                    api_key=config.vlm_service.api_key or "",
                    model=config.vlm_service.model,
                    base_url=config.vlm_service.base_url,
                )
            except Exception:
                vlm_service = None

        # Core agents required by MetaDataAgent
        metadata_cfg = agent_map.get("metadata")
        meta_tools = (
            metadata_cfg.tools_config if metadata_cfg and metadata_cfg.tools_config
            else global_tools
        )
        meta_agent = MetaDataAgent(
            config=_model("metadata"),
            tools_config=meta_tools,
        )

        writer = WriterAgent(
            config=_model("writer"),
            tools_config=global_tools,
        )

        # SkillRegistry is optional; load if skills config is present
        skill_registry = None
        if config.skills and config.skills.enabled:
            try:
                from .skills.loader import SkillLoader
                from .skills.registry import SkillRegistry
                skill_registry = SkillRegistry()
                loader = SkillLoader()
                skills = loader.load_directory(Path(config.skills.skills_dir))
                for skill in skills:
                    skill_registry.register(skill)
            except Exception:
                skill_registry = None

        reviewer = ReviewerAgent(
            config=_model("reviewer"),
            skill_registry=skill_registry,
        )

        planner = PlannerAgent(
            config=_model("planner"),
            vlm_service=vlm_service,
        )

        # VLMReviewAgent is optional
        vlm_review_agent = None
        vlm_cfg = agent_map.get("vlm_review")
        if vlm_cfg and vlm_cfg.vlm_review_config and vlm_cfg.vlm_review_config.enabled:
            try:
                from .agents.vlm_review_agent.vlm_review_agent import VLMReviewAgent
                vr = vlm_cfg.vlm_review_config
                vlm_review_agent = VLMReviewAgent(
                    model_config=vlm_cfg.model if vlm_cfg.model else None,
                    vlm_review_config=vr,
                    vlm_service=vlm_service,
                )
            except Exception:
                vlm_review_agent = None

        # Inject skill registry into MetaDataAgent
        meta_agent._skill_registry = skill_registry

        # Wire peer agents
        peers: Dict[str, Any] = {
            "writer": writer,
            "reviewer": reviewer,
            "planner": planner,
        }
        if vlm_review_agent is not None:
            peers["vlm_review"] = vlm_review_agent
        meta_agent.set_peers(peers)

        return meta_agent
