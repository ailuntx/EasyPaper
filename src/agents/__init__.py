"""
Agent initialization and FastAPI router registration.

- **Description**:
    - ``initialize_agents`` creates all agent instances from config.
    - ``register_agent_routers`` mounts their FastAPI routers on the app.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..config.schema import (
    AgentConfig,
    ModelConfig,
    ToolsConfig,
    VLMServiceConfig,
)

if TYPE_CHECKING:
    from fastapi import FastAPI
    from ..skills.registry import SkillRegistry


def initialize_agents(
    agent_configs: List[AgentConfig],
    *,
    skill_registry: Optional["SkillRegistry"] = None,
    global_tools_config: Optional[ToolsConfig] = None,
    vlm_service_config: Optional[VLMServiceConfig] = None,
) -> Dict[str, Any]:
    """
    Create all agent instances from the config list.

    - **Args**:
        - `agent_configs` (List[AgentConfig]): Per-agent configurations.
        - `skill_registry` (SkillRegistry, optional): Shared skill registry.
        - `global_tools_config` (ToolsConfig, optional): Global tool settings.
        - `vlm_service_config` (VLMServiceConfig, optional): Shared VLM service config.

    - **Returns**:
        - `Dict[str, BaseAgent]`: Mapping of agent name to instance.
    """
    from .metadata_agent.metadata_agent import MetaDataAgent
    from .writer_agent.writer_agent import WriterAgent
    from .reviewer_agent.reviewer_agent import ReviewerAgent
    from .planner_agent.planner_agent import PlannerAgent
    from .commander_agent.commander_agent import CommanderAgent
    from .parse_agent.parse_agent import ParseAgent
    from .template_agent.template_agent import TemplateAgent
    from .typesetter_agent.typesetter_agent import TypesetterAgent

    tools_cfg = global_tools_config or ToolsConfig()

    vlm_service = None
    if vlm_service_config and vlm_service_config.enabled:
        try:
            from .vlm_review_agent.vlm_review_agent import VLMService
            vlm_service = VLMService(
                provider=vlm_service_config.provider,
                api_key=vlm_service_config.api_key or "",
                model=vlm_service_config.model,
                base_url=vlm_service_config.base_url,
            )
        except Exception:
            vlm_service = None

    cfg_map: Dict[str, AgentConfig] = {a.name: a for a in agent_configs}
    agents: Dict[str, Any] = {}

    _BUILDERS = {
        "paper_parser": lambda c: ParseAgent(config=c.model),
        "template_parser": lambda c: TemplateAgent(config=c.model),
        "commander": lambda c: CommanderAgent(config=c.model, tools_config=tools_cfg),
        "writer": lambda c: WriterAgent(config=c.model, tools_config=tools_cfg),
        "typesetter": lambda c: TypesetterAgent(config=c.model),
        "metadata": lambda c: MetaDataAgent(config=c.model, tools_config=tools_cfg),
        "reviewer": lambda c: ReviewerAgent(config=c.model, skill_registry=skill_registry),
        "planner": lambda c: PlannerAgent(config=c.model, vlm_service=vlm_service),
    }

    for name, builder in _BUILDERS.items():
        cfg = cfg_map.get(name)
        if cfg and cfg.model:
            try:
                agents[name] = builder(cfg)
            except Exception as exc:
                print(f"[initialize_agents] Warning: failed to create '{name}': {exc}")

    # VLMReviewAgent (optional, may have no model)
    vlm_cfg = cfg_map.get("vlm_review")
    if vlm_cfg and vlm_cfg.vlm_review_config and vlm_cfg.vlm_review_config.enabled:
        try:
            from .vlm_review_agent.vlm_review_agent import VLMReviewAgent
            agents["vlm_review"] = VLMReviewAgent(
                model_config=vlm_cfg.model if vlm_cfg.model else None,
                vlm_review_config=vlm_cfg.vlm_review_config,
                vlm_service=vlm_service,
            )
        except Exception as exc:
            print(f"[initialize_agents] Warning: failed to create 'vlm_review': {exc}")

    # Inject skill registry into metadata agent
    meta = agents.get("metadata")
    if meta is not None:
        meta._skill_registry = skill_registry
        meta.set_peers(agents)

    return agents


def register_agent_routers(app: "FastAPI", agents: Dict[str, Any]) -> None:
    """
    Mount each agent's FastAPI router on *app*.

    - **Args**:
        - `app` (FastAPI): The FastAPI application instance.
        - `agents` (Dict[str, BaseAgent]): Agent instances from ``initialize_agents``.
    """
    for name, agent in agents.items():
        router = getattr(agent, "router", None)
        if router is not None:
            app.include_router(router)
            print(f"   Registered router for: {name}")
