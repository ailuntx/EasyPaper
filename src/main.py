# app/main.py
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from typing import Tuple, Set
from .agents.shared.llm_client import LLMClient
from .config import load_config
from .config.schema import SkillsConfig
from .agents import initialize_agents, register_agent_routers
from .skills.loader import SkillLoader
from .skills.registry import SkillRegistry

logger = logging.getLogger("uvicorn.error")


async def _validate_chat_connection(
    *,
    label: str,
    model_name: str,
    api_key: str,
    base_url: str,
) -> None:
    """
    Validate an OpenAI-compatible chat endpoint with a tiny completion call.
    """
    client = LLMClient(
        api_key=api_key,
        base_url=base_url,
    )
    try:
        await client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Return exactly: ok"},
                {"role": "user", "content": "connectivity check"},
            ],
            max_tokens=2,
            temperature=0,
        )
        print(f"[StartupCheck] OK: {label} ({model_name})")
    except Exception as e:
        raise RuntimeError(f"[StartupCheck] FAILED: {label} ({model_name}) -> {e}") from e


async def _validate_llm_and_vlm_connections(config) -> None:
    """
    Validate LLM and VLM connectivity before agent initialization.
    """
    # Validate all configured agent LLM endpoints (deduplicated by credentials+model).
    llm_seen: Set[Tuple[str, str, str]] = set()
    for agent_cfg in config.agents:
        if not agent_cfg.model:
            continue
        model_cfg = agent_cfg.model
        key = (model_cfg.model_name, model_cfg.base_url or "", model_cfg.api_key or "")
        if key in llm_seen:
            continue
        llm_seen.add(key)
        await _validate_chat_connection(
            label=f"LLM/{agent_cfg.name}",
            model_name=model_cfg.model_name,
            api_key=model_cfg.api_key,
            base_url=model_cfg.base_url,
        )

    # Validate shared VLM service endpoint (if enabled).
    if config.vlm_service and config.vlm_service.enabled:
        vlm_model = config.vlm_service.model or "gpt-4o"
        vlm_key = config.vlm_service.api_key or ""
        vlm_base = config.vlm_service.base_url or "https://api.openai.com/v1"
        if not vlm_key:
            raise RuntimeError("[StartupCheck] FAILED: shared VLM service enabled but api_key is empty")
        await _validate_chat_connection(
            label="VLM/shared_service",
            model_name=vlm_model,
            api_key=vlm_key,
            base_url=vlm_base,
        )

    # Validate agent-level VLM overrides when explicitly enabled and configured.
    vlm_seen: Set[Tuple[str, str, str]] = set()
    for agent_cfg in config.agents:
        vr = agent_cfg.vlm_review_config
        if not vr or not vr.enabled or not vr.vlm_model:
            continue
        model_cfg = agent_cfg.model
        vlm_model = vr.vlm_model
        vlm_key = vr.vlm_api_key or (model_cfg.api_key if model_cfg else "")
        vlm_base = vr.vlm_base_url or (model_cfg.base_url if model_cfg else "https://api.openai.com/v1")
        key = (vlm_model, vlm_base or "", vlm_key or "")
        if key in vlm_seen:
            continue
        if not vlm_key:
            raise RuntimeError(
                f"[StartupCheck] FAILED: VLM override for agent '{agent_cfg.name}' has no api_key"
            )
        vlm_seen.add(key)
        await _validate_chat_connection(
            label=f"VLM/{agent_cfg.name}",
            model_name=vlm_model,
            api_key=vlm_key,
            base_url=vlm_base,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config = load_config()
    app.state.config = config
    await _validate_llm_and_vlm_connections(config)

    # --- Skills system initialization ---
    skill_registry = SkillRegistry()
    skills_config = config.skills or SkillsConfig()
    if skills_config.enabled:
        loader = SkillLoader()
        skills = loader.load_directory(Path(skills_config.skills_dir))
        for skill in skills:
            skill_registry.register(skill)
        print(f"   Skills system: {len(skill_registry)} skills loaded from {skills_config.skills_dir}")
    app.state.skill_registry = skill_registry

    # Initialize agents (pass skill_registry for ReviewerAgent / MetaDataAgent,
    # and global tools config for ReAct-enabled agents)
    app.state.agents = initialize_agents(
        config.agents,
        skill_registry=skill_registry,
        global_tools_config=config.tools,
        vlm_service_config=config.vlm_service,
    )
    # Register agent routers
    register_agent_routers(app, app.state.agents)

    # Register skills API router
    from .skills.router import create_skills_router
    app.include_router(create_skills_router(skill_registry, config), tags=["skills"])

    print(f"Loaded config from env path with {len(app.state.agents)} agents.")
    # Print model info for each agent (never print api keys)
    for agent_cfg in config.agents:
        model = agent_cfg.model
        if model is None:
            print(f"   {agent_cfg.name:<20} model=(none)")
            continue
        base_host = model.base_url.rstrip("/").split("//")[-1].split("/")[0] if model.base_url else "default"
        extra = ""
        if agent_cfg.vlm_review_config and agent_cfg.vlm_review_config.vlm_model:
            vlm = agent_cfg.vlm_review_config
            vlm_host = vlm.vlm_base_url.rstrip("/").split("//")[-1].split("/")[0] if vlm.vlm_base_url else base_host
            extra = f"  vlm_model={vlm.vlm_model} vlm_host={vlm_host}"
        print(f"   {agent_cfg.name:<20} model={model.model_name:<30} base={base_host}{extra}")
    yield
    # Shutdown
    pass

app = FastAPI(title="Agent Service", version="1.0.0", lifespan=lifespan)

# --- Endpoints -------------------------------------------------------
@app.get("/config")
async def get_config():
    """Get the configuration of the app"""
    return app.state.config.dict()

@app.get("/list_agents")
async def list_agents():
    """List all available agents with their endpoints information"""
    agents_info = []
    for _, agent_instance in app.state.agents.items():
        agent_info = {
            "name": agent_instance.name,
            "description": agent_instance.description,
            "endpoints": agent_instance.endpoints_info,
            "status": "active"
        }
        agents_info.append(agent_info)

    return {"agents": agents_info}

@app.get("/healthz")
async def health():
    """Health check"""
    return {"status": "ok"}