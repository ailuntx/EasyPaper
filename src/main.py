# app/main.py
from pathlib import Path
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from .config import load_config
from .config.schema import SkillsConfig
from .agents import initialize_agents, register_agent_routers
from .skills.loader import SkillLoader
from .skills.registry import SkillRegistry

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config = load_config()
    app.state.config = config

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

    # Initialize agents (pass skill_registry for ReviewerAgent / MetaDataAgent)
    app.state.agents = initialize_agents(config.agents, skill_registry=skill_registry)
    # Register agent routers
    register_agent_routers(app, app.state.agents)

    # Register skills API router
    from .skills.router import create_skills_router
    app.include_router(create_skills_router(skill_registry, config), tags=["skills"])

    print(f"Loaded config from env path with {len(app.state.agents)} agents.")
    # Print model info for each agent (never print api keys)
    for agent_cfg in config.agents:
        model = agent_cfg.model
        # Mask base_url to show only the host
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

logger = logging.getLogger("uvicorn.error")

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