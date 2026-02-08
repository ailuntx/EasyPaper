# app/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import logging
from .config import load_config
from .agents import initialize_agents, register_agent_routers

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    config = load_config()
    app.state.config = config
    # Initialize agents
    app.state.agents = initialize_agents(config.agents)
    # Register agent routers
    register_agent_routers(app, app.state.agents)
    print(f"✅ Loaded config from env path with {len(app.state.agents)} agents.")
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