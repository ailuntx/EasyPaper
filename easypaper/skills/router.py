"""
Skills API Router
- **Description**:
    - FastAPI routes for skill management
    - POST /skills/register: Auto-register from instruction + URL
    - GET /skills: List all skills
    - GET /skills/{name}: Get single skill details
    - DELETE /skills/{name}: Remove a skill
"""
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from ..agents.shared.llm_client import LLMClient

from .registry import SkillRegistry
from .generator import SkillGenerator

logger = logging.getLogger("uvicorn.error")


# ---- Request / Response Models ----

class SkillRegisterRequest(BaseModel):
    """Request body for auto-registering a skill."""
    instruction: str = Field(..., description="What the skill should do")
    reference_url: str = Field(..., description="URL or file:// path to reference content")
    name: Optional[str] = Field(None, description="Override auto-generated name")
    skill_type: Optional[str] = Field(None, description="Override auto-detected type")


class SkillRegisterResponse(BaseModel):
    """Response after successful skill registration."""
    name: str
    type: str
    description: str
    source_url: Optional[str] = None
    message: str = "Skill registered successfully"


# ---- Router Factory ----

def create_skills_router(registry: SkillRegistry, config) -> APIRouter:
    """
    Create the FastAPI router for skills management.

    - **Args**:
        - `registry` (SkillRegistry): The global skill registry
        - `config` (AppConfig): Application config (for LLM client)

    - **Returns**:
        - `APIRouter`: The skills router
    """
    router = APIRouter(prefix="/skills")

    # Find an LLM client config from the agents list for the generator
    def _get_llm_config():
        """Pick the first available agent model config for LLM calls."""
        for agent_cfg in config.agents:
            return agent_cfg.model
        return None

    # ------------------------------------------------------------------
    # POST /skills/register
    # ------------------------------------------------------------------

    @router.post("/register", response_model=SkillRegisterResponse)
    async def register_skill(request: SkillRegisterRequest):
        """
        Auto-register a skill from an instruction + reference URL.
        The system fetches the URL, uses an LLM to extract the skill,
        saves it as YAML, and registers it in the active registry.
        """
        model_cfg = _get_llm_config()
        if model_cfg is None:
            raise HTTPException(
                status_code=500,
                detail="No LLM model configured for skill extraction",
            )

        skills_dir = Path("./skills")
        if config.skills:
            skills_dir = Path(config.skills.skills_dir)

        client = LLMClient(
            api_key=model_cfg.api_key,
            base_url=model_cfg.base_url,
        )
        generator = SkillGenerator(
            llm_client=client,
            model_name=model_cfg.model_name,
            skills_dir=skills_dir,
            registry=registry,
        )

        try:
            skill = await generator.register(
                instruction=request.instruction,
                reference_url=request.reference_url,
                name=request.name,
                skill_type=request.skill_type,
            )
            return SkillRegisterResponse(
                name=skill.name,
                type=skill.type,
                description=skill.description,
                source_url=skill.source_url,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logger.error("skills.router: register failed: %s", e)
            raise HTTPException(status_code=500, detail=f"Registration failed: {e}")

    # ------------------------------------------------------------------
    # GET /skills
    # ------------------------------------------------------------------

    @router.get("")
    async def list_skills():
        """List all registered skills."""
        return {
            "total": len(registry),
            "skills": registry.list_all(),
        }

    # ------------------------------------------------------------------
    # GET /skills/{name}
    # ------------------------------------------------------------------

    @router.get("/{name}")
    async def get_skill(name: str):
        """Get detailed information about a single skill."""
        if name not in registry:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
        skill = registry._skills[name]
        return skill.model_dump(exclude_none=True)

    # ------------------------------------------------------------------
    # DELETE /skills/{name}
    # ------------------------------------------------------------------

    @router.delete("/{name}")
    async def delete_skill(name: str):
        """Remove a skill from the registry (does not delete the YAML file)."""
        removed = registry.unregister(name)
        if not removed:
            raise HTTPException(status_code=404, detail=f"Skill '{name}' not found")
        return {"message": f"Skill '{name}' removed", "remaining": len(registry)}

    return router
