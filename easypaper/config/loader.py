import yaml
import os
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv
from .schema import ModelConfig, AgentConfig, AppConfig

# Resolve the project root (.env lives at the repo root)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


@lru_cache()
def load_config() -> AppConfig:
    """
    Load application config from a YAML file.
    - **Description**:
     - Reads AGENT_CONFIG_PATH from .env (or environment) and loads the YAML config.

    - **Returns**:
     - `AppConfig`: The parsed application configuration.
    """
    dotenv_path = _PROJECT_ROOT / ".env"
    load_dotenv(dotenv_path=dotenv_path, override=True)
    config_path = os.getenv("AGENT_CONFIG_PATH", "./configs/dev.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"❌ Config file not found: {config_path}. "
            f"Set environment variable AGENT_CONFIG_PATH to correct path."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AppConfig(**raw)