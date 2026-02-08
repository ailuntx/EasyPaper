import yaml
import os
from functools import lru_cache
from .schema import ModelConfig, AgentConfig, AppConfig


@lru_cache()
def load_config() -> AppConfig:
    """
    从环境变量中读取配置文件路径并加载。
    """
    config_path = os.getenv("AGENT_CONFIG_PATH", "./configs/dev.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"❌ Config file not found: {config_path}. "
            f"Set environment variable AGENT_CONFIG_PATH to correct path."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return AppConfig(**raw)