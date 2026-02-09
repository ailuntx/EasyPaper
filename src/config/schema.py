from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class ModelConfig(BaseModel):
    model_name: str
    api_key: str
    base_url: str = "https://api.openai.com/v1"


class WriterConfig(BaseModel):
    """Writer-specific configuration options."""
    max_review_iterations: int = 2
    enable_review: bool = True
    enable_tools: bool = True
    available_tools: List[str] = Field(
        default_factory=lambda: ["validate_citations", "count_words", "check_key_points"]
    )


class MetadataConfig(BaseModel):
    """Metadata agent-specific configuration options."""
    enable_mini_review: bool = True
    max_review_iterations: int = 2


class VLMReviewConfig(BaseModel):
    """VLM Review agent-specific configuration options."""
    enabled: bool = True
    provider: str = "openai"  # openai, claude, qwen
    # VLM model settings (can override model from ModelConfig)
    vlm_model: Optional[str] = None  # e.g., "gpt-4o", "google/gemini-2.0-flash-exp:free"
    vlm_api_key: Optional[str] = None  # If different from model.api_key
    vlm_base_url: Optional[str] = None  # If different from model.base_url
    # Analysis settings
    render_dpi: int = 150
    max_pages_to_analyze: int = 12
    check_overflow: bool = True
    check_underfill: bool = True
    check_layout: bool = False  # Disabled by default (expensive)
    # Thresholds
    min_fill_percentage: float = 0.85
    max_blank_area: float = 0.15


class AgentConfig(BaseModel):
    name: str
    model: ModelConfig
    max_tokens: int = 2000
    timeout: int = 20
    log_level: str = "INFO"
    # Agent-specific config (optional)
    writer_config: Optional[WriterConfig] = None
    metadata_config: Optional[MetadataConfig] = None
    vlm_review_config: Optional[VLMReviewConfig] = None


class SkillsConfig(BaseModel):
    """Skills system configuration."""
    enabled: bool = True
    skills_dir: str = "./skills"
    active_skills: List[str] = Field(default_factory=lambda: ["*"])  # "*" = all
    venue_profile: Optional[str] = None  # "neurips", "icml", etc.


class AppConfig(BaseModel):
    agents: List[AgentConfig]
    skills: Optional[SkillsConfig] = None