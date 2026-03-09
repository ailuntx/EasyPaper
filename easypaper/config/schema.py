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


class PaperSearchConfig(BaseModel):
    """Configuration for the paper search tool."""
    serpapi_api_key: Optional[str] = None
    semantic_scholar_api_key: Optional[str] = None
    timeout: int = 10
    search_results_per_round: int = 5  # Number of papers to return per search round
    planner_max_queries_per_section: int = 5
    planner_inter_round_delay_sec: float = 1.5
    planner_min_target_papers_per_section: int = 3
    semantic_scholar_min_results_before_fallback: int = 3
    enable_query_cache: bool = True
    cache_ttl_hours: int = 24
    citation_budget_enabled: bool = True
    citation_budget_soft_cap: bool = True
    citation_budget_export: bool = True
    citation_budget_reserve_size: int = 4


class ResearchContextConfig(BaseModel):
    """Configuration for research context generation."""
    enabled: bool = True  # Whether to generate Research Context
    detailed: bool = True  # Whether to use detailed mode
    top_k_key_papers: int = 10  # Number of key papers to include
    research_context_first_enabled: bool = True
    claim_evidence_enabled: bool = True
    contribution_ranking_enabled: bool = True
    export_planning_decision_trace: bool = False


class ToolsConfig(BaseModel):
    """Configuration for ReAct tool usage."""
    enabled: bool = True
    available_tools: List[str] = Field(
        default_factory=lambda: [
            "validate_citations",
            "count_words",
            "check_key_points",
            "search_papers",
        ]
    )
    max_react_iterations: int = 3
    planner_structure_signals_enabled: bool = True
    writer_structure_contract_enabled: bool = True
    review_structure_gate_enabled: bool = True
    structure_gate_min_paragraph_threshold: int = 5
    paper_search: Optional[PaperSearchConfig] = None
    research_context: Optional[ResearchContextConfig] = None


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


class VLMServiceConfig(BaseModel):
    """Shared VLM service configuration (used by Planner and VLMReviewAgent)."""
    enabled: bool = True
    provider: str = "openai"
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class AgentConfig(BaseModel):
    name: str
    model: Optional[ModelConfig] = None
    max_tokens: int = 2000
    timeout: int = 20
    log_level: str = "INFO"
    # Agent-specific config (optional)
    writer_config: Optional[WriterConfig] = None
    metadata_config: Optional[MetadataConfig] = None
    vlm_review_config: Optional[VLMReviewConfig] = None
    tools_config: Optional[ToolsConfig] = None


class SkillsConfig(BaseModel):
    """Skills system configuration."""
    enabled: bool = True
    skills_dir: str = "./skills"
    active_skills: List[str] = Field(default_factory=lambda: ["*"])  # "*" = all
    venue_profile: Optional[str] = None  # "neurips", "icml", etc.


class AppConfig(BaseModel):
    agents: List[AgentConfig]
    skills: Optional[SkillsConfig] = None
    tools: Optional[ToolsConfig] = None
    vlm_service: Optional[VLMServiceConfig] = None