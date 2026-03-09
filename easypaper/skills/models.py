"""
Skill Data Models
- **Description**:
    - Pydantic models for the WritingSkill system
    - Compatible with AI-Research-SKILLs YAML frontmatter conventions
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal


class WritingSkill(BaseModel):
    """
    A pluggable writing skill that can inject constraints into prompts or
    supply rules to reviewer checkers.

    - **Fields**:
        - `name` (str): Kebab-case identifier (e.g. "anti-ai-style")
        - `description` (str): What this skill does and when to use it
        - `version` (str): Semantic version
        - `tags` (List[str]): Tags for fuzzy matching / discovery
        - `type` (str): One of writing_constraint, reviewer_checker, venue_profile
        - `target_sections` (List[str]): Sections this skill applies to ("*" = all)
        - `priority` (int): Lower number = higher priority (injected earlier)
        - `system_prompt_append` (str): Text injected into the writer system prompt
        - `revision_guidance` (str): Guidance for fixing violations found by checkers
        - `anti_patterns` (List[str]): Words or phrases to flag / avoid
        - `required_patterns` (List[str]): Patterns that should appear in output
        - `venue_config` (Dict): Venue-specific config (words_per_page, etc.)
        - `source_url` (str): Where this skill was extracted from
    """

    # --- Metadata ---
    name: str
    description: str = ""
    version: str = "1.0.0"
    tags: List[str] = Field(default_factory=list)

    # --- Core fields ---
    type: Literal["writing_constraint", "reviewer_checker", "venue_profile"]
    target_sections: List[str] = Field(default_factory=lambda: ["*"])
    priority: int = 10

    # --- Content ---
    system_prompt_append: str = ""
    revision_guidance: str = ""
    anti_patterns: List[str] = Field(default_factory=list)
    required_patterns: List[str] = Field(default_factory=list)

    # --- Venue-specific (only for type=venue_profile) ---
    venue_config: Optional[Dict] = None

    # --- Source tracking ---
    source_url: Optional[str] = None
