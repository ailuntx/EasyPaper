"""
Skill Generator
- **Description**:
    - Auto-generates WritingSkill YAML from an instruction + reference URL
    - Fetches content from various URL types (GitHub, local, generic web)
    - Uses LLM to extract structured skill data from raw content
"""
import json
import logging
import re
from pathlib import Path
from typing import Optional

import httpx
import yaml

from .models import WritingSkill
from .registry import SkillRegistry

logger = logging.getLogger("uvicorn.error")


SKILL_EXTRACTION_PROMPT = """You are a writing-skill extraction system. Given:
1. An **instruction** describing what the user wants this skill to do
2. **Raw content** from a reference URL

Your task is to produce a JSON object representing a WritingSkill with these fields:

{
  "name": "<kebab-case-identifier>",
  "description": "<one-sentence description>",
  "version": "1.0.0",
  "tags": ["<tag1>", "<tag2>"],
  "type": "<writing_constraint | reviewer_checker | venue_profile>",
  "target_sections": ["*"],
  "priority": 10,
  "system_prompt_append": "<multi-line string: rules/instructions to inject into writer>",
  "revision_guidance": "<how to fix violations>",
  "anti_patterns": ["<word or phrase to avoid>"],
  "required_patterns": [],
  "venue_config": null,
  "source_url": "<the reference URL>"
}

Guidelines:
- `name` must be kebab-case, concise, and unique
- `type` defaults to "writing_constraint" unless the instruction clearly asks for
  a reviewer checker or venue profile
- `system_prompt_append` is the most important field — extract concrete, actionable
  rules from the reference content. Use numbered lists or bullet points.
- `anti_patterns` should list specific words/phrases to avoid (if applicable)
- Keep `priority` at 10 unless the instruction indicates urgency
- Output ONLY the JSON object — no markdown fences, no explanation."""


class SkillGenerator:
    """
    Generates WritingSkill YAML files from user instructions and reference URLs.

    - **Description**:
        - Fetches content from URLs (GitHub, local, web)
        - Uses LLM to extract structured skill fields
        - Saves as YAML and registers in the SkillRegistry

    - **Methods**:
        - `register()`: Main entry point for auto-registration
    """

    def __init__(
        self,
        llm_client,
        model_name: str,
        skills_dir: Path = Path("./skills"),
        registry: Optional[SkillRegistry] = None,
    ):
        self._client = llm_client
        self._model = model_name
        self._skills_dir = Path(skills_dir)
        self._registry = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def register(
        self,
        instruction: str,
        reference_url: str,
        name: Optional[str] = None,
        skill_type: Optional[str] = None,
    ) -> WritingSkill:
        """
        Auto-register a skill from instruction + reference URL.

        - **Args**:
            - `instruction` (str): What the skill should do
            - `reference_url` (str): URL or file:// path to reference content
            - `name` (str, optional): Override the auto-generated name
            - `skill_type` (str, optional): Override the auto-detected type

        - **Returns**:
            - `WritingSkill`: The registered skill
        """
        # 1. Fetch content
        content = await self._fetch_content(reference_url)
        if not content:
            raise ValueError(f"Failed to fetch content from: {reference_url}")

        # 2. Extract skill via LLM
        skill = await self._extract_skill(instruction, content, skill_type)

        # 3. Override name if provided
        if name:
            skill.name = name

        # 4. Record source
        skill.source_url = reference_url

        # 5. Save YAML
        type_dir = self._skills_dir / skill.type.replace("_", "/").rsplit("/", 1)[0]
        # Map type to subdirectory
        dir_map = {
            "writing_constraint": "writing",
            "reviewer_checker": "reviewing",
            "venue_profile": "venues",
        }
        subdir = dir_map.get(skill.type, "writing")
        save_dir = self._skills_dir / subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = save_dir / f"{skill.name}.yaml"

        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(
                skill.model_dump(exclude_none=True),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False,
            )
        logger.info("skills.generator: saved skill '%s' to %s", skill.name, yaml_path)

        # 6. Register in registry
        if self._registry is not None:
            self._registry.register(skill)

        return skill

    # ------------------------------------------------------------------
    # URL resolution
    # ------------------------------------------------------------------

    async def _fetch_content(self, url: str) -> str:
        """
        Fetch content from a URL. Handles GitHub, file://, and generic URLs.

        - **Args**:
            - `url` (str): The URL to fetch

        - **Returns**:
            - `str`: The fetched text content
        """
        # Local file
        if url.startswith("file://"):
            local_path = Path(url.replace("file://", ""))
            if local_path.exists():
                return local_path.read_text(encoding="utf-8")
            raise FileNotFoundError(f"Local file not found: {local_path}")

        # GitHub blob URL → raw
        raw_url = self._github_to_raw(url)

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(raw_url)
            response.raise_for_status()
            text = response.text

        # GitHub anchor extraction
        anchor = self._extract_anchor(url)
        if anchor and text:
            text = self._extract_section_by_anchor(text, anchor)

        # Truncate if too long
        max_chars = 15000
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[... truncated ...]"

        return text

    @staticmethod
    def _github_to_raw(url: str) -> str:
        """
        Convert GitHub blob URL to raw.githubusercontent.com URL.

        - **Args**:
            - `url` (str): Original URL

        - **Returns**:
            - `str`: Raw content URL (or original if not GitHub)
        """
        # github.com/user/repo/blob/branch/path → raw.githubusercontent.com/user/repo/branch/path
        match = re.match(
            r"https?://github\.com/([^/]+)/([^/]+)/blob/([^#]+)",
            url,
        )
        if match:
            user, repo, path = match.groups()
            return f"https://raw.githubusercontent.com/{user}/{repo}/{path}"

        # github.com/user/repo (no path) → README.md
        match = re.match(
            r"https?://github\.com/([^/]+)/([^/#]+)/?(?:#.*)?$",
            url,
        )
        if match:
            user, repo = match.groups()
            return f"https://raw.githubusercontent.com/{user}/{repo}/main/README.md"

        return url.split("#")[0]  # Strip anchor for direct fetch

    @staticmethod
    def _extract_anchor(url: str) -> Optional[str]:
        """Extract the #anchor portion from a URL."""
        if "#" in url:
            return url.split("#", 1)[1]
        return None

    @staticmethod
    def _extract_section_by_anchor(text: str, anchor: str) -> str:
        """
        Extract the section starting at ## anchor until the next ## heading.

        - **Args**:
            - `text` (str): Full markdown text
            - `anchor` (str): The anchor ID (e.g. "writing-tips")

        - **Returns**:
            - `str`: Extracted section text
        """
        # Normalize anchor: GitHub converts headings to lowercase, replaces spaces with -
        anchor_lower = anchor.lower().replace("-", " ")
        lines = text.split("\n")
        start = None
        end = None

        for i, line in enumerate(lines):
            if line.startswith("#"):
                heading_text = re.sub(r"^#+\s*", "", line).strip().lower()
                heading_slug = heading_text.replace(" ", "-")
                if heading_slug == anchor.lower() or heading_text == anchor_lower:
                    start = i
                    continue
            if start is not None and line.startswith("##") and i > start:
                end = i
                break

        if start is not None:
            return "\n".join(lines[start : end or len(lines)])
        return text  # Fallback: return full text

    # ------------------------------------------------------------------
    # LLM extraction
    # ------------------------------------------------------------------

    async def _extract_skill(
        self,
        instruction: str,
        content: str,
        skill_type: Optional[str] = None,
    ) -> WritingSkill:
        """
        Use LLM to extract a WritingSkill from instruction + content.

        - **Args**:
            - `instruction` (str): User instruction
            - `content` (str): Fetched reference content
            - `skill_type` (str, optional): Override the extracted type

        - **Returns**:
            - `WritingSkill`: The extracted skill
        """
        user_message = (
            f"## Instruction\n{instruction}\n\n"
            f"## Reference Content\n{content}"
        )

        if skill_type:
            user_message += f"\n\n## Constraint\nThe skill type MUST be: {skill_type}"

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SKILL_EXTRACTION_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=2000,
        )

        raw = response.choices[0].message.content or ""
        # Strip markdown fences if present
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", raw.strip())
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:500]}")

        # Override type if specified
        if skill_type:
            data["type"] = skill_type

        return WritingSkill(**data)
