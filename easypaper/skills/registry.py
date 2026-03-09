"""
Skill Registry
- **Description**:
    - Global in-memory registry of WritingSkill objects
    - Provides query methods for the PromptCompiler and ReviewerAgent
"""
import logging
from typing import Dict, List, Optional, Any

from .models import WritingSkill

logger = logging.getLogger("uvicorn.error")


class SkillRegistry:
    """
    Global registry that holds loaded WritingSkill instances and provides
    query helpers for downstream consumers (PromptCompiler, Checkers).

    - **Methods**:
        - `register()` / `unregister()`: Add or remove a skill
        - `get_writing_skills()`: Get skills matching a section type and optional venue
        - `get_checker_skills()`: Get skills intended for reviewer checkers
        - `get_venue_profile()`: Get a specific venue profile by name
        - `list_all()`: Dump all registered skills as dicts
    """

    def __init__(self) -> None:
        self._skills: Dict[str, WritingSkill] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, skill: WritingSkill) -> None:
        """
        Register a skill. Overwrites if the same name already exists.

        - **Args**:
            - `skill` (WritingSkill): The skill to register
        """
        if skill.name in self._skills:
            logger.info("skills.registry: overwriting skill '%s'", skill.name)
        self._skills[skill.name] = skill
        logger.debug(
            "skills.registry: registered '%s' (type=%s, priority=%d)",
            skill.name,
            skill.type,
            skill.priority,
        )

    def unregister(self, name: str) -> bool:
        """
        Remove a skill by name.

        - **Args**:
            - `name` (str): Skill name to remove

        - **Returns**:
            - `bool`: True if removed, False if not found
        """
        if name in self._skills:
            del self._skills[name]
            logger.info("skills.registry: removed skill '%s'", name)
            return True
        return False

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_writing_skills(
        self,
        section_type: str,
        venue: Optional[str] = None,
        active_names: Optional[List[str]] = None,
    ) -> List[WritingSkill]:
        """
        Get writing_constraint skills that match *section_type* and optionally
        a *venue* profile.  Results are sorted by priority (ascending).

        - **Args**:
            - `section_type` (str): Current section being written
            - `venue` (str, optional): Venue name to include venue_profile skills
            - `active_names` (List[str], optional): If provided and does not contain
              "*", only return skills whose name is in this list.

        - **Returns**:
            - `List[WritingSkill]`: Matched skills sorted by priority
        """
        results: List[WritingSkill] = []

        for skill in self._skills.values():
            # Filter by active_names if specified
            if active_names and "*" not in active_names:
                if skill.name not in active_names:
                    continue

            # Include writing_constraint skills that match section
            if skill.type == "writing_constraint":
                if "*" in skill.target_sections or section_type in skill.target_sections:
                    results.append(skill)

            # Include venue_profile if it matches the requested venue
            elif skill.type == "venue_profile" and venue:
                if self._venue_matches(skill_name=skill.name, venue=venue):
                    if "*" in skill.target_sections or section_type in skill.target_sections:
                        results.append(skill)

        results.sort(key=lambda s: s.priority)
        return results

    def get_checker_skills(self) -> List[WritingSkill]:
        """
        Get all reviewer_checker skills, sorted by priority.

        - **Returns**:
            - `List[WritingSkill]`: Checker skills sorted by priority
        """
        results = [
            s for s in self._skills.values() if s.type == "reviewer_checker"
        ]
        results.sort(key=lambda s: s.priority)
        return results

    def get_venue_profile(self, venue: str) -> Optional[WritingSkill]:
        """
        Get a venue_profile skill by venue name (case-insensitive partial match).

        - **Args**:
            - `venue` (str): Venue identifier (e.g. "neurips", "icml")

        - **Returns**:
            - `WritingSkill` or None
        """
        venue_lower = venue.lower()
        for skill in self._skills.values():
            if skill.type == "venue_profile":
                if self._venue_matches(skill_name=skill.name, venue=venue):
                    return skill
        return None

    @staticmethod
    def _venue_matches(skill_name: str, venue: str) -> bool:
        """
        Runtime venue matcher with tolerant normalization.
        """
        if not skill_name or not venue:
            return False
        s = str(skill_name).strip().lower()
        v = str(venue).strip().lower()
        if s == v:
            return True
        # Normalize separators and keep alnum tokens only.
        s_tokens = [t for t in "".join(ch if ch.isalnum() else " " for ch in s).split() if t]
        v_tokens = [t for t in "".join(ch if ch.isalnum() else " " for ch in v).split() if t]
        if not s_tokens or not v_tokens:
            return s in v or v in s
        s_join = " ".join(s_tokens)
        v_join = " ".join(v_tokens)
        if s_join in v_join or v_join in s_join:
            return True
        # Token overlap rule: venue mention contains the profile key token.
        # e.g. "nature portfolio" -> matches "nature"
        return any(tok == s_join for tok in v_tokens) or (s_tokens[0] in v_tokens)

    def list_all(self) -> List[Dict[str, Any]]:
        """
        Dump all registered skills as serializable dicts.

        - **Returns**:
            - `List[Dict]`: List of skill dictionaries
        """
        return [
            {
                "name": s.name,
                "type": s.type,
                "description": s.description,
                "version": s.version,
                "tags": s.tags,
                "priority": s.priority,
                "target_sections": s.target_sections,
                "has_prompt": bool(s.system_prompt_append),
                "anti_patterns_count": len(s.anti_patterns),
            }
            for s in sorted(self._skills.values(), key=lambda s: (s.type, s.priority))
        ]

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills
