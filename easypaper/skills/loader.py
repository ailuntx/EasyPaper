"""
Skill Loader
- **Description**:
    - Loads WritingSkill definitions from YAML files on disk
    - Recursively scans a skills directory for .yaml files
"""
import logging
from pathlib import Path
from typing import List, Optional

import yaml

from .models import WritingSkill

logger = logging.getLogger("uvicorn.error")


class SkillLoader:
    """
    Loads WritingSkill objects from YAML files.

    - **Methods**:
        - `load_directory()`: Recursively scan a directory and load all .yaml files
        - `load_single()`: Load a single YAML file into a WritingSkill
    """

    def load_directory(self, skills_dir: Path) -> List[WritingSkill]:
        """
        Recursively load all .yaml files under *skills_dir*.

        - **Args**:
            - `skills_dir` (Path): Root directory to scan

        - **Returns**:
            - `List[WritingSkill]`: All successfully loaded skills
        """
        skills: List[WritingSkill] = []
        skills_path = Path(skills_dir)

        if not skills_path.exists():
            logger.warning("skills.loader: directory not found: %s", skills_path)
            return skills

        for yaml_file in sorted(skills_path.rglob("*.yaml")):
            skill = self.load_single(yaml_file)
            if skill is not None:
                skills.append(skill)

        logger.info(
            "skills.loader: loaded %d skills from %s",
            len(skills),
            skills_path,
        )
        return skills

    def load_single(self, path: Path) -> Optional[WritingSkill]:
        """
        Load a single YAML file into a WritingSkill.

        - **Args**:
            - `path` (Path): Path to the .yaml file

        - **Returns**:
            - `WritingSkill` or None if loading fails
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict):
                logger.warning("skills.loader: invalid YAML (not a dict): %s", path)
                return None

            skill = WritingSkill(**data)
            logger.debug("skills.loader: loaded skill '%s' from %s", skill.name, path)
            return skill

        except Exception as e:
            logger.warning("skills.loader: failed to load %s: %s", path, e)
            return None
