"""
Skills CLI
- **Description**:
    - Command-line interface for skill management
    - register: Auto-register a skill from instruction + URL
    - list: List all loaded skills
    - remove: Remove a skill YAML file

Usage:
    python -m src.skills.cli register --instruction "..." --ref "URL" [--name NAME] [--type TYPE]
    python -m src.skills.cli list [--skills-dir ./skills]
    python -m src.skills.cli remove --name NAME [--skills-dir ./skills]
"""
import argparse
import asyncio
import sys
from pathlib import Path

import yaml


def cmd_register(args):
    """Register a new skill from instruction + reference URL."""
    from ..agents.shared.llm_client import LLMClient
    from .generator import SkillGenerator
    from .registry import SkillRegistry
    from .loader import SkillLoader

    skills_dir = Path(args.skills_dir)

    # Load existing registry
    registry = SkillRegistry()
    loader = SkillLoader()
    for skill in loader.load_directory(skills_dir):
        registry.register(skill)

    # Build LLM client from env
    import os
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("SKILL_LLM_MODEL", "gpt-4o-mini")

    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is required.")
        sys.exit(1)

    client = LLMClient(api_key=api_key, base_url=base_url)
    generator = SkillGenerator(
        llm_client=client,
        model_name=model_name,
        skills_dir=skills_dir,
        registry=registry,
    )

    async def _run():
        skill = await generator.register(
            instruction=args.instruction,
            reference_url=args.ref,
            name=args.name,
            skill_type=args.type,
        )
        print(f"Registered skill: {skill.name}")
        print(f"  Type: {skill.type}")
        print(f"  Description: {skill.description}")
        print(f"  Source: {skill.source_url}")
        print(f"  Tags: {skill.tags}")

    asyncio.run(_run())


def cmd_list(args):
    """List all skills in the skills directory."""
    from .loader import SkillLoader
    from .registry import SkillRegistry

    skills_dir = Path(args.skills_dir)
    loader = SkillLoader()
    skills = loader.load_directory(skills_dir)

    if not skills:
        print(f"No skills found in {skills_dir}")
        return

    # Group by type
    by_type = {}
    for s in skills:
        by_type.setdefault(s.type, []).append(s)

    for skill_type, items in sorted(by_type.items()):
        print(f"\n[{skill_type}]")
        for s in sorted(items, key=lambda x: x.priority):
            sections = ", ".join(s.target_sections)
            ap_count = len(s.anti_patterns)
            prompt_len = len(s.system_prompt_append)
            print(
                f"  {s.name:<30} prio={s.priority:<3} sections=[{sections}] "
                f"anti_patterns={ap_count} prompt_chars={prompt_len}"
            )

    print(f"\nTotal: {len(skills)} skills")


def cmd_remove(args):
    """Remove a skill YAML file."""
    skills_dir = Path(args.skills_dir)
    removed = False

    for yaml_file in skills_dir.rglob("*.yaml"):
        try:
            with open(yaml_file, "r") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict) and data.get("name") == args.name:
                yaml_file.unlink()
                print(f"Removed: {yaml_file}")
                removed = True
                break
        except Exception:
            continue

    if not removed:
        print(f"Skill '{args.name}' not found in {skills_dir}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="EasyPaper Skills CLI",
        prog="python -m src.skills.cli",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # register
    reg_parser = subparsers.add_parser("register", help="Auto-register a skill")
    reg_parser.add_argument("--instruction", required=True, help="What the skill should do")
    reg_parser.add_argument("--ref", required=True, help="Reference URL or file:// path")
    reg_parser.add_argument("--name", default=None, help="Override skill name")
    reg_parser.add_argument("--type", default=None, help="Override skill type")
    reg_parser.add_argument("--skills-dir", default="./skills", help="Skills directory")
    reg_parser.set_defaults(func=cmd_register)

    # list
    list_parser = subparsers.add_parser("list", help="List all skills")
    list_parser.add_argument("--skills-dir", default="./skills", help="Skills directory")
    list_parser.set_defaults(func=cmd_list)

    # remove
    rm_parser = subparsers.add_parser("remove", help="Remove a skill")
    rm_parser.add_argument("--name", required=True, help="Skill name to remove")
    rm_parser.add_argument("--skills-dir", default="./skills", help="Skills directory")
    rm_parser.set_defaults(func=cmd_remove)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
