from typing import Any

import yaml
from pathlib import Path

WORKDIR = Path.cwd()
SKILLS_DIR = Path.cwd() / "skills"
SKILL_REGISTRY: dict[str, dict] = {}


def _parse_frontmatter(content: str) -> tuple[dict, str]:
    """解析 markdown 个 YAML 信息(meta)，并返回：(meta, body)"""
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, parts[2].strip()

"""
仅执行一次
"""
def _scan_skills():
    if not SKILLS_DIR.exists():
        return

    for d in sorted(SKILLS_DIR.iterdir()):
        if not d.is_dir():
            continue
        manifest = d / "SKILL.md"
        if manifest.exists():
            raw = manifest.read_text()
            meta, body = _parse_frontmatter(raw)
            name = meta.get("name", d.name)
            desc = meta.get("description", "")
            SKILL_REGISTRY[name] = {
                "name": name,
                "description": desc,
                "content": raw,
            }

def scan_skills():
    print("\n\033[94m------------[SKILLS]------------\033[0m")
    _scan_skills()
    print(list_skills())
    print("\033[94m--------------------------------\033[0m\n")

def list_skills() -> str:
    return "\n".join(f"- {s['name']}" for s in SKILL_REGISTRY.values())

def list_skills_reminder() -> str:
    return "\n".join(f"<skill>{s['name']}</skill>" for s in SKILL_REGISTRY.values())

def load_skill(name: str) -> str | dict[str, Any]:
    skill = SKILL_REGISTRY.get(name, None)
    if not skill:
        return f"Skill '{name}' not found."
    return skill["content"]

def build_system_prompt_for_skills() -> str:
    catalog = list_skills_reminder()
    return (
        f"Skills available:\n{catalog}\n"
        "Use load_skill to get full details when needed."
    )
