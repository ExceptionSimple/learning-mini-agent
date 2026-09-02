#!/usr/bin/env python3
"""
ch11/prompt.py —— system prompt 运行时组装

把 system prompt 拆成分段（section），运行时按真实状态按需拼接：
  - identity / subagent / todo_reminder 始终加载，且顺序保持稳定，
    利于服务端 prefix 缓存命中；
  - tools 段由 TOOLS schema 运行时推导（而非写死工具名），工具增删自动生效；
  - skills / memory_index 段按真实状态按需加载，无对应资源时不占位。

记忆正文仍由 main.py 注入用户消息（ch10 设计，保持 system 静态以命中
服务端 prefix 缓存），这里只负责把「记忆索引目录」作为 memory_index 段
按需拼入。

对外 API：
  - update_context(context, messages) -> dict   按真实状态推导上下文
  - get_system_prompt(context) -> str           取组装好的 system prompt（带缓存）
"""
import json
from pathlib import Path

from tool import TOOLS
from skills import SKILL_REGISTRY, list_skills_reminder
from memory import read_memory_index

WORKDIR = Path.cwd()

PROMPT_SECTIONS = {
    "identity": f"你是一个编程助手，堪比 Claude Code！你的工作目录在 {WORKDIR} 下。",
    "subagent": "For complex sub-problems, use the task tool to spawn a subagent.",
    "todo_reminder": "<reminder>遇到编辑代码任务时，记得要更新 todo 列表</reminder>",
    "tools": "Available tools: {tool_list}.",
    "skills": "Skills available:\n{skill_catalog}\nUse load_skill to get full details when needed.",
    "memory_index": "Memories available:\n{memories}",
}

# 静态段：始终加载，且顺序保持稳定（利于服务端 prefix 缓存命中）
STATIC_SECTIONS = ["identity", "tools", "subagent", "todo_reminder"]


def update_context(context: dict, messages: list) -> dict:
    """从真实状态推导上下文：可调用的工具、技能注册表、记忆索引文件。

    enabled_tools 取 TOOLS schema：compact 在 agent_loop 里特判、不在
    TOOL_CALL_MAP 中，直接取 schema 才能列出全部可调用工具（8 个）。
    """
    return {
        "enabled_tools": [t["function"]["name"] for t in TOOLS],
        "skills": list_skills_reminder() if SKILL_REGISTRY else "",
        "memories": read_memory_index(),
    }


def assemble_system_prompt(context: dict) -> str:
    """按 context 真实状态拼接：始终加载的静态段 + 按需加载的动态段。"""
    sections = [
        PROMPT_SECTIONS["identity"],
        PROMPT_SECTIONS["tools"].format(
            tool_list=", ".join(context.get("enabled_tools", []))),
        PROMPT_SECTIONS["subagent"],
        PROMPT_SECTIONS["todo_reminder"],
    ]
    skills = context.get("skills", "")
    if skills:
        sections.append(PROMPT_SECTIONS["skills"].format(skill_catalog=skills))
    memories = context.get("memories", "")
    if memories:
        sections.append(PROMPT_SECTIONS["memory_index"].format(memories=memories))
    return "\n\n".join(sections)


_last_context_key = None
_last_prompt = None


def get_system_prompt(context: dict) -> str:
    """缓存包装 —— context 没变时跳过重复拼接。

    用 json.dumps 做确定性 cache key：Python 内置 hash() 有进程随机化，且对
    list/dict 报 unhashable，不适合。这里的缓存只避免进程内重复拼接字符串；
    真正的 API 级 prefix 缓存由服务端命中，前提是静态段顺序稳定。
    """
    global _last_context_key, _last_prompt
    key = json.dumps(context, sort_keys=True, ensure_ascii=False, default=str)
    if key == _last_context_key and _last_prompt:
        print("  \033[90m[cache hit] system prompt unchanged\033[0m")
        return _last_prompt
    _last_context_key = key
    _last_prompt = assemble_system_prompt(context)
    loaded = list(STATIC_SECTIONS)
    if context.get("skills"):
        loaded.append("skills")
    if context.get("memories"):
        loaded.append("memory_index")
    print(f"  \033[32m[assembled] sections: {', '.join(loaded)}\033[0m")
    return _last_prompt
