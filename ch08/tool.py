from pathlib import Path

from tools.structed_tool import tool_to_openai, StructuredTool
from tools.base_tools import run_bash, run_edit, run_glob, run_read, run_write
from subagent import spawn_subagent
from skills import load_skill

WORKDIR = Path.cwd()

# 用 StructuredTool 构造工具：schema 手动填写
TOOL_DEFINITIONS = [
    StructuredTool(
        name="bash",
        description="Run a shell command.",
        args_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to run."},
            },
            "required": ["command"],
        },
        func=run_bash,
    ),
    StructuredTool(
        name="read_file",
        description="Read file contents.",
        args_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to read."},
                "limit": {"type": "integer", "description": "Optional max number of lines to read."},
            },
            "required": ["path"],
        },
        func=run_read,
    ),
    StructuredTool(
        name="write_file",
        description="Write content to a file.",
        args_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to write."},
                "content": {"type": "string", "description": "Full content to write."},
            },
            "required": ["path", "content"],
        },
        func=run_write,
    ),
    StructuredTool(
        name="edit_file",
        description="Replace exact text in a file once.",
        args_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to edit."},
                "old_text": {"type": "string", "description": "Exact text to find."},
                "new_text": {"type": "string", "description": "Replacement text."},
            },
            "required": ["path", "old_text", "new_text"],
        },
        func=run_edit,
    ),
    StructuredTool(
        name="glob",
        description="Find files matching a glob pattern.",
        args_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'."},
            },
            "required": ["pattern"],
        },
        func=run_glob,
    ),
    StructuredTool(
        name="task",
        description="Launch a subagent to handle a complex subtask. Returns only the final conclusion.",
        args_schema={
            "type": "object",
            "properties": {
                "description": {
                    "type": "string"
                }
            },
            "required": ["description"]
        },
        func=spawn_subagent
    ),
    StructuredTool(
        name="load_skill",
        description="Load the full content of a skill by name.",
        args_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string"
                }
            },
            "required": ["name"]
        },
        func=load_skill
    )
]

# 交给自己的 core/llm.py 序列化成 OpenAI 工具格式
TOOLS = [tool_to_openai(t) for t in TOOL_DEFINITIONS]

# 工具名 → 可执行函数 的映射
TOOL_CALL_MAP = {t.name: t.func for t in TOOL_DEFINITIONS}
