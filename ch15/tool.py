from pathlib import Path

from tools.structed_tool import tool_to_openai, StructuredTool
from tools.base_tools import run_bash, run_edit, run_glob, run_read, run_write
from subagent import spawn_subagent
from skills import load_skill
from task import (
    run_create_task, run_list_tasks, run_get_task,
    run_claim_task, run_complete_task,
)
from scheduler import (
    run_schedule_cron, run_list_crons, run_cancel_cron,
)

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
                # s13: 对齐真实 Claude Code —— 模型可显式声明后台执行，should_run_background 据此命中
                "run_in_background": {"type": "boolean",
                                      "description": "Run in the background, return immediately."},
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
    ),
    StructuredTool(
        name="compact",
        description="Summarize earlier conversation to free context space.",
        args_schema= {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string"
                }
            }
        },
        func=None
    ),
    # ── 任务系统（ch13）：落盘到 .tasks/*.json，带 blockedBy 依赖门禁 ──
    StructuredTool(
        name="create_task",
        description="Create a new task with optional blockedBy dependencies.",
        args_schema={
            "type": "object",
            "properties": {
                "subject": {"type": "string"},
                "description": {"type": "string"},
                "blockedBy": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["subject"],
        },
        func=run_create_task,
    ),
    StructuredTool(
        name="list_tasks",
        description="List all tasks with status, owner, and dependencies.",
        args_schema={"type": "object", "properties": {}, "required": []},
        func=run_list_tasks,
    ),
    StructuredTool(
        name="get_task",
        description="Get full details of a specific task by ID.",
        args_schema={
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
        func=run_get_task,
    ),
    StructuredTool(
        name="claim_task",
        description="Claim a pending task. Sets owner, changes status to in_progress.",
        args_schema={
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
        func=run_claim_task,
    ),
    StructuredTool(
        name="complete_task",
        description="Complete an in-progress task. Reports unblocked downstream tasks.",
        args_schema={
            "type": "object",
            "properties": {"task_id": {"type": "string"}},
            "required": ["task_id"],
        },
        func=run_complete_task,
    ),
    # ── 定时任务（ch15）：cron 表达式注册 / 列出 / 取消，到点由 agent_loop 注入 prompt ──
    StructuredTool(
        name="schedule_cron",
        description="Schedule a cron job. cron is 5-field: min hour dom month dow.",
        args_schema={
            "type": "object",
            "properties": {
                "cron": {"type": "string",
                         "description": "5-field cron expression"},
                "prompt": {"type": "string",
                           "description": "Message to inject when fired"},
                "recurring": {"type": "boolean",
                              "description": "True=recurring, False=one-shot"},
                "durable": {"type": "boolean",
                            "description": "True=persist to disk"},
            },
            "required": ["cron", "prompt"],
        },
        func=run_schedule_cron,
    ),
    StructuredTool(
        name="list_crons",
        description="List all registered cron jobs.",
        args_schema={"type": "object", "properties": {},
                     "required": []},
        func=run_list_crons,
    ),
    StructuredTool(
        name="cancel_cron",
        description="Cancel a cron job by ID.",
        args_schema={
            "type": "object",
            "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        },
        func=run_cancel_cron,
    ),
]

# 交给自己的 core/llm.py 序列化成 OpenAI 工具格式
TOOLS = [tool_to_openai(t) for t in TOOL_DEFINITIONS]

# 工具名 → 可执行函数 的映射
TOOL_CALL_MAP = {t.name: t.func for t in TOOL_DEFINITIONS}
