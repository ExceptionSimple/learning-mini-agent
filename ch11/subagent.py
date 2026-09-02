import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from core.llm import DeepSeekLLM
from tools.structed_tool import StructuredTool, tool_to_openai
from tools.base_tools import run_bash, run_read, run_glob, run_write, run_edit
from hook import trigger_hook, ToolCallHookContext

WORKDIR = Path.cwd()

# 子代理专属工具集：与父 Agent 隔离，只暴露文件与命令类基础工具
SUB_TOOL_DEFINITIONS = [
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
]

SUB_TOOLS = [tool_to_openai(t) for t in SUB_TOOL_DEFINITIONS]

SUB_TOOL_CALL_MAP = {t.name: t.func for t in SUB_TOOL_DEFINITIONS}

SUB_SYSTEM = (
    f"You are a coding agent at {WORKDIR}. "
    "Complete the task you were given, then return a concise summary. "
    "Do not delegate further."
)

def extract_text(content) -> str:
    """Extract text from message content blocks."""
    if not isinstance(content, list):
        return content or ""
    return "\n".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")

def spawn_subagent(description: str) -> str:
    # 属于 subagent 的全新上下文，与父 agent 隔离
    messages = [
        {"role": "system", "content": SUB_SYSTEM},
        {"role": "user", "content": description}
    ]
    llm = DeepSeekLLM(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL"),
    )

    result = None
    for _ in range(30):
        # 无状态非流式调用：messages 由外部维护，invoke 不写实例状态
        response = llm.invoke(
            messages=messages,
            model=os.environ.get("DEEPSEEK_MODEL"),
            tools=SUB_TOOLS,
        )

        if response['finish_reason'] == "stop":
            result = extract_text(response['content'])
            break

        if response['finish_reason'] != "tool_calls":
            continue

        # invoke 已把原始 tool_calls 解析为 ToolCall 对象存入实例
        tool_calls = llm.tool_calls

        # 回传 assistant 消息：to_openai() 负责序列化，arguments 必须为 JSON 字符串
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [tc.to_openai() for tc in tool_calls],
        })

        for tool in tool_calls:
            tool_call_context = ToolCallHookContext(
                tool_name=tool.name,
                tool_input=tool.tool_input,
                tool_result='',
                messages=messages,
                tool=tool,
            )
            if not trigger_hook("before_tool_call", tool_call_context):
                continue
            print(f"\033[33m[tool] {tool.name}\033[0m")
            handler = SUB_TOOL_CALL_MAP[tool.name]
            result = handler(**tool.tool_input)
            messages.append({
                "role": "tool",
                "tool_call_id": tool.tool_id,
                "content": result,
            })

    if not result:
        result = "Subagent stopped after 30 turns without final answer."

    print(f"\033[35m[Subagent done]\033[0m")
    return result
