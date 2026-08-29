from typing import Any

from core.chat_model import DeepSeekChat
from tool import StructuredTool

HOOKS = {
    # 调用工具之前
    "before_tool_call": [],
    # 调用工具之后
    "after_tool_call": [],
    # 调用 LLM 之前
    "before_llm_call": [],
    # 调用 LLM 之后
    "after_llm_call": [],
}

class ToolCallHookContext:
    def __init__(
            self,
            tool_name: str,
            tool_input: dict[str, Any],
            tool: StructuredTool | None = None,
            agent: DeepSeekChat | None = None,
            # task: Task | None = None,
            tool_result: str | None = None,
            raw_tool_result: Any | None = None,
    ) -> None:
        """Initialize tool call hook context.

        Args:
            tool_name: Name of the tool being called
            tool_input: Tool input parameters (mutable)
            tool: Tool instance reference
            agent: Optional agent executing the tool
            task: Optional current task
            crew: Optional crew instance
            tool_result: Optional tool result (for after hooks)
            raw_tool_result: Optional raw tool result (for after hooks)
        """
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.tool = tool
        self.agent = agent
        # self.task = task
        # self.crew = crew
        self.tool_result = tool_result
        self.raw_tool_result = raw_tool_result

def register_hook(event: str, callback):
    HOOKS[event].append(callback)

def trigger_hook(event: str, *args):
    if event not in HOOKS:
        print(f"{event} 事件不存在")
        return None

    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result

    return None
