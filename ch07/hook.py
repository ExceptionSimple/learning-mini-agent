from __future__ import annotations

from typing import TYPE_CHECKING, Any

# 以下仅用于类型注解，运行时不需要，避免与 core.agent 形成循环导入
if TYPE_CHECKING:
    from core.agent import Agent
    from core.llm import DeepSeekLLM
    from tool import StructuredTool
    from tools.structed_tool import ToolCall

HOOKS = {
    # 调用工具之前
    "before_tool_call": [],
    # 调用工具之后
    "after_tool_call": [],
    # 调用 LLM 之前
    "before_llm_call": [],
    # 调用 LLM 之后
    "after_llm_call": [],
    # 一轮 Turn 之后
    "after_turn_call": []
}

class ToolCallHookContext:
    def __init__(
            self,
            tool_name: str,
            tool_input: dict[str, Any],
            tool: ToolCall = None,
            agent: Agent = None,
            # task: Task | None = None,
            tool_result: str = None,
            raw_tool_result: Any | None = None,
            messages: list = None
    ) -> None:
        """Initialize tool call hook context.

        Args:
            tool_name: Name of the tool being called
            tool_input: Tool input parameters (mutable)
            tool: Tool instance reference
            agent: Optional agent executing the tool
            tool_result: Optional tool result (for after hooks)
            raw_tool_result: Optional raw tool result (for after hooks)
        """
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.tool = tool
        self.agent = agent
        self.tool_result = tool_result
        self.raw_tool_result = raw_tool_result
        self.messages = messages

class LLMCallHookContext:
    def __init__(self,
        agent: Agent = None,
        llm: DeepSeekLLM = None,
        messages: list = None,
        session_id: str = None
    ):
        self.agent = agent
        self.llm = llm
        self.messages = messages
        self.response: str | None = None
        self.session_id = session_id


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
