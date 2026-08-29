

import json


class StructuredTool:
    """极简自定义工具类：手动填写参数 schema，包装一个执行函数（自实现，不依赖第三方）。"""

    def __init__(self, name: str, description: str, args_schema: dict, func):
        self.name = name
        self.description = description
        self.args_schema = args_schema
        self.func = func

class ToolCall:
    """封装一次 LLM 返回的工具调用（对齐 OpenAI tool_calls 结构）。"""

    def __init__(self, tool_id: str, name: str, tool_input: dict):
        self.tool_id = tool_id
        self.name = name
        self.tool_input = tool_input

    def to_openai(self) -> dict:
        """转回 OpenAI 原始格式，用于随 assistant 消息回传给 API（arguments 必须为 JSON 字符串）。"""
        return {
            "id": self.tool_id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.tool_input),
            },
        }

def tool_to_openai(tool) -> dict:
    """把自定义工具实例转成 OpenAI 工具格式，供 tools 参数使用。"""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.args_schema,
        },
    }
