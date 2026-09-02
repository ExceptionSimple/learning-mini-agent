from fileinput import FileInput
from typing import TypedDict, Any, NotRequired, Literal


class LLMMessage(TypedDict):

    role: Literal["user", "assistant", "system", "tool"]
    content: str | list[dict[str, Any]] | None
    tool_call_id: NotRequired[str]
    name: NotRequired[str]
    tool_calls: NotRequired[list[dict[str, Any]]]
    raw_tool_call_parts: NotRequired[list[Any]]
    # files: NotRequired[dict[str, FileInput]]
