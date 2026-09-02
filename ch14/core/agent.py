from dotenv import load_dotenv

from utils.color_print import cprint

load_dotenv()

from .llm import DeepSeekLLM
from tools.structed_tool import StructuredTool
from tool import TOOLS, TOOL_CALL_MAP
from permission import check_permission
from hook import register_hook, trigger_hook, ToolCallHookContext
from context import compact_history
from error_recovery import (
    RecoveryState,
    DEFAULT_MAX_TOKENS,
    ESCALATED_MAX_TOKENS,
    MAX_RECOVERY_RETRIES,
    CONTINUATION_PROMPT,
    stream_with_recovery,
    is_prompt_too_long_error,
    reactive_compact,
)

def before_tool_call(context: ToolCallHookContext) -> bool:
    cprint("[HOOK] before_tool_call", color="bright_cyan")
    if not check_permission(context.tool):
        context.messages.append({
            "role": "tool",
            "tool_call_id": context.tool.tool_id,
            "content": "Permission denied."
        })
        return False
    return True

register_hook("before_tool_call", before_tool_call)

class Agent:
    def __init__(self, llm: DeepSeekLLM, tools: list[StructuredTool]):
        self.llm = llm
        self.tools = tools

    def agent_loop(self, messages: list):
        # 本轮错误恢复状态：升级 / 续写 / 应急压缩 / 繁忙退避 都只在一轮内计数
        state = RecoveryState()
        max_tokens = DEFAULT_MAX_TOKENS
        while True:
            in_reasoning = False
            reasoning_content = ""
            answer_content = ""
            try:
                # 流式调用自带 429/5xx 指数退避重试（error_recovery.stream_with_recovery）；
                # 非瞬时错误在此 catch，按 Path 2（prompt 超长 -> 应急压缩）或不可恢复处理
                for kind, delta in stream_with_recovery(
                        self.llm,
                        state,
                        messages=messages,
                        model=state.current_model,
                        thinking=True,
                        tools=self.tools,
                        max_tokens=max_tokens,
                ):
                    if kind == 'reasoning':
                        if not in_reasoning:
                            print('\n\033[94m-----------------[思考]------------------\033[0m\n')
                        in_reasoning = True
                        reasoning_content += delta
                    else:
                        if in_reasoning:
                            print('\n\n\033[94m-----------------[回答]------------------\033[0m\n')
                        in_reasoning = False
                        answer_content += delta
                    print(delta, end='')
            except Exception as e:
                # Path 2: prompt/上下文超长 -> 应急压缩一次后重试
                if is_prompt_too_long_error(e) and not state.has_attempted_reactive_compact:
                    state.has_attempted_reactive_compact = True
                    cprint("[reactive compact] 上下文超长，应急压缩后重试", color="red")
                    system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
                    rest = messages[1:] if system_msg else messages
                    compacted = reactive_compact(rest)
                    messages[:] = ([system_msg] + compacted) if system_msg else compacted
                    continue
                # 其余错误不可自动恢复：记录错误消息并结束本轮（不让 while 死循环）
                cprint(f"[unrecoverable] {type(e).__name__}: {str(e)[:200]}", color="red")
                messages.append({
                    "role": "assistant",
                    "content": f"[Error] {type(e).__name__}: {str(e)[:200]}",
                })
                break
            print()

            # Path 1: max_tokens 截断（finish_reason == "length"）
            if self.llm.finish_reason == "length":
                if not state.has_escalated:
                    # 首次截断：只升级 max_tokens，丢弃截断输出、重发同一请求（不 append）
                    max_tokens = ESCALATED_MAX_TOKENS
                    state.has_escalated = True
                    cprint(f"[max_tokens] 升级 {DEFAULT_MAX_TOKENS} -> {ESCALATED_MAX_TOKENS}，重发同一请求",
                           color="yellow")
                    continue
                if state.recovery_count < MAX_RECOVERY_RETRIES:
                    # 升档后仍截断：保存截断输出并追加续写提示，让模型接着写
                    state.recovery_count += 1
                    if answer_content is not None and answer_content.strip() != "":
                        messages.append({"role": "assistant", "content": answer_content})
                    messages.append({"role": "user", "content": CONTINUATION_PROMPT})
                    cprint(f"[max_tokens] 追加续写提示 {state.recovery_count}/{MAX_RECOVERY_RETRIES}",
                           color="yellow")
                    continue
                cprint("[max_tokens] 达到续写上限，放弃本轮", color="red")
                break

            if answer_content is not None and answer_content.strip() != "":
                messages.append({
                    "role": "assistant",
                    "content": answer_content,
                    "reasoning_content": reasoning_content
                })

            if self.llm.finish_reason == 'stop':
                break

            if self.llm.finish_reason != "tool_calls":
                continue

            tool_calls = self.llm.tool_calls
            messages.append({
                'role': 'assistant',
                'content': None,
                "tool_calls": [tool.to_openai() for tool in tool_calls]
            })

            for tool in tool_calls:
                tool_call_context = ToolCallHookContext(
                    tool_name=tool.name,
                    tool_input=tool.tool_input,
                    tool_result='',
                    messages=messages,
                    tool=tool
                )
                if not trigger_hook("before_tool_call", tool_call_context):
                    continue

                print(f"\033[33m[tool] {tool.name}\033[0m")

                if tool.name == "compact":
                    system_msg = messages[0]
                    compact_history_msg = compact_history(messages[1:])
                    messages[:] = [system_msg] + compact_history_msg
                    break

                handler = TOOL_CALL_MAP[tool.name]
                result = handler(**tool.tool_input)
                tool_call_context.tool_result = result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool.tool_id,
                    "content": result
                })
