import os
import uuid
import json

from dotenv import load_dotenv
load_dotenv()

from .llm import DeepSeekLLM
from tools.structed_tool import StructuredTool
from tool import TOOLS, TOOL_CALL_MAP
from permission import check_permission

class Agent:
    def __init__(self, llm: DeepSeekLLM, tools: list[StructuredTool]):
        self.llm = llm
        self.tools = tools

    def agent_loop(self, messages: list):
        while True:
            in_reasoning = False
            reasoning_content = ""
            answer_content = ""
            for kind, delta in self.llm.stream(
                    messages=messages,
                    model=os.environ.get("DEEPSEEK_MODEL"),
                    thinking=True,
                    tools=self.tools
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
            print()

            if answer_content is not None and answer_content.strip() != "":
                messages.append({
                    "role": "assistant",
                    "content": answer_content,
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
                if not check_permission(tool):
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool.tool_id,
                        "content": "Permission denied."
                    })
                    continue

                print(f"\033[33m[tool] {tool.name}\033[0m")
                handler = TOOL_CALL_MAP[tool.name]
                result = handler(**tool.tool_input)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool.tool_id,
                    "content": result
                })
