import os
import json
from tools import TOOLS, TOOL_CALL_MAP

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
WORKDIR = Path.cwd()

from llm.chat_model import DeepSeekChat


chat = DeepSeekChat(
    api_key=os.environ.get("DEEPSEEK_API_KEY"),
    base_url=os.environ.get("DEEPSEEK_BASE_URL")
)

def agent_loop(messages: list):
    while True:
        in_reasoning = False
        for kind, delta in chat.stream(
            messages=messages,
            model="deepseek-v4-pro",
            thinking=True,
            tools=TOOLS
        ):
            if kind == 'reasoning':
                if not in_reasoning:
                    print('\n\033[94m-----------------[思考]------------------\033[0m\n')
                in_reasoning = True
            else:
                if in_reasoning:
                    print('\n\n\033[94m-----------------[回答]------------------\033[0m\n')
                in_reasoning = False
            print(delta, end='')
        print()

        if chat.finish_reason == 'stop':
            break

        if chat.finish_reason != "tool_calls":
            continue

        tool_calls = chat.tool_calls
        messages.append({
            'role': 'assistant',
            'content': None,
            "tool_calls": tool_calls
        })

        for tool in tool_calls:
            print(f"\033[33m[tool] {tool['function']['name']}\033[0m")
            handler = TOOL_CALL_MAP[tool['function']['name']]
            result = handler(**json.loads(tool['function']['arguments']))
            messages.append({
                "role": "tool",
                "tool_call_id": tool['id'],
                "content": result
            })

def main():
    print('====================================')
    print('=             Mini Agent           =')
    print('====================================')
    messages = [{ 'role': 'system', 'content': f'你是一个编程助手，堪比 Claude Code！你的工作目录在 {WORKDIR} 下' }]
    while True:
        query = input('\033[94m[USER] > \033[0m')
        messages.append({
            'role': 'user',
            'content': query
        })
        agent_loop(messages=messages)

if __name__ == '__main__':
    main()
