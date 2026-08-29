import os
import uuid

from dotenv import load_dotenv

from tool import TOOLS

load_dotenv()

from pathlib import Path
WORKDIR = Path.cwd()

from core.llm import DeepSeekLLM
from session_manage import recovery_sessions, write_sessions
from core.agent import Agent

def main():
    session_id = uuid.uuid4().hex[:8]
    print('====================================')
    print('=             Mini Agent           =')
    print('====================================')
    print(f'Model: {os.environ.get("DEEPSEEK_MODEL")}')
    print(f'Session ID: {session_id}')
    print('====================================')
    messages = recovery_sessions(session_id)

    if len(messages) == 0:
        messages.append({ 'role': 'system', 'content': f'你是一个编程助手，堪比 Claude Code！你的工作目录在 {WORKDIR} 下' })

    chat = DeepSeekLLM(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL")
    )

    agent = Agent(llm=chat, tools=TOOLS)

    while True:
        query = input('\033[94m[USER] > \033[0m')
        messages.append({
            'role': 'user',
            'content': query
        })
        agent.agent_loop(messages=messages)
        write_sessions(session_id=session_id, messages=messages)

if __name__ == '__main__':
    main()
