import os
import uuid

from dotenv import load_dotenv
load_dotenv()

from utils.color_print import cprint

from pathlib import Path
WORKDIR = Path.cwd()

from tool import TOOLS
from core.llm import DeepSeekLLM
from session_manage import recovery_sessions, write_sessions
from core.agent import Agent
from hook import register_hook, trigger_hook, LLMCallHookContext
from skills import scan_skills, build_system_prompt_for_skills

def after_turn_call(context: LLMCallHookContext):
    cprint("[HOOK] after_turn_call", color="bright_cyan")
    return write_sessions(session_id=context.session_id, messages=context.messages)

register_hook("after_turn_call", after_turn_call)

def main():
    session_id = uuid.uuid4().hex[:8]
    print('====================================')
    print('=             Mini Agent           =')
    print('====================================')
    print(f'Model: {os.environ.get("DEEPSEEK_MODEL")}')
    print(f'Session ID: {session_id}')
    print('====================================')
    scan_skills()
    messages = recovery_sessions(session_id)

    if len(messages) == 0:
        messages.append({
            'role': 'system',
            'content': (
                f'你是一个编程助手，堪比 Claude Code！你的工作目录在 {WORKDIR} 下\n'
                f'{build_system_prompt_for_skills()}'
            )
        })

    llm = DeepSeekLLM(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL")
    )

    agent = Agent(llm=llm, tools=TOOLS)

    while True:
        query = input('\033[94m[USER] > \033[0m')
        messages.append({
            'role': 'user',
            'content': query
        })
        agent.agent_loop(messages=messages)
        llm_context = LLMCallHookContext(
            llm=llm,
            agent=agent,
            session_id=session_id,
            messages=messages,
        )
        if not trigger_hook("after_turn_call", llm_context):
            break

if __name__ == '__main__':
    main()
