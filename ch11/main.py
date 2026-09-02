import os
import uuid

from dotenv import load_dotenv
load_dotenv()

from utils.color_print import cprint

from tool import TOOLS
from core.llm import DeepSeekLLM
from session_manage import recovery_sessions, write_sessions
from core.agent import Agent
from hook import register_hook, trigger_hook, LLMCallHookContext
from skills import scan_skills
from memory import load_memories, extract_memories
from prompt import update_context, get_system_prompt

def after_turn_call(context: LLMCallHookContext):
    cprint("[HOOK] after_turn_call: 保存会话", color="bright_cyan")
    return write_sessions(session_id=context.session_id, messages=context.messages)

register_hook("after_turn_call", after_turn_call)

def before_turn_call(context: LLMCallHookContext):
    """每轮对话开始前：把与当前问题相关的记忆注入最新一条 user 消息。"""
    cprint("[HOOK] before_turn_call: 注入相关记忆", color="bright_magenta")
    msg = context.messages
    if not msg:
        return True
    block = load_memories(msg)                       # 相关记忆正文；无记忆文件时返回 ""
    if block and msg[-1].get("role") == "user":
        msg[-1]["content"] = f"{msg[-1].get('content', '')}\n\n{block}"
    return True

register_hook("before_turn_call", before_turn_call)

def after_turn_memory(context: LLMCallHookContext):
    """每轮对话结束后：从对话中抽取新记忆并落盘（LLM 失败时静默）。"""
    cprint("[HOOK] after_turn_call: 抽取记忆", color="bright_magenta")
    extract_memories(context.messages)
    return True

register_hook("after_turn_call", after_turn_memory)

def main():
    session_id = uuid.uuid4().hex[:8]
    # session_id = ""
    print('====================================')
    print('=             Mini Agent           =')
    print('====================================')
    print(f'Model: {os.environ.get("DEEPSEEK_MODEL")}')
    print(f'Session ID: {session_id}')
    print('====================================')
    scan_skills()
    messages = recovery_sessions(session_id)

    if len(messages) == 0:
        # 分段组装 system prompt：真实工具表 / 技能 / 记忆索引由 prompt.py 运行时推导
        prompt_context = update_context({}, messages)
        messages.append({
            'role': 'system',
            'content': get_system_prompt(prompt_context),
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
        llm_context = LLMCallHookContext(
            llm=llm,
            agent=agent,
            session_id=session_id,
            messages=messages,
        )
        trigger_hook("before_turn_call", llm_context)      # 注入相关记忆
        agent.agent_loop(messages=messages)
        if not trigger_hook("after_turn_call", llm_context):  # 保存会话 + 抽取记忆
            break

if __name__ == '__main__':
    main()
