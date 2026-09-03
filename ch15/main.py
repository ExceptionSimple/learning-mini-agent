import contextlib
import os
import threading
import time
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
# s15: cron 调度 —— scheduler 是纯调度库（只负责"到点入队"，不驱动 agent）；
# "空闲时把到点任务投递成一轮新回合"的职责在 main，用户回合与 cron 投递
# 共用 scheduler.agent_lock 串行化，绝不打断正在进行的对话。
from scheduler import (
    has_cron_queue, consume_cron_queue, agent_lock, cron_log, LOG_PATH,
)

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

    def run_turn(query: str) -> bool:
        """驱动一轮 agent 回合（调用方须已持有 agent_lock）。

        query 由调用方给出：用户回合 = 原话；cron 投递 = "[cron] " 拼接的任务 prompt。
        流程与旧 REPL 主循环一致：追加用户消息 -> before_turn 注记忆 -> agent_loop
        -> after_turn（保存会话 + 抽取记忆）；返回 False 表示应退出主循环。
        """
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
        return trigger_hook("after_turn_call", llm_context)  # 保存会话 + 抽取记忆

    def run_turn_logged(query: str) -> bool:
        """投递线程专用回合：把 run_turn 整条链路的 stdout（钩子打印、agent_loop
        流式输出、记忆抽取…）全部重定向进 scheduler.log，终端保持干净。

        为什么不影响用户回合：重定向期间 sys.stdout 全局指向日志文件，但 cron
        投递在 agent_lock 下独占运行（此刻主线程阻塞在 input() 或锁上、不打印），
        用户回合走的是下方原版 run_turn，输出照常上终端。
        """
        with open(LOG_PATH, "a", encoding="utf-8") as f, \
                contextlib.redirect_stdout(f):
            f.write("\n--- cron turn ---\n")
            return run_turn(query)

    def queue_processor_loop():
        """s15: 后台投递线程 —— agent 空闲（主循环阻塞在 input、锁空闲）时把到点的
        cron 任务自动投递成一轮新回合，不用等用户下一句输入。

        为什么 agent_lock 非阻塞抢占：拿不到锁说明用户回合正在跑，跳过本轮继续等，
        绝不打断正在进行的对话。为什么拿到锁后再查一次队列：acquire 之前队列非空，
        但等待期间可能已被消费光，双检避免空转一轮 agent 回合。
        """
        while True:
            time.sleep(0.2)
            if not has_cron_queue():
                continue
            if not agent_lock.acquire(blocking=False):
                continue
            try:
                if not has_cron_queue():
                    continue
                fired = consume_cron_queue()
                prompts = [f"[cron] {j.prompt}" for j in fired]
                # 投递详情先落盘（即使回合中途失败也留痕）；回合 stdout 在
                # run_turn_logged 里整体重定向进 scheduler.log
                cron_log(f"[queue processor] deliver {len(fired)} → "
                         + " || ".join(prompts))
                try:
                    run_turn_logged("\n".join(prompts))
                except Exception as e:
                    # 单次投递失败不杀线程：打印告警，后续 cron 仍能正常投递
                    cprint(f"\n[queue processor] delivery failed: {e}", color="red")
                else:
                    # 终端只留一条极简反馈，回合详情见 scheduler.log
                    print("\n\033[90m[cron turn done — 详见 scheduler.log]\033[0m")
            finally:
                agent_lock.release()

    # s15: 启动 cron 自动投递线程（daemon：主进程退出随之中止）
    threading.Thread(target=queue_processor_loop, daemon=True).start()

    while True:
        query = input('\033[94m[USER] > \033[0m')
        if not query.strip():
            # 空输入（裸回车）不触发回合：避免 append 空 user 消息让模型空转一轮
            continue
        with agent_lock:                      # 与 cron 投递回合串行化
            if not run_turn(query):           # 保存会话/抽记忆返回 False 才退出
                break

if __name__ == '__main__':
    main()
