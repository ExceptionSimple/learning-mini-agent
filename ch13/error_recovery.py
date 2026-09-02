#!/usr/bin/env python3
"""
ch12/error_recovery.py —— LLM 调用过程的错误修复原语（s11: Error Recovery）

把 LLM 调用包上三条恢复路径，避免因瞬时错误 / 输出截断 / 上下文超长中断整轮对话：

  Path 1  max_tokens 截断（DeepSeek finish_reason == "length"）
          -> 首次把 max_tokens 从 DEFAULT_MAX_TOKENS 升到 ESCALATED_MAX_TOKENS，重发同一请求
             （不 append 截断输出，避免把半截话当真）；仍截断则保存截断输出并追加一条
             CONTINUATION_PROMPT 让模型接着写，最多 MAX_RECOVERY_RETRIES 次。
  Path 2  prompt / 上下文超长（DeepSeek 以 400 返回，message 含 "maximum context length" 等字样）
          -> reactive_compact()（复用 ch09 context.py 的应急压缩：先落盘留痕、再 LLM 摘要、
             保留最近几条），同一轮只触发一次；仍超长则判定不可恢复。
  Path 3  限流 / 服务器繁忙（瞬时错误）
          -> 指数退避 + 抖动重试（MAX_RETRIES 次）；连续 MAX_CONSECUTIVE_OVERLOAD 次
             "繁忙"自动切换到备用模型 DEEPSEEK_FALLBACK_MODEL。

对照 DeepSeek 官方错误码（api-docs.deepseek.com/quick_start/error_codes）：
    400 格式错误       —— 非瞬时：按 Path 2 关键字判断（上下文超长）或直接放弃
    401 认证失败        —— 非瞬时：不重试（重试无意义）
    402 余额不足        —— 非瞬时：不重试
    422 参数错误        —— 非瞬时：不重试
    429 请求速率达上限  —— 瞬时：指数退避后重试（Path 3）
    500 / 503 服务器故障/繁忙 —— 瞬时：退避 + 连续计数，达阈值切备用模型（Path 3）
    其余连接类异常      —— 视为瞬时（服务端不可达/超时），按 503 路径退避
  注：DeepSeek 没有 529；529 是 Anthropic 语义。本模块对"繁忙"统一按 HTTP 5xx 判断。

适配说明（相对 s11 参考脚本）：
  - 参考脚本是独立可跑的 REPL（内置 Anthropic client + 自带 tools/prompt 组装 + agent loop），
    本项目已把各层拆在 core/llm.py / tool.py / prompt.py / context.py。故本文件只保留
    「错误修复原语」，接入方是 core/agent.py 的 agent_loop（流式 LLM 调用）。
  - LLM 由 Anthropic client 换成项目 DeepSeekLLM（requests 直连）：
      · 错误语义按 requests/HTTP 判断（HTTPError.status_code + 响应体关键字）；
      · finish_reason == "length" 对应参考版的 stop_reason == "max_tokens"；
      · prompt 超长的应急压缩直接复用 ch09 的 context.reactive_compact（参考版因 s08/s09
        已覆盖 LLM 压缩，退化为"保留末 N 条"；本项目用真正的 LLM 摘要版）。
  - 模型配置：主模型仍读环境变量 DEEPSEEK_MODEL；可选 DEEPSEEK_FALLBACK_MODEL 作繁忙兜底，
    不配置则退化为持续重试同一模型。
  - 打印统一走 utils.color_print.cprint，颜色语义：黄 = 退避重试，红 = 降级 / 放弃。
"""

import os
import time
import random

import requests

from dotenv import load_dotenv
load_dotenv()

from context import reactive_compact as _context_reactive_compact
from utils.color_print import cprint

# ── 常量 ──────────────────────────────────────────────
DEFAULT_MAX_TOKENS = 4096                 # 默认 max_tokens（与 DeepSeekLLM.invoke/stream 默认对齐）
ESCALATED_MAX_TOKENS = 8192               # max_tokens 截断后升档的目标值（按模型输出上限自行调整）
MAX_RECOVERY_RETRIES = 3                  # 续写提示最多追加次数
MAX_RETRIES = 10                          # 瞬时错误退避重试的总次数上限
BASE_DELAY_MS = 500                       # 指数退避的初始延迟（毫秒）
MAX_CONSECUTIVE_OVERLOAD = 3              # 连续多少次"服务器繁忙"(5xx) 后切换到备用模型

# 模型配置：主模型读 DEEPSEEK_MODEL；备用模型可选（不配则繁忙时保持原模型继续重试）
PRIMARY_MODEL = os.environ.get("DEEPSEEK_MODEL")
FALLBACK_MODEL = os.environ.get("DEEPSEEK_FALLBACK_MODEL")

# max_tokens 截断后的续写提示：让模型不带客套地接着上一段继续输出
CONTINUATION_PROMPT = (
    "输出长度达到上限被截断。请紧接着上一段直接继续输出，不要重复、不要道歉、不要小结。"
)


class RecoveryState:
    """单轮对话（一次 agent_loop）内、跨多次 LLM 调用的恢复状态。

    main 每问一句调一次 agent_loop，agent_loop 内部可能多次请求 LLM（工具循环 /
    截断重试 / 续写）。这里用实例字段记录"哪些恢复动作已经做过"，避免同一轮里
    反复升级、反复压缩造成浪费或死循环。
    """

    def __init__(self):
        self.has_escalated = False                    # 是否已把 max_tokens 升过档（只升一次）
        self.recovery_count = 0                       # 已追加的续写提示次数（上限 MAX_RECOVERY_RETRIES）
        self.consecutive_overload = 0                 # 连续"服务器繁忙"(5xx) 计数（达阈值切备用模型）
        self.has_attempted_reactive_compact = False   # 本轮是否已做过应急压缩（只做一次）
        self.current_model = PRIMARY_MODEL            # 当前生效的模型（可能因繁忙被切到备用模型）


def _response_text(e: Exception) -> str:
    """从异常里尽量取出可判读的文本（HTTP 响应体优先，退化到 str(e)）。"""
    resp = getattr(e, "response", None)
    if resp is not None:
        return getattr(resp, "text", None) or str(e)
    return str(e)


def retry_delay(attempt: int, retry_after=None) -> float:
    """计算退避等待时长（秒）：指数退避 + 抖动。

    参数:
        attempt: 当前已重试的次数（从 0 起），决定 2^attempt 的指数基底。
        retry_after: 服务端 Retry-After 指定的秒数，给定时优先采用，忽略退避公式。

    返回:
        float: 应睡眠的秒数。上限 32s + 25% 抖动，避免多实例同时打爆服务端。
    """
    if retry_after:
        return float(retry_after)
    base = min(BASE_DELAY_MS * (2 ** attempt), 32000) / 1000
    jitter = random.uniform(0, base * 0.25)
    return base + jitter


def classify_transient(e: Exception) -> str | None:
    """判断异常是否属于"可自动重试的瞬时错误"，并给出类别。

    对照 DeepSeek 错误码，瞬时错误分两类：
        ratelimit  -> HTTP 429（请求速率达上限），单纯退避即可；
        overloaded -> HTTP 5xx（500 故障 / 503 繁忙）及连接类异常，退避之外还要累计
                      连续次数，达阈值时切备用模型（见 _handle_transient）。

    非瞬时错误（400 格式/401 认证/402 余额/422 参数等）返回 None，应当原样上抛，
    由外层做专门处理（如 400 上下文超长 -> Path 2）或直接放弃，不应盲目重试。

    参数:
        e: 调用抛出的异常（多为 requests.exceptions.HTTPError，也可能是连接类异常）。

    返回:
        "ratelimit" | "overloaded" | None。
    """
    status = None
    resp = getattr(e, "response", None)
    if resp is not None:
        status = getattr(resp, "status_code", None)
    text = _response_text(e).lower()

    # 连接类异常：读不到响应，视为瞬时（服务端不可达/超时），按 overloaded 退避
    if isinstance(e, (requests.exceptions.ConnectionError,
                      requests.exceptions.Timeout,
                      requests.exceptions.ChunkedEncodingError)):
        return "overloaded"

    # 429 限流（含个别代理把错误文本放进 message 的情况）
    if status == 429 or "ratelimit" in text or "rate limit" in text or "429" in text:
        return "ratelimit"
    # 5xx：500 服务器故障 / 503 服务器繁忙
    if isinstance(e, requests.exceptions.HTTPError) and status is not None and status >= 500:
        return "overloaded"
    # 兜底关键字：代理/网关可能把繁忙语义放进响应文本
    if ("overloaded" in text or "server busy" in text
            or "temporarily unavailable" in text):
        return "overloaded"
    return None


def _handle_transient(state: RecoveryState, category: str, attempt: int) -> float:
    """瞬时错误的公共处理：更新状态并返回应睡眠的秒数。

    对 overloaded 额外累计连续繁忙次数：达到 MAX_CONSECUTIVE_OVERLOAD 且配置了备用
    模型时，切换 state.current_model 并清零计数（连续过载视为当前模型不可用）。

    参数:
        state: 恢复状态，会被就地更新。
        category: classify_transient 的返回值（"ratelimit"/"overloaded"）。
        attempt: 已重试次数，用于计算退避时长。

    返回:
        float: 调用方应 time.sleep 的秒数。
    """
    if category == "overloaded":
        state.consecutive_overload += 1
        if state.consecutive_overload >= MAX_CONSECUTIVE_OVERLOAD:
            if FALLBACK_MODEL and state.current_model != FALLBACK_MODEL:
                state.current_model = FALLBACK_MODEL
                state.consecutive_overload = 0
                cprint(f"[5xx x{MAX_CONSECUTIVE_OVERLOAD}] 切换模型 -> {FALLBACK_MODEL}", color="red")
            else:
                state.consecutive_overload = 0
                cprint(f"[5xx x{MAX_CONSECUTIVE_OVERLOAD}] 未配置 DEEPSEEK_FALLBACK_MODEL，保持原模型重试",
                       color="red")
    delay = retry_delay(attempt)
    tag = "429 rate limit" if category == "ratelimit" else "5xx server busy"
    cprint(f"[{tag}] 重试 {attempt + 1}/{MAX_RETRIES}，等待 {delay:.1f}s", color="yellow")
    return delay


def with_retry(fn, state: RecoveryState):
    """非流式 LLM 调用（DeepSeekLLM.invoke）的退避重试封装。

    参考脚本用它包 client.messages.create；本项目 invoke 是同步返回 dict 的非流式
    调用，同样适用（如记忆抽取 / 上下文摘要等旁路调用若也想抗瞬时错误）。

    参数:
        fn: 无参可调用对象，执行一次 LLM 调用并返回结果。
        state: 恢复状态（记录连续繁忙与备用模型切换）。

    返回:
        fn() 的成功结果。

    异常:
        - 非瞬时错误原样向上抛（由外层按 Path 2 / 不可恢复处理）；
        - 重试超过 MAX_RETRIES 抛 RuntimeError。
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = fn()
            state.consecutive_overload = 0   # 调用成功，重置连续繁忙计数
            return result
        except Exception as e:
            category = classify_transient(e)
            if category is None:
                raise
            time.sleep(_handle_transient(state, category, attempt))
    raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded")


def stream_with_recovery(llm, state: RecoveryState, *,
                         messages: list, model: str = None,
                         thinking: bool = True, tools: list = None,
                         max_tokens: int = DEFAULT_MAX_TOKENS):
    """流式 LLM 调用（DeepSeekLLM.stream）的退避重试封装 —— agent_loop 用它。

    参考脚本把重试写在 while 里包 messages.create；本项目主循环走 llm.stream()（生成器），
    因此这里用生成器包装：把 llm.stream(...) 逐段 yield ('reasoning'|'content', delta)。

    参数:
        llm: DeepSeekLLM 实例。
        state: 恢复状态，被内部更新（连续繁忙 / 备用模型）。
        messages: 本轮消息列表（重试时原样重发同一请求）。
        model: 模型名；缺省取主模型 DEEPSEEK_MODEL（agent_loop 会显式传 state.current_model）。
        thinking: 是否开启思考（reasoning_content），默认 True，与 agent_loop 原调用一致。
        tools: 工具 schema 列表，透传给 llm.stream。
        max_tokens: 本次请求的输出上限；首次截断后由 agent_loop 升档再传入。

    yield:
        (kind, delta)：与 llm.stream 一致，kind ∈ {"reasoning", "content"}。

    说明:
        - 瞬时错误（429 限流 / 5xx 繁忙 / 连接类）自动退避后从头重新请求。流式无断点
          续传，重试意味着重新生成；好在这些错误大多在输出第一段前就被服务端拒绝，
          不会出现重复正文。
        - 非瞬时错误向上抛，由 agent_loop 的 except 分支处理（Path 2 prompt 超长 ->
          reactive_compact；其余判为不可恢复）。
        - 重试超过 MAX_RETRIES 抛 RuntimeError。
    """
    if model is None:
        model = PRIMARY_MODEL
    for attempt in range(MAX_RETRIES):
        try:
            for kind, delta in llm.stream(
                messages=messages,
                model=model,
                thinking=thinking,
                tools=tools,
                max_tokens=max_tokens,
            ):
                yield kind, delta
            return
        except Exception as e:
            category = classify_transient(e)
            if category is None:
                raise
            time.sleep(_handle_transient(state, category, attempt))
    raise RuntimeError(f"Max retries ({MAX_RETRIES}) exceeded")


def is_prompt_too_long_error(e: Exception) -> bool:
    """判断异常是否属于 prompt / 上下文超长类错误（Path 2 的触发条件）。

    DeepSeek 没有独立的上下文超长错误码，这类错误以 HTTP 400 返回，message 形如
    "This model's maximum context length is ... tokens. However, your messages
    resulted in ... tokens. Please reduce the length of the messages."。
    这里按关键字命中判断（兼容 OpenAI 风格的 context_length_exceeded 等代理文案）。

    参数:
        e: 调用抛出的异常。

    返回:
        bool: True 表示该错误是上下文超长，应走 reactive_compact 应急压缩。
    """
    text = _response_text(e).lower()
    keywords = (
        "maximum context length",
        "context_length_exceeded",
        "context length exceeded",
        "prompt is too long",
        "prompt_too_long",
        "prompt_is_too_long",
        "too many tokens",
        "max context window",
        "reduce the length",      # DeepSeek 超长报错固定文案："Please reduce the length of the messages"
    )
    return any(k in text for k in keywords)


def reactive_compact(messages: list) -> list:
    """prompt / 上下文超长时的应急压缩（Path 2，同一轮只调用一次）。

    作用: 把过长历史压缩成可继续工作的形态，然后让 agent_loop 用压缩结果重发请求。

    本项目 ch09 的 context.py 已实现真正的应急压缩：先把消息落盘留痕，再交给 LLM 生成
    摘要，并保留最近几条（含配对的工具调用/结果）供接着干活。参考 s11 版是简化为"保留末
    N 条"（因为它所在章节还没覆盖 LLM 压缩）；本项目直接复用现有实现，不重复造轮子。

    参数:
        messages: 待压缩的历史消息（调用方应传入去掉 system 后的部分，由调用方保留 system）。

    返回:
        list: 压缩后的消息列表（[Compacted 摘要, ...保留尾部]）。
    """
    return _context_reactive_compact(messages)
