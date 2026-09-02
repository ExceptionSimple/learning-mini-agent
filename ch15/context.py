import os

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
import json
import time

from core.llm import DeepSeekLLM

CONTEXT_LIMIT = 50000
KEEP_RECENT = 3
PERSIST_THRESHOLD = 30000
TOOL_RESULT_DIR = Path("tool_results")
TRANSCRIPT_DIR = Path("transcript")

def estimate_size(msgs):
    return len(str(msgs))

def _message_has_tool_use(msg):
    """
    判断一条消息是不是：工具调用
    OpenAI/DeepSeek 格式：role=assistant 且带 tool_calls 字段
    """
    return msg.get("role") == "assistant" and bool(msg.get("tool_calls"))

def _is_tool_result_message(msg):
    """
    判断一条消息是不是：工具调用结果
    OpenAI/DeepSeek 格式：role=tool
    """
    return msg.get("role") == "tool"

# ================================================
# L1: 保留头尾、裁剪中间结果
# ================================================
def snip_compact(messages, max_messages=50):
    size = len(messages)
    if size <= max_messages: return messages
    # 保留 3 条开头，保留 keep_tail 条结尾
    keep_head, keep_tail = 3, max_messages - 3
    # 重新调整位置
    head_end, tail_start = keep_head, size - keep_tail

    # 如果 head_end - 1 既要开头保留的最后一条是一个 工具调用
    # 则需要收集后面连着的所有 result，以确保消息的完整性
    if head_end > 0 and _message_has_tool_use(messages[head_end - 1]):
        while head_end < size and _is_tool_result_message(messages[head_end]):
            head_end += 1

    # 如果要保留的尾部消息的最后一条是：工具返回结果
    # 则前面必然是 工具调用，因此需要 tail_start - 1，一起收集进来，以确保消息的完整性。
    if tail_start > 0 and tail_start < size \
        and _is_tool_result_message(messages[tail_start]) \
        and _message_has_tool_use(messages[tail_start - 1]):
        tail_start -= 1

    # 两个边界相交，则表示：中间没有压缩空间
    # 因此，直接返回整个 messages
    if head_end >= tail_start:
        return messages

    snipped = tail_start - head_end

    return messages[:head_end] + [{"role": "user", "content": f"[snipped {snipped} messages]"}] + messages[tail_start:]

# ================================================
# L2: 旧的工具调用结果 占位符
# ================================================
def collect_tool_results(messages):
    """
    收集消息中的所有工具
    :param messages:
    :return:
    """
    blocks = []

    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        blocks.append(msg)

    return blocks

def micro_compact(messages):
    """
    只保留最近的 KEEP_RECENT 条工具调用结果
    其余的全部压缩
    """
    tool_results = collect_tool_results(messages)

    if len(tool_results) <= KEEP_RECENT:
        return messages

    for block in tool_results[:-KEEP_RECENT]:
        if len(block.get("content")) > 120:
            block["content"] = "[Earlier tool result compacted. Re-run if needed.]"

    return messages

# ================================================
# L3: 当工具返回结果非常之大时，提供预览，并且将完整结果落盘
# ================================================
def persist_large_output(tool_call_id, output):
    """
    将工具的结果保存到本地文件中，只预留前 2000 字符作为预览
    :param tool_call_id:
    :param output:
    :return:
    """
    if len(output) <= PERSIST_THRESHOLD:
        return output
    path = TOOL_RESULT_DIR / f"{tool_call_id}.txt"
    TOOL_RESULT_DIR.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(output)
    return f"<persisted-output>\nFull output: {path}\nPreview:\n{output[:2000]}\n</persisted-output>"

def tool_result_budget(messages: list, max_bytes=200_000):
    last = messages[-1] if messages else None
    if not last or last.get("role") != "tool":
        return messages
    # 只统计工具返回结果（assistant 工具调用消息 content 为 None，且不属于结果）
    results = [b for b in messages if b.get("role") == "tool"]
    total = sum(len(str(b.get("content", ""))) for b in results)
    if total <= max_bytes:
        return messages
    # 按照返回结果的大小进行排序，从大到小
    ranked = sorted(results, key=lambda p: len(p.get("content") or ""), reverse=True)
    for b in ranked:
        if total <= max_bytes: # 直到结果的长度总和不超过 max_bytes 阈值时，就退出
            break
        content = str(b.get("content", ""))
        # 只有当 content（既工具 result）超过阈值时，才进行裁剪
        if len(content) <= PERSIST_THRESHOLD: continue
        tool_call_id = b.get("tool_call_id", "Unknown")
        b["content"] = persist_large_output(tool_call_id, content)
        # 裁剪完后，重新计算 total
        total = sum(len(str(b.get("content", ""))) for b in results)
    return messages

# ================================================
# L4: 自动压缩 —— LLM 全量压缩总结 —— 摘要
# ================================================
def write_transcript(messages):
    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPT_DIR / f"transcript_{int(time.time())}.jsonl"
    with path.open("w") as f:
        for msg in messages:
            f.write(json.dumps(msg, default=str) + "\n")
    return path

def summarize_history(messages: list):
    conversation = json.dumps(messages, default=str)
    MAX = 80000
    if len(conversation) > MAX:
        # 保头尾、弃中间：开头保留原始目标/约束，结尾保留当前进度
        head, tail = MAX // 2, MAX // 2
        conversation = conversation[:head] + "\n...[middle truncated]...\n" + conversation[-tail:]
    prompt = ("Summarize this coding-agent conversation so work can continue.\n"
              "Preserve: 1. current goal, 2. key findings/decisions, 3. files read/changed, "
              "4. remaining work, 5. user constraints.\nBe compact but concrete.\n\n" + conversation)

    llm = DeepSeekLLM(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL")
    )
    response = llm.invoke(messages = [{"role": "system", "content": prompt}], model=os.environ.get("DEEPSEEK_MODEL"))
    return response['content']

def compact_history(messages):
    transcript_path = write_transcript(messages)
    print(f"[transcript saved: {transcript_path}]")
    summary = summarize_history(messages)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}]

# ================================================
# 兜底-应急方案: 如果 API 返回提示词太长的报错时
# ================================================
def reactive_compact(messages):
    transcript = write_transcript(messages)
    summary = summarize_history(messages)
    tail_start = max(0, len(messages) - 5)
    if (tail_start > 0 and tail_start < len(messages)
            and _is_tool_result_message(messages[tail_start])
            and _message_has_tool_use(messages[tail_start - 1])):
        tail_start -= 1
    return [{"role": "user", "content": f"[Reactive compact]\n\n{summary}"}, *messages[tail_start:]]
