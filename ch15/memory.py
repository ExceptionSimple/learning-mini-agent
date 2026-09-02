# ========================================================
# ch10/memory.py —— 记忆系统（持久化 / 检索 / 抽取 / 合并）
#
# 适配说明（相对参考实现）：
#  1. LLM 层改用本项目 core/llm.DeepSeekLLM（DeepSeek API）：参考实现导入的
#     llm.chat_model_v2.ChatModel 在本项目不存在，且其 invoke 参数为消息列表而非裸字符串。
#  2. 补齐缺失导入：os / json / re / time / dotenv；颜色输出统一走 utils.color_print。
#  3. frontmatter 改为本项目约定格式（metadata: 嵌套 type），解析器同步支持。
#  4. consolidate_memories 增加空结果保护，避免 LLM 返回 [] 时清空全部记忆。
#  5. 接线方式：main.py 将 build_memory_system_prompt() 拼入 SYSTEM；
#     load_memories / extract_memories 由 main.py 注册的 before_turn_call /
#     after_turn_call 钩子回调调用（见本文件末尾 build_memory_system_prompt）。
# ========================================================

import os
import json
import re
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from core.llm import DeepSeekLLM
from utils.color_print import cprint

WORKDIR = Path.cwd()
MEMORY_DIR = WORKDIR / ".memory"
MEMORY_DIR.mkdir(exist_ok=True)          # 确保 .memory 目录存在
MEMORY_INDEX = MEMORY_DIR / "MEMORY.md"
MEMORY_TYPES = ["user", "feedback", "project", "reference"]


def extract_text(content) -> str:
    """从模型回复中提取纯文本。

    参数:
        content: 模型回复的 content 字段。本项目为普通字符串；
                 兼容 list（Anthropic block）结构，逐块拼接其中的 text。

    返回:
        str: 纯文本回复；content 为空或 None 时返回 ""。
    """
    if isinstance(content, list):
        return "\n".join(
            str(getattr(b, "text", "")) for b in content
            if getattr(b, "type", None) == "text"
        )
    return str(content or "")


def _llm_text(prompt: str) -> str:
    """调用本项目自己的 DeepSeekLLM，返回纯文本回复。

    参数:
        prompt: 发送给 LLM 的完整提示词文本（将作为单条 user 消息）。

    返回:
        str: 模型的文本回复；调用失败或 content 为空时返回 ""。

    说明: 参考实现为 client.messages.create(...)，此处适配为本项目 core/llm.py 的
          DeepSeekLLM.invoke()——消息为列表格式、模型名读环境变量 DEEPSEEK_MODEL。
    """
    llm = DeepSeekLLM(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url=os.environ.get("DEEPSEEK_BASE_URL"),
    )
    response = llm.invoke(
        messages=[{"role": "user", "content": prompt}],
        model=os.environ.get("DEEPSEEK_MODEL"),
    )
    return extract_text(response.get("content"))


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    """解析记忆文件的 frontmatter。

    参数:
        text: 记忆文件全文（以 "---" 开头的 markdown 文本）。

    返回:
        tuple[dict, str]: (meta, body)
            - meta: 解析出的元数据字典，形如
              {"name":..., "description":..., "metadata": {"type":...}}
            - body: 去掉 frontmatter 后的正文；无 frontmatter 时返回 (空字典, 原文)。

    支持本项目格式（metadata 为嵌套子块）：
        ---
        name: xxx
        description: xxx
        metadata:
          type: user
        ---
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    meta, current = {}, None
    for line in parts[1].strip().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if val == "":
            current = key                      # 进入子块（如 metadata:）
            meta.setdefault(key, {})
        elif current and line.startswith(" "):
            meta[current][key] = val           # 子块字段（如 metadata 下的 type:）
        else:
            meta[key] = val
            current = None
    return meta, parts[2].strip()


def write_memory_file(name: str, mem_type: str, description: str, body: str):
    """写入单个记忆文件并重建索引。frontmatter 采用本项目 metadata 嵌套格式。

    参数:
        name: 记忆名称，用作文件名 slug（如 "user-pref-tabs"）。
        mem_type: 记忆类型，须为 MEMORY_TYPES 之一（user/feedback/project/reference）。
        description: 一行摘要，用于索引目录检索。
        body: 记忆正文（markdown）。

    返回:
        Path: 写入的文件路径（如 .memory/user-pref-tabs.md）。
    """
    slug = name.lower().replace(" ", "-").replace("/", "-")
    filepath = MEMORY_DIR / f"{slug}.md"
    filepath.write_text(
        f"---\nname: {name}\ndescription: {description}\n"
        f"metadata:\n  type: {mem_type}\n---\n\n{body}\n"
    )
    _rebuild_index()
    return filepath


def _rebuild_index():
    """重建 MEMORY.md 索引（列出所有记忆文件的 name — description）。

    参数: 无。

    返回: None。副作用：重写 .memory/MEMORY.md。
    """
    lines = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter(raw)
        name = meta.get("name", f.stem)
        desc = meta.get("description", body.split("\n")[0][:80])
        lines.append(f"- [{name}]({f.name}) — {desc}")
    MEMORY_INDEX.write_text("\n".join(lines) + "\n" if lines else "")


def read_memory_index() -> str:
    """读取 MEMORY.md 索引（每轮注入 SYSTEM）。

    参数: 无。

    返回:
        str: 索引内容；索引文件不存在时返回 ""。
    """
    if not MEMORY_INDEX.exists():
        return ""
    return MEMORY_INDEX.read_text().strip()


def read_memory_file(filename: str) -> str | None:
    """读取单个记忆文件全文。

    参数:
        filename: 记忆文件名（如 "user-pref-tabs.md"）。

    返回:
        str | None: 文件全文；文件不存在时返回 None。
    """
    path = MEMORY_DIR / filename
    if not path.exists():
        return None
    return path.read_text()


def list_memory_files() -> list[dict]:
    """列出所有记忆文件及元信息（排除 MEMORY.md）。

    参数: 无。

    返回:
        list[dict]: 按文件名排序，每条含
            {"filename", "name", "description", "type", "body"}。
            type 优先取 metadata.type，兼容旧扁平格式。
    """
    result = []
    for f in sorted(MEMORY_DIR.glob("*.md")):
        if f.name == "MEMORY.md":
            continue
        raw = f.read_text()
        meta, body = _parse_frontmatter(raw)
        result.append({
            "filename": f.name,
            "name": meta.get("name", f.stem),
            "description": meta.get("description", ""),
            "type": meta.get("metadata", {}).get("type", meta.get("type", "user")),
            "body": body,
        })
    return result


def select_relevant_memories(messages: list, max_items: int = 5) -> list[str]:
    """按最近对话挑选相关记忆文件名。

    参数:
        messages: 会话消息列表（每项为含 role/content 的 dict）。
        max_items: 最多返回的记忆文件数，默认 5。

    返回:
        list[str]: 相关记忆文件名列表；无记忆文件或无可选时返回 []。

    说明: 优先 LLM 选择，失败则退化为关键词匹配。
    """
    files = list_memory_files()
    if not files:
        return []
    # 收集最近 3 条 user 文本作上下文
    recent_texts = []
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if isinstance(content, list):   # 兼容 block 结构，本项目为 str
                content = " ".join(
                    str(getattr(b, "text", "")) for b in content
                    if getattr(b, "type", None) == "text"
                )
            if isinstance(content, str):
                recent_texts.append(content)
            if len(recent_texts) >= 3:
                break
    recent = " ".join(reversed(recent_texts))[:2000]
    if not recent.strip():
        return []
    # 候选目录：name — description
    catalog = "\n".join(
        f"{i}: {f['name']} — {f['description']}" for i, f in enumerate(files)
    )
    prompt = (
        "Given the recent conversation and the memory catalog below, "
        "select the indices of memories that are clearly relevant. "
        "Return ONLY a JSON array of integers, e.g. [0, 3]. "
        "If none are relevant, return [].\n\n"
        f"Recent conversation:\n{recent}\n\n"
        f"Memory catalog:\n{catalog}"
    )
    try:
        text = _llm_text(prompt)
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            indices = json.loads(match.group())
            selected = []
            for idx in indices:
                if isinstance(idx, int) and 0 <= idx < len(files):
                    selected.append(files[idx]["filename"])
                    if len(selected) >= max_items:
                        break
            return selected
    except Exception:
        pass
    # 兜底：关键词匹配 name + description
    keywords = [w.lower() for w in recent.split() if len(w) > 3]
    selected = []
    for f in files:
        text = (f["name"] + " " + f["description"]).lower()
        if any(kw in text for kw in keywords):
            selected.append(f["filename"])
            if len(selected) >= max_items:
                break
    return selected


def load_memories(messages: list) -> str:
    """加载相关记忆正文，供注入用户消息。

    参数:
        messages: 会话消息列表（从中挑选相关记忆）。

    返回:
        str: "<relevant_memories>...</relevant_memories>" 包裹的记忆正文；
             没有选中记忆时返回 ""。
    """
    selected_files = select_relevant_memories(messages)
    if not selected_files:
        return ""
    parts = ["<relevant_memories>"]
    for filename in selected_files:
        content = read_memory_file(filename)
        if content:
            parts.append(content)
    parts.append("</relevant_memories>")
    return "\n\n".join(parts)


def extract_memories(messages: list):
    """从最近对话抽取新记忆并落盘。每轮对话结束后调用。

    参数:
        messages: 会话消息列表（取最近 10 条作为抽取素材）。

    返回:
        None。副作用：调用 LLM 抽取新记忆，写入 .memory/ 并重建索引；
        LLM 失败或返回空列表时静默返回，不影响主流程。
    """
    # 收集最近 10 条消息文本
    dialogue_parts = []
    for msg in messages[-10:]:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                str(getattr(b, "text", "")) for b in content
                if getattr(b, "type", None) == "text"
            )
        if isinstance(content, str) and content.strip():
            dialogue_parts.append(f"{role}: {content}")
    dialogue = "\n".join(dialogue_parts)
    if not dialogue.strip():
        return
    # 已存在记忆清单，供 LLM 去重
    existing = list_memory_files()
    existing_desc = "\n".join(
        f"- {m['name']}: {m['description']}" for m in existing
    ) if existing else "(none)"
    prompt = (
        "Extract user preferences, constraints, or project facts from this dialogue.\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n"
        "- name: short kebab-case identifier (e.g. 'user-preference-tabs')\n"
        "- type: one of 'user' (user preference), 'feedback' (guidance), "
        "'project' (project fact), 'reference' (external pointer)\n"
        "- description: one-line summary for index lookup\n"
        "- body: full detail in markdown\n"
        "If nothing new or already covered by existing memories, return [].\n\n"
        f"Existing memories:\n{existing_desc}\n\n"
        f"Dialogue:\n{dialogue[:4000]}"
    )
    try:
        text = _llm_text(prompt)
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())
        if not items:
            return
        count = 0
        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                write_memory_file(name, mem_type, desc, body)
                count += 1
        if count:
            cprint(f"[Memory: extracted {count} new memories]", color="yellow")
    except Exception:
        pass


CONSOLIDATE_THRESHOLD = 10


def consolidate_memories():
    """合并重复/过期记忆。

    参数: 无。

    返回: None。副作用：当记忆文件数 ≥ CONSOLIDATE_THRESHOLD 时，
          调用 LLM 合并并重写全部记忆文件；LLM 返回空结果时跳过，
          不会删除任何已有记忆。
    """
    files = list_memory_files()
    if len(files) < CONSOLIDATE_THRESHOLD:
        return
    catalog = "\n\n".join(
        f"## {f['filename']}\nname: {f['name']}\ndescription: {f['description']}\n{f['body']}"
        for f in files
    )
    prompt = (
        "Consolidate the following memory files. Rules:\n"
        "1. Merge duplicates into one\n"
        "2. Remove outdated/contradicted memories\n"
        "3. Keep the total under 30 memories\n"
        "4. Preserve important user preferences above all\n"
        "Return a JSON array. Each item: {name, type, description, body}.\n\n"
        f"{catalog[:16000]}"
    )
    try:
        text = _llm_text(prompt)
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if not match:
            return
        items = json.loads(match.group())
        # 空结果保护：LLM 可能返回 []，此时绝不能删除已有记忆（原版会先 unlink 再写空）
        if not items:
            cprint("[Memory: consolidate skipped — empty result]", color="yellow")
            return
        # 删除旧记忆文件（保留 MEMORY.md），再写入合并结果
        for f in MEMORY_DIR.glob("*.md"):
            if f.name != "MEMORY.md":
                f.unlink()
        for mem in items:
            name = mem.get("name", f"memory_{int(time.time())}")
            mem_type = mem.get("type", "user")
            desc = mem.get("description", "")
            body = mem.get("body", "")
            if desc and body:
                write_memory_file(name, mem_type, desc, body)
        cprint(f"[Memory: consolidated {len(files)} → {len(items)} memories]", color="yellow")
    except Exception:
        pass


# main.py 把 build_memory_system_prompt() 拼进 SYSTEM（与 skills 目录并列）；
# load_memories / extract_memories 由 main.py 注册的 before_turn_call / after_turn_call 回调调用。
def build_memory_system_prompt() -> str:
    """构建带记忆索引的 SYSTEM 提示词片段（由 main.py 的 SYSTEM 引入）。

    参数: 无。

    返回:
        str: SYSTEM 提示词片段，含 "Memories available:" 索引目录；
             无记忆文件时索引目录部分为空，但仍返回提示模板。
    """
    index = read_memory_index()
    memories_section = f"\n\nMemories available:\n{index}" if index else ""
    return (
        f"{memories_section}\n"
        "Relevant memories are injected below. Respect user preferences from memory.\n"
        "When the user says 'remember' or expresses a clear preference, extract it as a memory."
    )
