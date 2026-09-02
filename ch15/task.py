"""
ch13/task.py —— 落盘的任务系统（Task + 依赖 + 认领/完成）。

为什么从 todo_write 升级：ch06 的 todo_write 只存在内存的 CURRENT_TODOS，
agent 一退出就丢，也无法表达"任务 A 依赖任务 B"的前置关系。task.py 把每个
任务存成 .tasks/ 下的一个 JSON 文件，支持 blockedBy 依赖门禁，供 agent
（尤其是多 agent / subagent 场景）规划、认领、完成任务。
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import random
import time

from utils.color_print import cprint

WORKDIR = Path.cwd()

# 任务仓库：每任务一个 JSON 文件；启动即建目录，避免首次写盘报错
TASKS_DIR = WORKDIR / ".tasks"
TASKS_DIR.mkdir(exist_ok=True)

@dataclass
class Task:
    """一条任务。status 流转：pending -> in_progress -> completed。"""
    id: str
    subject: str
    description: str
    status: str          # pending | in_progress | completed
    owner: str | None    # Agent name (multi-agent scenarios)
    blockedBy: list[str] # Dependency task IDs（都 completed 才允许 claim）

# 任务文件路径约定：.tasks/{task_id}.json
def _task_path(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"

def create_task(subject: str, description: str = "",
                blockedBy: list[str] | None = None) -> Task:
    """新建任务：id 用时间戳+随机数保证唯一，默认 pending、无 owner，立即落盘。"""
    task = Task(
        id=f"task_{int(time.time())}_{random.randint(0, 9999):04d}",
        subject=subject,
        description=description,
        status="pending",
        owner=None,
        blockedBy=blockedBy or [],
    )
    save_task(task)
    return task

def save_task(task: Task):
    """asdict 把 dataclass 转 dict 后写 JSON；indent=2 便于人读和 diff。"""
    _task_path(task.id).write_text(json.dumps(asdict(task), indent=2))

def load_task(task_id: str) -> Task:
    """从 JSON 还原 Task（字段与 dataclass 一一对应）。"""
    return Task(**json.loads(_task_path(task_id).read_text()))

def list_tasks() -> list[Task]:
    """按文件名排序列出全部任务（glob 限定 task_*.json，排除杂散文件）。"""
    return [Task(**json.loads(p.read_text()))
            for p in sorted(TASKS_DIR.glob("task_*.json"))]

def get_task(task_id: str) -> str:
    """Return full task details as JSON."""
    task = load_task(task_id)
    return json.dumps(asdict(task), indent=2)

def can_start(task_id: str) -> bool:
    """Check if all blockedBy dependencies are completed.
    Missing dependencies are treated as blocked.

    为什么"缺失也算阻塞"：依赖任务不存在说明规划有问题，宁可让任务停在
    pending 等人修，也不允许跳过依赖提前开跑。
    """
    task = load_task(task_id)
    for dep_id in task.blockedBy:
        if not _task_path(dep_id).exists():
            return False
        if load_task(dep_id).status != "completed":
            return False
    return True

def claim_task(task_id: str, owner: str = "agent") -> str:
    """认领任务（多 agent 场景表示"我要干这个"）。
    门禁：仅 pending 且所有依赖 completed 才能认领；成功则置 owner + in_progress。
    """
    task = load_task(task_id)
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"
    if not can_start(task_id):
        deps = [d for d in task.blockedBy
                if not _task_path(d).exists() or load_task(d).status != "completed"]
        return f"Blocked by: {deps}"
    task.owner = owner
    task.status = "in_progress"
    save_task(task)
    cprint(f"[claim] {task.subject} → in_progress (owner: {owner})", color="cyan")
    return f"Claimed {task.id} ({task.subject})"

def complete_task(task_id: str) -> str:
    """完成任务，并报告因此解锁的下游任务。
    为什么返回 unblocked：依赖解锁是"完成"的连带价值，要喂回给 LLM，
    让它知道接下来可以认领哪些新任务。
    """
    task = load_task(task_id)
    if task.status != "in_progress":
        return f"Task {task_id} is {task.status}, cannot complete"
    task.status = "completed"
    save_task(task)
    unblocked = [t.subject for t in list_tasks()
                 if t.status == "pending" and t.blockedBy and can_start(t.id)]
    cprint(f"[complete] {task.subject} ✓", color="green")
    msg = f"Completed {task.id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked: {', '.join(unblocked)}"
        cprint(f"[unblocked] {', '.join(unblocked)}", color="yellow")
    return msg

# ============================================================
# Task Tools —— 给 LLM 通过 tool_use 调用的入口
# 约定：每个 run_* 接收工具参数、返回「给模型看的字符串」；
#      彩色日志打印给人看，返回值才是模型拿到的结果。
# ============================================================
def run_create_task(subject: str, description: str = "",
                    blockedBy: list[str] | None = None) -> str:
    """工具入口：建任务，返回带 id 的摘要，让模型知道新任务的引用名。"""
    task = create_task(subject, description, blockedBy)
    deps = f" (blockedBy: {', '.join(blockedBy)})" if blockedBy else ""
    cprint(f"[create] {task.subject}{deps}", color="blue")
    return f"Created {task.id}: {task.subject}{deps}"

def run_list_tasks() -> str:
    """工具入口：全部任务概览。○=pending ●=in_progress ✓=completed。"""
    tasks = list_tasks()
    if not tasks:
        return "No tasks. Use create_task to add some."
    lines = []
    for t in tasks:
        icon = {"pending": "○", "in_progress": "●",
                "completed": "✓"}.get(t.status, "?")
        deps = f" (blockedBy: {', '.join(t.blockedBy)})" if t.blockedBy else ""
        owner = f" [{t.owner}]" if t.owner else ""
        lines.append(f"  {icon} {t.id}: {t.subject} "
                     f"[{t.status}]{owner}{deps}")
    return "\n".join(lines)

def run_get_task(task_id: str) -> str:
    """工具入口：单个任务的完整 JSON 详情；找不到时给模型友好错误而不是抛异常。"""
    try:
        return get_task(task_id)
    except FileNotFoundError:
        return f"Error: Task {task_id} not found"

def run_claim_task(task_id: str) -> str:
    """工具入口：单 agent 场景固定 owner="agent"（多 agent 时由系统分配身份）。
    为什么兜底：模型可能传残缺 task_id（如把 task_xxx 提取成 xxx），找不到时
    返回友好错误而不是让整个 agent 崩溃（对齐 run_get_task）。
    """
    try:
        return claim_task(task_id, owner="agent")
    except FileNotFoundError:
        return f"Error: Task {task_id} not found. Use list_tasks to see valid IDs."

def run_complete_task(task_id: str) -> str:
    """工具入口：透传给 complete_task，返回值已含解锁的下游任务。
    为什么兜底：同 run_claim_task，错误 id 友好返回而非崩溃。
    """
    try:
        return complete_task(task_id)
    except FileNotFoundError:
        return f"Error: Task {task_id} not found. Use list_tasks to see valid IDs."
