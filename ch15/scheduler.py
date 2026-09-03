"""
ch15/scheduler.py —— Cron 定时任务调度器。

为什么需要：agent 有些事不要求立刻做，而是"到点做"（每天早上 9 点巡检、
每 5 分钟同步一次、某个时间点后自动清理等）。scheduler 让模型能预约一条
cron 表达式，后台 daemon 线程每秒轮询，到点把这条 prompt 塞进 cron_queue，
agent_loop 在每轮循环顶部消费并注入对话，模型下一轮就会"想起"该做的事。

与 background_task 的区别：后台任务是"现在跑、跑完通知"（立即执行，见
ch14/background_task.py）；scheduler 是"定时触发、到点才投递"（预约执行）。
二者都用 daemon 线程 + 锁保护共享状态，但职责互补。
"""
import json
import random
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

WORKDIR = Path.cwd()
DURABLE_PATH = WORKDIR / ".scheduled_tasks.json"  # durable job 落盘文件


@dataclass
class CronJob:
    """一条定时任务。cron 为 5 段表达式：分 时 日 月 周。"""
    id: str
    cron: str        # "0 9 * * *"：分 时 日 月 周
    prompt: str      # 触发时要注入给模型的消息
    recurring: bool  # True = 周期重复；False = 一次性（触发后自动删除）
    durable: bool    # True = 落盘持久化（重启后恢复）


# 全局状态 + 锁
scheduled_jobs: dict[str, CronJob] = {}   # job_id → job
cron_queue: list[CronJob] = []            # 已到点、待 agent_loop 消费的 job
cron_lock = threading.Lock()              # 保护 scheduled_jobs / cron_queue 的并发读写
agent_lock = threading.Lock()             # 预留：保护 agent 侧读队列（当前未用）
_last_fired: dict[str, str] = {}          # job_id → "YYYY-MM-DD HH:MM"，防同分钟重复触发

LOG_PATH = WORKDIR / "scheduler.log"   # cron 调度日志


def cron_log(msg: str):
    """cron 日志统一写入当前目录 scheduler.log（时间戳、无 ANSI 色）。

    为什么落盘而非 print：register/fire/cancel 等诊断对终端实时价值低，反而会
    骑在悬空的 "[USER] > " input 提示符上污染界面；写文件可事后 tail 排查，
    且日志写失败不阻塞调用方线程。
    """
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError as e:
        # 日志写失败属极少数异常：退化为一次终端告警，不吞掉
        print(f"  \033[31m[cron log write failed: {e}]\033[0m")


def _cron_field_matches(field: str, value: int) -> bool:
    """匹配单个 cron 字段：支持 * 、*/n 步进、, 列表、- 区间与整数值。"""
    if field == "*":
        return True
    if field.startswith("*/"):
        step = int(field[2:])
        return step > 0 and value % step == 0
    if "," in field:
        return any(_cron_field_matches(f.strip(), value)
                   for f in field.split(","))
    if "-" in field:
        lo, hi = field.split("-", 1)
        return int(lo) <= value <= int(hi)
    return value == int(field)


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """5 段 cron 表达式是否命中给定时间。

    标准 cron 语义：分/时/月必须全中；日与周两者都受限时取 OR（满足其一
    即可命中）——这是 cron 的经典规则，避免出现"每月 1 号且周一的第 1 号"
    这类永远不触发、或意外双重限定的情况。
    """
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return False
    minute, hour, dom, month, dow = fields
    dow_val = (dt.weekday() + 1) % 7  # Python 周一=0 → cron 周日=0
    m = _cron_field_matches(minute, dt.minute)
    h = _cron_field_matches(hour, dt.hour)
    dom_ok = _cron_field_matches(dom, dt.day)
    month_ok = _cron_field_matches(month, dt.month)
    dow_ok = _cron_field_matches(dow, dow_val)
    # 分/时/月必须全中
    if not (m and h and month_ok):
        return False
    # 日与周：两者都受限时，满足其一即可（OR）
    dom_unconstrained = dom == "*"
    dow_unconstrained = dow == "*"
    if dom_unconstrained and dow_unconstrained:
        return True
    if dom_unconstrained:
        return dow_ok
    if dow_unconstrained:
        return dom_ok
    return dom_ok or dow_ok


def _validate_cron_field(field: str, lo: int, hi: int) -> str | None:
    """校验单个字段取值在 [lo, hi] 内；返回错误信息或 None。"""
    if field == "*":
        return None
    if field.startswith("*/"):
        step_str = field[2:]
        if not step_str.isdigit():
            return f"Invalid step: {field}"
        step = int(step_str)
        if step <= 0:
            return f"Step must be > 0: {field}"
        return None
    if "," in field:
        for part in field.split(","):
            err = _validate_cron_field(part.strip(), lo, hi)
            if err:
                return err
        return None
    if "-" in field:
        parts = field.split("-", 1)
        if not parts[0].isdigit() or not parts[1].isdigit():
            return f"Invalid range: {field}"
        a, b = int(parts[0]), int(parts[1])
        if a < lo or a > hi or b < lo or b > hi:
            return f"Range {field} out of bounds [{lo}-{hi}]"
        if a > b:
            return f"Range start > end: {field}"
        return None
    if not field.isdigit():
        return f"Invalid field: {field}"
    val = int(field)
    if val < lo or val > hi:
        return f"Value {val} out of bounds [{lo}-{hi}]"
    return None


def validate_cron(cron_expr: str) -> str | None:
    """校验整个 cron 表达式（5 段、各段取值范围）；返回错误信息或 None。"""
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        return f"Expected 5 fields, got {len(fields)}"
    bounds = [(0, 59), (0, 23), (1, 31), (1, 12), (0, 6)]
    names = ["minute", "hour", "day-of-month", "month", "day-of-week"]
    for i, (field, (lo, hi), name) in enumerate(zip(fields, bounds, names)):
        err = _validate_cron_field(field, lo, hi)
        if err:
            return f"{name}: {err}"
    return None


def save_durable_jobs():
    """把 durable job 落盘到 .scheduled_tasks.json。

    为什么只存 durable：一次性（session-only）任务重启后本来就该消失，
    持久化只对"想跨重启保留"的任务有意义。
    """
    durable = [asdict(j) for j in scheduled_jobs.values() if j.durable]
    DURABLE_PATH.write_text(json.dumps(durable, indent=2))


def load_durable_jobs():
    """启动时从磁盘恢复 durable job。

    为什么跳过无效 job：cron 语法可能被手改坏或跨版本变更，坏 job 直接
    打印告警跳过，不阻止其余合法 job 恢复。
    """
    if not DURABLE_PATH.exists():
        return
    try:
        jobs = json.loads(DURABLE_PATH.read_text())
        for j in jobs:
            job = CronJob(**j)
            err = validate_cron(job.cron)
            if err:
                cron_log(f"[cron] skipping invalid job {job.id}: {err}")
                continue
            scheduled_jobs[job.id] = job
        valid = [j for j in jobs if j["id"] in scheduled_jobs]
        if valid:
            cron_log(f"[cron] loaded {len(valid)} durable job(s)")
    except Exception:
        pass  # 文件损坏等：静默忽略，空调度器也能正常跑


def schedule_job(cron: str, prompt: str, recurring: bool = True,
                 durable: bool = True) -> CronJob | str:
    """注册一条 cron 任务，返回 CronJob；校验失败返回错误字符串。

    为什么先校验再登记：非法 cron 放进调度器会在每次轮询都报错，
    不如在入口就拦下。id 用随机数，避免跨会话/跨进程冲突。
    """
    err = validate_cron(cron)
    if err:
        return err
    job = CronJob(
        id=f"cron_{random.randint(0, 999999):06d}",
        cron=cron, prompt=prompt,
        recurring=recurring, durable=durable,
    )
    with cron_lock:
        scheduled_jobs[job.id] = job
    if durable:
        save_durable_jobs()
    cron_log(f"[cron register] {job.id} '{cron}' → {prompt[:40]}")
    return job


def cancel_job(job_id: str) -> str:
    """取消一条 cron 任务；不存在返回错误信息。"""
    with cron_lock:
        job = scheduled_jobs.pop(job_id, None)
    if not job:
        return f"Job {job_id} not found"
    if job.durable:
        save_durable_jobs()  # durable job 被取消也要同步落盘
    cron_log(f"[cron cancel] {job_id}")
    return f"Cancelled {job_id}"


def cron_scheduler_loop():
    """daemon 线程主循环：每秒轮询，命中且本分钟未触发过的 job 入队。

    为什么独立线程：轮询不能阻塞 agent 主循环；daemon=True 使主进程退出时
    线程自动中止。为什么按分钟去重：_last_fired 记录 "YYYY-MM-DD HH:MM"，
    避免同一分钟内多次轮询把同一 job 重复入队。为什么逐个 try/except：
    单个 job 抛错不能拖垮整个调度线程。
    """
    while True:
        time.sleep(1)
        now = datetime.now()
        minute_marker = now.strftime("%Y-%m-%d %H:%M")  # 带日期，跨天不误判
        with cron_lock:
            for job in list(scheduled_jobs.values()):
                try:
                    if cron_matches(job.cron, now):
                        if _last_fired.get(job.id) != minute_marker:
                            cron_queue.append(job)
                            _last_fired[job.id] = minute_marker
                            cron_log(f"[cron fire] {job.id} → {job.prompt[:40]}")
                        if not job.recurring:
                            scheduled_jobs.pop(job.id, None)
                            if job.durable:
                                save_durable_jobs()
                except Exception as e:
                    cron_log(f"[cron error] {job.id}: {e}")


def consume_cron_queue() -> list[CronJob]:
    """取走 cron_queue 里已触发的 job（agent_loop 在每轮循环顶部调用）。

    为什么取走即清空：同一批触发通知只投递一次，模型消费后不重复打扰。
    """
    with cron_lock:
        fired = list(cron_queue)
        cron_queue.clear()
    return fired


def has_cron_queue() -> bool:
    """是否有待投递的 cron 任务（供 agent_loop 判断本轮要不要注入）。"""
    with cron_lock:
        return bool(cron_queue)


# 模块级启动：先恢复 durable job，再起 daemon 线程轮询。
# 为什么放模块级而非 main 里：与 background_task 一致，import 即启动，
# 任何进程入口（main / 测试脚本）都无需手动拉线程。
load_durable_jobs()
threading.Thread(target=cron_scheduler_loop, daemon=True).start()
cron_log("[cron] scheduler thread started")


# ============================================================
# Cron Tools —— 给 LLM 通过 tool_use 调用的入口
# 约定：与 task.py 一致，每个 run_* 接收工具参数、返回「给模型看的字符串」
# ============================================================
def run_schedule_cron(cron: str, prompt: str,
                      recurring: bool = True, durable: bool = True) -> str:
    """工具入口：注册 cron 任务，返回带 id 的结果让模型知道引用名。"""
    result = schedule_job(cron, prompt, recurring, durable)
    if isinstance(result, str):
        return f"Error: {result}"
    return f"Scheduled {result.id}: '{cron}' → {prompt}"


def run_list_crons() -> str:
    """工具入口：列出全部 cron 任务（含 recurring / durable 标记）。"""
    with cron_lock:
        jobs = list(scheduled_jobs.values())
    if not jobs:
        return "No cron jobs. Use schedule_cron to add one."
    lines = []
    for j in jobs:
        tag = "recurring" if j.recurring else "one-shot"
        dur = "durable" if j.durable else "session"
        lines.append(f"  {j.id}: '{j.cron}' → {j.prompt[:40]} "
                     f"[{tag}, {dur}]")
    return "\n".join(lines)


def run_cancel_cron(job_id: str) -> str:
    """工具入口：取消一条 cron 任务。"""
    return cancel_job(job_id)