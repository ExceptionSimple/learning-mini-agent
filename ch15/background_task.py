"""
ch14/background_task.py —— 后台任务（Background Tasks）。

为什么需要：agent 的某些工具调用很耗时（npm install、build、deploy 等），
同步执行会阻塞主循环，模型只能干等。这里把这类调用丢进 daemon 线程异步跑，
立即给模型返回"[Background task started with ID bg_xxxx]"占位结果，主循环继续转；
任务完成后，再由主循环用 <task_notification> 把结果通知给模型。

对应 s13：真实 Claude Code 的 Bash 工具 background:true 机制（后台执行 +
后台任务 ID + task_notification 通知）。本实现只模拟 bash 慢命令的启发式后台。

适配说明（上游参考版 → 本仓库）：
  - TOOL_CALL_MAP 在 tool.py（ch13 分层后不再放 tools/__init__.py），改从这里导入；
  - 上游遍历的工具块带 .name/.input/.id 属性，本仓库是 tools/structed_tool.py 的
    ToolCall（.name/.tool_input/.tool_id），execute/start 两处随之改名；
  - 打印统一走 utils.color_print.cprint，与 error_recovery / task 一致。
"""
import threading

from tool import TOOL_CALL_MAP
from utils.color_print import cprint

# 后台任务三件套：计数器 + 状态表 + 结果表，外加一把锁。
# 为什么需要锁：worker 跑在 daemon 线程里，主循环在别处读写这两张表，
# 跨线程共享 dict 不加锁会读到半写状态（甚至抛 RuntimeError）。
_bg_counter = 0
background_tasks: dict[str, dict] = {}   # bg_id → {tool_use_id, command, status}
background_results: dict[str, str] = {}   # bg_id → output
background_lock = threading.Lock()

# -----------------------------------
# 目前仅仅针对 bash 工具模拟
# -----------------------------------
def is_slow_operation(tool_name: str, tool_input: dict) -> bool:
    """启发式：按命令关键字判断是否可能耗时（install/build/test/deploy…）。

    为什么启发式：模型不总是显式声明 run_in_background，慢命令需要被"猜"出来。
    注意：目前仅针对 bash 工具模拟；关键字是包含匹配，误判最坏只是多开一个
    后台线程，不会执行错命令。
    """
    if tool_name != "bash":
        return False
    cmd = tool_input.get("command", "").lower()
    slow_keywords = ["install", "build", "test", "deploy", "compile",
                     "docker build", "pip install", "npm install",
                     "cargo build", "pytest", "make"]
    return any(kw in cmd for kw in slow_keywords)

def should_run_background(tool_name: str, tool_input: dict) -> bool:
    """是否后台执行：模型显式声明（run_in_background）优先，未声明才走启发式。

    为什么显式值优先：显式 true 是想后台、显式 false 是想前台，都是明确意图，
    启发式只该兜底"模型没表态"的情况。若只认 true，显式 false 会被启发式劫持
    （npm install 又拖回后台），违背模型意图。
    """
    if "run_in_background" in tool_input:
        return bool(tool_input["run_in_background"])
    return is_slow_operation(tool_name, tool_input)

def execute_tool(block) -> str:
    """执行一个工具调用块并返回输出。
    block 需带 .name（工具名）与 .tool_input（参数字典，本仓库 ToolCall 形态）。
    复用的是 TOOL_CALL_MAP 里的同步 handler —— 后台线程里跑的就是它，
    主流程与后台执行共用同一套逻辑，不会出现两套行为不一致。
    """
    handler = TOOL_CALL_MAP.get(block.name)
    if handler:
        # run_in_background 仅供主循环做前后台决策，不传给底层 handler
        # （run_bash 等签名不接收该 kwarg，硬传会 TypeError）
        tool_input = {k: v for k, v in block.tool_input.items()
                      if k != "run_in_background"}
        return handler(**tool_input)
    return f"Unknown tool: {block.name}"

def start_background_task(block) -> str:
    """把工具调用丢进 daemon 线程执行，立即返回后台任务 ID（bg_0001…）。

    为什么 daemon：主进程退出时后台线程随之中止，不会残留僵尸线程。
    时序：先在锁内登记状态 running（让主循环立刻可见、可查询），再启动线程；
    worker 完成后在锁内把状态改 completed、结果写进 background_results，
    collect_background_results 才能拾取到。
    为什么 worker 套 try/except：handler 万一抛异常（非 bash 工具误入后台等），
    也要把错误写回并标记 completed，而不是留下永久 running 的僵尸任务。
    """
    global _bg_counter
    _bg_counter += 1
    bg_id = f"bg_{_bg_counter:04d}"
    cmd = block.tool_input.get("command", block.name)
    def worker():
        try:
            result = execute_tool(block)
        except Exception as e:                       # 兜底：错误也记为 completed
            result = f"Error in background task: {type(e).__name__}: {e}"
        with background_lock:
            background_tasks[bg_id]["status"] = "completed"
            background_results[bg_id] = result
    with background_lock:
        background_tasks[bg_id] = {
            "tool_use_id": block.tool_id,
            "command": cmd,
            "status": "running",
        }
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    cprint(f"[background] dispatched {bg_id}: {cmd[:40]}", color="yellow")
    return bg_id

def collect_background_results() -> list[str]:
    """收集所有已完成的后台任务，打包成 <task_notification> 消息返回给主循环。

    为什么 pop：通知过一次即移除，不会反复喂给模型造成重复干扰。
    为什么 summary 截到 200 字：长输出不该全文塞进上下文（完整结果已存
    background_results，模型需要时可再查）；command 也截 40 字做日志。
    为什么锁内再取一次：ready_ids 在锁外判断，取出时状态可能已被并发变更，
    pop 必须与状态判断在同一个锁临界区里才安全。
    """
    with background_lock:
        ready_ids = [bid for bid, task in background_tasks.items()
                     if task["status"] == "completed"]
    notifications = []
    for bg_id in ready_ids:
        with background_lock:
            task = background_tasks.pop(bg_id)
            output = background_results.pop(bg_id, "")
        summary = output[:200] if len(output) > 200 else output
        notifications.append(
            f"<task_notification>\n"
            f"  <task_id>{bg_id}</task_id>\n"
            f"  <status>completed</status>\n"
            f"  <command>{task['command']}</command>\n"
            f"  <summary>{summary}</summary>\n"
            f"</task_notification>")
        cprint(f"[background done] {bg_id}: "
               f"{task['command'][:40]} ({len(output)} chars)", color="green")
    return notifications
