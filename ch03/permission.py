import json
from pathlib import Path

WORKDIR = Path.cwd()

# ==================================================
# 阀门 1: 必须拒绝的命令列表
# ==================================================
DENY_LIST = [
    "rm -rf /", "sudo", "shutdown", "reboot",
    "mkfs", "dd if=", "> /dev/sda",
]

def check_deny_list(command: str) -> str | None:
    for pattern in DENY_LIST:
        if pattern in command:
            return f"Blocked: '{pattern}' is on the deny list"
    return None

# ==================================================
# 阀门 2: 规则匹配
# ==================================================
PERMISSION_RULES = [
    {
        "tools": ["write_file", "edit_file"],
        "check": lambda args: not (WORKDIR / args.get("path", "")).resolve().is_relative_to(WORKDIR),
        "message": "Writing outside workspace",
    },
    {
        "tools": ["bash"],
        "check": lambda args: any(kw in args.get("command", "") for kw in ["rm ", "> /etc/", "chmod 777"]),
        "message": "Potentially destructive command",
    },
    {
        "tools": ["asklinux"],
        "check": True,
        "message": "asklinux is private.",
    }
]

def check_rules(tool_name: str, args: dict) -> str | None:
    for rule in PERMISSION_RULES:
        if tool_name in rule["tools"] and rule["check"](args):
            return rule["message"]
    return None


# ==================================================
# 阀门 3: 当阀门2的规则命中后，需要人工确认
# ==================================================
def ask_user(tool_name: str, args: dict, reason: str) -> str:
    print(f"\n\033[33m⚠  {reason}\033[0m")
    print(f"   Tool: {tool_name}({args})")
    choice = input("   Allow? [y/N] ").strip().lower()
    return "allow" if choice in ("y", "yes") else "deny"

# ==================================================
# 主函数
# ==================================================

def check_permission(tool) -> bool:
    name = tool['function']['name']
    params = json.loads(tool['function']['arguments'] or '{}')
    # 闸门 1: 硬拒绝
    if name == "bash":
        reason = check_deny_list(params.get("command", ""))
        if reason:
            print(f"\n⛔ {reason}")
            return False

    # 闸门 2 + 3: 规则匹配 → 用户审批
    reason = check_rules(name, params)
    if reason:
        decision = ask_user(name, params, reason)
        if decision == "deny":
            return False

    return True
