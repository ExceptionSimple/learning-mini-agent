import json
import logging
from pathlib import Path

SESSIONS_DIR = Path(".sessions")

def recovery_sessions(session_id: str):
    messages = []
    path = SESSIONS_DIR / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.touch()
    with open(path, "r", encoding="UTF-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            msg = json.loads(line)
            messages.append(msg)
    return messages

def write_sessions(session_id: str, messages: list):
    path = SESSIONS_DIR / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="UTF-8") as f:
        for msg in messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")
    return True

def append_session(session_id: str, message: dict):
    path = SESSIONS_DIR / f"{session_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        logging.error(f"Session {session_id}.jsonl not found")
        return False
    with open(path, "a", encoding="UTF-8") as f:
        f.write(json.dumps(message, ensure_ascii=False) + "\n")
    return True
