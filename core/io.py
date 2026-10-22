"""JSON/JSONL 和原子文件发布。"""
import json
import os
import tempfile
from pathlib import Path


def read_rows(path):
    p = Path(path)
    if p.suffix.lower() == ".json":
        rows = json.loads(p.read_text(encoding="utf-8-sig"))
        if not isinstance(rows, list):
            raise ValueError("JSON 顶层必须为数组")
        return rows
    return [json.loads(line) for line in p.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def atomic_text(path, text):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=p.parent, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, p)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_rows(path, rows):
    atomic_text(path, "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))


def write_json(path, data):
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2))
