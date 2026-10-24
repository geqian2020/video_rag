"""Filename-based completed-transcript registry; never reads source media.

CLI:
  python -m workflow.transcript_registry rebuild REGISTRY SOURCE CACHE_NUL TRANSCRIPTS
  python -m workflow.transcript_registry reuse REGISTRY FILENAME DESTINATION VIDEO_ID
  python -m workflow.transcript_registry register REGISTRY FILENAME SOURCE_TRANSCRIPT
A filename match is an explicit user policy, not proof of identical video contents.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
from core.io import atomic_text, write_json, write_rows


def valid(path):
    p = Path(path)
    audit = Path(str(p) + ".audit.json")
    if not p.is_file() or not audit.is_file() or not p.stat().st_size:
        return False
    try:
        rows = [json.loads(s) for s in p.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        info = json.loads(audit.read_text(encoding="utf-8"))
        if not rows or not all(x.get("video_id") and x.get("text") and
                               float(x["end"]) > float(x["start"]) for x in rows):
            return False
        return info.get("transcript_sha256") == hashlib.sha256(p.read_bytes()).hexdigest()
    except (OSError, ValueError, KeyError, TypeError, OverflowError):
        return False


def key(filename):
    return Path(filename.replace("\\", "/")).name.casefold()


def load(registry):
    try:
        data = json.loads(Path(registry).read_text(encoding="utf-8"))
        if data.get("version") == 1 and isinstance(data.get("entries"), dict):
            return data["entries"]
    except (OSError, ValueError, TypeError):
        pass
    return {}


def save(registry, entries):
    atomic_text(registry, json.dumps({"version": 1, "entries": entries},
                                     ensure_ascii=False, indent=2))


def register(registry, filename, transcript):
    if not valid(transcript):
        raise ValueError("Cannot register incomplete transcript")
    entries = load(registry)
    name = key(filename)
    # Keep the first already-verified complete copy as canonical.
    old = entries.get(name, {}).get("transcript")
    canonical = old if old and valid(old) else str(Path(transcript).resolve())
    entries[name] = {"transcript": canonical}
    save(registry, entries)


def rebuild(registry, root, cache_nul, transcripts):
    root = Path(root)
    entries = load(registry)
    media = [os.fsdecode(p) for p in Path(cache_nul).read_bytes().split(b"\0") if p]
    for path in media:
        relative = Path(path).relative_to(root).as_posix()
        video_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]
        dst = Path(transcripts) / (video_id + ".jsonl")
        name = key(path)
        if valid(dst):
            # Prefer a source from this scan; register is atomic on completion.
            entries[name] = {"transcript": str(dst.resolve())}
        else:
            previous = entries.get(name, {}).get("transcript")
            if not previous or not valid(previous):
                entries.pop(name, None)
    save(registry, entries)
    print(f"REGISTERED {len(entries)} completed unique filenames", flush=True)


def reuse(registry, filename, destination, video_id):
    entries = load(registry)
    previous = entries.get(key(filename), {}).get("transcript")
    if not previous or not valid(previous):
        return False
    source = Path(previous)
    target = Path(destination)
    if source.resolve() == target.resolve():
        return True
    rows = [json.loads(s) for s in source.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    audit = json.loads(Path(str(source) + ".audit.json").read_text(encoding="utf-8"))
    for row in rows:
        row["video_id"] = video_id
    audit["video_id"] = video_id
    # Publish the derived output without ever opening the duplicate cloud media.
    write_rows(target, rows)
    audit["transcript_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    write_json(str(target) + ".audit.json", audit)
    return valid(target)


def main():
    command, *args = sys.argv[1:]
    if command == "rebuild" and len(args) == 4:
        rebuild(*args)
    elif command == "reuse" and len(args) == 4:
        sys.exit(0 if reuse(*args) else 1)
    elif command == "register" and len(args) == 3:
        register(*args)
    else:
        raise SystemExit("Usage: rebuild registry root cache_nul transcripts | reuse registry filename dest video_id | register registry filename transcript")


if __name__ == "__main__":
    main()
