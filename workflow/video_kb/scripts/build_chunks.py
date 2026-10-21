#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_chunks.py

扫描指定根目录下由 transcribe_meetings.py 生成的带时间戳 Markdown 转录文件
（[HH:MM:SS] 文本 的格式），对文本做清洗（简繁统一 + 术语修正 + 俚语标注），
再按累计字符数把细粒度的转录行合并成适合检索的片段（chunk），输出为
JSONL 文件，供后续 build_index.py 建立向量索引。

设计要点：
    - 不修改原始 .md 文件，输出单独写入 index_data/chunks.jsonl。
    - 合并时按累计字符数判断是否达到目标片段长度，不是简单按行数或固定
      时间窗口合并——真实转录里有的行只有几个字，有的行有上百字，长度
      极不均匀（已用真实数据验证）。
    - 片段之间保留最后一条原始转录行作为重叠，避免语义被切断在片段边界。
    - 不跨视频合并。
    - 过滤空白/占位符内容（如 [无法识别]）。
    - 视频与转录文件的匹配只依赖同名 stem（同目录下 X.mp4 对应 X.md），
      不依赖父文件夹命名是否规范。
    - 文件夹名解析得到的 session_topics_hint / associated_ppts 仅作为
      弱提示元数据，缺失或内容不准确都不影响主流程，不参与检索排序或过滤。

用法：
    python build_chunks.py --root <OneDrive子目录> --out index_data/chunks.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

# Windows 终端默认代码页（如 cp1252/cp936）无法打印部分中文或混排字符，
# 这里强制将 stdout/stderr 重新配置为 UTF-8，避免 print() 时崩溃。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from clean_text import TextCleaner  # noqa: E402


DEFAULT_TARGET_CHARS = 300  # 目标片段累计字符数，落在 150~400 字区间的中点
DEFAULT_MAX_CHARS = 400     # 超过此字符数强制切断，避免单个片段过长
DEFAULT_MIN_SECONDS = 20    # 片段最短时长（秒），过短的片段倾向于继续合并
LINE_PATTERN = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]\s*(.*)$")
PLACEHOLDER_TEXTS = {"[无法识别]", ""}


def parse_timestamp_line(line: str):
    """解析形如 '[00:12:30] 文本内容' 的一行，返回 (seconds, text) 或 None。"""
    match = LINE_PATTERN.match(line.strip())
    if not match:
        return None
    h, m, s, text = match.groups()
    seconds = int(h) * 3600 + int(m) * 60 + int(s)
    return seconds, text.strip()


def format_timestamp(seconds: int) -> str:
    h, rem = divmod(max(0, seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def extract_transcript_lines(md_text: str):
    """从 Markdown 全文中提取时间戳行，忽略标题/元信息部分。
    返回 [(seconds, text), ...]，按文件原始顺序（已天然按时间递增）。"""
    lines = []
    in_content = False
    for raw_line in md_text.splitlines():
        if raw_line.strip() == "## 转录内容":
            in_content = True
            continue
        if not in_content:
            continue
        parsed = parse_timestamp_line(raw_line)
        if parsed is None:
            continue
        seconds, text = parsed
        if text in PLACEHOLDER_TEXTS:
            continue
        lines.append((seconds, text))
    return lines


def merge_lines_into_chunks(
    lines,
    target_chars: int = DEFAULT_TARGET_CHARS,
    max_chars: int = DEFAULT_MAX_CHARS,
    min_seconds: int = DEFAULT_MIN_SECONDS,
):
    """
    将 (seconds, text) 列表按累计字符数合并为片段。

    规则：
        - 不断累加原始行，直到累计字符数达到 target_chars，且当前片段
          时长已达到 min_seconds，才允许收尾；若单次累加后超过 max_chars，
          无论时长是否达标都强制收尾，避免长句导致片段无限增长。
        - 收尾后，下一个片段以本片段最后一条原始行作为起始重叠行，
          确保跨片段边界的语义不会被完全切断。
        - 仅由重叠行构成、没有任何新增内容的尾部片段会被丢弃，不单独输出，
          避免产生与前一个片段高度重复的"空壳"片段。

    返回 [(start_seconds, end_seconds, text), ...]
    """
    if not lines:
        return []

    chunks = []
    current = []  # [(seconds, text), ...]，当前正在累积的片段
    new_line_count_in_current = 0  # current 中真正"新增"的行数（不含重叠行）

    def flush():
        start = current[0][0]
        end = current[-1][0]
        text = "".join(t for _, t in current)
        chunks.append((start, end, text))

    for seconds, text in lines:
        current.append((seconds, text))
        new_line_count_in_current += 1

        char_count = sum(len(t) for _, t in current)
        duration = current[-1][0] - current[0][0]

        should_flush = char_count >= max_chars or (
            char_count >= target_chars and duration >= min_seconds
        )
        if should_flush:
            flush()
            overlap_line = current[-1]
            current = [overlap_line]
            new_line_count_in_current = 0

    # 收尾：只有当当前缓冲里包含至少一条"新增"内容时才输出最后一个片段，
    # 避免输出一个只含重叠行、没有任何新信息的尾部片段。
    if new_line_count_in_current > 0:
        flush()

    return chunks


def resolve_video_for_md(md_path: Path):
    """根据同名 stem 在同目录下查找对应的 .mp4 文件。找不到返回 None。"""
    candidate = md_path.with_suffix(".mp4")
    if candidate.exists():
        return candidate
    return None


def collect_folder_metadata(md_path: Path, root: Path):
    """
    解析父文件夹名作为弱提示元数据。

    注意：文件夹命名不保证规范（已确认部分文件夹名与内容无关），
    这里只做尽力而为的提取，解析失败或内容不准确都不影响主流程，
    下游也不应依赖这些字段做任何过滤或排序。
    """
    folder = md_path.parent
    folder_name = folder.name
    ppt_files = sorted(
        p.name for p in folder.glob("*.pptx")
    ) + sorted(p.name for p in folder.glob("*.ppt"))
    try:
        rel_folder = str(folder.relative_to(root))
    except ValueError:
        rel_folder = str(folder)
    return {
        "folder_name": folder_name,
        "folder_relative_path": rel_folder,
        "associated_ppts": ppt_files,
    }


def build_chunks_for_file(
    md_path: Path,
    root: Path,
    cleaner: TextCleaner,
    target_chars: int,
    max_chars: int,
    min_seconds: int,
):
    """处理单个 .md 转录文件，返回该文件产生的 chunk 字典列表。"""
    md_text = md_path.read_text(encoding="utf-8")
    raw_lines = extract_transcript_lines(md_text)

    cleaned_lines = []
    all_jargon = set()
    for seconds, text in raw_lines:
        cleaned_text, jargon_terms = cleaner.clean(text)
        if not cleaned_text:
            continue
        cleaned_lines.append((seconds, cleaned_text))
        all_jargon.update(jargon_terms)

    merged = merge_lines_into_chunks(
        cleaned_lines, target_chars=target_chars, max_chars=max_chars, min_seconds=min_seconds
    )

    video_path = resolve_video_for_md(md_path)
    try:
        video_rel = str(video_path.relative_to(root)) if video_path else None
    except ValueError:
        video_rel = str(video_path) if video_path else None
    try:
        transcript_rel = str(md_path.relative_to(root))
    except ValueError:
        transcript_rel = str(md_path)

    folder_meta = collect_folder_metadata(md_path, root)
    modified_time = md_path.stat().st_mtime

    chunks = []
    for idx, (start, end, text) in enumerate(merged):
        if not text.strip():
            continue
        _, jargon_terms = cleaner.clean(text)
        chunk_id = f"{md_path.stem}__{idx:06d}"
        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "source_type": "video_transcript",
                "video_name": video_path.name if video_path else None,
                "video_path": video_rel,
                "transcript_path": transcript_rel,
                "start_time": format_timestamp(start),
                "end_time": format_timestamp(end),
                "start_seconds": start,
                "end_seconds": end,
                "modified_time": modified_time,
                "jargon_terms": jargon_terms,
                "folder_name": folder_meta["folder_name"],
                "folder_relative_path": folder_meta["folder_relative_path"],
                "associated_ppts": folder_meta["associated_ppts"],
                "related_ppt": None,
            }
        )
    return chunks


def find_md_files(root: Path):
    return sorted(root.rglob("*.md"))


def main():
    parser = argparse.ArgumentParser(
        description="扫描带时间戳的转录 Markdown，清洗并切片为检索片段，输出 JSONL"
    )
    parser.add_argument("--root", type=str, required=True, help="扫描的根目录")
    parser.add_argument(
        "--out", type=str, default="index_data/chunks.jsonl", help="输出 JSONL 文件路径"
    )
    parser.add_argument(
        "--config-dir", type=str, default=None,
        help="术语表/俚语表所在目录（默认: video_kb/config）",
    )
    parser.add_argument("--target-chars", type=int, default=DEFAULT_TARGET_CHARS)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--min-seconds", type=int, default=DEFAULT_MIN_SECONDS)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    config_dir = Path(args.config_dir) if args.config_dir else None
    cleaner = TextCleaner(config_dir=config_dir) if config_dir else TextCleaner()

    md_files = find_md_files(root)
    print(f"找到 {len(md_files)} 个 Markdown 转录文件。")

    total_chunks = 0
    skipped_no_video = 0
    with out_path.open("w", encoding="utf-8") as f:
        for md_path in md_files:
            chunks = build_chunks_for_file(
                md_path, root, cleaner, args.target_chars, args.max_chars, args.min_seconds
            )
            if chunks and chunks[0]["video_path"] is None:
                skipped_no_video += 1
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            total_chunks += len(chunks)
            print(f"  {md_path.name}: {len(chunks)} 个片段")

    print("=" * 60)
    print(f"处理完成。共生成 {total_chunks} 个片段，写入 {out_path}")
    if skipped_no_video:
        print(f"注意: {skipped_no_video} 个转录文件未找到对应的同名 .mp4 视频文件。")


if __name__ == "__main__":
    main()
