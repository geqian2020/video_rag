"""将旧 [HH:MM:SS] Markdown 转为 JSONL；缺失 end 的事实以 estimated 标记。"""
import argparse
import re
from pathlib import Path

from core.io import write_rows
from core.schema import Segment

PATTERN = re.compile(r"^\[(\d+):([0-5]\d):([0-5]\d)\]\s*(.+)$")


def convert(text, video_id, duration):
    lines = []
    for line in text.splitlines():
        match = PATTERN.match(line.strip())
        if match:
            h, m, s, value = match.groups()
            lines.append((int(h) * 3600 + int(m) * 60 + int(s), value))
    if not lines:
        raise ValueError("未发现时间戳行")
    if any(a[0] > b[0] for a, b in zip(lines, lines[1:])):
        raise ValueError("时间戳必须递增")
    # 旧稿只保留整数秒，同秒多行合并，不制造零时长。
    grouped = []
    for start, value in lines:
        if grouped and grouped[-1][0] == start:
            grouped[-1] = (start, grouped[-1][1] + "\n" + value)
        else:
            grouped.append((start, value))
    if duration <= grouped[-1][0]:
        raise ValueError("--duration 必须大于最后一个时间戳；不能凭空估计视频结尾")
    rows = []
    for i, (start, value) in enumerate(grouped):
        if value.strip() == "[无法识别]":
            continue
        end = grouped[i + 1][0] if i + 1 < len(grouped) else duration
        rows.append(Segment(video_id=video_id, text=value, start=start, end=end,
                            timestamp_quality="estimated").model_dump())
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--video-id", required=True, help="使用唯一稳定的编号或视频相对路径，勿只用文件 stem")
    p.add_argument("--duration", required=True, type=float, help="实际视频总秒数，可由 ffprobe 获取")
    a = p.parse_args()
    rows = convert(Path(a.input).read_text(encoding="utf-8-sig"), a.video_id, a.duration)
    write_rows(a.output, rows)
    print(f"转换完成: {len(rows)} 行，所有 end 均标记为 estimated")


if __name__ == "__main__":
    main()
