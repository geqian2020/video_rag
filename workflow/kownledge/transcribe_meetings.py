#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
transcribe_meetings.py

递归扫描项目目录（及所有子目录）中的 .mp4 文件，使用 ffmpeg 提取音频，
并调用本地 Whisper 模型进行中文语音转录，为每个视频生成同名的 Markdown
转录文件。

用法:
    python transcribe_meetings.py [--root DIR] [--model small] [--force]

详见同目录下的 README.md。
"""

import argparse
import datetime
import logging
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

try:
    import whisper
except ImportError:
    print(
        "错误: 未安装 openai-whisper。请先运行: pip install -r requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)


LOG_FILE_NAME = "transcribe_log.txt"
DEFAULT_MODEL = "small"
DEFAULT_CHUNK_SECONDS = 600  # 每片音频 10 分钟，避免整段音频一次性占用内存
DEFAULT_LANGUAGE = "zh"


def find_ffmpeg_tools():
    """定位 ffmpeg / ffprobe 可执行文件，要求其已在 PATH 中。"""
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError(
            "未找到 ffmpeg/ffprobe，请先安装 ffmpeg 并确认其在 PATH 中"
            "（例如 winget install --id Gyan.FFmpeg -e --source winget），"
            "安装完成后请重启终端再运行本脚本。"
        )
    return ffmpeg, ffprobe


def format_timestamp(seconds: float) -> str:
    """将秒数格式化为 [HH:MM:SS]。"""
    total_seconds = max(0, int(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"[{hours:02d}:{minutes:02d}:{secs:02d}]"


def get_duration_seconds(ffprobe: str, media_path: Path) -> float:
    """使用 ffprobe 获取媒体文件时长（秒）。"""
    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(media_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def extract_audio_chunk(ffmpeg: str, video_path: Path, start: float, duration: float, out_wav: Path) -> None:
    """使用 ffmpeg 从视频中截取 [start, start+duration) 的音频，转换为 16kHz 单声道 WAV。"""
    cmd = [
        ffmpeg, "-y",
        "-ss", str(start),
        "-t", str(duration),
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-f", "wav",
        str(out_wav),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_tail = result.stderr.strip()[-500:]
        raise RuntimeError(f"ffmpeg 提取音频失败: {stderr_tail}")


class Transcriber:
    """封装 Whisper 模型加载与转录，模型只加载一次并在多个文件间复用。"""

    def __init__(self, model_name: str, language: str):
        self.language = language
        logging.info("正在加载 Whisper 模型 '%s' (CPU 模式)...", model_name)
        self.model = whisper.load_model(model_name, device="cpu")
        logging.info("模型加载完成。")

    def transcribe_wav(self, wav_path: Path):
        """转录单个 wav 文件，返回 whisper 的 segments 列表 [{start, end, text}, ...]。"""
        result = self.model.transcribe(
            str(wav_path),
            language=self.language,
            task="transcribe",
            fp16=False,  # CPU 不支持 fp16，显式关闭以避免警告并强制使用 fp32
            verbose=False,
        )
        return result.get("segments", [])


def process_video(
    video_path: Path,
    transcriber: Transcriber,
    ffmpeg: str,
    ffprobe: str,
    chunk_seconds: int,
    temp_dir: Path,
) -> str:
    """
    处理单个视频文件：按 chunk_seconds 分片提取音频 -> 逐片转录 -> 按时间戳拼接文本。

    分片处理可以避免将整段（可能长达数小时）的音频一次性解码进内存，
    每个分片对应的临时 wav 文件在转录完成后立即删除。

    单个分片转录失败不会中断整个文件的处理，会记录为 [无法识别] 并继续下一片。
    仅当整个视频完全无法读取时才会向上抛出异常，导致该文件被记为失败。
    """
    duration = get_duration_seconds(ffprobe, video_path)
    lines = []
    chunk_start = 0.0
    chunk_index = 0

    while chunk_start < duration:
        chunk_len = min(chunk_seconds, duration - chunk_start)
        temp_wav = temp_dir / f"chunk_{uuid.uuid4().hex}.wav"
        try:
            extract_audio_chunk(ffmpeg, video_path, chunk_start, chunk_len, temp_wav)
            segments = transcriber.transcribe_wav(temp_wav)
            if not segments:
                lines.append(f"{format_timestamp(chunk_start)} [无法识别]")
            for seg in segments:
                ts = format_timestamp(chunk_start + seg["start"])
                text = seg["text"].strip()
                lines.append(f"{ts} {text}" if text else f"{ts} [无法识别]")
        except Exception as exc:
            logging.warning(
                "分片 %d (%.0fs~%.0fs) 转录失败，标记为无法识别并继续: %s",
                chunk_index, chunk_start, chunk_start + chunk_len, exc,
            )
            lines.append(f"{format_timestamp(chunk_start)} [无法识别]")
        finally:
            if temp_wav.exists():
                try:
                    temp_wav.unlink()
                except OSError:
                    pass

        chunk_start += chunk_len
        chunk_index += 1

    return "\n\n".join(lines) if lines else "[无法识别]"


def build_markdown(video_path: Path, project_root: Path, transcript_text: str) -> str:
    try:
        rel_path = video_path.relative_to(project_root)
    except ValueError:
        rel_path = video_path
    generated_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return (
        f"# 会议转录：{video_path.stem}\n\n"
        f"* 原始文件：`{rel_path}`\n"
        f"* 转录语言：中文\n"
        f"* 生成时间：{generated_at}\n\n"
        f"## 转录内容\n\n"
        f"{transcript_text}\n"
    )


def find_mp4_files(root: Path):
    return sorted(root.rglob("*.mp4"))


def main():
    parser = argparse.ArgumentParser(
        description="递归扫描目录中的 MP4 会议视频并生成中文转录 Markdown（CPU 模式）"
    )
    parser.add_argument("--root", type=str, default=".", help="扫描的根目录 (默认: 当前目录)")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Whisper 模型名称 (默认: {DEFAULT_MODEL})，可选 tiny/base/small/medium/large",
    )
    parser.add_argument(
        "--chunk-seconds", type=int, default=DEFAULT_CHUNK_SECONDS,
        help=f"音频分片长度，单位秒 (默认: {DEFAULT_CHUNK_SECONDS})",
    )
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE, help="转录语言代码 (默认: zh)")
    parser.add_argument(
        "--force", action="store_true", help="即使已存在同名 .md 文件也重新转录并覆盖"
    )
    parser.add_argument(
        "--only", type=str, default=None,
        help="仅处理路径中包含该子字符串的文件（用于单文件/小规模测试）",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(root / LOG_FILE_NAME, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    try:
        ffmpeg, ffprobe = find_ffmpeg_tools()
    except RuntimeError as exc:
        logging.error(str(exc))
        sys.exit(1)

    mp4_files = find_mp4_files(root)
    if args.only:
        mp4_files = [p for p in mp4_files if args.only in str(p)]

    found = len(mp4_files)
    succeeded = 0
    skipped = 0
    failed = []

    logging.info("找到 %d 个 MP4 文件。", found)

    if found == 0:
        logging.info("未找到任何 MP4 文件，退出。")
        return

    transcriber = None

    with tempfile.TemporaryDirectory(prefix="transcribe_tmp_") as temp_dir_str:
        temp_dir = Path(temp_dir_str)

        for video_path in mp4_files:
            md_path = video_path.with_suffix(".md")

            if md_path.exists() and not args.force:
                logging.info("跳过 (已存在 Markdown): %s", video_path)
                skipped += 1
                continue

            logging.info("开始处理: %s", video_path)

            if transcriber is None:
                transcriber = Transcriber(args.model, args.language)

            try:
                transcript_text = process_video(
                    video_path, transcriber, ffmpeg, ffprobe, args.chunk_seconds, temp_dir
                )
                markdown = build_markdown(video_path, root, transcript_text)
                md_path.write_text(markdown, encoding="utf-8")
                logging.info("完成: %s -> %s", video_path, md_path)
                succeeded += 1
            except Exception as exc:
                logging.error("处理失败: %s | 原因: %s", video_path, exc)
                failed.append((str(video_path), str(exc)))

    logging.info("=" * 60)
    logging.info("处理完成统计:")
    logging.info("  找到: %d", found)
    logging.info("  成功: %d", succeeded)
    logging.info("  跳过: %d", skipped)
    logging.info("  失败: %d", len(failed))
    if failed:
        logging.info("失败详情:")
        for path, reason in failed:
            logging.info("  - %s: %s", path, reason)
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
