"""本地媒体工具；仅使用 ffmpeg/ffprobe，不允许网络媒体 URL。"""
import json
import math
import shutil
import subprocess
from pathlib import Path


def local_media(path):
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError("媒体文件不存在，请提供本地路径")
    return p


def executable(name):
    value = shutil.which(name)
    if not value:
        raise RuntimeError(f"未找到 {name}，请使用已安装的本地 ffmpeg/ffprobe")
    return value


def probe(path):
    source = local_media(path)
    r = subprocess.run([executable('ffprobe'), '-v', 'error', '-show_entries',
        'format=duration:stream=width,height,codec_type', '-of', 'json', '-protocol_whitelist', 'file,pipe', str(source)],
        check=True, capture_output=True, text=True)
    data = json.loads(r.stdout)
    duration = float(data['format']['duration'])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('媒体时长非法')
    stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), {})
    return {'duration': duration, 'width': stream.get('width'), 'height': stream.get('height')}


def windows(duration, seconds):
    if not all(math.isfinite(x) and x > 0 for x in [duration, seconds]):
        raise ValueError('时长和窗口必须为有限正数')
    count = math.ceil(duration / seconds)
    for i in range(count):
        start = i * seconds
        yield start, min(duration, start + seconds)


def extract_audio(path, output, start, end):
    subprocess.run([executable('ffmpeg'), '-nostdin', '-v', 'error', '-y', '-ss', str(start),
        '-protocol_whitelist', 'file,pipe', '-i', str(local_media(path)), '-t', str(end - start), '-vn', '-ac', '1', '-ar', '16000',
        '-f', 'wav', str(output)], check=True, capture_output=True)
