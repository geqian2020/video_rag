"""Bounded OneDrive audio prefetch + checkpointed offline Whisper batch.

One background worker prepares at most one *next* audio file while the current
Whisper job runs. Original cloud video files are NEVER deleted. Disk cleanup
is limited to private, generated audio staging files after verified ASR.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

from core.config import load_config
from core.io import write_json
from core.media import probe
from workflow import transcript_registry as registry


@dataclass(frozen=True)
class Video:
    path: Path
    relative: str
    id: str
    video_id: str
    transcript: Path
    basename: str


@dataclass
class Group:
    basename: str
    videos: list[Video]
    representative: Video


class Pipeline:
    def __init__(self, source: Path, data: Path, videos: list[Video], *,
                 python: str = sys.executable, ffmpeg: str | None = None,
                 interval: float = 20.0, stop: threading.Event | None = None):
        self.source = source
        self.data = data
        self.videos = videos
        self.python = python
        self.ffmpeg = ffmpeg or shutil.which('ffmpeg')
        if not self.ffmpeg:
            raise RuntimeError('ffmpeg not found')
        self.interval = interval
        self.stop = stop or threading.Event()
        self.registry = data / 'transcript_registry.json'
        self.stage_dir = data / 'temp' / 'audio_stage'
        self.logs = data / 'logs'
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.log_file = self.logs / ('pipeline-' + datetime.now().strftime('%Y%m%d-%H%M%S') + '.log')
        self.config = load_config()

    def log(self, message: str):
        message = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}'
        with self._lock:
            print(message, flush=True)
            with self.log_file.open('a', encoding='utf-8') as f:
                f.write(message + '\n')

    def is_complete(self, video: Video):
        return registry.valid(video.transcript)

    def groups(self):
        # Complete outputs later in the manifest can satisfy earlier duplicates.
        for video in self.videos:
            if self.is_complete(video):
                registry.register(str(self.registry), video.basename, str(video.transcript))
        grouped = OrderedDict()
        for video in self.videos:
            grouped.setdefault(video.basename, []).append(video)
        planned = []
        for basename, members in grouped.items():
            # Pick the copy with most partial checkpoints; run.py validates each.
            rep = max(members, key=lambda v: len(list(Path(str(v.transcript) + '.windows').glob('*.json'))))
            planned.append(Group(basename, members, rep))
        return planned

    def stage_paths(self, video: Video):
        return (self.stage_dir / (video.id + '.wav'),
                self.stage_dir / (video.id + '.json'))

    def prepared(self, video: Video):
        wav, meta = self.stage_paths(video)
        try:
            info = json.loads(meta.read_text(encoding='utf-8'))
            if not wav.is_file() or wav.stat().st_size < 44:
                return False
            if (info.get('source') != str(video.path) or
                info.get('video_id') != video.video_id or
                info.get('bytes') != wav.stat().st_size):
                return False
            return probe(wav)['duration'] > 0  # local WAV only
        except (OSError, ValueError, KeyError, TypeError, subprocess.CalledProcessError):
            return False

    def prepare_audio(self, video: Video) -> Path:
        wav, meta = self.stage_paths(video)
        if self.prepared(video):
            self.log(f'[PREFETCH] 复用已准备的本地音轨：{video.relative} | {wav.stat().st_size / 1048576:.1f} MiB')
            return wav
        partial = wav.with_suffix('.partial.wav')
        partial.unlink(missing_ok=True)
        self.log(f'[PREFETCH] 开始顺序读取并提取音轨：{video.relative}')
        begun = time.monotonic()
        cmd = [self.ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
               '-i', str(video.path), '-map', '0:a:0', '-vn', '-ar', '16000',
               '-ac', '1', '-c:a', 'pcm_s16le', '-f', 'wav', str(partial)]
        errlog = self.logs / ('extract-' + video.id + '.log')
        rc = None
        try:
            with errlog.open('wb') as err:
                proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err)
                try:
                    while True:
                        try:
                            rc = proc.wait(timeout=self.interval)
                            break
                        except subprocess.TimeoutExpired:
                            if self.stop.is_set():
                                proc.terminate()
                                raise InterruptedError('Stopped during audio prefetch')
                            size = partial.stat().st_size if partial.exists() else 0
                            self.log(f'[PREFETCH] {video.relative}: {time.monotonic()-begun:.0f}s, '
                                     f'本地已写音轨 {size/1048576:.1f} MiB（不是视频下载百分比）')
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        try:
                            proc.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            proc.wait()
            if rc != 0:
                details = errlog.read_text(encoding='utf-8', errors='replace')[-1800:]
                raise RuntimeError(f'Audio extraction failed (ffmpeg rc={rc}): {details}')
            duration = probe(partial)['duration']
            if duration <= 0:
                raise ValueError('No valid audio duration')
            os.replace(partial, wav)
            write_json(meta, {'version': 1, 'source': str(video.path),
                              'video_id': video.video_id, 'bytes': wav.stat().st_size,
                              'duration': duration})
            self.log(f'[PREFETCH] 完成：{video.relative} | 提取耗时 {time.monotonic()-begun:.1f}s '
                     f'| 本地音轨 {wav.stat().st_size/1048576:.1f} MiB | 时长 {duration:.1f}s')
            return wav
        except BaseException:
            partial.unlink(missing_ok=True)
            raise

    def run_asr(self, video: Video, audio: Path):
        # --audio-source changes FFmpeg input only. Window-checkpoint
        # fingerprints remain tied to original video ID and settings.
        cmd = [self.python, '-m', 'transcribe.run', '--video', str(video.path),
               '--video-id', video.video_id, '--no-vocab', '--audio-source', str(audio),
               '--output', str(video.transcript)]
        logfile = self.logs / ('asr-' + video.id + '.log')
        begun = time.monotonic()
        self.log(f'[ASR] 开始：{video.relative} | 复用窗口检查点；日志 {logfile}')
        with logfile.open('wb') as f:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                    env={**os.environ, 'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8'})
            try:
                previous_size = 0
                while True:
                    try:
                        rc = proc.wait(timeout=self.interval)
                        break
                    except subprocess.TimeoutExpired:
                        if self.stop.is_set():
                            proc.terminate()
                            raise InterruptedError('ASR interrupted')
                        size = logfile.stat().st_size
                        if size > previous_size:
                            with logfile.open('rb') as logread:
                                logread.seek(previous_size)
                                chunk = logread.read()
                            previous_size = size
                            lines = chunk.decode('utf-8', errors='replace').replace('\r','\n').splitlines()
                            for line in lines[-5:]:
                                if '窗口' in line or 'ERROR' in line or 'Traceback' in line:
                                    self.log('[ASR] ' + line[:300])
                        self.log(f'[ASR] 运行中 {time.monotonic()-begun:.0f}s：{video.relative}')
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        if rc:
            raise RuntimeError(f'Whisper failed rc={rc}; see {logfile}. Window checkpoints are retained.')
        if not self.is_complete(video):
            raise RuntimeError(f'Whisper returned 0 without a valid JSONL/audit: {video.relative}')
        self.log(f'[ASR] 完成并通过 SHA256 审计：{video.relative} | 耗时 {time.monotonic()-begun:.1f}s')

    def release_original(self, video: Video):
        # This requests de-hydration only. Never unlink/delete OneDrive files.
        if os.name != 'nt':
            self.log(f'[RELEASE] 非 Windows 环境，跳过 OneDrive 属性操作：{video.relative}')
            return
        t0 = time.monotonic()
        completed = subprocess.run(['attrib.exe', '+U', '-P', str(video.path)],
                                   capture_output=True, text=True, errors='replace')
        attrs = subprocess.run(['attrib.exe', str(video.path)], capture_output=True,
                               text=True, errors='replace')
        state = (attrs.stdout or attrs.stderr).strip()
        if completed.returncode:
            self.log(f'[RELEASE] 请求失败 rc={completed.returncode}：{video.relative} | {completed.stderr.strip()}')
        else:
            self.log(f'[RELEASE] 仅联机请求已发送：{video.relative} | {time.monotonic()-t0:.1f}s '
                     f'| 属性：{state} | 不能据此保证已回收磁盘空间')

    def cleanup_audio(self, video: Video):
        wav, meta = self.stage_paths(video)
        for path in (wav, meta):
            path.unlink(missing_ok=True)  # only private audio artifacts
        self.log(f'[CLEAN] 已清理私有临时音轨：{video.relative}')

    def process(self) -> dict:
        groups = self.groups()
        self.log(f'[PLAN] 扫描文件 {len(self.videos)}，按名称分组 {len(groups)}。'
                 '优先复用完整转录；每次仅后台预取一个下一个视频。')
        pending = []
        reused = 0
        for group in groups:
            canonical = registry.load(str(self.registry)).get(group.basename, {}).get('transcript')
            if canonical and registry.valid(canonical):
                for v in group.videos:
                    if not self.is_complete(v):
                        if not registry.reuse(str(self.registry), v.basename, str(v.transcript), v.video_id):
                            raise RuntimeError('Filename registry reuse failed: ' + v.relative)
                    reused += 1
                self.log(f'[CACHE] 命中：{group.basename}；复用 {len(group.videos)} 个来源，未下载原视频')
            else:
                pending.append(group)
        self.log(f'[PLAN] 需首次/续转 {len(pending)} 个唯一文件名，已复用 {reused} 个来源。')
        new = 0
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix='onedrive-prefetch') as executor:
            next_future: Future | None = None
            try:
                for i, group in enumerate(pending):
                    video = group.representative
                    if next_future is None:
                        next_future = executor.submit(self.prepare_audio, video)
                    self.log(f'[QUEUE] [{i+1}/{len(pending)}] 等待音轨可用：{video.relative}')
                    while not next_future.done():
                        if self.stop.wait(self.interval):
                            raise InterruptedError('Interrupted before stage ready')
                        self.log(f'[QUEUE] 仍在准备：{video.relative}')
                    audio = next_future.result()
                    next_group = pending[i+1] if i+1 < len(pending) else None
                    next_future = (executor.submit(self.prepare_audio, next_group.representative)
                                   if next_group is not None else None)
                    self.run_asr(video, audio)
                    registry.register(str(self.registry), video.basename, str(video.transcript))
                    new += 1
                    self.release_original(video)
                    self.cleanup_audio(video)
                    for duplicate in group.videos:
                        if duplicate == video:
                            continue
                        if not registry.reuse(str(self.registry), duplicate.basename,
                                              str(duplicate.transcript), duplicate.video_id):
                            raise RuntimeError('Cannot publish duplicate provenance: ' + duplicate.relative)
                        reused += 1
                    self.log(f'[SUCCESS] {group.basename}: 新转录 1 个，另有 {len(group.videos)-1} 个同名来源已复用')
                return {'new': new, 'reused': reused, 'sources': len(self.videos),
                        'unique_names': len(groups)}
            finally:
                self.stop.set()


def load_manifest(path: Path, source: Path, transcripts: Path) -> list[Video]:
    # Git Bash manifest stores /c/... paths; native Windows Python uses C:/...
    def native(text: str) -> Path:
        if os.name == 'nt' and re.match(r'^/[a-zA-Z]/', text):
            text = text[1].upper() + ':' + text[2:]
        return Path(text)
    items = [native(b.decode('utf-8')) for b in path.read_bytes().split(b'\0') if b]
    result = []
    for item in items:
        try:
            relative = item.relative_to(source).as_posix()
        except ValueError:
            raise ValueError(f'Scanned path outside source root: {item}') from None
        digest = hashlib.sha256(relative.encode('utf-8')).hexdigest()[:24]
        result.append(Video(item, relative, digest, 'sharepoint/' + digest,
                            transcripts / (digest + '.jsonl'), item.name.casefold()))
    if not result:
        raise ValueError('Empty scan manifest')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--interval', type=float, default=20.0)
    args = parser.parse_args(argv)
    if args.interval <= 0:
        parser.error('interval must be positive')
    os.environ.setdefault('PYTHONUTF8', '1')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    args.data.mkdir(parents=True, exist_ok=True)
    root = args.data / 'transcripts'
    root.mkdir(parents=True, exist_ok=True)
    videos = load_manifest(args.manifest, args.source, root)
    pipeline = Pipeline(args.source, args.data, videos, interval=args.interval)
    try:
        result = pipeline.process()
    except KeyboardInterrupt:
        pipeline.stop.set()
        pipeline.log('[STOP] 用户中断；保留已有窗口检查点及已完成音轨，未删除 OneDrive 视频')
        raise SystemExit(130)
    except Exception as exc:
        pipeline.stop.set()
        pipeline.log(f'[ERROR] {type(exc).__name__}: {exc}；批次已停止，未完成窗口和音轨可复用')
        raise SystemExit(1) from exc
    pipeline.log('[DONE] ' + json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
