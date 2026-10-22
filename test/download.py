"""从公开原始页面按 manifest 下载；默认只取字幕，视频需显式 --mode video/both。"""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys

from core.io import write_json


def load_manifest(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    videos = data['videos']
    ids = [v['video_id'] for v in videos]
    if len(ids) != len(set(ids)) or not 4 <= len(ids) <= 5:
        raise ValueError('manifest 必须包含 4-5 个不重复视频')
    for v in videos:
        if not re.fullmatch(r'[A-Za-z0-9_-]{11}', v['video_id']):
            raise ValueError('非法 YouTube ID')
        if v['url'] != 'https://www.youtube.com/watch?v='+v['video_id']:
            raise ValueError('本测试集只接受 manifest 中的 YouTube 原始页面')
        if not 3600 <= v['duration_seconds'] <= 7200:
            raise ValueError('测试视频时长须在 1-2 小时')
    return videos


def command(video, output, mode, system_ca=False):
    args = [sys.executable, '-m', 'yt_dlp', '--no-playlist', '--socket-timeout', '15',
        '--retries', '1', '--extractor-retries', '1', '--continue', '--no-overwrites',
        '-o', str(Path(output)/'raw/%(id)s/%(id)s.%(ext)s')]
    if system_ca:
        args += ['--compat-options', 'no-certifi']  # 使用系统可信 CA，仍验证 TLS
    if mode in {'subtitles', 'both'}:
        args += ['--write-subs', '--write-auto-subs', '--sub-langs', 'en', '--sub-format', 'json3']
    if mode == 'subtitles':
        args += ['--skip-download']
    else:
        args += ['-f', 'bv*[height<=720]+ba/b[height<=720]', '--merge-output-format', 'mp4']
    return args + [video['url']]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='test/videos.json')
    p.add_argument('--output', default='data/public_benchmark')
    p.add_argument('--mode', choices=['subtitles', 'video', 'both'], default='subtitles')
    p.add_argument('--system-ca', action='store_true', help='代理证书在系统 CA 时使用，不关闭 TLS 校验')
    a = p.parse_args()
    root = Path(a.output);root.mkdir(parents=True, exist_ok=True)
    results = []
    for video in load_manifest(a.manifest):
        cached = root/'raw'/video['video_id']/(video['video_id']+'.en.json3')
        if a.mode == 'subtitles' and cached.is_file():
            try:
                data=json.loads(cached.read_text(encoding='utf-8'))
                if any(e.get('segs') for e in data.get('events', [])):
                    results.append({'video_id':video['video_id'],'returncode':0,'success':True, 'cached':True,
                        'subtitles':[{'file':str(cached),'bytes':cached.stat().st_size,'sha256':hashlib.sha256(cached.read_bytes()).hexdigest()}]})
                    print(video['video_id'], '复用本地字幕', flush=True)
                    continue
            except (ValueError,OSError):
                pass
        log = root/(video['video_id']+'.download.log')
        with log.open('w', encoding='utf-8') as f:
            run = subprocess.run(command(video, root, a.mode, a.system_ca), stdout=f, stderr=subprocess.STDOUT)
        files = list((root/'raw'/video['video_id']).glob('*.json3'))
        valid = []
        for file in files:
            try:
                data = json.loads(file.read_text(encoding='utf-8'))
                if not any(e.get('segs') for e in data.get('events', [])):
                    continue
                valid.append({'file':str(file),'bytes':file.stat().st_size,
                    'sha256':hashlib.sha256(file.read_bytes()).hexdigest()})
            except (ValueError, OSError):
                continue
        ok = run.returncode == 0 and (bool(valid) if a.mode != 'video' else
             any((root/'raw'/video['video_id']).glob('*.mp4')))
        results.append({'video_id':video['video_id'],'returncode':run.returncode,
                        'success':ok,'subtitles':valid,'log':str(log)})
        print(video['video_id'], '成功' if ok else '失败，请查看本地日志', flush=True)
    write_json(root/'downloads.json', results)
    if not all(r['success'] for r in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
