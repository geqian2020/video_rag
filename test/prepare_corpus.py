"""将本地 JSON3 字幕转成严格 JSONL；字幕测试不等同于 OCR/ASR 测试。"""
import argparse
import hashlib
import html
import json
from pathlib import Path
import re

from core.io import write_json, write_rows
from core.schema import Segment
from test.download import load_manifest


def convert_json3(data, video_id, duration):
    lines = []
    for event in data.get('events', []):
        text = html.unescape(''.join(s.get('utf8', '') for s in event.get('segs', [])))
        text = re.sub(r'\s+', ' ', text).strip()
        if text:
            lines.append((float(event['tStartMs'])/1000, event.get('dDurationMs'), text))
    lines.sort(key=lambda x:x[0])
    # 同一时刻的多个字幕事件合并；不同时间的相同话语不自动删去。
    grouped = []
    for start, length, text in lines:
        if grouped and start == grouped[-1][0]:
            prev = grouped.pop()
            grouped.append((start, max(float(prev[1] or 0), float(length or 0)), prev[2]+' '+text))
        else:
            grouped.append((start, length, text))
    rows = []
    for i,(start,length,text) in enumerate(grouped):
        next_start = grouped[i+1][0] if i+1 < len(grouped) else duration
        end = min(duration, next_start, start+float(length)/1000 if length else next_start)
        if end <= start:
            continue
        rows.append(Segment(video_id=video_id, text=text, start=start, end=end,
                            timestamp_quality='estimated').model_dump())
    if not rows:
        raise ValueError('字幕未解析出有效片段')
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', default='test/videos.json')
    p.add_argument('--root', default='data/public_benchmark')
    p.add_argument('--allow-missing', action='store_true', help='显式允许部分语料试跑，报告记录缺失来源')
    a = p.parse_args();root=Path(a.root)
    all_rows, summary, missing = [], [], []
    for v in load_manifest(a.manifest):
        raw = root/'raw'/v['video_id']/(v['video_id']+'.en.json3')
        if not raw.is_file():
            if not a.allow_missing:
                raise FileNotFoundError('缺少字幕，请先下载；部分试跑需显式 --allow-missing')
            missing.append(v['video_id'])
            continue
        rows = convert_json3(json.loads(raw.read_text(encoding='utf-8')),v['video_id'],v['duration_seconds'])
        write_rows(root/'segments'/(v['video_id']+'.jsonl'), rows)
        summary.append({'video_id':v['video_id'],'duration_seconds':v['duration_seconds'],
            'segments':len(rows),'characters':sum(len(r['text']) for r in rows),
            'subtitle_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()})
        all_rows.extend(rows)
    write_rows(root/'transcripts.jsonl', all_rows)
    write_json(root/'corpus_stats.json', {'videos':summary,'total_segments':len(all_rows),
        'total_duration_seconds':sum(v['duration_seconds'] for v in summary),
        'missing_video_ids':missing,'complete_manifest':not missing,
        'source':'public English captions; timestamp_quality=estimated'})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
