"""导出连续段落校对稿；人工修改后导回新的 JSONL，无需媒体或模型。"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

from core.io import atomic_text, read_rows, write_rows
from core.schema import Segment


def joined(rows):
    text = ''
    for row in rows:
        part = row['text'].strip()
        # Whisper 的英文分段通常需要空格，中文相邻汉字之间不额外插空格。
        cjk = lambda c: '\u3400' <= c <= '\u9fff'
        separator = '' if (not text or cjk(text[-1]) and cjk(part[0])
                           or text[-1] in '，。！？；：、（《【「『'
                           or part[0] in '，。！？；：、）》】」』') else ' '
        text += separator + part
    return text


def paragraphs(rows, target_seconds=30, max_seconds=60, gap_seconds=10):
    if not all(math.isfinite(x) for x in (target_seconds, max_seconds, gap_seconds)):
        raise ValueError('段落参数必须为有限数')
    if not 0 < target_seconds <= max_seconds or gap_seconds < 0:
        raise ValueError('要求 0 < target_seconds <= max_seconds 且 gap_seconds >= 0')
    groups, current = [], []
    for row in rows:
        if row['end'] - row['start'] > max_seconds:
            raise ValueError('单个原始片段超过段落上限；请提高上限或提供更细粒度来源')
        if current:
            start = min(r['start'] for r in current)
            end = max(r['end'] for r in current)
            boundary = (row['video_id'] != current[0]['video_id']
                        or row['start'] < current[-1]['start']
                        or row['start'] - end > gap_seconds
                        or max(end, row['end']) - min(start, row['start']) > max_seconds
                        or end - start >= target_seconds
                        and re.search(r'[。！？.!?][”"\u2019\u201d]*\s*$', current[-1]['text']))
            if boundary:
                groups.append(current)
                current = []
        current.append(row)
    if current:
        groups.append(current)
    return groups


def timestamp(seconds):
    ms = round(seconds * 1000)
    hours, ms = divmod(ms, 3600000)
    minutes, ms = divmod(ms, 60000)
    sec, ms = divmod(ms, 1000)
    return f'{hours:02}:{minutes:02}:{sec:02}.{ms:03}'


def heading(index, rows):
    return (f'## P{index:06} | {timestamp(min(r["start"] for r in rows))}'
            f' → {timestamp(max(r["end"] for r in rows))}'
            f' | {json.dumps(rows[0]["video_id"], ensure_ascii=False)}')


def source_rows(source):
    rows = [Segment.model_validate(r).model_dump() for r in read_rows(source)]
    if not rows:
        raise ValueError('没有可校对的转录')
    return rows


def digest(source):
    return hashlib.sha256(Path(source).read_bytes()).hexdigest()


def export_review(source, output=None, *, target_seconds=30, max_seconds=60, gap_seconds=10):
    source = Path(source)
    output = Path(output) if output is not None else source.with_suffix('.review.md')
    if output.resolve() == source.resolve() or output.exists():
        raise FileExistsError(f'保留已有文件，不覆盖：{output}')
    rows = source_rows(source)
    options = dict(target_seconds=target_seconds, max_seconds=max_seconds, gap_seconds=gap_seconds)
    groups = paragraphs(rows, **options)
    metadata = dict(version=1, source_sha256=digest(source), **options)
    text = '<!-- video-rag-review ' + json.dumps(metadata, sort_keys=True) + ' -->\n'
    text += '# 转录校对稿\n\n只修改各段正文；保留文件头、段落编号、时间及视频标识。\n'
    text += '听不清的地方请标注，不按常识补写。段落仅便于校对，不代表完整话题或检索分块。\n'
    for i, group in enumerate(groups, 1):
        body = joined(group)
        if re.search(r'^## P', body, re.M):
            raise ValueError('转录正文包含保留段落标记 ## P，请先检查原始数据')
        text += '\n' + heading(i, group) + '\n\n' + body + '\n'
    atomic_text(output, text)
    return output


def import_review(source, review, output):
    source, review, output = Path(source), Path(review), Path(output)
    if output.suffix.lower() != '.jsonl':
        raise ValueError('校正版输出请使用 .jsonl 扩展名')
    if output.resolve() in {source.resolve(), review.resolve()} or output.exists():
        raise FileExistsError(f'保留已有文件，不覆盖：{output}')
    text = review.read_text(encoding='utf-8-sig')
    first, _, rest = text.partition('\n')
    if not first.startswith('<!-- video-rag-review ') or not first.endswith(' -->'):
        raise ValueError('缺少校对稿文件头')
    metadata = json.loads(first[len('<!-- video-rag-review '):-len(' -->')])
    if metadata.pop('version') != 1 or metadata.pop('source_sha256') != digest(source):
        raise ValueError('原始转录已改变或文件不匹配，请重新导出并迁移校对内容')
    groups = paragraphs(source_rows(source), **metadata)
    markers = list(re.finditer(r'^## P[^\n]*$', rest, re.M))
    if [m.group() for m in markers] != [heading(i, g) for i, g in enumerate(groups, 1)]:
        raise ValueError('段落编号、顺序、时间或视频标识被修改；请保留全部段落标题')
    rows = []
    for i, (marker, group) in enumerate(zip(markers, groups)):
        end = markers[i + 1].start() if i + 1 < len(markers) else len(rest)
        body = rest[marker.end():end].strip()
        if not body:
            raise ValueError('段落正文不可清空；听不清可标注 [无法识别]')
        if body == joined(group):
            rows.extend(group)  # 未改动段落保留每个原始片段及时间。
        else:
            # 正文可增删、合并句子；不能假装仍有逐句 ASR 对齐。
            rows.append(Segment(video_id=group[0]['video_id'], text=body,
                                start=min(r['start'] for r in group),
                                end=max(r['end'] for r in group),
                                timestamp_quality='estimated').model_dump())
    write_rows(output, rows)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    export = commands.add_parser('export', help='导出单文件，或递归批量导出目录内的 JSONL')
    export.add_argument('--input', required=True)
    export.add_argument('--output')
    export.add_argument('--target-seconds', type=float, default=30)
    export.add_argument('--max-seconds', type=float, default=60)
    export.add_argument('--gap-seconds', type=float, default=10)
    imp = commands.add_parser('import', help='导回新的 JSONL；不覆盖原件')
    imp.add_argument('--source', required=True)
    imp.add_argument('--review', required=True)
    imp.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'import':
        print(import_review(args.source, args.review, args.output))
    else:
        path = Path(args.input)
        if path.is_dir() and args.output:
            parser.error('批量导出时不使用 --output，稿件保存于各转录文件旁')
        sources = sorted(path.rglob('*.jsonl')) if path.is_dir() else [path]
        if not sources:
            parser.error('目录中没有 JSONL')
        for source in sources:
            try:
                print(export_review(source, args.output, target_seconds=args.target_seconds,
                                    max_seconds=args.max_seconds, gap_seconds=args.gap_seconds))
            except FileExistsError as exc:
                print(exc)


if __name__ == '__main__':
    main()
