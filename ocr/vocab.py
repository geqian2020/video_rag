"""候选词 CSV -> 人工确认 -> 可累积的 JSON 词汇库；不会把 OCR 结果自动认证。"""
import argparse
import csv
from difflib import SequenceMatcher
import io
import json
from pathlib import Path
import re
import unicodedata
import uuid

import jieba.posseg as pseg
from pydantic import Field

from core.config import Strict, load_config
from core.io import atomic_text, read_rows, write_json
from core.vocabulary import Term, Vocabulary, load_vocabulary


class OCRLine(Strict):
    video_id: str = Field(min_length=1)
    timestamp: float = Field(ge=0, allow_inf_nan=False)
    frame_path: str
    text: str
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    box: list[list[float]] = []


def normalize_key(term):
    return unicodedata.normalize('NFKC', term).casefold().strip()


def confusable_key(term):
    return normalize_key(term).translate(str.maketrans({'i':'1','l':'1','o':'0'}))


def extract_terms(text):
    value = unicodedata.normalize('NFKC', text)
    # 英文缩写/产品词保留大小写、数字与连字符。不把 O/0、I/l 等混淆项自动改写。
    out = set(re.findall(r'(?<![A-Za-z0-9])(?=[A-Za-z0-9_+./-]*[A-Za-z])[A-Za-z0-9]+(?:[-_+./][A-Za-z0-9]+)*', value))
    for word, flag in pseg.cut(value, HMM=False):
        if flag.startswith('n') and re.fullmatch(r'[\u4e00-\u9fff]{2,16}', word):
            out.add(word)
    # 短中文标题完整保留供人工检查，避免分词器把新词拆散；候选不代表专有名词。
    for phrase in re.findall(r'[\u4e00-\u9fff]{2,20}', value):
        out.add(phrase)
    return sorted(t for t in out if 2 <= len(t) <= 64)


def candidates(rows, cfg, vocabulary=None):
    groups = {}
    for raw in rows:
        row = OCRLine.model_validate(raw)
        if row.confidence < cfg.ocr.min_confidence:
            continue
        for term in extract_terms(row.text):
            key = (row.video_id, normalize_key(term))
            group = groups.setdefault(key, {'term': term, 'video_id': row.video_id,
                'timestamps': set(), 'frames': set(), 'confidence': 0, 'evidence': []})
            group['timestamps'].add(row.timestamp)
            group['frames'].add((row.timestamp, row.frame_path))
            group['confidence'] = max(group['confidence'], row.confidence)
            if len(group['evidence']) < 8:
                group['evidence'].append(row.model_dump())
    output = []
    for (video_id, key), group in sorted(groups.items()):
        output.append({'candidate_id': str(uuid.uuid5(uuid.NAMESPACE_URL, video_id+'\0'+key)),
            'ocr_term': group['term'], 'canonical_term': group['term'], 'approved': '',
            'scope': 'video', 'video_id': video_id, 'priority': 0,
            'frame_count': len(group['frames']), 'max_confidence': round(group['confidence'], 4),
            'evidence_strength': 'repeated' if len(group['frames']) >= cfg.ocr.min_frames else 'single_or_rare',
            'similar_candidates': '', 'timestamps_json': json.dumps(sorted(group['timestamps'])),
            'evidence_json': json.dumps(group['evidence'], ensure_ascii=False)})
    # 相近词仅提示审核，不合并；仅比较英文混淆项，避免海量中文短语两两比较。
    buckets = {}
    for row in output:
        if re.fullmatch(r'[A-Za-z0-9_+./-]+', row['ocr_term']):
            key = (row['video_id'], len(row['ocr_term']))
            buckets.setdefault(key, []).append(row['ocr_term'])
    for row in output:
        term = row['ocr_term']
        peers = buckets.get((row['video_id'], len(term)), [])
        row['similar_candidates'] = ' | '.join(p for p in peers[:500]
            if p != term and SequenceMatcher(None, term.casefold(), p.casefold()).ratio() >= .75)[:500]
    for row in output:
        term=row['ocr_term'];ascii_term=bool(re.fullmatch(r'[A-Za-z0-9_+./-]+',term))
        row['confusable_group']=(row['video_id']+':'+confusable_key(term)) if ascii_term else ''
        confirmed=[e.term for e in (vocabulary.entries if vocabulary else [])
                   if e.approved and (e.scope=='global' or e.video_id==row['video_id'])
                   and confusable_key(e.term)==confusable_key(term)] if ascii_term else []
        row['confirmed_spelling_suggestions']=' | '.join(sorted(set(confirmed)))
    return output


def export_csv(path, rows):
    if not rows:
        raise ValueError('没有候选词；先检查 OCR 原文、裁剪区域和置信度设置')
    if Path(path).exists():
        raise FileExistsError('候选 CSV 已存在；请换输出文件名，避免覆盖已审核内容')
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    # Excel 兼容 UTF-8 BOM。
    atomic_text(path, '\ufeff' + buffer.getvalue())


def import_review(path, previous=None):
    vocab = load_vocabulary(previous)
    entries = {(e.scope, e.video_id, normalize_key(e.term)): e for e in vocab.entries}
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            approval = (row.get('approved') or '').strip().lower()
            if approval in {'', 'false', 'no', '0'}:
                continue
            if approval not in {'true', 'yes', '1'}:
                raise ValueError('approved 只能填写 true/yes/1 或留空/false/no/0')
            evidence = json.loads(row.get('evidence_json') or '[]')
            # 对证据沿用严格格式检查，但不假称确认拼写必须和错误 OCR 相同。
            evidence = [OCRLine.model_validate(e).model_dump() for e in evidence]
            term = Term(term=row['canonical_term'], approved=True, scope=row.get('scope') or 'video',
                video_id=(row.get('video_id') or None), priority=int(row.get('priority') or 0),
                timestamps=json.loads(row.get('timestamps_json') or '[]'), evidence=evidence)
            key = (term.scope, term.video_id, normalize_key(term.term))
            if key in entries:
                old = entries[key]
                term.timestamps = sorted(set(old.timestamps + term.timestamps))
                term.evidence = (old.evidence + term.evidence)[-20:]
                term.aliases = old.aliases
            entries[key] = term
    return Vocabulary(entries=list(entries.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    subs = p.add_subparsers(dest='command', required=True)
    make = subs.add_parser('candidates')
    make.add_argument('--input', required=True, action='append', help='OCR observations.jsonl；可重复指定')
    make.add_argument('--output', default='data/vocab_candidates.csv')
    make.add_argument('--config', default=None)
    make.add_argument('--vocabulary',default=None,help='既有确认词库，仅提供混淆拼写建议')
    approve = subs.add_parser('approve')
    approve.add_argument('--review', required=True)
    approve.add_argument('--merge', default=None, help='合并已有词汇库，保留已确认词条')
    approve.add_argument('--output', default='data/vocabulary.json')
    a = p.parse_args()
    if a.command == 'candidates':
        rows = candidates([r for file in a.input for r in read_rows(file)], load_config(a.config),load_vocabulary(a.vocabulary))
        export_csv(a.output, rows)
        print(f'已输出 {len(rows)} 条待审核候选；尚无条目被确认')
    else:
        vocab = import_review(a.review, a.merge)
        write_json(a.output, vocab.model_dump())
        print(f'词汇库中已确认 {sum(e.approved for e in vocab.entries)} 条')


if __name__ == '__main__':
    main()
