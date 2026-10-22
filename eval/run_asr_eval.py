"""同段音频无词表/有词表对比：CER、标准术语召回与额外术语。小样本评估，不调用模型。"""
import argparse
from collections import defaultdict
import json
import re
import unicodedata

from core.io import read_rows
from core.vocabulary import load_vocabulary


def normalize(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold()
                   if not c.isspace() and not unicodedata.category(c).startswith('P'))


def distance(a, b):
    if len(b) > len(a):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ac in enumerate(a, 1):
        current = [i]
        for j, bc in enumerate(b, 1):
            current.append(min(current[-1]+1, previous[j]+1, previous[j-1]+(ac != bc)))
        previous = current
    return previous[-1]


def contains(text, term):
    text, term = unicodedata.normalize('NFKC', text).casefold(), unicodedata.normalize('NFKC', term).casefold()
    # PA 不应因为出现在 capacity 中而被当作正确术语。
    if re.fullmatch(r'[a-z0-9_+./-]+', term):
        return bool(re.search(r'(?<![a-z0-9])'+re.escape(term)+r'(?![a-z0-9])', text))
    return term in text


def group(rows):
    out = defaultdict(list)
    for row in rows:
        if not row.get('video_id') or not isinstance(row.get('text'), str):
            raise ValueError('评估输入必须有 video_id 和 text')
        out[row['video_id']].append(row['text'])
    return {key: '\n'.join(value) for key, value in out.items()}


def evaluate(reference, hypothesis, vocab):
    refs, preds = group(reference), group(hypothesis)
    if not refs:
        raise ValueError('参考文本不能为空')
    edits = chars = expected_count = recognized_count = 0
    extra = {}
    for video_id, text in refs.items():
        ref, pred = normalize(text), normalize(preds.get(video_id, ''))
        if not ref or max(len(ref), len(pred)) > 5000:
            raise ValueError('每个 video_id 应为 1-5000 字的同一人工校对短片段')
        edits += distance(ref, pred)
        chars += len(ref)
        relevant = {e.term for e in vocab.entries if e.approved
                    and (e.scope == 'global' or e.video_id == video_id)}
        expected = {term for term in relevant if contains(text, term)}
        found = {term for term in relevant if contains(preds.get(video_id, ''), term)}
        expected_count += len(expected)
        recognized_count += len(expected & found)
        extra[video_id] = sorted(found - expected)
    return {'cer': edits / chars, 'term_recall': recognized_count / expected_count if expected_count else None,
            'expected_terms': expected_count, 'recognized_terms': recognized_count,
            'extra_terms_not_in_reference': extra}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', required=True)
    p.add_argument('--baseline', required=True)
    p.add_argument('--assisted', required=True)
    p.add_argument('--vocabulary', required=True)
    a = p.parse_args()
    reference, vocab = read_rows(a.reference), load_vocabulary(a.vocabulary)
    print(json.dumps({'baseline': evaluate(reference, read_rows(a.baseline), vocab),
                      'assisted': evaluate(reference, read_rows(a.assisted), vocab)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
