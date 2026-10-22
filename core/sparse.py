"""jieba + 标准 BM25；TF 长度归一化存在文档向量，IDF 存在查询向量。
两者在 Qdrant 的 sparse dot product 即 BM25，不使用模型推断的稀疏词扩展。
"""
import math
import re
from collections import Counter

import jieba
from qdrant_client import models

TOKENIZER_VERSION = "jieba-0.42.1-default-hmm-false-v1"


def tokenize(text):
    return [x.lower() for x in jieba.cut(text, HMM=False) if re.search(r"[\w\u4e00-\u9fff]", x)]


def fit(texts, k1=1.5, b=0.75):
    documents = [tokenize(t) for t in texts]
    df = Counter()
    for tokens in documents:
        df.update(set(tokens))
    vocab = {t: i for i, t in enumerate(sorted(df))}
    n = len(documents)
    return {"vocab": vocab, "idf": {t: math.log(1 + (n - f + .5) / (f + .5))
            for t, f in df.items()}, "avgdl": sum(map(len, documents)) / max(n, 1) or 1,
            "k1": k1, "b": b, "tokenizer": TOKENIZER_VERSION}


def vector(text, stats, query=False):
    counts = Counter(tokenize(text))
    items = []
    dl = sum(counts.values())
    for word, count in counts.items():
        idx = stats["vocab"].get(word)
        if idx is None:
            continue
        k1, b = stats["k1"], stats["b"]
        score = (stats["idf"][word] if query else
                 count * (k1 + 1) / (count + k1 * (1 - b + b * dl / stats["avgdl"])))
        items.append((idx, score))
    items.sort()
    return models.SparseVector(indices=[i for i, _ in items], values=[v for _, v in items])
