"""句子向量相似度 + 时间窗口分块，保留原始 ASR 时间范围而不插值伪造对齐。"""
import argparse
import re
import uuid
from collections import defaultdict

import numpy as np

from core.config import load_config
from core.io import read_rows, write_rows
from core.models import Embedder
from core.schema import Chunk, Segment


def sentence_units(segment):
    # 同一 ASR segment 内拆出的句子继承整个 segment 的时间范围。
    # source_segments 在此记录原文的句子级摘录，时间仍为来源 segment 的包络。
    parts = re.findall(r"[^。！？!?\n]+[。！？!?]?|[。！？!?]", segment.text)
    return [segment.model_copy(update={"text": p}) for p in parts if p.strip()]


def build_chunks(rows, embedder, cfg):
    groups = defaultdict(list)
    for row in rows:
        seg = Segment.model_validate(row)
        if seg.text.strip() == "[无法识别]":
            continue
        groups[seg.video_id].extend(sentence_units(seg))
    result = []
    for video_id, units in sorted(groups.items()):
        units.sort(key=lambda s: (s.start, s.end))
        for s in units:
            if s.end - s.start > cfg.chunking.max_seconds:
                raise ValueError("单条输入时间窗超过 max_seconds，请提供更细粒度转录")
            if embedder.token_count(s.text) > cfg.models.max_tokens:
                raise ValueError("单句超过模型 token 上限，请先人工或按 ASR 细分")
        vectors = embedder.encode([s.text for s in units])
        current = []
        indices = []

        def emit():
            if not current:
                return
            text = "\n".join(s.text for s in current)
            start, end = min(s.start for s in current), max(s.end for s in current)
            key = f"{video_id}\0{start}\0{end}\0{text}"
            chunk = Chunk(chunk_id=str(uuid.uuid5(uuid.NAMESPACE_URL, key)), video_id=video_id,
                          start_ts=start, end_ts=end, text=text, source_segments=list(current),
                          timestamp_quality="estimated" if any(s.timestamp_quality == "estimated"
                              for s in current) else "asr")
            result.append(chunk.model_dump())

        for i, unit in enumerate(units):
            if current:
                end = max(s.end for s in current)
                duration = end - current[0].start
                centroid = np.mean(vectors[indices[-3:]], axis=0)
                similarity = float(np.dot(centroid, vectors[i]) / max(np.linalg.norm(centroid), 1e-12))
                hard_boundary = (max(end, unit.end) - current[0].start > cfg.chunking.max_seconds
                    or unit.start - end > cfg.chunking.gap_seconds
                    or embedder.token_count("\n".join([s.text for s in current] + [unit.text]))
                        > cfg.models.max_tokens)
                semantic_boundary = duration >= cfg.chunking.min_seconds and similarity < cfg.chunking.similarity_threshold
                if hard_boundary or semantic_boundary:
                    emit()
                    current, indices = [], []
            current.append(unit)
            indices.append(i)
        emit()
    # 完全重复的输入不能静默制造重复点。
    ids = [c["chunk_id"] for c in result]
    if len(set(ids)) != len(ids):
        raise ValueError("出现重复 chunk_id，请先去除重复转录")
    if not result:
        raise ValueError("没有可用转录")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="data/chunks.jsonl")
    p.add_argument("--config", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    rows = build_chunks(read_rows(a.input), Embedder(cfg), cfg)
    write_rows(a.output, rows)
    print(f"已生成 {len(rows)} 个块")


if __name__ == "__main__":
    main()
