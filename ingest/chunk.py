"""句子向量相似度 + 时间窗口分块，保留原始 ASR 时间范围而不插值伪造对齐。"""
import argparse
import re
import sys
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from core.config import load_config
from core.io import read_rows, write_rows
from core.models import Embedder
from core.schema import Chunk, Segment
from ingest.embedding_cache import EmbeddingCache, model_signature


def progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [CHUNK] {message}",
          file=sys.stderr, flush=True)


@contextmanager
def stage(label):
    """A heartbeat is elapsed time, not a claim that work has advanced."""
    started = time.monotonic()
    stopped = threading.Event()

    def heartbeat():
        while not stopped.wait(15):
            progress(f"{label}: still running, elapsed={time.monotonic() - started:.0f}s")

    progress(f"{label}: start")
    worker = threading.Thread(target=heartbeat, daemon=True)
    worker.start()
    try:
        yield
    except BaseException:
        progress(f"{label}: interrupted/failed, elapsed={time.monotonic() - started:.1f}s")
        raise
    else:
        progress(f"{label}: done, elapsed={time.monotonic() - started:.1f}s")
    finally:
        stopped.set()
        worker.join()


def sentence_units(segment):
    # 同一 ASR segment 内拆出的句子继承整个 segment 的时间范围。
    # source_segments 在此记录原文的句子级摘录，时间仍为来源 segment 的包络。
    parts = re.findall(r"[^。！？!?\n]+[。！？!?]?|[。！？!?]", segment.text)
    return [segment.model_copy(update={"text": p}) for p in parts if p.strip()]


def build_chunks(rows, embedder, cfg, embedding_cache=None):
    started = time.monotonic()
    groups = defaultdict(list)
    input_count = 0
    for input_count, row in enumerate(rows, 1):
        seg = Segment.model_validate(row)
        if seg.text.strip() == "[无法识别]":
            continue
        groups[seg.video_id].extend(sentence_units(seg))
        if input_count % 1000 == 0:
            progress(f"Grouped input segments={input_count}")
    total_units = sum(len(units) for units in groups.values())
    progress(f"Input segments={input_count}, videos={len(groups)}, sentence units={total_units}")
    result = []
    completed_units = 0
    for video_number, (video_id, units) in enumerate(sorted(groups.items()), 1):
        label = f"Video {video_number}/{len(groups)} id={video_id!r}"
        progress(f"{label}: sentences={len(units)}, validation start")
        before = len(result)
        units.sort(key=lambda s: (s.start, s.end))
        for checked, s in enumerate(units, 1):
            if s.end - s.start > cfg.chunking.max_seconds:
                raise ValueError("单条输入时间窗超过 max_seconds，请提供更细粒度转录")
            if embedder.token_count(s.text) > cfg.models.max_tokens:
                raise ValueError("单句超过模型 token 上限，请先人工或按 ASR 细分")
            if checked % 1000 == 0:
                progress(f"{label}: validated={checked}/{len(units)}")
        with stage(f"{label}: embedding {len(units)} sentences"):
            texts = [s.text for s in units]
            if embedding_cache is None:
                vectors = embedder.encode(texts)
            else:
                vectors = embedding_cache.encode(texts, embedder, cfg.models.batch_size)
        progress(f"{label}: assembling chunks")
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
            if (i + 1) % 1000 == 0:
                progress(f"{label}: assembled={i + 1}/{len(units)}, chunks={len(result) - before}")
        emit()
        completed_units += len(units)
        progress(f"{label}: done, chunks={len(result) - before}, "
                 f"completed sentences={completed_units}/{total_units}, "
                 f"total chunks={len(result)}, elapsed={time.monotonic() - started:.1f}s")
    # 完全重复的输入不能静默制造重复点。
    ids = [c["chunk_id"] for c in result]
    if len(set(ids)) != len(ids):
        raise ValueError("出现重复 chunk_id，请先去除重复转录")
    if not result:
        raise ValueError("没有可用转录")
    progress(f"Chunk validation passed: chunks={len(result)}")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="data/chunks.jsonl")
    p.add_argument("--config", default=None)
    p.add_argument("--embedding-cache", default=None,
                   help="Persistent SQLite vector checkpoints; default: beside output")
    p.add_argument("--no-embedding-cache", action="store_true",
                   help="Explicitly disable embedding checkpoints")
    a = p.parse_args()
    cfg = load_config(a.config)
    with stage(f"Read input {a.input}"):
        source_rows = read_rows(a.input)
    progress(f"Loaded input segments={len(source_rows)}")
    with stage(f"Load embedding model={cfg.models.embedding}, device={cfg.models.device}"):
        embedder = Embedder(cfg)
    if a.no_embedding_cache:
        embedder.show_progress = True
        rows = build_chunks(source_rows, embedder, cfg)
    else:
        # Our saved-batch counters replace an inner progress bar per tiny batch.
        embedder.show_progress = False
        cache_path = a.embedding_cache or Path(a.output).with_name("chunk_embeddings.sqlite3")
        with stage("Resolve embedding cache model identity"):
            signature = model_signature(cfg, embedder)
        with EmbeddingCache(cache_path, signature, progress) as cache:
            rows = build_chunks(source_rows, embedder, cfg, embedding_cache=cache)
    with stage(f"Write output {a.output}"):
        write_rows(a.output, rows)
    progress(f"Complete: chunks={len(rows)}, output={a.output}")
    print(f"已生成 {len(rows)} 个块", flush=True)


if __name__ == "__main__":
    main()

