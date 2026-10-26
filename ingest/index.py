"""可恢复的完整快照建库；每批保存向量及 Qdrant 校验记录，最后原子发布。"""
import argparse
import hashlib
import json
import re
import sys
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock
import numpy as np
from qdrant_client import models

from core.config import load_config
from core.io import atomic_text, read_rows, write_json, write_rows
from core.models import Embedder, fingerprint
from core.schema import Chunk
from core.sparse import TOKENIZER_VERSION, fit, vector
from core.store import open_store
from ingest.embedding_cache import EmbeddingCache, model_signature


def progress(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INDEX] {message}",
          file=sys.stderr, flush=True)


@contextmanager
def stage(label):
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
    finally:
        stopped.set()
        worker.join()
    progress(f"{label}: done, elapsed={time.monotonic() - started:.1f}s")


def digest(value):
    def convert(obj):
        return obj.model_dump()
    raw = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                     separators=(",", ":"), default=convert).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def read_checkpoint(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def record_hash(record):
    if not isinstance(record.vector, dict) or not {"dense", "bm25"} <= record.vector.keys():
        raise ValueError("Qdrant record is missing its dense or BM25 vector")
    # Local Qdrant may return normalized vectors before close and the persisted
    # source representation after reopen. Cosine direction is the invariant;
    # tolerate float32 rounding rather than rewriting every healthy batch.
    dense = np.asarray(record.vector["dense"], dtype=np.float64)
    if dense.ndim != 1 or not np.isfinite(dense).all():
        raise ValueError("Invalid Qdrant dense vector")
    norm = np.linalg.norm(dense)
    dense = np.round(dense / norm if norm else dense, 6)
    dense[dense == 0] = 0.0
    sparse = record.vector["bm25"]
    sparse = sparse.model_dump() if hasattr(sparse, "model_dump") else sparse
    vectors = {"dense": dense.tolist(), "bm25": sparse}
    return digest({"id": str(record.id), "payload": record.payload, "vectors": vectors})


def verified_batch(client, collection, batch, marker):
    saved = read_checkpoint(marker)
    if (not saved or saved.get("input") != digest(batch)
            or not isinstance(saved.get("records"), dict)):
        return False
    records = client.retrieve(collection_name=collection,
                              ids=[c["chunk_id"] for c in batch],
                              with_payload=True, with_vectors=True)
    if len(records) != len(batch):
        return False
    expected = {c["chunk_id"]: c for c in batch}
    if {str(record.id) for record in records} != expected.keys():
        return False
    try:
        return all(record.payload == expected.get(str(record.id))
                   and saved.get("records", {}).get(str(record.id)) == record_hash(record)
                   for record in records)
    except (ValueError, TypeError):
        return False


def save_batch(client, collection, batch, marker):
    records = client.retrieve(collection_name=collection,
                              ids=[c["chunk_id"] for c in batch],
                              with_payload=True, with_vectors=True)
    expected = {c["chunk_id"]: c for c in batch}
    if (len(records) != len(batch) or {str(record.id) for record in records} != expected.keys()
            or any(record.payload != expected.get(str(record.id)) for record in records)):
        raise RuntimeError("Qdrant batch read-back validation failed; checkpoint not saved")
    write_json(marker, {"version": 1, "input": digest(batch),
                        "records": {str(record.id): record_hash(record) for record in records}})


def build_index(chunks, cfg, embedder, model_identity=None):
    started = time.monotonic()
    progress(f"Validate input: blocks={len(chunks)}, root={cfg.storage.root}")
    chunks = [Chunk.model_validate(c).model_dump() for c in chunks]
    if not chunks or any(c["enrichment_status"] == "pending" for c in chunks):
        raise ValueError("需要非空且已完成增强的 chunks")
    ids = [c["chunk_id"] for c in chunks]
    if len(set(ids)) != len(ids):
        raise ValueError("重复 chunk_id")
    # Qdrant 只接受 UUID 或整数点 ID。
    for cid in ids:
        uuid.UUID(cid)
    root = Path(cfg.storage.root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "BUILD.lock"), timeout=0):
        with stage("Resolve model identity"):
            if model_identity is None:
                if isinstance(embedder, Embedder):
                    model_identity = model_signature(cfg, embedder)
                else:
                    # Custom adapters must provide a stable, truthful identity to
                    # opt into reuse. Unknown adapters remain isolated per call.
                    model_identity = {"dimension": embedder.dimension,
                                      "unverified_adapter": uuid.uuid4().hex}
        if model_identity.get("dimension") != embedder.dimension:
            raise ValueError("Checkpoint identity dimension differs from embedder")
        identity = {"format": 1, "chunks": digest(chunks), "model": fingerprint(cfg),
                    "embedding_identity": model_identity, "dimension": embedder.dimension,
                    "batch_size": cfg.models.batch_size,
                    "bm25": [TOKENIZER_VERSION, cfg.retrieval.bm25_k1, cfg.retrieval.bm25_b],
                    "qdrant_url": cfg.storage.qdrant_url,
                    "collection_prefix": cfg.storage.collection_prefix}
        build_key = digest(identity)
        registry = root / "builds" / (build_key + ".json")
        state = read_checkpoint(registry)
        current_file = root / "CURRENT"
        current = current_file.read_text(encoding="utf-8").strip() if current_file.is_file() else None
        generation = state.get("generation") if state else None
        reuse = (state is not None and state.get("identity") == identity
                 and isinstance(generation, str) and re.fullmatch(r"[0-9a-f]{32}", generation)
                 and generation != current)
        if not reuse:
            generation = uuid.uuid4().hex
        folder = root / generation
        folder.mkdir(exist_ok=True)
        collection = cfg.storage.collection_prefix + "_" + generation
        progress(f"{'Resume' if reuse else 'New'} unpublished snapshot: generation={generation}, blocks={len(chunks)}")
        write_json(registry, {"version": 1, "generation": generation,
                              "identity": identity, "status": "building"})
        sparse_texts = [c["text"] + "\n" + "\n".join(q["question"] for q in c["questions"])
                        for c in chunks]
        with stage("Fit BM25 statistics"):
            stats = fit(sparse_texts, cfg.retrieval.bm25_k1, cfg.retrieval.bm25_b)
        progress(f"Open Qdrant: collection={collection}")
        client = open_store(cfg, folder)
        try:
            if not client.collection_exists(collection):
                client.create_collection(collection_name=collection,
                    vectors_config={"dense": models.VectorParams(size=embedder.dimension, distance=models.Distance.COSINE)},
                    sparse_vectors_config={"bm25": models.SparseVectorParams()})
            if cfg.storage.qdrant_url:
                client.create_payload_index(collection_name=collection, field_name="video_id",
                    field_schema=models.PayloadSchemaType.KEYWORD, wait=True)
            size = cfg.models.batch_size
            total_batches = (len(chunks) + size - 1) // size
            reused_blocks = 0
            cache_identity = {**model_identity, "configuration": fingerprint(cfg),
                              "purpose": "index-block-original"}
            with EmbeddingCache(root / "index_embeddings.sqlite3", cache_identity, progress) as cache:
                for number, start in enumerate(range(0, len(chunks), size), 1):
                    batch = chunks[start:start + size]
                    end = start + len(batch)
                    label = f"Batch {number}/{total_batches}, blocks={start + 1}-{end}/{len(chunks)}"
                    marker = folder / "checkpoints" / f"batch-{number:08d}.json"
                    with stage(f"{label}: verify saved batch"):
                        reusable = verified_batch(client, collection, batch, marker)
                    if reusable:
                        reused_blocks += len(batch)
                        progress(f"{label}: verified reuse, completed={end / len(chunks):.1%}")
                        continue
                    with stage(f"{label}: encode/load saved vectors"):
                        dense = cache.encode([c["text"] for c in batch], embedder, size)
                    points = [models.PointStruct(id=c["chunk_id"], payload=c,
                        vector={"dense": v.tolist(), "bm25": vector(sparse_texts[start + i], stats)})
                        for i, (c, v) in enumerate(zip(batch, dense))]
                    with stage(f"{label}: write Qdrant and verify"):
                        client.upsert(collection_name=collection, points=points, wait=True)
                        save_batch(client, collection, batch, marker)
                    progress(f"{label}: saved, completed={end / len(chunks):.1%}, "
                             f"reused blocks={reused_blocks}, elapsed={time.monotonic() - started:.1f}s")
            progress("Validate final point count")
            if client.count(collection_name=collection, exact=True).count != len(chunks):
                raise RuntimeError("写入条数校验失败")
            with stage("Write snapshot manifest and source chunks"):
                write_json(folder / "manifest.json", {"schema_version": 1, "collection": collection,
                    "model": fingerprint(cfg), "dimension": embedder.dimension, "count": len(chunks),
                    "qdrant_url": cfg.storage.qdrant_url, "bm25": stats,
                    "embedding_identity": model_identity, "build_signature": build_key})
                write_rows(folder / "chunks.jsonl", chunks)
        except BaseException:
            progress(f"Build interrupted/failed; retaining snapshot={generation} and vector checkpoints. "
                     "Rerun with the same input/config to resume; CURRENT was not changed.")
            raise
        finally:
            client.close()
        write_json(registry, {"version": 1, "generation": generation,
                              "identity": identity, "status": "ready"})
        with stage("Publish CURRENT"):
            atomic_text(root / "CURRENT", generation + "\n")
        progress(f"Complete: generation={generation}, blocks={len(chunks)}, elapsed={time.monotonic() - started:.1f}s")
    return generation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/enriched.jsonl")
    p.add_argument("--config", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    with stage(f"Read input {a.input}"):
        chunks = read_rows(a.input)
    with stage(f"Load embedding model={cfg.models.embedding}, device={cfg.models.device}"):
        embedder = Embedder(cfg)
    generation = build_index(chunks, cfg, embedder)
    print(f"索引发布成功: {generation}；重启 API 加载新快照", flush=True)


if __name__ == "__main__":
    main()

