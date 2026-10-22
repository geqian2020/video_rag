"""完整快照建库：独立 generation 写完后原子发布 CURRENT；失败不影响旧索引。"""
import argparse
import uuid
from pathlib import Path

from filelock import FileLock
from qdrant_client import models

from core.config import load_config
from core.io import atomic_text, read_rows, write_json, write_rows
from core.models import Embedder, fingerprint
from core.schema import Chunk
from core.sparse import fit, vector
from core.store import open_store


def build_index(chunks, cfg, embedder):
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
        generation = uuid.uuid4().hex
        folder = root / generation
        folder.mkdir()
        collection = cfg.storage.collection_prefix + "_" + generation
        sparse_texts = [c["text"] + "\n" + "\n".join(q["question"] for q in c["questions"])
                        for c in chunks]
        stats = fit(sparse_texts, cfg.retrieval.bm25_k1, cfg.retrieval.bm25_b)
        client = open_store(cfg, folder)
        try:
            client.create_collection(collection_name=collection,
                vectors_config={"dense": models.VectorParams(size=embedder.dimension, distance=models.Distance.COSINE)},
                sparse_vectors_config={"bm25": models.SparseVectorParams()})
            if cfg.storage.qdrant_url:
                client.create_payload_index(collection_name=collection, field_name="video_id",
                    field_schema=models.PayloadSchemaType.KEYWORD, wait=True)
            size = cfg.models.batch_size
            for start in range(0, len(chunks), size):
                batch = chunks[start:start + size]
                dense = embedder.encode([c["text"] for c in batch])  # 主向量只含原文
                points = [models.PointStruct(id=c["chunk_id"], payload=c,
                    vector={"dense": v.tolist(), "bm25": vector(sparse_texts[start + i], stats)})
                    for i, (c, v) in enumerate(zip(batch, dense))]
                client.upsert(collection_name=collection, points=points, wait=True)
            if client.count(collection_name=collection, exact=True).count != len(chunks):
                raise RuntimeError("写入条数校验失败")
            write_json(folder / "manifest.json", {"schema_version": 1, "collection": collection,
                "model": fingerprint(cfg), "dimension": embedder.dimension, "count": len(chunks),
                "qdrant_url": cfg.storage.qdrant_url, "bm25": stats})
            write_rows(folder / "chunks.jsonl", chunks)
        except Exception:
            # 只清理本次生成的 collection，绝不删旧库。磁盘失败目录留作本地排查。
            if client.collection_exists(collection):
                client.delete_collection(collection)
            raise
        finally:
            client.close()
        atomic_text(root / "CURRENT", generation + "\n")
    return generation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default="data/enriched.jsonl")
    p.add_argument("--config", default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    generation = build_index(read_rows(a.input), cfg, Embedder(cfg))
    print(f"索引发布成功: {generation}；重启 API 加载新快照")


if __name__ == "__main__":
    main()
