"""Dense top-30 + BM25 top-30 -> RRF top-30 -> cross-encoder top-5。"""
import argparse
import json
import re
from pathlib import Path
from threading import Lock

from qdrant_client import models

from core.config import load_config
from core.models import Embedder, fingerprint
from core.sparse import TOKENIZER_VERSION, vector
from core.store import open_store
from retrieve.rerank import Reranker


def rrf(rankings, k=60, limit=30):
    merged = {}
    for ranking in rankings:
        seen = set()
        for rank, point in enumerate(ranking, 1):
            cid = str(point.id)
            if cid in seen:
                continue
            seen.add(cid)
            if cid not in merged:
                merged[cid] = {**point.payload, "rrf_score": 0.0}
            merged[cid]["rrf_score"] += 1 / (k + rank)
    return sorted(merged.values(), key=lambda p: (-p["rrf_score"], p["chunk_id"]))[:limit]


class Searcher:
    def __init__(self, cfg, embedder=None, reranker=None):
        self.cfg = cfg
        self.lock = Lock()  # embedded Qdrant 与模型单实例串行，避免并发内存峰值
        root = Path(cfg.storage.root)
        generation = (root / "CURRENT").read_text(encoding="utf-8").strip()
        if not re.fullmatch(r"[0-9a-f]{32}", generation):
            raise ValueError("CURRENT 非法")
        folder = root / generation
        self.manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest["schema_version"] != 1 or self.manifest["model"] != fingerprint(cfg):
            raise ValueError("模型/索引配置不一致，请恢复原配置或重新建库")
        if self.manifest["qdrant_url"] != cfg.storage.qdrant_url:
            raise ValueError("Qdrant 地址与索引不一致")
        if self.manifest["bm25"]["tokenizer"] != TOKENIZER_VERSION:
            raise ValueError("分词器版本变化，请重建索引")
        self.embedder = embedder or Embedder(cfg)
        self.reranker = reranker or Reranker(cfg)
        if self.embedder.dimension != self.manifest["dimension"]:
            raise ValueError("向量维度不一致")
        self.client = open_store(cfg, folder)
        self.collection = self.manifest["collection"]
        if not self.client.collection_exists(self.collection):
            self.client.close()
            raise ValueError("索引 collection 不存在")

    def close(self):
        self.client.close()

    def _recall(self, query, video_ids=None):
        filt = (models.Filter(must=[models.FieldCondition(key="video_id",
                match=models.MatchAny(any=video_ids))]) if video_ids is not None else None)
        dense = self.client.query_points(collection_name=self.collection, using="dense",
            query=self.embedder.encode([query], query=True)[0].tolist(), query_filter=filt,
            limit=self.cfg.retrieval.recall_k, with_payload=True).points
        sparse_vector = vector(query, self.manifest["bm25"], query=True)
        sparse = []
        if sparse_vector.indices:
            sparse = self.client.query_points(collection_name=self.collection, using="bm25",
                query=sparse_vector, query_filter=filt, limit=self.cfg.retrieval.recall_k,
                with_payload=True).points
        return rrf([dense, sparse], self.cfg.retrieval.rrf_k, self.cfg.retrieval.recall_k)

    def search(self, query, top_k=None, video_ids=None, include_candidates=False):
        if not query.strip() or len(query) > 2000:
            raise ValueError("query 长度必须为 1-2000 字符")
        top_k = self.cfg.retrieval.top_k if top_k is None else top_k
        if not 1 <= top_k <= self.cfg.retrieval.recall_k:
            raise ValueError("top_k 超出范围")
        if video_ids is not None and not video_ids:
            return {"results": [], "candidates": []} if include_candidates else []
        with self.lock:
            candidates = self._recall(query, video_ids)
            results = self.reranker.rerank(query, candidates, top_k)
        if include_candidates:
            return {"results": results, "candidates": candidates}
        return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--video-id", action="append", default=None)
    p.add_argument("--config", default=None)
    a = p.parse_args()
    searcher = Searcher(load_config(a.config))
    try:
        print(json.dumps(searcher.search(a.query, a.top_k, a.video_id), ensure_ascii=False, indent=2))
    finally:
        searcher.close()


if __name__ == "__main__":
    main()
