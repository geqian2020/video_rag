"""Cross-encoder 对 query/原文成对打分；失败直接报错，不静默取消重排。"""
import numpy as np


class Reranker:
    def __init__(self, cfg):
        from sentence_transformers import CrossEncoder
        self.cfg = cfg
        self.model = CrossEncoder(cfg.models.reranker, device=cfg.models.device,
            revision=cfg.models.reranker_revision, local_files_only=cfg.models.offline,
            trust_remote_code=False, max_length=cfg.models.rerank_tokens)

    def rerank(self, query, candidates, top_k):
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        # 显式检查，避免长 query/文本静默截断导致时间引用对应错误。
        for q, text in pairs:
            encoded = self.model.tokenizer(q, text, truncation=False)
            if len(encoded["input_ids"]) > self.cfg.models.rerank_tokens:
                raise ValueError("query + chunk 超过 reranker token 上限，请缩短问题或分块")
        scores = np.asarray(self.model.predict(pairs, batch_size=self.cfg.models.batch_size,
                            show_progress_bar=False)).reshape(-1)
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise RuntimeError("reranker 返回非法分数")
        ranked = [{**c, "rerank_score": float(s)} for c, s in zip(candidates, scores)]
        return sorted(ranked, key=lambda c: (-c["rerank_score"], c["chunk_id"]))[:top_k]
