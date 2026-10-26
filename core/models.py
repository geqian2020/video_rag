"""所有推理仅从本机模型/缓存加载，不启用远程自定义代码。"""
import numpy as np


class Embedder:
    def __init__(self, cfg):
        from sentence_transformers import SentenceTransformer
        self.cfg = cfg
        # CLI ingestion enables this; retrieval stays quiet by default.
        self.show_progress = False
        self.model = SentenceTransformer(
            cfg.models.embedding, device=cfg.models.device,
            revision=cfg.models.embedding_revision,
            local_files_only=cfg.models.offline, trust_remote_code=False,
        )
        if cfg.models.max_tokens > self.model.max_seq_length:
            raise ValueError("max_tokens 超过模型声明的上限")
        self.model.max_seq_length = cfg.models.max_tokens
        self.dimension = self.model.get_sentence_embedding_dimension()

    def token_count(self, text):
        return len(self.model.tokenizer.encode(text, add_special_tokens=True))

    def encode(self, texts, query=False):
        kwargs = {}
        if query and self.cfg.models.query_prompt_name:
            kwargs["prompt_name"] = self.cfg.models.query_prompt_name
        prompt = self.model.prompts.get(kwargs.get("prompt_name"), "")
        if any(self.token_count(prompt + t) > self.cfg.models.max_tokens for t in texts):
            raise ValueError("文本超过 embedding token 上限，拒绝静默截断")
        return np.asarray(self.model.encode(
            texts, normalize_embeddings=True, batch_size=self.cfg.models.batch_size,
            show_progress_bar=self.show_progress, **kwargs), dtype=np.float32)


def fingerprint(cfg):
    # 同维度的不同模型不能混用；本地路径内容更改也必须重建。
    return {k: getattr(cfg.models, k) for k in
            ("embedding", "embedding_revision", "max_tokens", "query_prompt_name")}

