"""配置优先级：RAG_* 环境变量 > YAML > 默认值；不自动读取 .env。"""
import os
from pathlib import Path
from urllib.parse import urlparse

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Models(Strict):
    embedding: str = "BAAI/bge-m3"
    embedding_revision: str | None = None
    reranker: str = "BAAI/bge-reranker-v2-m3"
    reranker_revision: str | None = None
    device: str = "cpu"
    offline: bool = True
    batch_size: int = Field(8, ge=1)
    max_tokens: int = Field(1024, ge=128, le=8192)
    rerank_tokens: int = Field(2048, ge=256, le=8192)
    # Qwen3 Embedding 切换时设置为 query，BGE-M3 保持 null。
    query_prompt_name: str | None = None


class Chunking(Strict):
    similarity_threshold: float = Field(0.65, ge=-1, le=1)
    min_seconds: float = Field(15, ge=0)
    max_seconds: float = Field(90, gt=0)
    gap_seconds: float = Field(10, ge=0)
    @model_validator(mode="after")
    def limits(self):
        if self.min_seconds > self.max_seconds:
            raise ValueError("min_seconds 不可大于 max_seconds")
        return self


class Enrichment(Strict):
    url: str = "http://127.0.0.1:11434"
    model: str = "qwen3:4b"
    timeout: float = Field(180, gt=0)
    retries: int = Field(2, ge=0, le=5)
    # fail：不满足 3-5 个有证据问题就停止；extractive：显式降级且记录状态。
    on_failure: str = Field("fail", pattern="^(fail|extractive)$")


class Storage(Strict):
    root: str = "data/index"
    # null 为 Qdrant embedded，无需 Docker；服务端模式填写内网 URL。
    qdrant_url: str | None = None
    collection_prefix: str = Field("video_rag", pattern="^[a-zA-Z][a-zA-Z0-9_]{0,39}$")


class Retrieval(Strict):
    recall_k: int = Field(30, ge=30, le=200)
    top_k: int = Field(5, ge=1, le=30)
    rrf_k: int = Field(60, ge=1)
    bm25_k1: float = Field(1.5, gt=0)
    bm25_b: float = Field(0.75, ge=0, le=1)


class OCR(Strict):
    interval_seconds: float = Field(5, gt=0, allow_inf_nan=False)
    min_confidence: float = Field(0.7, ge=0, le=1, allow_inf_nan=False)
    min_frames: int = Field(2, ge=1)
    # 相对画面的 x0,y0,x1,y1；只识别 PPT 区域通常比整屏更有效。
    crop: tuple[float, float, float, float] = (0, 0, 1, 1)
    upscale: float = Field(1.5, ge=1, le=4, allow_inf_nan=False)
    max_side: int = Field(2560, ge=256, le=4096)
    det_model: str | None = None
    rec_model: str | None = None
    cls_model: str | None = None
    rec_keys: str | None = None

    @model_validator(mode="after")
    def crop_limits(self):
        x0, y0, x1, y1 = self.crop
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
            raise ValueError("ocr.crop 必须为 0-1 的 x0,y0,x1,y1")
        return self


class ASR(Strict):
    # 已有 small.pt 路径或已缓存模型名；禁止缺失时自动下载。
    model: str = "small"
    model_cache: str | None = None
    device: str = "cpu"
    language: str = "zh"
    window_seconds: float = Field(30, gt=0, le=60, allow_inf_nan=False)
    context_margin: float = Field(30, ge=0, allow_inf_nan=False)
    prompt_tokens: int = Field(180, ge=1, le=200)
    max_terms: int = Field(30, ge=1, le=100)


class Settings(Strict):
    models: Models = Field(default_factory=Models)
    chunking: Chunking = Field(default_factory=Chunking)
    enrichment: Enrichment = Field(default_factory=Enrichment)
    storage: Storage = Field(default_factory=Storage)
    retrieval: Retrieval = Field(default_factory=Retrieval)
    ocr: OCR = Field(default_factory=OCR)
    asr: ASR = Field(default_factory=ASR)
    allowed_hosts: list[str] = ["127.0.0.1", "localhost", "::1"]


def local_endpoint(url: str, cfg: Settings):
    parsed = urlparse(url)
    if (parsed.scheme not in {"http", "https"} or parsed.hostname not in cfg.allowed_hosts
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path not in {"", "/"}):
        raise ValueError("服务地址必须为 allowed_hosts 中的主机根 URL，禁止携带凭据")


def load_config(path: str | None = None) -> Settings:
    chosen = Path(path or os.getenv("RAG_CONFIG", "config.yaml"))
    if not chosen.is_file():
        raise FileNotFoundError(f"配置文件不存在: {chosen}")
    data = yaml.safe_load(chosen.read_text(encoding="utf-8")) or {}
    for key, value in os.environ.items():
        if key.startswith("RAG_") and "__" in key:
            section, field = key[4:].lower().split("__", 1)
            data.setdefault(section, {})[field] = yaml.safe_load(value)
    cfg = Settings.model_validate(data)
    # 在模型库导入之前应用，离线模式强制覆盖外部相反设置。
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1" if cfg.models.offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if cfg.models.offline else "0"
    local_endpoint(cfg.enrichment.url, cfg)
    if cfg.storage.qdrant_url:
        local_endpoint(cfg.storage.qdrant_url, cfg)
    return cfg
