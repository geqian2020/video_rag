"""只检索 API：无文件上传、无自动回答、无公网 LLM 请求。"""
import hmac
import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.config import load_config
from retrieve.search import Searcher


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=30)
    video_ids: list[str] | None = Field(None, max_length=100)

    @field_validator("query")
    @classmethod
    def query_nonblank(cls, value):
        if not value.strip():
            raise ValueError("query 不能为空白")
        return value


class Hit(BaseModel):
    chunk_id: str
    video_id: str
    start_ts: float
    end_ts: float
    text: str
    summary: str
    timestamp_quality: str
    enrichment_status: str
    rrf_score: float
    rerank_score: float


class SearchResponse(BaseModel):
    results: list[Hit]
    context: str
    context_is_untrusted: bool = True


def create_app(factory=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.api_key = os.environ.get("RAG_API_KEY", "")
        if not app.state.api_key:
            raise RuntimeError("请设置 RAG_API_KEY；服务默认仍应只监听 127.0.0.1")
        app.state.searcher = factory() if factory else Searcher(load_config())
        try:
            yield
        finally:
            app.state.searcher.close()

    app = FastAPI(title="本地视频知识库检索", version="1.0.0", lifespan=lifespan)

    def authenticate(request: Request, x_api_key: str = Header(default="")):
        if not hmac.compare_digest(x_api_key.encode(), request.app.state.api_key.encode()):
            raise HTTPException(401, "API key 无效")

    @app.get("/health")
    def health():
        return {"status": "ok"}  # lifespan 成功说明模型和索引已加载

    @app.post("/search", response_model=SearchResponse, dependencies=[Depends(authenticate)])
    def search(body: SearchRequest, request: Request):
        try:
            hits = request.app.state.searcher.search(body.query, body.top_k, body.video_ids)
        except ValueError:
            raise HTTPException(422, "输入超过模型限制，请缩短问题或调整分块配置") from None
        except Exception:
            # 不向客户端暴露本机路径、原文和底层异常；不记录查询正文。
            logging.getLogger(__name__).error("检索失败；请本地检查服务状态")
            raise HTTPException(503, "检索服务暂时不可用") from None
        results = [Hit.model_validate(h) for h in hits]
        context = "\n\n".join(
            f"[{i}] video_id={h.video_id} time={h.start_ts:.3f}-{h.end_ts:.3f}s\n{h.text}"
            for i, h in enumerate(results, 1))
        return SearchResponse(results=results, context=context)

    return app


app = create_app()
