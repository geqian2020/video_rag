import os
from qdrant_client import QdrantClient


def open_store(cfg, generation_dir):
    if cfg.storage.qdrant_url:
        return QdrantClient(url=cfg.storage.qdrant_url,
                            api_key=os.getenv("QDRANT_API_KEY"), timeout=60, trust_env=False)
    return QdrantClient(path=str(generation_dir / "qdrant"))
