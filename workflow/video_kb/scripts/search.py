#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
search.py

对本地 Chroma 向量数据库做查询，返回最相似的片段，展示原文、视频路径、
起止时间，不做任何 LLM 总结——检索结果必须是可验证的原文，而不是生成的
答案，这样才能判断结果真实可信，也便于发现转录错误。

用法：
    python search.py "PA校准的数据采集流程是什么？"
    python search.py "PA校准的数据采集流程是什么？" --top-k 10
"""

import argparse
import json
import os
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 见 build_index.py 中的说明：必须在相关库 import 之前关闭遥测/联网检查，
# 否则在网络受限环境下 import chromadb / sentence_transformers 会长时间挂起
# （实测 sentence_transformers 的 import 曾因此耗时超过 10 分钟）。
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_IMPL", "none")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("DO_NOT_TRACK", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import chromadb
except ImportError:
    print("错误: 未安装 chromadb。请先运行: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print(
        "错误: 未安装 sentence-transformers。请先运行: pip install -r requirements.txt",
        file=sys.stderr,
    )
    sys.exit(1)


DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
DEFAULT_COLLECTION_NAME = "video_transcripts"


def format_result(rank: int, doc: str, meta: dict, distance: float) -> str:
    # Chroma 默认用余弦距离，向量已归一化时 distance = 1 - cosine_similarity
    similarity = max(0.0, 1.0 - distance)
    jargon = meta.get("jargon_terms", "[]")
    try:
        jargon_list = json.loads(jargon) if jargon else []
    except (json.JSONDecodeError, TypeError):
        jargon_list = []

    lines = [
        f"结果 {rank}",
        f"相似度: {similarity:.3f}",
        f"视频: {meta.get('video_name') or '(未找到对应视频文件)'}",
        f"路径: {meta.get('video_path') or meta.get('transcript_path')}",
        f"时间: {meta.get('start_time')} - {meta.get('end_time')}",
    ]
    if meta.get("associated_ppts"):
        try:
            ppts = json.loads(meta["associated_ppts"])
        except (json.JSONDecodeError, TypeError):
            ppts = []
        if ppts:
            lines.append(f"同目录相关材料(非对应关系，仅供参考): {', '.join(ppts)}")
    if jargon_list:
        lines.append(f"命中内部术语: {', '.join(jargon_list)}")
    lines.append(f"原文: {doc}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="在本地视频转录知识库中检索相似片段，返回原文和定位信息（不做 LLM 总结）"
    )
    parser.add_argument("query", type=str, help="查询问题/关键词")
    parser.add_argument("--db-dir", type=str, default="index_data/chroma_db")
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION_NAME)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    db_dir = Path(args.db_dir)
    if not db_dir.exists():
        print(f"错误: 找不到数据库目录 {db_dir}，请先运行 build_index.py", file=sys.stderr)
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(db_dir))
    try:
        collection = client.get_collection(name=args.collection)
    except Exception:
        print(f"错误: 找不到 collection '{args.collection}'，请先运行 build_index.py", file=sys.stderr)
        sys.exit(1)

    if collection.count() == 0:
        print("索引为空，没有可检索的内容。请先运行 build_chunks.py 和 build_index.py。")
        return

    model = SentenceTransformer(args.model, device="cpu")
    query_embedding = model.encode([args.query], normalize_embeddings=True)[0]

    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=args.top_k,
    )

    docs = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not docs:
        print("未找到相关结果。")
        return

    print(f"查询: {args.query}\n")
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances), start=1):
        print(format_result(i, doc, meta, dist))
        print("-" * 60)


if __name__ == "__main__":
    main()
