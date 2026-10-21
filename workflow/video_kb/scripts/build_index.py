#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_index.py

读取 build_chunks.py 产出的 chunks.jsonl，用本地中文 Embedding 模型
（默认 BAAI/bge-small-zh-v1.5，CPU 推理）计算向量，写入本地持久化的
Chroma 向量数据库。

增量更新：
    manifest.json 记录每个 chunk_id 对应的转录文件修改时间（modified_time）。
    重新运行时，若某个转录文件的 modified_time 未变化，其对应的 chunk 不会
    重复计算 embedding 或重复写入；若变化（文件被重新转录过），会先删除该
    转录文件产生的旧 chunk，再写入新的。

用法：
    python build_index.py --chunks index_data/chunks.jsonl --db-dir index_data/chroma_db
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

# chromadb / huggingface_hub / transformers 在 import 时都会尝试联网做
# 遥测上报或版本检查；在网络受限/被过滤的环境中这些请求会长时间挂起
# （实测 sentence_transformers 的 import 曾因此耗时超过 10 分钟），
# 必须在对应库 import 之前设置这些环境变量关闭遥测/联网检查。
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
BATCH_SIZE = 32


def load_chunks(chunks_path: Path):
    chunks = []
    with chunks_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def save_manifest(manifest_path: Path, manifest: dict):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def chunk_metadata_for_chroma(chunk: dict) -> dict:
    """Chroma 的 metadata 值必须是 str/int/float/bool/None，不能是 list，
    所以这里把 list 字段序列化为 JSON 字符串，查询时再反序列化回来。"""
    meta = dict(chunk)
    text = meta.pop("text")
    chunk_id = meta.pop("chunk_id")
    for key in ("jargon_terms", "associated_ppts"):
        if key in meta and isinstance(meta[key], list):
            meta[key] = json.dumps(meta[key], ensure_ascii=False)
    # Chroma 不允许 metadata 值为 None，统一转换为空字符串
    for key in list(meta.keys()):
        if meta[key] is None:
            meta[key] = ""
    return chunk_id, text, meta


def main():
    parser = argparse.ArgumentParser(
        description="将 chunks.jsonl 编码为向量并写入本地 Chroma 数据库（增量更新）"
    )
    parser.add_argument("--chunks", type=str, default="index_data/chunks.jsonl")
    parser.add_argument("--db-dir", type=str, default="index_data/chroma_db")
    parser.add_argument("--manifest", type=str, default="index_data/manifest.json")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--collection", type=str, default=DEFAULT_COLLECTION_NAME)
    parser.add_argument(
        "--rebuild", action="store_true",
        help="忽略 manifest，强制重新计算并写入所有 chunk",
    )
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"错误: 找不到 {chunks_path}，请先运行 build_chunks.py", file=sys.stderr)
        sys.exit(1)

    chunks = load_chunks(chunks_path)
    print(f"读取到 {len(chunks)} 个片段。")

    manifest_path = Path(args.manifest)
    manifest = {} if args.rebuild else load_manifest(manifest_path)

    db_dir = Path(args.db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_dir))
    collection = client.get_or_create_collection(name=args.collection)

    # 按 transcript_path 分组，判断哪些转录文件自上次建索引后发生了变化
    by_transcript = {}
    for chunk in chunks:
        by_transcript.setdefault(chunk["transcript_path"], []).append(chunk)

    to_embed = []
    unchanged_count = 0
    for transcript_path, group in by_transcript.items():
        current_mtime = group[0]["modified_time"]
        prev_mtime = manifest.get(transcript_path)
        if prev_mtime is not None and prev_mtime == current_mtime and not args.rebuild:
            unchanged_count += len(group)
            continue
        # 文件是新增的或已变化：先删除该转录文件产生的旧 chunk（若存在），再重新写入
        try:
            collection.delete(where={"transcript_path": transcript_path})
        except Exception:
            pass
        to_embed.extend(group)
        manifest[transcript_path] = current_mtime

    print(f"跳过未变化的片段: {unchanged_count}")
    print(f"需要（重新）计算 embedding 的片段: {len(to_embed)}")

    if not to_embed:
        print("没有需要更新的内容，索引已是最新。")
        save_manifest(manifest_path, manifest)
        return

    print(f"正在加载 Embedding 模型 '{args.model}' (CPU 模式)...")
    model = SentenceTransformer(args.model, device="cpu")
    print("模型加载完成。")

    for i in range(0, len(to_embed), BATCH_SIZE):
        batch = to_embed[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]
        embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        ids, docs, metadatas = [], [], []
        for chunk, emb in zip(batch, embeddings):
            chunk_id, text, meta = chunk_metadata_for_chroma(chunk)
            ids.append(chunk_id)
            docs.append(text)
            metadatas.append(meta)

        collection.upsert(
            ids=ids,
            documents=docs,
            metadatas=metadatas,
            embeddings=[e.tolist() for e in embeddings],
        )
        print(f"  已写入 {min(i + BATCH_SIZE, len(to_embed))}/{len(to_embed)} 个片段")

    save_manifest(manifest_path, manifest)
    print("=" * 60)
    print(f"索引构建完成。collection '{args.collection}' 当前共有 {collection.count()} 个片段。")
    print(f"数据库目录: {db_dir}")


if __name__ == "__main__":
    main()
