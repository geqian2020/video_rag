"""Local, transactional sentence-vector checkpoints; never store transcript text.

Only fully committed inference batches are reusable. A killed in-flight batch may
be recomputed. The cache is a derived artifact, not a replacement for transcripts.
"""
import hashlib
import json
import re
import sqlite3
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from core.models import fingerprint


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def model_signature(cfg, embedder):
    """Identify loaded Hub revision, or hash actual local model artifacts."""
    model_path = Path(cfg.models.embedding)
    if model_path.is_dir():
        artifacts = {}
        for path in sorted(model_path.rglob("*")):
            relative = path.relative_to(model_path)
            if (not path.is_file() or any(p.startswith(".") for p in relative.parts)
                    or path.suffix not in {".json", ".txt", ".model", ".bin", ".pt", ".safetensors"}):
                continue
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            artifacts[relative.as_posix()] = digest.hexdigest()
        if not artifacts:
            raise ValueError(f"No model artifacts found for checkpoint identity: {model_path}")
        resolved = {"local_artifacts": artifacts}
    else:
        # Prefer the loaded model's revision over a mutable main ref on disk.
        revision = None
        model = getattr(embedder, "model", None)
        if model is not None and hasattr(model, "_first_module"):
            module = model._first_module()
            for name in ("auto_model", "model"):
                config = getattr(getattr(module, name, None), "config", None)
                revision = getattr(config, "_commit_hash", None)
                if revision:
                    break
        if not revision:
            # Pure cache lookup: no HTTP request and no download.
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(
                cfg.models.embedding, "config.json", revision=cfg.models.embedding_revision)
            if isinstance(cached, str):
                revision = Path(cached).parent.name
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError("Cannot resolve embedding checkpoint model revision; use a local model directory")
        resolved = {"hub_revision": revision.lower()}
    packages = {}
    for name in ("sentence-transformers", "transformers", "torch", "numpy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    return {
        "format": 1, "encoding": "dense-normalized-float32",
        "model": fingerprint(cfg), "resolved": resolved,
        "dimension": int(embedder.dimension), "device": cfg.models.device,
        "packages": packages,
    }


class EmbeddingCache:
    def __init__(self, path, signature, report=lambda message: None):
        self.path = Path(path)
        self.report = report
        self.dimension = int(signature["dimension"])
        if self.dimension < 1:
            raise ValueError("Embedding dimension must be positive")
        manifest = json.dumps(signature, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        self.scope = _sha(manifest.encode("utf-8"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=60)
        try:
            # DELETE journal also works on Windows; FULL commits each batch to disk.
            self.connection.execute("PRAGMA journal_mode=DELETE")
            self.connection.execute("PRAGMA synchronous=FULL")
            with self.connection:
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS models (scope TEXT PRIMARY KEY, manifest TEXT NOT NULL)")
                self.connection.execute(
                    "CREATE TABLE IF NOT EXISTS embeddings ("
                    "scope TEXT NOT NULL, text_hash TEXT NOT NULL, vector BLOB NOT NULL, "
                    "checksum TEXT NOT NULL, PRIMARY KEY(scope, text_hash))")
                existing = self.connection.execute(
                    "SELECT manifest FROM models WHERE scope=?", (self.scope,)).fetchone()
                if existing is not None and existing[0] != manifest:
                    raise ValueError("Checkpoint model manifest mismatch")
                self.connection.execute("INSERT OR IGNORE INTO models VALUES (?, ?)",
                                        (self.scope, manifest))
        except BaseException:
            self.connection.close()
            raise
        self.report(f"Embedding cache={self.path}, model scope={self.scope[:12]}")

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def encode(self, texts, embedder, batch_size):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        # Deduplicate identical sentences, including across videos and sources.
        unique = {}
        for text in texts:
            key = _sha(text.encode("utf-8"))
            if key in unique and unique[key] != text:
                raise ValueError("Sentence hash collision")
            unique[key] = text
        vectors = {}
        missing = []
        corrupt = 0
        for key, text in unique.items():
            row = self.connection.execute(
                "SELECT vector, checksum FROM embeddings WHERE scope=? AND text_hash=?",
                (self.scope, key)).fetchone()
            if row is not None:
                raw, checksum = row
                if (len(raw) == self.dimension * 4 and _sha(raw) == checksum):
                    vector = np.frombuffer(raw, dtype="<f4")
                    if np.isfinite(vector).all():
                        vectors[key] = vector
                        continue
                corrupt += 1
            missing.append((key, text))
        reused = len(vectors)
        self.report(f"Embedding checkpoint: sentences={len(texts)}, unique={len(unique)}, "
                    f"reused={reused}, pending={len(missing)}, invalid={corrupt}")
        # Similar to SentenceTransformer's length ordering, but every inference
        # batch is committed independently instead of waiting for a whole video.
        missing.sort(key=lambda item: len(item[1]), reverse=True)
        last_report = time.monotonic()
        for start in range(0, len(missing), batch_size):
            batch = missing[start:start + batch_size]
            encoded = np.asarray(embedder.encode([text for _, text in batch]), dtype="<f4")
            if encoded.shape != (len(batch), self.dimension) or not np.isfinite(encoded).all():
                raise ValueError("Invalid embedding batch; checkpoint was not written")
            records = []
            for (key, _), vector in zip(batch, encoded):
                raw = vector.tobytes()
                records.append((self.scope, key, raw, _sha(raw)))
            # An interruption before commit preserves all previously saved batches.
            with self.connection:
                self.connection.executemany(
                    "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?)", records)
            for (key, _), vector in zip(batch, encoded):
                vectors[key] = vector
            done = start + len(batch)
            now = time.monotonic()
            if start == 0 or done == len(missing) or now - last_report >= 10:
                self.report(f"Embedding saved: new={done}/{len(missing)}, "
                            f"ready unique={reused + done}/{len(unique)}")
                last_report = now
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack([vectors[_sha(text.encode("utf-8"))] for text in texts])
