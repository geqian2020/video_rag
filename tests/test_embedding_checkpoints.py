"""Checkpoint recovery tests use deterministic fake models, never real weights."""
import contextlib
import hashlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from core.config import Settings
from ingest.chunk import build_chunks
from ingest.embedding_cache import EmbeddingCache, model_signature


class FakeEmbedder:
    dimension = 3

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def token_count(self, text):
        return len(text)

    def encode(self, texts):
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_on:
            raise RuntimeError("injected interruption")
        return np.array([[1., float("calibration" in text), float("fruit" in text)]
                         for text in texts], dtype=np.float32)


class Checkpoints(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "checkpoints.sqlite3"
        self.signature = {"format": 1, "dimension": 3, "model": "test-revision-1"}
        self.texts = ["alpha", "beta", "gamma", "delta", "epsilon"]

    def cache(self, signature=None, report=lambda message: None):
        return EmbeddingCache(self.path, signature or self.signature, report)

    def test_resume_after_failed_batch_only_encodes_missing(self):
        interrupted = FakeEmbedder(fail_on=2)
        with self.cache() as cache:
            with self.assertRaisesRegex(RuntimeError, "interruption"):
                cache.encode(self.texts, interrupted, 2)
        saved_texts = set(interrupted.calls[0])
        resumed = FakeEmbedder()
        with self.cache() as cache:
            actual = cache.encode(self.texts, resumed, 2)
        self.assertEqual(set(sum(resumed.calls, [])), set(self.texts) - saved_texts)
        np.testing.assert_array_equal(actual, FakeEmbedder().encode(self.texts))

    def test_full_cache_reordered_and_duplicate_texts_need_no_inference(self):
        with self.cache() as cache:
            cache.encode(self.texts, FakeEmbedder(), 2)
        model = FakeEmbedder(fail_on=1)
        reordered = self.texts[::-1] + self.texts[:2]
        with self.cache() as cache:
            actual = cache.encode(reordered, model, 2)
        self.assertEqual(model.calls, [])
        np.testing.assert_array_equal(actual, FakeEmbedder().encode(reordered))

    def test_identical_new_sentences_encoded_once(self):
        model = FakeEmbedder()
        with self.cache() as cache:
            cache.encode(["repeat"] * 10, model, 8)
        self.assertEqual(model.calls, [["repeat"]])

    def test_changed_text_only_encodes_that_sentence(self):
        with self.cache() as cache:
            cache.encode(self.texts, FakeEmbedder(), 2)
        model = FakeEmbedder()
        with self.cache() as cache:
            cache.encode(self.texts[:-1] + ["edited"], model, 2)
        self.assertEqual(model.calls, [["edited"]])

    def test_model_signature_change_keeps_old_scope_but_reencodes(self):
        with self.cache() as cache:
            cache.encode(self.texts, FakeEmbedder(), 2)
        model = FakeEmbedder()
        with self.cache({**self.signature, "model": "revision-2"}) as cache:
            cache.encode(self.texts, model, 2)
        self.assertEqual(set(sum(model.calls, [])), set(self.texts))
        old_model = FakeEmbedder(fail_on=1)
        with self.cache() as cache:
            cache.encode(self.texts, old_model, 2)
        self.assertEqual(old_model.calls, [])

    def test_corrupt_row_is_recomputed_without_discarding_good_rows(self):
        messages = []
        with self.cache() as cache:
            cache.encode(self.texts, FakeEmbedder(), 2)
            cache.connection.execute("UPDATE embeddings SET vector=? WHERE text_hash=?",
                                     (b"broken", hashlib.sha256(b"beta").hexdigest()))
            cache.connection.commit()
        model = FakeEmbedder()
        with self.cache(report=messages.append) as cache:
            cache.encode(self.texts, model, 2)
        self.assertEqual(model.calls, [["beta"]])
        self.assertTrue(any("invalid=1" in m for m in messages))

    def test_bad_new_vector_never_gets_a_checkpoint(self):
        class Bad(FakeEmbedder):
            def encode(self, texts):
                return np.full((len(texts), 3), np.nan)
        with self.cache() as cache:
            with self.assertRaisesRegex(ValueError, "Invalid embedding"):
                cache.encode(self.texts, Bad(), 2)
            self.assertEqual(cache.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0], 0)

    def test_failed_sql_transaction_keeps_previous_batch_and_rolls_back_current(self):
        with self.cache() as cache:
            cache.encode(["saved"], FakeEmbedder(), 2)
            bad_key = hashlib.sha256(b"bad").hexdigest()
            cache.connection.execute(
                f"CREATE TRIGGER fail_batch BEFORE INSERT ON embeddings "
                f"WHEN NEW.text_hash='{bad_key}' BEGIN SELECT RAISE(ABORT, 'injected'); END")
            cache.connection.commit()
            with self.assertRaises(sqlite3.IntegrityError):
                cache.encode(["good", "bad"], FakeEmbedder(), 2)
            self.assertEqual(cache.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0], 1)

    def test_abrupt_process_exit_preserves_committed_batch(self):
        script = '''
import os, sys
import numpy as np
from ingest.embedding_cache import EmbeddingCache
class Model:
    calls = 0
    def encode(self, texts):
        self.calls += 1
        if self.calls == 2:
            os._exit(9)
        return np.ones((len(texts), 3), dtype=np.float32)
with EmbeddingCache(sys.argv[1], {"format": 1, "dimension": 3, "model": "test-revision-1"}) as cache:
    cache.encode(["alpha", "beta", "gamma", "delta", "epsilon"], Model(), 2)
'''
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        result = subprocess.run([sys.executable, "-c", script, str(self.path)], env=env,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 9, result.stderr)
        model = FakeEmbedder()
        with self.cache() as cache:
            cache.encode(self.texts, model, 2)
            self.assertEqual(cache.connection.execute("PRAGMA quick_check").fetchone()[0], "ok")
        self.assertEqual(len(sum(model.calls, [])), 3)

    def test_chunk_output_and_cross_video_reuse(self):
        cfg = Settings()
        cfg.models.batch_size = 2
        cfg.chunking.min_seconds = 0
        cfg.chunking.similarity_threshold = .99
        rows = [{"video_id": video, "start": i * 10, "end": i * 10 + 5, "text": text}
                for video in ("first", "second")
                for i, text in enumerate(["calibration!", "fruit!", "calibration!"])]
        model = FakeEmbedder()
        with contextlib.redirect_stderr(io.StringIO()):
            expected = build_chunks(rows, FakeEmbedder(), cfg)
            with self.cache() as cache:
                actual = build_chunks(rows, model, cfg, cache)
            resumed = FakeEmbedder(fail_on=1)
            with self.cache() as cache:
                again = build_chunks(rows, resumed, cfg, cache)
        self.assertEqual(actual, expected)
        self.assertEqual(again, expected)
        self.assertEqual(len(sum(model.calls, [])), 2)
        self.assertEqual(resumed.calls, [])

    def test_local_model_content_change_invalidates_identity(self):
        cfg = Settings()
        directory = Path(self.temp.name) / "model"
        directory.mkdir()
        weights = directory / "pytorch_model.bin"
        weights.write_bytes(b"old")
        cfg.models.embedding = str(directory)
        before = model_signature(cfg, FakeEmbedder())
        weights.write_bytes(b"new")
        after = model_signature(cfg, FakeEmbedder())
        self.assertNotEqual(before, after)
        cfg.models.max_tokens = 512
        self.assertNotEqual(after, model_signature(cfg, FakeEmbedder()))

    def test_loaded_hub_revision_used_without_network(self):
        cfg = Settings()
        model = FakeEmbedder()
        model.model = SimpleNamespace(_first_module=lambda: SimpleNamespace(
            auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash="a" * 40))))
        before = model_signature(cfg, model)
        self.assertEqual(before["resolved"], {"hub_revision": "a" * 40})
        model.model = SimpleNamespace(_first_module=lambda: SimpleNamespace(
            model=SimpleNamespace(config=SimpleNamespace(_commit_hash="b" * 40))))
        self.assertNotEqual(before, model_signature(cfg, model))


if __name__ == "__main__":
    unittest.main()
