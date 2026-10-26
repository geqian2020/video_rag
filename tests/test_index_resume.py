"""Index recovery uses real persistent Qdrant and deterministic model substitutes."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

from core.config import Settings
from core.store import open_store
from ingest.index import build_index


IDENTITY = {"dimension": 3, "model": "test-index-model-v1"}


class FakeEmbedder:
    dimension = 3

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on

    def encode(self, texts):
        self.calls.append(list(texts))
        if len(self.calls) == self.fail_on:
            raise RuntimeError("injected encode failure")
        return np.array([[1., float(len(text)), .5] for text in texts], dtype=np.float32)


def source_chunks(count=6):
    chunks = []
    for i in range(count):
        text = f"PA calibration measurement number {i}."
        source = {"video_id": "demo", "start": i * 5, "end": i * 5 + 4, "text": text}
        chunks.append({"chunk_id": str(uuid.uuid5(uuid.NAMESPACE_URL, text)), "video_id": "demo",
                       "start_ts": source["start"], "end_ts": source["end"], "text": text,
                       "source_segments": [source], "timestamp_quality": "asr", "summary": text,
                       "questions": [], "enrichment_status": "extractive_fallback"})
    return chunks


class IndexResume(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "index"
        self.cfg = Settings()
        self.cfg.storage.root = str(self.root)
        self.cfg.models.batch_size = 2
        self.chunks = source_chunks()
        self.log = io.StringIO()

    def run_build(self, model=None, chunks=None, identity=None):
        with contextlib.redirect_stderr(self.log):
            return build_index(self.chunks if chunks is None else chunks, self.cfg,
                               model or FakeEmbedder(), model_identity=identity or IDENTITY)

    def partial(self):
        with self.assertRaisesRegex(RuntimeError, "encode failure"):
            self.run_build(FakeEmbedder(fail_on=2))
        self.assertFalse((self.root / "CURRENT").exists())
        record = next((self.root / "builds").glob("*.json"))
        return json.loads(record.read_text())["generation"]

    def open_partial(self, generation):
        client = open_store(self.cfg, self.root / generation)
        self.addCleanup(client.close)
        return client, self.cfg.storage.collection_prefix + "_" + generation

    def test_resume_skips_verified_batches_and_publishes_same_generation(self):
        generation = self.partial()
        model = FakeEmbedder()
        self.assertEqual(self.run_build(model), generation)
        self.assertEqual(sum(model.calls, []), [c["text"] for c in self.chunks[2:]])
        self.assertEqual((self.root / "CURRENT").read_text().strip(), generation)
        self.assertIn("verified reuse", self.log.getvalue())
        self.assertIn("completed=100.0%", self.log.getvalue())

    def test_failure_after_vector_save_before_batch_marker_needs_no_reencoding(self):
        with patch("ingest.index.save_batch", side_effect=RuntimeError("marker failure")):
            with self.assertRaisesRegex(RuntimeError, "marker failure"):
                self.run_build()
        model = FakeEmbedder()
        self.run_build(model)
        self.assertEqual(sum(model.calls, []), [c["text"] for c in self.chunks[2:]])

    def test_deleted_qdrant_record_is_repaired_from_vector_cache(self):
        generation = self.partial()
        client, collection = self.open_partial(generation)
        client.delete(collection_name=collection, points_selector=[self.chunks[0]["chunk_id"]], wait=True)
        client.close()
        model = FakeEmbedder()
        self.assertEqual(self.run_build(model), generation)
        self.assertEqual(sum(model.calls, []), [c["text"] for c in self.chunks[2:]])
        self.assertNotIn("Batch 1/3, blocks=1-2/6: verified reuse", self.log.getvalue())

    def test_tampered_payload_is_repaired(self):
        generation = self.partial()
        client, collection = self.open_partial(generation)
        client.set_payload(collection_name=collection, payload={"text": "tampered"},
                           points=[self.chunks[0]["chunk_id"]], wait=True)
        client.close()
        self.run_build()
        client = open_store(self.cfg, self.root / generation)
        try:
            point = client.retrieve(collection, [self.chunks[0]["chunk_id"]], with_payload=True)[0]
            self.assertEqual(point.payload["text"], self.chunks[0]["text"])
        finally:
            client.close()

    def test_corrupt_batch_marker_is_rebuilt_without_reencoding(self):
        generation = self.partial()
        marker = self.root / generation / "checkpoints" / "batch-00000001.json"
        marker.write_text('{"records": []}', encoding="utf-8")
        model = FakeEmbedder()
        self.run_build(model)
        self.assertEqual(sum(model.calls, []), [c["text"] for c in self.chunks[2:]])
        self.assertIsInstance(json.loads(marker.read_text())["records"], dict)

    def test_tampered_vector_is_repaired_without_reencoding_saved_text(self):
        generation = self.partial()
        client, collection = self.open_partial(generation)
        from qdrant_client.models import PointVectors
        client.update_vectors(collection, [PointVectors(id=self.chunks[0]["chunk_id"],
                                                       vector={"dense": [0., 0., 1.]})], wait=True)
        client.close()
        model = FakeEmbedder()
        self.run_build(model)
        self.assertEqual(sum(model.calls, []), [c["text"] for c in self.chunks[2:]])
        self.assertNotIn("Batch 1/3, blocks=1-2/6: verified reuse", self.log.getvalue())

    def test_changed_input_uses_new_snapshot_but_reuses_unchanged_vectors(self):
        old = self.partial()
        changed = source_chunks(7)
        model = FakeEmbedder()
        new = self.run_build(model, changed)
        self.assertNotEqual(old, new)
        self.assertTrue((self.root / old).is_dir())
        self.assertEqual(sum(model.calls, []), [c["text"] for c in changed[2:]])

    def test_model_identity_change_does_not_reuse_old_vectors(self):
        old = self.partial()
        model = FakeEmbedder()
        new = self.run_build(model, identity={**IDENTITY, "model": "test-index-model-v2"})
        self.assertNotEqual(old, new)
        self.assertEqual(sum(model.calls, []), [c["text"] for c in self.chunks])

    def test_previous_published_snapshot_survives_new_failure(self):
        published = self.run_build()
        with self.assertRaisesRegex(RuntimeError, "encode failure"):
            self.run_build(FakeEmbedder(fail_on=1), source_chunks(7))
        self.assertEqual((self.root / "CURRENT").read_text().strip(), published)
        client, collection = self.open_partial(published)
        self.assertEqual(client.count(collection, exact=True).count, 6)

    def test_published_snapshot_is_never_modified_in_place(self):
        first = self.run_build()
        model = FakeEmbedder(fail_on=1)
        second = self.run_build(model)
        self.assertNotEqual(first, second)
        self.assertEqual(model.calls, [])

    def test_publish_failure_resumes_ready_snapshot(self):
        from core.io import atomic_text
        def fail_publish(path, text):
            if Path(path).name == "CURRENT":
                raise OSError("injected publish failure")
            return atomic_text(path, text)
        with patch("ingest.index.atomic_text", side_effect=fail_publish):
            with self.assertRaisesRegex(OSError, "publish failure"):
                self.run_build()
        record = next((self.root / "builds").glob("*.json"))
        generation = json.loads(record.read_text())["generation"]
        model = FakeEmbedder(fail_on=1)
        self.assertEqual(self.run_build(model), generation)
        self.assertEqual(model.calls, [])

    def test_abrupt_exit_keeps_qdrant_and_vector_checkpoints(self):
        script = '''
import os, sys
from tests.test_index_resume import FakeEmbedder, IDENTITY, source_chunks
from core.config import Settings
from ingest.index import build_index
class Crash(FakeEmbedder):
    def encode(self, texts):
        if len(self.calls) == 1:
            os._exit(9)
        return super().encode(texts)
cfg = Settings()
cfg.models.batch_size = 2
cfg.storage.root = sys.argv[1]
build_index(source_chunks(), cfg, Crash(), model_identity=IDENTITY)
'''
        env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])}
        # Load the test helper by its file directory, avoiding a third-party tests package.
        script = script.replace("from tests.test_index_resume", "from test_index_resume")
        env["PYTHONPATH"] += os.pathsep + str(Path(__file__).resolve().parent)
        result = subprocess.run([sys.executable, "-c", script, str(self.root)], env=env,
                                capture_output=True, timeout=30)
        self.assertEqual(result.returncode, 9, result.stderr)
        model = FakeEmbedder()
        self.run_build(model)
        self.assertEqual(sum(model.calls, []), [c["text"] for c in self.chunks[2:]])
        self.assertIn("verified reuse", self.log.getvalue())

    def test_extra_qdrant_point_prevents_publication(self):
        generation = self.partial()
        client, collection = self.open_partial(generation)
        from qdrant_client.models import PointStruct, SparseVector
        client.upsert(collection, [PointStruct(id=str(uuid.uuid4()), payload={}, vector={
            "dense": [1., 0., 0.], "bm25": SparseVector(indices=[0], values=[1.])})], wait=True)
        client.close()
        with self.assertRaisesRegex(RuntimeError, "写入条数"):
            self.run_build()
        self.assertFalse((self.root / "CURRENT").exists())


if __name__ == "__main__":
    unittest.main()
