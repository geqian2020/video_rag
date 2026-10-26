"""No-Ollama deployment: fake embeddings, real local Qdrant, no model downloads."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from core.config import Settings, load_config
from core.io import read_rows, write_rows
from core.schema import Chunk, Question
from ingest.enrich import extractive_chunk, main


def chunk(text="设备校准需要采集测试数据。随后进行验证。"):
    source = {"video_id": "training/demo", "start": 12.5, "end": 21.8, "text": text}
    return Chunk.model_validate({
        "chunk_id": "e33ed57e-76aa-4f22-ae97-529c083483e8", "video_id": source["video_id"],
        "start_ts": source["start"], "end_ts": source["end"], "text": text,
        "source_segments": [source], "timestamp_quality": "asr",
    })


class ExtractiveMode(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "chunks.jsonl"
        self.output = self.root / "enriched.jsonl"
        self.cfg = Settings()
        write_rows(self.source, [chunk().model_dump()])

    def execute(self, mode=None):
        args = ["enrich", "--input", str(self.source), "--output", str(self.output)]
        if mode is not None:
            args += ["--mode", mode]
        with patch("sys.argv", args), \
                patch("ingest.enrich.load_config", return_value=self.cfg), \
                patch("ingest.enrich.httpx.Client", side_effect=AssertionError("HTTP must not be opened")), \
                patch("core.ollama.ensure_ollama", side_effect=AssertionError("Ollama must not be started")), \
                contextlib.redirect_stdout(io.StringIO()) as log:
            main()
        return log.getvalue()

    def test_preserves_original_and_provenance(self):
        original = chunk()
        extracted = extractive_chunk(original)
        for key in ("chunk_id", "video_id", "start_ts", "end_ts", "text", "source_segments", "timestamp_quality"):
            self.assertEqual(extracted[key], original.model_dump()[key])
        self.assertEqual(extracted["summary"], "设备校准需要采集测试数据。")
        self.assertEqual(extracted["questions"], [])
        self.assertEqual(extracted["enrichment_status"], "extractive_fallback")
        Chunk.model_validate(extracted)

    def test_english_and_newlines_not_rewritten(self):
        for text in ("PA calibration requires measurements", "PA calibration\nThen verify"):
            extracted = extractive_chunk(chunk(text))
            self.assertIn(extracted["summary"], text)
            self.assertEqual(extracted["text"], text)

    def test_stale_enhancement_removed(self):
        original = chunk().model_copy(update={"questions": [Question(question="old question", evidence="old")], "summary": "old"})
        extracted = extractive_chunk(original)
        self.assertNotEqual(extracted["summary"], "old")
        self.assertEqual(extracted["questions"], [])

    def test_cli_mode_bypasses_http_and_ollama(self):
        log = self.execute("extractive")
        self.assertIn("no Ollama, no generated questions", log)
        self.assertEqual(read_rows(self.output)[0]["enrichment_status"], "extractive_fallback")

    def test_config_mode_without_shell_change(self):
        self.cfg.enrichment.mode = "extractive"
        self.execute()
        self.assertEqual(len(read_rows(self.output)), 1)

    def test_environment_overrides_yaml(self):
        config = self.root / "config.yaml"
        config.write_text("enrichment:\n  mode: ollama\n", encoding="utf-8")
        with patch.dict(os.environ, {"RAG_ENRICHMENT__MODE": "extractive"}):
            self.assertEqual(load_config(str(config)).enrichment.mode, "extractive")

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValidationError):
            Settings.model_validate({"enrichment": {"mode": "unknown"}})

    def test_failed_input_preserves_previous_output(self):
        self.output.write_text("previous result", encoding="utf-8")
        bad = chunk().model_dump()
        bad["end_ts"] = 1
        write_rows(self.source, [bad])
        with self.assertRaises(ValidationError):
            self.execute("extractive")
        self.assertEqual(self.output.read_text(encoding="utf-8"), "previous result")

    def test_all_chunks_publish_with_progress(self):
        rows = [chunk().model_copy(update={"chunk_id": str(i)}).model_dump() for i in range(501)]
        write_rows(self.source, rows)
        log = self.execute("extractive")
        self.assertIn("Extracted 500/501", log)
        self.assertIn("Extracted 501/501", log)
        self.assertEqual(len(read_rows(self.output)), 501)

    def test_real_qdrant_accepts_extractive_results(self):
        from ingest.index import build_index
        from core.store import open_store
        class FakeEmbedder:
            dimension = 3
            def encode(self, texts):
                self.texts = texts
                return np.array([[1., 0., 0.] for _ in texts], dtype=np.float32)
        self.execute("extractive")
        self.cfg.storage.root = str(self.root / "index")
        model = FakeEmbedder()
        generation = build_index(read_rows(self.output), self.cfg, model)
        self.assertEqual(model.texts, [chunk().text])
        folder = Path(self.cfg.storage.root) / generation
        manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        client = open_store(self.cfg, folder)
        try:
            records, _ = client.scroll(manifest["collection"], with_payload=True)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].payload["questions"], [])
            self.assertEqual(records[0].payload["text"], chunk().text)
            self.assertEqual(records[0].payload["enrichment_status"], "extractive_fallback")
        finally:
            client.close()


if __name__ == "__main__":
    unittest.main()
