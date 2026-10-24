"""Persistent filename-dedup regression tests (no cloud media access)."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from core.io import write_json, write_rows
from workflow import transcript_registry as registry


def _completed(path, video_id="original", model="small", prompt_style="terms"):
    rows = [{"video_id": video_id, "start": 0.5, "end": 1.5, "text": "hello"}]
    write_rows(path, rows)
    write_json(str(path) + ".audit.json", {
        "video_id": video_id, "model": model, "prompt_style": prompt_style,
        "transcript_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()
    })


def test_reuse_filename_preserves_derived_identity(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "load_config", lambda: SimpleNamespace(
        asr=SimpleNamespace(model="small", prompt_style="terms")))
    canonical = tmp_path / "source.jsonl"
    target = tmp_path / "target.jsonl"
    index = tmp_path / "registry.json"
    _completed(canonical)
    registry.register(index, "A/Training.MP4", canonical)
    assert registry.reuse(index, "B/training.mp4", target, "sharepoint/new")
    rows = [json.loads(s) for s in target.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["video_id"] == "sharepoint/new"
    assert registry.valid(target)
    assert registry.key("A/Training.MP4") == registry.key("B/training.mp4")


def test_reject_corrupt_or_incompatible_transcript(monkeypatch, tmp_path):
    settings = SimpleNamespace(asr=SimpleNamespace(model="small", prompt_style="terms"))
    monkeypatch.setattr(registry, "load_config", lambda: settings)
    source = tmp_path / "source.jsonl"
    index = tmp_path / "registry.json"
    _completed(source)
    registry.register(index, "same.mp4", source)
    settings.asr.model = "medium"
    assert not registry.reuse(index, "Other/SAME.MP4", tmp_path / "copy.jsonl", "new")
    settings.asr.model = "small"
    source.write_text("corrupt", encoding="utf-8")
    assert not registry.reuse(index, "same.mp4", tmp_path / "copy.jsonl", "new")


def test_rebuild_registers_existing_transcript_without_opening_media(monkeypatch, tmp_path):
    monkeypatch.setattr(registry, "load_config", lambda: SimpleNamespace(
        asr=SimpleNamespace(model="small", prompt_style="terms")))
    root = tmp_path / "cloud"
    root.mkdir()
    media = root / "ChineseVideo.mp4"  # intentionally absent: cloud must not be opened
    manifest = tmp_path / "scan.nul"
    manifest.write_bytes(str(media).encode("utf-8") + b"\0")
    transcripts = tmp_path / "transcripts"
    transcripts.mkdir()
    video_id = hashlib.sha256("ChineseVideo.mp4".encode("utf-8")).hexdigest()[:24]
    _completed(transcripts / (video_id + ".jsonl"))
    reg = tmp_path / "registry.json"
    registry.rebuild(reg, str(root), str(manifest), str(transcripts))
    assert "chinesevideo.mp4" in registry.load(reg)
