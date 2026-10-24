"""Regression tests for crash-safe window-level ASR checkpoints."""
import json

import pytest

from core.config import Settings
from core.io import read_rows
from core.vocabulary import Vocabulary
from transcribe import run


def test_resume_skips_completed_window(monkeypatch, tmp_path):
    cfg = Settings()
    cfg.asr.window_seconds = 30
    monkeypatch.setattr(run, 'probe', lambda _: {'duration': 65})
    extracted = []
    def extract(video, output, start, end):
        extracted.append((start, end))
        output.write_bytes(b'audio')
    monkeypatch.setattr(run, 'extract_audio', extract)
    output = tmp_path / 'transcript.jsonl'

    class InterruptedModel:
        calls = 0
        def transcribe(self, _audio, **_kwargs):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError('interrupted on second window')
            return {'segments': [{'start': 1, 'end': 2, 'text': 'one'}]}

    with pytest.raises(RuntimeError, match='interrupted'):
        run.transcribe_video('video.mp4', 'video-1', Vocabulary(), cfg,
                             InterruptedModel(), len, output)
    assert not output.exists()
    assert (tmp_path / 'transcript.jsonl.windows/000000.json').is_file()
    assert not (tmp_path / 'transcript.jsonl.windows/000001.json').exists()

    class ResumedModel:
        calls = 0
        def transcribe(self, _audio, **_kwargs):
            self.calls += 1
            return {'segments': [{'start': 1, 'end': 2, 'text': 'rest'}]}

    model = ResumedModel()
    rows = run.transcribe_video('video.mp4', 'video-1', Vocabulary(), cfg, model, len, output)
    assert model.calls == 2
    assert extracted == [(0, 30), (30, 60), (30, 60), (60, 65)]
    assert [row['text'] for row in rows] == ['one', 'rest', 'rest']
    assert read_rows(output) == rows
    audit = json.loads((tmp_path / 'transcript.jsonl.audit.json').read_text(encoding='utf-8'))
    assert len(audit['windows']) == 3


def test_corrupt_or_changed_fingerprint_forces_retranscription(monkeypatch, tmp_path):
    cfg = Settings()
    cfg.asr.window_seconds = 30
    monkeypatch.setattr(run, 'probe', lambda _: {'duration': 10})
    monkeypatch.setattr(run, 'extract_audio', lambda v, p, s, e: p.write_bytes(b'audio'))
    calls = []
    class Model:
        def transcribe(self, _audio, **_kwargs):
            calls.append(True)
            return {'segments': [{'start': 1, 'end': 3, 'text': 'hello'}]}
    output = tmp_path / 'output.jsonl'
    for expected in (1, 1):
        run.transcribe_video('video.mp4', 'v', Vocabulary(), cfg, Model(), len, output)
        assert len(calls) == expected
    checkpoint = tmp_path / 'output.jsonl.windows/000000.json'
    checkpoint.write_text('truncated', encoding='utf-8')
    run.transcribe_video('video.mp4', 'v', Vocabulary(), cfg, Model(), len, output)
    assert len(calls) == 2
    cfg.asr.language = 'en'
    run.transcribe_video('video.mp4', 'v', Vocabulary(), cfg, Model(), len, output)
    assert len(calls) == 3
