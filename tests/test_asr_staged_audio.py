"""Staged audio must preserve the old ASR window checkpoint fingerprint."""
from pathlib import Path

from core.config import Settings
from core.vocabulary import Vocabulary
from transcribe import run


def test_audio_stage_and_original_use_same_window_checkpoint(monkeypatch, tmp_path):
    cfg = Settings()
    cfg.asr.window_seconds = 30
    wav = tmp_path / 'prepared.wav'
    wav.write_bytes(b'local audio')
    probes, extractions, calls = [], [], []

    def fake_probe(path):
        probes.append(str(path))
        return {'duration': 30}

    def fake_extract(path, output, start, end):
        extractions.append((str(path), start, end))
        Path(output).write_bytes(b'window')

    monkeypatch.setattr(run, 'probe', fake_probe)
    monkeypatch.setattr(run, 'extract_audio', fake_extract)

    class Model:
        def transcribe(self, _audio, **_kwargs):
            calls.append(1)
            return {'segments': [{'start': 1, 'end': 2, 'text': 'training'}]}

    original = 'cloud-video.mp4'
    output = tmp_path / 'result.jsonl'
    run.transcribe_video(original, 'sharepoint/same', Vocabulary(), cfg,
                         Model(), len, output, audio_source=wav)
    assert len(calls) == 1
    assert probes[0] == str(wav.resolve())
    assert extractions[0][0] == str(wav.resolve())
    assert (tmp_path / 'result.jsonl.windows/000000.json').is_file()

    # A rerun without staged audio must still reuse the exact same checkpoint.
    run.transcribe_video(original, 'sharepoint/same', Vocabulary(), cfg,
                         Model(), len, output)
    assert len(calls) == 1
    assert len(extractions) == 1
