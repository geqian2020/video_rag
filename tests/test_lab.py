"""No model downloads: validate launch paths, resume, and human edits reaching ingestion."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from core.config import Settings
from core.io import read_rows, write_rows
from core.vocabulary import Vocabulary
from tools import lab
from transcribe import run
from transcribe.review import export_review


def test_single_input_path_and_media_only_scan(tmp_path, monkeypatch):
    root = tmp_path / 'repository with spaces'
    root.mkdir()
    media = root / '中文 videos'
    media.mkdir()
    (media / 'one.MP4').write_bytes(b'video')
    (media / 'slides.pptx').write_bytes(b'never opened')
    sub = media / 'sub'
    sub.mkdir()
    (sub / 'one.mp4').write_bytes(b'other')
    (root / 'lab_input.txt').write_text('"中文 videos"\n', encoding='utf-8-sig')
    monkeypatch.setattr(lab, 'ROOT', root)
    source = lab.input_path()
    found = lab.discover(source)
    assert len(found) == 2 and all(p.suffix.lower() == '.mp4' for p in found)
    manifest = lab.manifest_for(source, found, lab.workspace(source))
    assert len({m['output'] for m in manifest}) == 2
    assert {m['video_id'] for m in manifest} == {'one.MP4', 'sub/one.mp4'}
    assert lab.workspace(source).parent == root / 'data' / 'lab'


def test_reject_non_media_and_multiple_paths(tmp_path):
    document = tmp_path / 'slides.pptx'
    document.touch()
    with pytest.raises(ValueError, match='supported'):
        lab.input_path(str(document))
    with pytest.raises(ValueError, match='exactly one'):
        lab.input_path('one.mp4\ntwo.mp4')


def test_review_edit_is_imported_and_invalidates_downstream(tmp_path, monkeypatch):
    source = tmp_path / 'transcripts' / 'video.jsonl'
    write_rows(source, [{'video_id': 'video', 'start': 0, 'end': 20,
                         'text': 'Incorrect term.', 'timestamp_quality': 'asr'}])
    review = export_review(source)
    entries = [{'output': str(source)}]
    effective = lab.assemble(entries, tmp_path)
    assert read_rows(effective)[0]['text'] == 'Incorrect term.'
    config, output = tmp_path / 'config.json', tmp_path / 'chunks.jsonl'
    config.write_text('{}')
    calls = []
    def stage(*_args):
        calls.append(True)
        output.write_bytes(effective.read_bytes())
    monkeypatch.setattr(lab, 'run_stage', stage)
    for _ in range(2):
        lab.cached_stage('chunk', [], [effective], [output], tmp_path, config)
    assert len(calls) == 1
    review.write_text(review.read_text().replace('Incorrect term.', 'Corrected term.'), encoding='utf-8')
    lab.assemble(entries, tmp_path)
    lab.cached_stage('chunk', [], [effective], [output], tmp_path, config)
    rows = read_rows(output)
    assert len(calls) == 2
    assert rows[0]['text'] == 'Corrected term.'
    assert rows[0]['timestamp_quality'] == 'estimated'
    assert read_rows(source)[0]['text'] == 'Incorrect term.'
    assert len(list((tmp_path / 'corrected').glob('*.jsonl'))) == 2
    output.write_text('corrupt')
    lab.cached_stage('chunk', [], [effective], [output], tmp_path, config)
    assert len(calls) == 3


def test_faster_audio_windows_resume_and_detect_media_change(tmp_path, monkeypatch):
    cfg = Settings()
    cfg.asr.backend = 'faster-whisper'
    source = tmp_path / 'source.mp4'
    source.write_bytes(b'original')
    output = tmp_path / 'transcript.jsonl'
    monkeypatch.setitem(sys.modules, 'faster_whisper.audio',
                        SimpleNamespace(decode_audio=lambda *_a, **_kw: np.zeros(65*16000)))
    def forbidden(*_args):
        pytest.fail('faster-whisper must not require system ffmpeg/ffprobe')
    monkeypatch.setattr(run, 'probe', forbidden)
    monkeypatch.setattr(run, 'extract_audio', forbidden)
    class Model:
        calls = 0
        def transcribe(self, audio, **_kwargs):
            assert isinstance(audio, np.ndarray) and len(audio) <= 30*16000
            self.calls += 1
            return {'segments': [{'start': 0, 'end': 2, 'text': 'Recognized.'}]}
    model = Model()
    for _ in range(2):
        rows = run.transcribe_video(source, 'v', Vocabulary(), cfg, model, len, output)
    assert model.calls == 3
    assert [(r['start'], r['end']) for r in rows] == [(0, 2), (30, 32), (60, 62)]
    source.write_bytes(b'replaced media, same path')
    run.transcribe_video(source, 'v', Vocabulary(), cfg, model, len, output)
    assert model.calls == 6
    cfg.asr.compute_type = 'float32'
    run.transcribe_video(source, 'v', Vocabulary(), cfg, model, len, output)
    assert model.calls == 9
    cfg.asr.multilingual = True
    run.transcribe_video(source, 'v', Vocabulary(), cfg, model, len, output)
    assert model.calls == 12
    audit = json.loads(Path(str(output) + '.audit.json').read_text())
    assert audit['multilingual'] is True and audit['backend'] == 'faster-whisper'


def test_complete_asr_reuses_output_without_model_loading(tmp_path, monkeypatch):
    from transcribe import faster
    cfg = Settings()
    source = tmp_path / 'video.mp4'
    source.write_bytes(b'video')
    output = tmp_path / 'transcripts/video.jsonl'
    config, manifest = tmp_path / 'config.json', tmp_path / 'manifest.json'
    config.write_text(json.dumps(cfg.model_dump()), encoding='utf-8')
    manifest.write_text(json.dumps([{'video': str(source), 'video_id': 'v', 'output': str(output)}]))
    calls = []
    class Model:
        def __init__(self, _cfg):
            calls.append('load')
        token_count = staticmethod(len)
    def transcribe(*_args):
        write_rows(output, [{'video_id': 'v', 'start': 0, 'end': 20, 'text': 'Complete.'}])
        Path(str(output) + '.audit.json').write_text('{}')
    monkeypatch.setattr(lab, 'configure_cuda_libraries', lambda: None)
    monkeypatch.setattr(faster, 'FasterWhisperAdapter', Model)
    monkeypatch.setattr(run, 'transcribe_video', transcribe)
    lab.asr_worker(manifest, config)
    review = output.with_suffix('.review.md')
    review.write_text(review.read_text().replace('Complete.', 'My edit.'), encoding='utf-8')
    lab.asr_worker(manifest, config)
    assert calls == ['load']
    assert 'My edit.' in review.read_text()


def test_lab_worker_environment_does_not_inherit_old_rag_paths(monkeypatch):
    monkeypatch.setenv('RAG_CONFIG', 'old-config.yaml')
    monkeypatch.setenv('RAG_STORAGE__ROOT', 'old-index')
    monkeypatch.setenv('RAG_MODELS__DEVICE', 'cuda')
    monkeypatch.setenv('RAG_API_KEY', 'keep-local-key')
    env = lab.child_env()
    assert 'RAG_CONFIG' not in env and 'RAG_STORAGE__ROOT' not in env
    assert 'RAG_MODELS__DEVICE' not in env
    assert env['RAG_API_KEY'] == 'keep-local-key'


def test_real_audio_decode_with_lab_dependencies(tmp_path):
    import wave
    audio = pytest.importorskip('faster_whisper.audio')
    path = tmp_path / '中文 sample.wav'
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(8000)
        output.writeframes(b'\x00\x00' * 8000)
    samples = audio.decode_audio(str(path), sampling_rate=16000)
    assert samples.shape == (16000,)
    assert samples.dtype == np.float32
    assert not np.any(samples)


def test_default_quality_profile_and_explicit_medium(monkeypatch):
    monkeypatch.delenv('LAB_ASR_MODEL', raising=False)
    assert lab.chosen_asr_model() == 'large-v3'
    monkeypatch.setenv('LAB_ASR_MODEL', 'medium')
    assert lab.chosen_asr_model() == 'medium'
    monkeypatch.setenv('LAB_ASR_MODEL', 'medium.en')
    with pytest.raises(ValueError):
        lab.chosen_asr_model()


def test_model_profiles_keep_previous_human_reviews(tmp_path):
    source = tmp_path / 'video.mp4'
    source.write_bytes(b'media')
    cfg = Settings()
    cfg.asr.language, cfg.asr.multilingual = 'auto', True
    folder = tmp_path / 'output'
    old = lab.manifest_for(source, [source], folder)[0]
    old_review = Path(old['output']).with_suffix('.review.md')
    old_review.parent.mkdir(parents=True)
    old_review.write_text('Previously corrected by a human.')
    large = lab.manifest_for(source, [source], folder, lab.asr_profile('large-v3', cfg))[0]
    medium = lab.manifest_for(source, [source], folder, lab.asr_profile('medium', cfg))[0]
    cfg.asr.language, cfg.asr.multilingual = 'zh', False
    fixed = lab.manifest_for(source, [source], folder, lab.asr_profile('large-v3', cfg))[0]
    assert len({x['output'] for x in (old, large, medium, fixed)}) == 4
    assert old_review.read_text() == 'Previously corrected by a human.'
    assert 'large-v3-auto-mixed-v1' in large['output']


@pytest.mark.parametrize('language, expected_multilingual', [('auto', True), ('zh', False), ('en', False)])
def test_large_model_config_language_policy(tmp_path, monkeypatch, language, expected_multilingual):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setenv('LAB_LANGUAGE', language)
    cfg = lab.make_config(tmp_path, {'asr': 'local-large-v3', 'embedding': 'bge', 'reranker': 'reranker'})
    assert cfg.asr.model == 'local-large-v3'
    assert cfg.asr.device == 'cpu' and cfg.asr.compute_type == 'int8'
    assert cfg.asr.language == language and cfg.asr.multilingual is expected_multilingual


@pytest.mark.parametrize('language, multilingual', [('auto', True), ('zh', False)])
def test_faster_adapter_preserves_original_language_and_disables_history(language, multilingual):
    from transcribe.faster import FasterWhisperAdapter
    cfg = Settings()
    cfg.asr.multilingual = True
    adapter = FasterWhisperAdapter.__new__(FasterWhisperAdapter)
    adapter.cfg = cfg
    called = {}
    def transcribe(audio, **kwargs):
        called.update(kwargs)
        segment = SimpleNamespace(start=0, end=2, text='先做 calibration 再 verification。',
                                  compression_ratio=1, avg_logprob=0, no_speech_prob=0)
        return iter([segment]), None
    adapter.model = SimpleNamespace(transcribe=transcribe)
    result = adapter.transcribe(np.zeros(16000), language=language)
    assert called['task'] == 'transcribe'
    assert called['multilingual'] is multilingual
    assert called['language'] == (None if language == 'auto' else language)
    assert called['condition_on_previous_text'] is False
    assert result['segments'][0]['text'] == '先做 calibration 再 verification。'
