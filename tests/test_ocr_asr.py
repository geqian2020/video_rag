"""OCR 词汇审核、作用域、prompt 与 Whisper 输出对接测试。"""
import csv
import io
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

from core.config import Settings
from core.io import read_rows
from core.media import windows
from core.vocabulary import Term, Vocabulary, build_prompt
from eval.run_asr_eval import contains, evaluate
from ocr.extract import extract
from ocr.vocab import candidates, export_csv, import_review
from transcribe.run import transcribe_video


def observation(term='Qdrant', ts=0, video='a', confidence=.95):
    return {'video_id': video, 'timestamp': ts, 'frame_path': f'frame-{ts}.png',
            'text': term, 'confidence': confidence, 'box': [[0, 0], [40, 0], [40, 20], [0, 20]]}


def test_repeated_frames_never_auto_approve():
    cfg = Settings()
    out = candidates([observation(ts=i) for i in range(5)], cfg)
    assert len(out) == 1 and out[0]['approved'] == ''
    assert out[0]['frame_count'] == 5
    assert out[0]['evidence_strength'] == 'repeated'


def test_same_timestamp_duplicate_does_not_inflate_support():
    out = candidates([observation(), observation()], Settings())
    assert out[0]['frame_count'] == 1


def test_low_confidence_retained_only_in_raw():
    assert candidates([observation(confidence=.3)], Settings()) == []
    with pytest.raises(ValueError):
        candidates([observation(confidence=float('nan'))], Settings())


def test_confusable_names_not_silently_merged():
    out = candidates([observation('Qdrant'), observation('Qdranl')], Settings())
    assert len(out) == 2
    assert all(row['similar_candidates'] for row in out)


def test_csv_review_corrects_spelling_and_requires_approval(tmp_path):
    path = tmp_path/'review.csv'
    rows = candidates([observation('Qdranl'), observation('NoiseTerm')], Settings())
    export_csv(path, rows)
    assert import_review(path).entries == []
    with pytest.raises(FileExistsError):
        export_csv(path, rows)
    for row in rows:
        if row['ocr_term'] == 'Qdranl':
            row.update(canonical_term='Qdrant', approved='yes')
    path.unlink()
    export_csv(path, rows)
    vocab = import_review(path)
    assert len(vocab.entries) == 1 and vocab.entries[0].term == 'Qdrant'
    assert vocab.entries[0].evidence[0]['text'] == 'Qdranl'
    prompt, selected = build_prompt(vocab, 'a', 0, 30, Settings(), len)
    assert selected == ['Qdrant'] and 'Qdranl' not in prompt


def test_context_and_video_scope_and_token_budget():
    cfg = Settings()
    cfg.asr.context_margin = 0
    cfg.asr.prompt_tokens = 20
    vocab = Vocabulary(entries=[Term(term='Qdrant', approved=True, video_id='a', timestamps=[10]),
        Term(term='WrongVideo', approved=True, video_id='b'),
        Term(term='DistantTerm', approved=True, video_id='a', timestamps=[500]),
        Term(term='Unreviewed', approved=False, video_id='a'),
        Term(term='BM25', approved=True, scope='global')])
    prompt, selected = build_prompt(vocab, 'a', 0, 30, cfg, len)
    assert selected == ['Qdrant', 'BM25'] and len(prompt) <= 20
    assert build_prompt(vocab, 'b', 0, 30, cfg, len)[1][0] == 'WrongVideo'
    cfg.asr.prompt_tokens = 3
    assert build_prompt(vocab, 'a', 0, 30, cfg, len) == ('', [])


def test_invalid_crop_and_vocabulary_time():
    with pytest.raises(ValueError):
        Settings(ocr={'crop': [0.8, 0, .2, 1]})
    with pytest.raises(ValueError):
        Term(term='Qdrant', approved=True, video_id='a', timestamps=[float('nan')])
    assert list(windows(61, 30)) == [(0, 30), (30, 60), (60, 61)]


def test_offline_whisper_refuses_missing_checkpoint(monkeypatch, tmp_path):
    import sys
    from transcribe.run import load_local_whisper
    fake = SimpleNamespace(available_models=lambda: ['small'],
        load_model=lambda *args, **kwargs: pytest.fail('不得触发自动下载'))
    monkeypatch.setitem(sys.modules, 'whisper', fake)
    cfg = Settings(asr={'model_cache': str(tmp_path)})
    with pytest.raises(FileNotFoundError):
        load_local_whisper(cfg)


def test_whisper_local_file_reuse(monkeypatch, tmp_path):
    import sys
    from transcribe.run import load_local_whisper
    model = tmp_path/'small.pt'
    model.write_bytes(b'checkpoint stub')
    def loader(path, device):
        assert path == str(model) and device == 'cpu'
        return 'loaded'
    monkeypatch.setitem(sys.modules, 'whisper', SimpleNamespace(load_model=loader))
    assert load_local_whisper(Settings(asr={'model': str(model)})) == 'loaded'


def test_asr_receives_only_confirmed_prompt_and_preserves_output(monkeypatch, tmp_path):
    import transcribe.run as run
    cfg = Settings()
    cfg.asr.window_seconds = 30
    monkeypatch.setattr(run, 'probe', lambda _: {'duration': 45})
    monkeypatch.setattr(run, 'extract_audio', lambda _v, p, _s, _e: p.write_bytes(b'audio stub'))
    calls = []
    class Model:
        def transcribe(self, audio, **kwargs):
            calls.append(kwargs)
            return {'segments': [{'start': 1, 'end': 3, 'text': '模型原始输出，不做替换。'}]}
    vocab = Vocabulary(entries=[Term(term='Qdrant', approved=True, video_id='a', timestamps=[0]),
                                  Term(term='Qdranl', approved=False, video_id='a')])
    output = tmp_path/'transcript.jsonl'
    rows = transcribe_video('stub.mp4', 'a', vocab, cfg, Model(), len, output)
    assert 'Qdrant' in calls[0]['initial_prompt'] and 'Qdranl' not in calls[0]['initial_prompt']
    assert calls[0]['carry_initial_prompt'] is True
    assert rows[1]['start'] == 31 and rows[1]['end'] == 33
    assert rows[0]['text'] == '模型原始输出，不做替换。'
    assert read_rows(output) == rows
    audit = json.loads(Path(str(output)+'.audit.json').read_text())
    assert len(audit['windows']) == 2


def test_asr_failure_does_not_overwrite_existing_output(monkeypatch, tmp_path):
    import transcribe.run as run
    monkeypatch.setattr(run, 'probe', lambda _: {'duration': 10})
    monkeypatch.setattr(run, 'extract_audio', lambda _v, p, _s, _e: p.write_bytes(b'x'))
    class Broken:
        def transcribe(self, *_args, **_kwargs):
            raise RuntimeError('故障注入')
    output = tmp_path/'output.jsonl'; output.write_text('old')
    with pytest.raises(RuntimeError):
        transcribe_video('stub', 'a', Vocabulary(), Settings(), Broken(), len, output)
    assert output.read_text() == 'old'


def test_asr_quality_metrics_penalize_inserted_terms():
    vocab = Vocabulary(entries=[Term(term='Qdrant', approved=True, scope='global'),
                                Term(term='BM25', approved=True, scope='global')])
    ref = [{'video_id': 'a', 'text': '使用Qdrant'}]
    better = evaluate(ref, ref, vocab)
    worse = evaluate(ref, [{'video_id': 'a', 'text': '使用Qdranl BM25'}], vocab)
    assert better['cer'] == 0 and better['term_recall'] == 1
    assert worse['term_recall'] == 0 and worse['extra_terms_not_in_reference']['a'] == ['BM25']
    assert contains('PA test', 'PA') and not contains('capacity', 'PA')


def test_real_ffmpeg_sampling_with_injected_ocr(tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg 未安装')
    video = tmp_path/'demo.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=white:s=320x180:r=1',
                    '-t', '2', '-c:v', 'mpeg4', str(video)], check=True, capture_output=True)
    cfg = Settings(ocr={'interval_seconds': 1, 'crop': [.1, .1, .9, .9]})
    def engine(path):
        assert Path(path).read_bytes().startswith(b'\x89PNG')
        return [[[[0, 0], [50, 0], [50, 20], [0, 20]], 'Qdrant', .9]], [0.01]
    folder = extract(video, 'a', tmp_path/'ocr', cfg, engine)
    rows = read_rows(folder/'observations.jsonl')
    assert [r['timestamp'] for r in rows] == [0, 1]
    assert all(Path(r['frame_path']).is_file() for r in rows)
