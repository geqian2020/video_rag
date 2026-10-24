"""Offline tests for V7 bounded audio prefetch, dedup and restart."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time

import pytest

from core.io import write_json, write_rows
from workflow.pipelined_asr import Pipeline, load_manifest


def _complete(video):
    rows = [{'video_id': video.video_id, 'start': 1, 'end': 2, 'text': 'transcribed'}]
    write_rows(video.transcript, rows)
    write_json(str(video.transcript) + '.audit.json', {
        'video_id': video.video_id, 'model': 'small', 'prompt_style': 'terms',
        'transcript_sha256': hashlib.sha256(video.transcript.read_bytes()).hexdigest()})


def _plan(tmp_path, names):
    source = tmp_path / 'cloud'
    source.mkdir()
    data = tmp_path / 'data'
    (data / 'transcripts').mkdir(parents=True)
    manifest = tmp_path / 'list.nul'
    manifest.write_bytes(b''.join(str(source / n).encode('utf-8') + b'\0' for n in names))
    videos = load_manifest(manifest, source, data / 'transcripts')
    return source, data, videos


def test_old_complete_and_same_filename_never_read_cloud(monkeypatch, tmp_path):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local'))
    source, data, videos = _plan(tmp_path, ['x/中文培训.mp4', 'y/中文培训.mp4'])
    _complete(videos[1])  # completed source comes later in scan list
    pipeline = Pipeline(source, data, videos, interval=.01)
    monkeypatch.setattr(pipeline, 'prepare_audio',
                        lambda v: pytest.fail('Must not read cloud media'))
    released = []
    monkeypatch.setattr(pipeline, 'release_original',
                        lambda v: released.append(v.relative))
    summary = pipeline.process()
    assert summary['new'] == 0 and summary['reused'] == 2
    assert len(released) == 2  # attrib-only requests, no media bytes read
    assert all(pipeline.is_complete(v) for v in videos)
    assert videos[0].video_id in videos[0].transcript.read_text(encoding='utf-8')


def test_only_one_serial_prefetch_with_foreground_asr(monkeypatch, tmp_path):
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local'))
    source, data, videos = _plan(tmp_path, ['a/A.mp4', 'b/A.mp4', 'c/B.mp4', 'd/C.mp4'])
    pipeline = Pipeline(source, data, videos, interval=.01)
    prepared, transcribed, released, cleaned = [], [], [], []
    current = [0]
    maximum = [0]
    mutex = threading.Lock()

    def stage(video):
        with mutex:
            current[0] += 1
            maximum[0] = max(maximum[0], current[0])
            prepared.append(video.relative)
        time.sleep(.02)
        with mutex:
            current[0] -= 1
        return tmp_path / 'fake-stage.wav'

    def asr(video, audio):
        assert audio.name == 'fake-stage.wav'
        transcribed.append(video.relative)
        _complete(video)
    monkeypatch.setattr(pipeline, 'prepare_audio', stage)
    monkeypatch.setattr(pipeline, 'run_asr', asr)
    monkeypatch.setattr(pipeline, 'release_original',
                        lambda v: released.append(v.relative))
    monkeypatch.setattr(pipeline, 'cleanup_audio',
                        lambda v: cleaned.append(v.relative))
    result = pipeline.process()
    assert result == {'new': 3, 'reused': 1, 'sources': 4, 'unique_names': 3}
    assert maximum[0] == 1
    assert len(prepared) == len(transcribed) == len(cleaned) == 3
    assert len(released) == 4  # includes the completed same-name duplicate
    assert all(pipeline.is_complete(v) for v in videos)


def test_completed_audio_staging_survives_missing_cloud_video(monkeypatch, tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe not available')
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local'))
    source, data, videos = _plan(tmp_path, ['demo.mp4'])
    mp4 = source / 'demo.mp4'
    subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-f', 'lavfi',
                    '-i', 'sine=frequency=440:duration=2', '-c:a', 'aac', '-y', str(mp4)],
                   check=True, capture_output=True)
    pipeline = Pipeline(source, data, videos, interval=.05)
    wav = pipeline.prepare_audio(videos[0])
    assert wav.is_file() and pipeline.prepared(videos[0])
    assert str(wav).startswith(str(data / 'temp' / 'audio_stage'))
    mp4.unlink()  # local test: cloud no longer available; reuse only cached WAV
    assert pipeline.prepare_audio(videos[0]) == wav
    pipeline.cleanup_audio(videos[0])
    assert not wav.exists()


def test_stage_dir_recreated_while_cloud_input_is_slow(monkeypatch, tmp_path):
    """Regression: FFmpeg waited on OneDrive, output directory vanished."""
    if not shutil.which('ffmpeg'):
        pytest.skip('FFmpeg unavailable for output-path smoke test')
    import workflow.pipelined_asr as module
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local'))
    source, data, videos = _plan(tmp_path, ['demo.mp4'])
    pipeline = Pipeline(source, data, videos, interval=.01)
    monkeypatch.setattr(module, 'probe', lambda _: {'duration': 1.0})

    class Hydrating:
        def __init__(self, cmd, stdin, stdout, stderr):
            self.output = Path(cmd[-1])
            self.iteration = 0
            self.finished = False
            shutil.rmtree(self.output.parent)  # folder removed during cloud read

        def wait(self, timeout=None):
            if not self.iteration:
                self.iteration = 1
                raise subprocess.TimeoutExpired('ffmpeg', timeout)
            assert self.output.parent.is_dir()
            self.output.write_bytes(b'RIFF' + bytes(128))
            self.finished = True
            return 0

        def poll(self):
            return 0 if self.finished else None

        def terminate(self):
            self.finished = True

    monkeypatch.setattr(module.subprocess, 'Popen', Hydrating)
    wav = pipeline.prepare_audio(videos[0])
    assert wav.is_file()
    assert wav.stat().st_size > 44
    assert pipeline.prepared(videos[0])


def test_retry_once_on_missing_ffmpeg_output_path(monkeypatch, tmp_path):
    """A missing output directory must trigger one retry, not skip the video."""
    if not shutil.which('ffmpeg'):
        pytest.skip('FFmpeg unavailable for output-path smoke test')
    import workflow.pipelined_asr as module
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'local'))
    source, data, videos = _plan(tmp_path, ['demo.mp4'])
    pipeline = Pipeline(source, data, videos, interval=.01)
    monkeypatch.setattr(module, 'probe', lambda _: {'duration': 1.0})
    attempts = []

    class IntermittentOutput:
        def __init__(self, cmd, stdin, stdout, stderr):
            attempts.append(1)
            self.output = Path(cmd[-1])
            self.err = stderr

        def wait(self, timeout=None):
            if len(attempts) == 1:
                self.err.write(b'Error opening output file: No such file or directory\\n')
                return 2
            self.output.write_bytes(b'RIFF' + bytes(128))
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(module.subprocess, 'Popen', IntermittentOutput)
    wav = pipeline.prepare_audio(videos[0])
    assert wav.is_file()
    assert len(attempts) == 2
    assert pipeline.prepared(videos[0])


def test_audio_paths_follow_configured_data_root_not_appdata(monkeypatch, tmp_path):
    """Corporate machines may deny LOCALAPPDATA; DATA_IN owns all temp files."""
    monkeypatch.setenv('LOCALAPPDATA', str(tmp_path / 'it-restricted'))
    monkeypatch.setenv('RAG_AUDIO_STAGE_DIR', str(tmp_path / 'stale-override'))
    source, data, videos = _plan(tmp_path, ['video.mp4'])
    pipeline = Pipeline(source, data, videos, interval=.01)
    assert pipeline.stage_dir == data / 'temp' / 'audio_stage'
    assert pipeline.window_temp == data / 'temp' / 'asr_windows'
    assert pipeline.stage_dir.is_dir()
    assert pipeline.window_temp.is_dir()
    assert not (tmp_path / 'it-restricted').exists()
    assert not (tmp_path / 'stale-override').exists()
