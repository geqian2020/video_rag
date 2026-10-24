"""使用已确认的局部 OCR 术语提示本地 Whisper，直接输出 RAG 所需 JSONL。"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

from core.config import load_config
from core.io import write_json, write_rows
from core.media import extract_audio, local_media, probe, windows
from core.schema import Segment
from core.vocabulary import build_prompt, load_vocabulary


def log_progress(message):
    """A console encoding error must never abort a successful ASR window."""
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        # Some Windows redirected streams default to cp1252 instead of UTF-8.
        # Use ASCII escapes as a fallback; do not suppress other I/O failures.
        print(message.encode('ascii', errors='backslashreplace').decode('ascii'), flush=True)


def load_local_whisper(cfg):
    import whisper
    name = cfg.asr.model
    if Path(name).is_file():
        checkpoint = Path(name).resolve()
    else:
        # 与 Whisper 官方的默认缓存规则一致；先检查，再只用本地路径加载。
        cache = Path(cfg.asr.model_cache or (Path(os.getenv('XDG_CACHE_HOME', Path.home()/'.cache')) / 'whisper'))
        if name not in whisper.available_models():
            raise ValueError('未知模型名；请使用已有 Whisper 模型名或本地 .pt 路径')
        checkpoint = cache / (name + '.pt')
        if not checkpoint.is_file():
            raise FileNotFoundError('Whisper 权重未缓存；请指定已有 .pt 路径，不自动下载')
    # .pt 为 torch checkpoint，仅加载来源可信的已批准权重。
    return whisper.load_model(str(checkpoint), device=cfg.asr.device)


def _window_fingerprint(video, video_id, cfg, start, end, prompt, selected):
    # Deliberately do not hash video contents: path identity is enough for this workflow.
    settings = {
        'schema': 1, 'video': str(video), 'video_id': video_id,
        'model': cfg.asr.model, 'device': cfg.asr.device,
        'language': cfg.asr.language, 'window_seconds': cfg.asr.window_seconds,
        'prompt_style': cfg.asr.prompt_style, 'prompt_tokens': cfg.asr.prompt_tokens,
        'start': start, 'end': end, 'prompt': prompt, 'selected_terms': selected,
    }
    return hashlib.sha256(json.dumps(settings, ensure_ascii=False, sort_keys=True).encode('utf-8')).hexdigest()


def _checkpoint_digest(rows, audit):
    payload = {'rows': rows, 'audit': audit}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode('utf-8')).hexdigest()


def _read_window_checkpoint(path, fingerprint, video_id, start, end):
    if not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding='utf-8'))
        rows, audit = cached['rows'], cached['audit']
        if cached['version'] != 1 or cached['fingerprint'] != fingerprint:
            return None
        if not isinstance(rows, list) or not isinstance(audit, dict):
            return None
        if cached['digest'] != _checkpoint_digest(rows, audit):
            return None
        if audit['window_start'] != start or audit['window_end'] != end:
            return None
        if audit['segment_count'] < len(rows):
            return None
        # Validate cached fields and ensure a checkpoint cannot inject other video IDs or times.
        for row in rows:
            seg = Segment.model_validate(row)
            if seg.video_id != video_id or seg.start < start or seg.end > end or seg.end <= seg.start:
                return None
        return rows, audit
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def transcribe_video(video, video_id, vocab, cfg, model, token_count, output, audio_source=None):
    if not video_id.strip():
        raise ValueError('video_id 不可为空')
    # A prepared local audio source prevents repeated reads of a OneDrive cloud video.
    # The original video identity is retained for window checkpoint fingerprints.
    media_for_audio = local_media(audio_source) if audio_source is not None else video
    duration = probe(media_for_audio)['duration']
    rows, audit = [], []
    checkpoints = Path(str(output) + '.windows')
    with tempfile.TemporaryDirectory(prefix='video-kb-asr-') as temp:
        for i, (start, end) in enumerate(windows(duration, cfg.asr.window_seconds)):
            prompt, selected = build_prompt(vocab, video_id, start, end, cfg, token_count)
            fingerprint = _window_fingerprint(video, video_id, cfg, start, end, prompt, selected)
            checkpoint = checkpoints / f'{i:06d}.json'
            recovered = _read_window_checkpoint(checkpoint, fingerprint, video_id, start, end)
            if recovered is not None:
                saved_rows, saved_audit = recovered
                rows.extend(saved_rows)
                audit.append(saved_audit)
                log_progress(f'复用已完成窗口 {i+1}')
                continue

            audio = Path(temp) / 'window.wav'
            extract_audio(media_for_audio, audio, start, end)
            kwargs = {'language': cfg.asr.language, 'task': 'transcribe', 'verbose': False,
                'fp16': cfg.asr.device.startswith('cuda'), 'condition_on_previous_text': False,
                'temperature': 0.0, 'initial_prompt': prompt or None, 'carry_initial_prompt': True}
            result = model.transcribe(str(audio), **kwargs)
            # Release temporary audio before persisting the completed window.
            audio.unlink()
            window_rows = []
            for seg in result.get('segments', []):
                text = str(seg['text'])
                if not text.strip():
                    continue
                begin, finish = float(seg['start']), float(seg['end'])
                if not math.isfinite(begin) or not math.isfinite(finish) or begin < 0 or finish <= begin:
                    raise ValueError('ASR 返回非法 segment 时间')
                absolute_start = start + min(begin, end-start)
                absolute_end = start + min(finish, end-start)
                if absolute_end <= absolute_start:
                    continue
                window_rows.append(Segment(video_id=video_id, start=absolute_start, end=absolute_end,
                                           text=text, timestamp_quality='asr').model_dump())
            quality = [{k: seg.get(k) for k in ['compression_ratio', 'avg_logprob', 'no_speech_prob']}
                       for seg in result.get('segments', [])]
            review = any((q['compression_ratio'] is not None and q['compression_ratio'] > 2.4)
                         or (q['avg_logprob'] is not None and q['avg_logprob'] < -1)
                         or (q['no_speech_prob'] is not None and q['no_speech_prob'] > .6) for q in quality)
            window_audit = {'quality': quality, 'review_required': review, 'window_start': start,
                            'window_end': end, 'selected_terms': selected,
                            'prompt_tokens': token_count(prompt) if prompt else 0,
                            'segment_count': len(result.get('segments', []))}
            # Atomic publish: if interrupted during the write, the checkpoint is not reused.
            write_json(checkpoint, {'version': 1, 'fingerprint': fingerprint,
                                   'rows': window_rows, 'audit': window_audit,
                                   'digest': _checkpoint_digest(window_rows, window_audit)})
            rows.extend(window_rows)
            audit.append(window_audit)
            log_progress(f'转录已完成窗口 {i+1}（断点已保存）')
    if not rows:
        raise ValueError('未识别到有效语音；未发布新转录，请检查音轨与模型')
    # Only publish final transcript and audit after every window has a verified result.
    write_rows(output, rows)
    audit_path = str(output) + '.audit.json'
    write_json(audit_path, {'video_id': video_id, 'model': cfg.asr.model,
        'mode': 'vocabulary_assisted' if any(w['selected_terms'] for w in audit) else 'no_terms_selected',
        'prompt_style': cfg.asr.prompt_style,
        'review_required_windows': sum(w['review_required'] for w in audit),
        'warning': '提示词仅影响识别偏好；不证明音频确实说了术语，需抽查', 'windows': audit,
        'transcript_sha256': hashlib.sha256(Path(output).read_bytes()).hexdigest()})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--video-id', required=True)
    p.add_argument('--audio-source', default=None,
                   help='Prepared local PCM audio; preserves original --video identity and window checkpoints')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--vocabulary', help='仅加载 approved=true 的词汇')
    mode.add_argument('--no-vocab', action='store_true', help='无词表基线，供同段音频对比')
    p.add_argument('--output', default='data/transcripts.jsonl')
    p.add_argument('--config', default=None)
    a = p.parse_args()
    cfg = load_config(a.config)
    video = local_media(a.video)
    vocab = load_vocabulary(a.vocabulary)
    model = load_local_whisper(cfg)
    if not model.is_multilingual and cfg.asr.language != 'en':
        raise ValueError('中文转录不能使用 .en 英语专用模型')
    from whisper.tokenizer import get_tokenizer
    tokenizer = get_tokenizer(model.is_multilingual, num_languages=model.num_languages,
                              language=cfg.asr.language, task='transcribe')
    cfg.asr.prompt_tokens = min(cfg.asr.prompt_tokens, model.dims.n_text_ctx//2 - 16)
    count = lambda text: len(tokenizer.encode(' ' + text.strip()))
    transcribe_video(video, a.video_id, vocab, cfg, model, count, a.output,
                     audio_source=a.audio_source)
    log_progress('转录完成；输出 JSONL 与 .audit.json，可继续 ingest.chunk')


if __name__ == '__main__':
    main()
