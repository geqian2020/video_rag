"""使用已确认的局部 OCR 术语提示本地 Whisper，直接输出 RAG 所需 JSONL。"""
import argparse
import hashlib
import math
import os
from pathlib import Path
import tempfile

from core.config import load_config
from core.io import write_json, write_rows
from core.media import extract_audio, local_media, probe, windows
from core.schema import Segment
from core.vocabulary import build_prompt, load_vocabulary


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


def transcribe_video(video, video_id, vocab, cfg, model, token_count, output):
    if not video_id.strip():
        raise ValueError('video_id 不可为空')
    duration = probe(video)['duration']
    rows, audit = [], []
    with tempfile.TemporaryDirectory(prefix='video-kb-asr-') as temp:
        for i, (start, end) in enumerate(windows(duration, cfg.asr.window_seconds)):
            audio = Path(temp) / 'window.wav'
            extract_audio(video, audio, start, end)
            prompt, selected = build_prompt(vocab, video_id, start, end, cfg, token_count)
            kwargs = {'language': cfg.asr.language, 'task': 'transcribe', 'verbose': False,
                'fp16': cfg.asr.device.startswith('cuda'), 'condition_on_previous_text': False,
                'temperature': 0.0, 'initial_prompt': prompt or None, 'carry_initial_prompt': True}
            result = model.transcribe(str(audio), **kwargs)
            # 逐窗口清理，避免长视频临时音频累积；异常时 TemporaryDirectory 仍清理。
            audio.unlink()
            for seg in result.get('segments', []):
                text = str(seg['text'])
                if not text.strip():
                    continue
                begin, finish = float(seg['start']), float(seg['end'])
                if not math.isfinite(begin) or not math.isfinite(finish) or begin < 0 or finish <= begin:
                    raise ValueError('ASR 返回非法 segment 时间')
                # Whisper 窗口结束可能略越过文件末尾，限制到实际窗口包络。
                absolute_start = start + min(begin, end-start)
                absolute_end = start + min(finish, end-start)
                if absolute_end <= absolute_start:
                    continue
                rows.append(Segment(video_id=video_id, start=absolute_start, end=absolute_end,
                                    text=text, timestamp_quality='asr').model_dump())
            audit.append({'window_start': start, 'window_end': end, 'selected_terms': selected,
                          'prompt_tokens': token_count(prompt) if prompt else 0,
                          'segment_count': len(result.get('segments', []))})
            print(f'转录已完成窗口 {i+1}', flush=True)
    if not rows:
        raise ValueError('未识别到有效语音；未发布新转录，请检查音轨与模型')
    # 首先保留本次完整输出；无任何字符串替换，不将 OCR 原文作为 spoken evidence。
    write_rows(output, rows)
    audit_path = str(output) + '.audit.json'
    write_json(audit_path, {'video_id': video_id, 'model': cfg.asr.model,
        'mode': 'vocabulary_assisted' if any(w['selected_terms'] for w in audit) else 'no_terms_selected',
        'warning': '提示词仅影响识别偏好；不证明音频确实说了术语，需抽查', 'windows': audit,
        'transcript_sha256': hashlib.sha256(Path(output).read_bytes()).hexdigest()})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--video-id', required=True)
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
    transcribe_video(video, a.video_id, vocab, cfg, model, count, a.output)
    print('转录完成；输出 JSONL 与 .audit.json，可继续 ingest.chunk')


if __name__ == '__main__':
    main()
