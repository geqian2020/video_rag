"""抽取 PPT/屏幕帧并使用离线 RapidOCR；保留每条 OCR 的时间、置信度、图像与框。"""
import argparse
import importlib.util
from pathlib import Path
import subprocess
import uuid

from core.config import load_config
from core.io import write_json, write_rows
from core.media import executable, local_media, probe, windows


def load_engine(cfg):
    # 固定 1.x 兼容适配器，默认 ONNX 模型随 wheel 提供，不调用在线下载。
    spec = importlib.util.find_spec('rapidocr_onnxruntime')
    if spec is None or spec.origin is None:
        raise RuntimeError('请在现有环境安装 requirements-ocr.txt，或复用已安装的 rapidocr-onnxruntime 1.4.4')
    bundled = Path(spec.origin).parent / 'models'
    if (cfg.ocr.det_model is None or cfg.ocr.rec_model is None) and not list(bundled.glob('*.onnx')):
        raise FileNotFoundError('包内 OCR 模型缺失；请准备本地 ONNX 文件，不允许自动下载')
    kwargs = {}
    for field, argument in [('det_model', 'det_model_path'), ('rec_model', 'rec_model_path'),
                            ('cls_model', 'cls_model_path'), ('rec_keys', 'rec_keys_path')]:
        value = getattr(cfg.ocr, field)
        if value:
            p = local_media(value)
            kwargs[argument] = str(p)
    from rapidocr_onnxruntime import RapidOCR
    return RapidOCR(max_side_len=cfg.ocr.max_side, **kwargs)


def frame_filter(info, cfg):
    x0, y0, x1, y1 = cfg.ocr.crop
    width, height = info['width'], info['height']
    if not width or not height:
        raise ValueError('文件没有可识别视频画面')
    return (f'crop=iw*{x1-x0}:ih*{y1-y0}:iw*{x0}:ih*{y0},'
            f"scale=w='min({cfg.ocr.max_side},iw*{cfg.ocr.upscale})':"
            f"h='min({cfg.ocr.max_side},ih*{cfg.ocr.upscale})':force_original_aspect_ratio=decrease")


def extract(video, video_id, output, cfg, engine=None):
    if not video_id.strip():
        raise ValueError('video_id 不能为空')
    source = local_media(video)
    info = probe(source)
    run = Path(output) / uuid.uuid4().hex
    frames = run / 'frames'
    frames.mkdir(parents=True)
    engine = engine if engine is not None else load_engine(cfg)
    rows = []
    for i, (ts, _) in enumerate(windows(info['duration'], cfg.ocr.interval_seconds)):
        frame = frames / f'{i:06d}_{ts:.3f}.png'
        subprocess.run([executable('ffmpeg'), '-nostdin', '-v', 'error', '-y', '-ss', str(ts),
            '-protocol_whitelist', 'file,pipe', '-i', str(source), '-frames:v', '1', '-vf', frame_filter(info, cfg), str(frame)],
            check=True, capture_output=True)
        if not frame.is_file():
            raise RuntimeError(f'抽帧失败，采样序号 {i}')
        result, _ = engine(str(frame))
        for box, text, confidence in (result if result is not None else []):
            rows.append({'video_id': video_id, 'timestamp': ts, 'frame_path': str(frame.resolve()),
                'text': str(text), 'confidence': float(confidence),
                'box': [[float(x), float(y)] for x, y in box]})
        print(f'OCR 已处理第 {i+1} 帧', flush=True)
    write_rows(run / 'observations.jsonl', rows)
    write_json(run / 'manifest.json', {'video_id': video_id, 'duration': info['duration'],
        'crop': cfg.ocr.crop, 'interval_seconds': cfg.ocr.interval_seconds,
        'upscale': cfg.ocr.upscale, 'engine': 'rapidocr-onnxruntime-1.4.4',
        'frame_timestamps': '采样目标时间；非逐字对齐', 'observations': len(rows)})
    return run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', required=True)
    p.add_argument('--video-id', required=True)
    p.add_argument('--output', default='data/ocr')
    p.add_argument('--config', default=None)
    a = p.parse_args()
    print('OCR 输出目录：', extract(a.video, a.video_id, a.output, load_config(a.config)))


if __name__ == '__main__':
    main()
