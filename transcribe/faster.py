"""Local-only faster-whisper adapter; preserve the existing checkpoint contract."""


class FasterWhisperAdapter:
    def __init__(self, cfg):
        from faster_whisper import WhisperModel
        from ctranslate2 import get_supported_compute_types
        self.cfg = cfg
        supported = get_supported_compute_types(cfg.asr.device)
        requested = cfg.asr.compute_type
        if requested not in supported:
            requested = next((x for x in ('int8_float16', 'int8_float32', 'int8', 'float32')
                              if x in supported), 'float32')
            print(f'[ASR] compute_type={cfg.asr.compute_type} unavailable; using {requested}', flush=True)
        self.model = WhisperModel(cfg.asr.model, device=cfg.asr.device,
                                  compute_type=requested, cpu_threads=cfg.asr.cpu_threads,
                                  num_workers=1, local_files_only=True)
        if not self.model.model.is_multilingual and cfg.asr.language != 'en':
            raise ValueError('非英语转录不能使用 .en 模型')

    def token_count(self, text):
        return len(self.model.hf_tokenizer.encode(' ' + text.strip()).ids)

    def transcribe(self, audio, **kwargs):
        segments, _ = self.model.transcribe(
            audio, language=None if kwargs['language'] == 'auto' else kwargs['language'], task='transcribe',
            beam_size=self.cfg.asr.beam_size, temperature=0.0,
            condition_on_previous_text=False, initial_prompt=kwargs.get('initial_prompt'),
            word_timestamps=False, vad_filter=False,
        )
        keys = ('start', 'end', 'text', 'compression_ratio', 'avg_logprob', 'no_speech_prob')
        return {'segments': [{key: getattr(seg, key) for key in keys} for seg in segments]}
