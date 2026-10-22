"""确认词汇与提示词选择。只有人工 approved 的标准拼写可进入 ASR。"""
import json
import re
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from core.config import Strict


class Term(Strict):
    term: str = Field(min_length=1, max_length=128)
    approved: bool = False
    scope: str = Field('video', pattern='^(global|video)$')
    video_id: str | None = None
    timestamps: list[float] = []
    priority: int = Field(0, ge=-100, le=100)
    aliases: list[str] = []  # 仅供审核保留；ASR 只使用确认后的标准 term
    evidence: list[dict] = []

    @field_validator('term')
    @classmethod
    def clean_term(cls, value):
        if not value.strip() or re.search(r'[\x00-\x1f\x7f]', value):
            raise ValueError('term 必须为非空单行文字')
        return value.strip()

    @field_validator('timestamps')
    @classmethod
    def valid_timestamps(cls, value):
        import math
        if any(not math.isfinite(v) or v < 0 for v in value):
            raise ValueError('词汇时间戳必须为非负有限数')
        return sorted(set(value))

    @model_validator(mode='after')
    def valid_scope(self):
        if self.scope == 'video' and not self.video_id:
            raise ValueError('video 范围词汇必须指定 video_id')
        if self.scope == 'global' and self.video_id is not None:
            raise ValueError('global 词汇的 video_id 必须为空')
        return self


class Vocabulary(Strict):
    schema_version: int = Field(1, ge=1, le=1)
    entries: list[Term] = []


def load_vocabulary(path):
    if path is None:
        return Vocabulary()
    return Vocabulary.model_validate(json.loads(Path(path).read_text(encoding='utf-8-sig')))


def build_prompt(vocab, video_id, start, end, cfg, token_count):
    # 局部画面术语优先。无 timestamps 的人工 video 词条适用于整段视频。
    candidates = []
    for entry in vocab.entries:
        if not entry.approved:
            continue
        if entry.scope == 'video':
            if entry.video_id != video_id:
                continue
            if entry.timestamps and not any(start-cfg.asr.context_margin <= t <= end+cfg.asr.context_margin
                                            for t in entry.timestamps):
                continue
        candidates.append(entry)
    candidates.sort(key=lambda t: (t.scope != 'video', -t.priority, t.term.casefold()))
    selected, seen = [], set()
    for entry in candidates:
        key = entry.term.casefold()
        if key in seen:
            continue
        proposal = '术语拼写：' + '、'.join(selected + [entry.term]) + '。'
        if token_count(proposal) > cfg.asr.prompt_tokens:
            continue  # 不截断词或依赖 Whisper 在尾部静默裁切
        selected.append(entry.term)
        seen.add(key)
        if len(selected) >= cfg.asr.max_terms:
            break
    return ('术语拼写：' + '、'.join(selected) + '。' if selected else ''), selected
