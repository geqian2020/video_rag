"""输入严格校验；时间单位统一为秒，禁止 NaN、负数、空文本。"""
import re
from pydantic import Field, field_validator, model_validator
from core.config import Strict


class Segment(Strict):
    video_id: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1)
    start: float = Field(ge=0, allow_inf_nan=False)
    end: float = Field(gt=0, allow_inf_nan=False)
    timestamp_quality: str = Field("asr", pattern="^(asr|estimated)$")

    @field_validator("text", "video_id")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("不允许纯空白")
        return value

    @model_validator(mode="after")
    def interval(self):
        if self.end <= self.start:
            raise ValueError("end 必须大于 start")
        return self


class Question(Strict):
    question: str = Field(min_length=4, max_length=200)
    evidence: str = Field(min_length=2)


class Enhancement(Strict):
    summary: str = Field(min_length=1)
    questions: list[Question] = Field(min_length=3, max_length=5)


def validate_enhancement(text, enhancement):
    """摘要采用原文摘录；证据字面检查不等于语义蕴含证明。"""
    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?])|\n", text) if s.strip()]
    if enhancement.summary not in sentences:
        raise ValueError("摘要必须为原文中的完整一句摘录")
    if len({q.question for q in enhancement.questions}) != len(enhancement.questions):
        raise ValueError("问题不可重复")
    for q in enhancement.questions:
        if q.evidence not in text:
            raise ValueError("问题证据必须来自原文")
        for number in re.findall(r"\d+(?:\.\d+)?", q.question):
            if number not in text:
                raise ValueError("问题包含原文没有的数字")
    return enhancement


class Chunk(Strict):
    chunk_id: str
    video_id: str
    start_ts: float = Field(ge=0, allow_inf_nan=False)
    end_ts: float = Field(gt=0, allow_inf_nan=False)
    text: str = Field(min_length=1)
    source_segments: list[Segment] = Field(min_length=1)
    timestamp_quality: str
    summary: str = ""
    questions: list[Question] = []
    enrichment_status: str = Field("pending", pattern="^(pending|validated|extractive_fallback)$")

    @model_validator(mode="after")
    def provenance(self):
        segments = self.source_segments
        if any(s.video_id != self.video_id for s in segments):
            raise ValueError("chunk 不可跨视频")
        if self.start_ts != min(s.start for s in segments) or self.end_ts != max(s.end for s in segments):
            raise ValueError("chunk 时间必须包围其原始 segments")
        if self.text != "\n".join(s.text for s in segments):
            raise ValueError("chunk 原文必须与 source_segments 一致")
        quality = "estimated" if any(s.timestamp_quality == "estimated" for s in segments) else "asr"
        if self.timestamp_quality != quality:
            raise ValueError("时间质量标记不一致")
        if self.enrichment_status == "validated":
            validate_enhancement(self.text, Enhancement(summary=self.summary, questions=self.questions))
        elif self.enrichment_status == "extractive_fallback":
            if self.summary not in self.text or not self.summary or self.questions:
                raise ValueError("降级摘要必须为原文摘录，且无问题")
        return self
