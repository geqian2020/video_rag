"""gold set 支持 chunk_id 或 video_id+时间范围，评估召回阶段与重排阶段。"""
import argparse
import json

from core.config import load_config
from core.io import read_rows
from retrieve.search import Searcher


def relevant(hit, gold):
    if "chunk_id" in gold:
        return hit["chunk_id"] == gold["chunk_id"]
    if hit["video_id"] != gold["video_id"]:
        return False
    # 至少覆盖 gold 窗口的一半，避免仅相交一个瞬间就算命中。
    overlap = max(0, min(hit["end_ts"], gold["end_ts"]) - max(hit["start_ts"], gold["start_ts"]))
    return overlap / (gold["end_ts"] - gold["start_ts"]) >= .5


def metrics(hits, gold, k):
    # Recall 以 gold 证据数为分母，多条返回命中同一证据只计一次。
    matched = {i for i, item in enumerate(gold) if any(relevant(h, item) for h in hits[:k])}
    reciprocal_rank = next((1 / rank for rank, hit in enumerate(hits[:k], 1)
                            if any(relevant(hit, g) for g in gold)), 0.0)
    return {"recall": len(matched) / len(gold), "mrr": reciprocal_rank}


def validate_gold(rows):
    if not rows:
        raise ValueError("gold set 不能为空")
    for row in rows:
        if not isinstance(row.get("query"), str) or not row["query"].strip() or not row.get("relevant"):
            raise ValueError("每条 gold 必须有 query 和非空 relevant")
        for g in row["relevant"]:
            if "chunk_id" not in g:
                from core.schema import Segment
                Segment(video_id=g["video_id"], text="gold", start=g["start_ts"], end=g["end_ts"])
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gold", default="examples/gold.jsonl")
    p.add_argument("--config", default=None)
    a = p.parse_args()
    rows = validate_gold(read_rows(a.gold))
    searcher = Searcher(load_config(a.config))
    measurements = []
    try:
        for row in rows:
            out = searcher.search(row["query"], top_k=5, video_ids=row.get("video_ids"), include_candidates=True)
            measurements.append({"recall_stage@30": metrics(out["candidates"], row["relevant"], 30),
                                 "rerank@5": metrics(out["results"], row["relevant"], 5)})
    finally:
        searcher.close()
    report = {stage: {name: sum(m[stage][name] for m in measurements) / len(rows)
                     for name in ("recall", "mrr")} for stage in measurements[0]}
    print(json.dumps({"queries": len(rows), "metrics": report}, indent=2))


if __name__ == "__main__":
    main()
