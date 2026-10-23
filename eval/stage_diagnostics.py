"""Stage-attribution diagnostics for an existing eval/run_eval.py JSONL record file.

This is a *retrieval* diagnostic, not answer correctness or a model benchmark.
Do not treat alternative evidence locations as interchangeable until reviewed.
"""
import argparse
import json

from core.io import read_rows, write_json


def classify(record):
    """Classify answerable queries by whether evidence survives each stage."""
    if not record["answerable"]:
        return "unanswerable_false_accept" if record.get("false_accept") else "unanswerable_rejected"
    recalled = record["recall_stage_at30"]["hit"] > 0
    final = record["rerank_at5"]["hit"] > 0
    if final and not recalled:
        # Usually a data / labeling / pipeline mismatch. Never silently hide it.
        return "inconsistent_stage_records"
    if not recalled:
        return "recall_miss"
    if not final:
        return "rerank_loss"
    if record["rerank_at5"]["recall"] < 1:
        return "partial_evidence"
    return "full_evidence"


def summarize(records):
    counts = {}
    details = []
    for row in records:
        category = classify(row)
        counts[category] = counts.get(category, 0) + 1
        details.append({
            "query": row["query"], "category": category,
            "video_id": row.get("video_id"), "label_status": row.get("label_status"),
            "reviewed": row.get("reviewed"),
        })
    answerable = sum(bool(row["answerable"]) for row in records)
    recall_misses = counts.get("recall_miss", 0)
    rerank_losses = counts.get("rerank_loss", 0)
    return {
        "queries": len(records), "answerable": answerable,
        "counts": counts,
        "recall_miss_rate": recall_misses / answerable if answerable else None,
        "rerank_loss_rate": rerank_losses / answerable if answerable else None,
        "details": details,
        "caveat": (
            "Rates use answerable questions as denominator; they are not "
            "end-to-end answer accuracy. Labels must be reviewed, and partial "
            "multi-evidence coverage is distinct from a full hit. "
            "This tool does not repair or modify gold labels."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details", required=True,
                        help="JSONL produced by eval.run_eval --details")
    parser.add_argument("--output", help="Optional local JSON; avoid publishing private queries")
    args = parser.parse_args()
    result = summarize(read_rows(args.details))
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
