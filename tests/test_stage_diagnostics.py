"""No model, index, or network access required."""
from eval.stage_diagnostics import classify, summarize


def row(recall, final, final_recall=1.0):
    return {
        "query": "Q", "answerable": True, "reviewed": False,
        "recall_stage_at30": {"hit": recall, "recall": float(recall)},
        "rerank_at5": {"hit": final, "recall": final_recall if final else 0.0},
    }


def test_classify_stage_losses():
    assert classify(row(0, 0)) == "recall_miss"
    assert classify(row(1, 0)) == "rerank_loss"
    assert classify(row(1, 1)) == "full_evidence"
    assert classify(row(1, 1, 0.5)) == "partial_evidence"
    assert classify(row(0, 1)) == "inconsistent_stage_records"
    assert classify({"answerable": False, "false_accept": True}) == "unanswerable_false_accept"


def test_summary_denominator_and_empty():
    report = summarize([row(0, 0), row(1, 0), row(1, 1),
                        {"query": "none", "answerable": False, "false_accept": False}])
    assert report["answerable"] == 3
    assert report["recall_miss_rate"] == 1 / 3
    assert report["rerank_loss_rate"] == 1 / 3
    assert report["counts"]["unanswerable_rejected"] == 1
    assert summarize([])["recall_miss_rate"] is None
