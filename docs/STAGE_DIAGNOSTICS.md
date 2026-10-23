# Stage-loss diagnosis (development data only)

This lightweight diagnostic isolates candidate-generation failures from
reranking failures. It is useful when comparing architectural choices such as
the multi-stage retrieval flow of [HKUDS/VideoRAG](https://github.com/HKUDS/VideoRAG)
and the modular retrieval/analysis pipeline in
[NVIDIA VSS](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization).
It **does not** import their code or assert benchmark equivalence.

Run on an existing local index and reviewed evaluation file:

```bash
python -m eval.run_eval --gold examples/gold.jsonl --details data/diagnostic_details.jsonl --output data/summary.json
python -m eval.stage_diagnostics --details data/diagnostic_details.jsonl --output data/stage_diagnostics.json
pytest -q tests/test_stage_diagnostics.py
```

| Category | Meaning | Next experiment |
| --- | --- | --- |
| recall_miss | No labeled evidence found in the original candidate pool | Chunking, query expansion, embedding / BM25 coverage |
| rerank_loss | Evidence recalled, but absent after top-5 reranking | Cross-encoder, candidate ordering, reranking budgets |
| partial_evidence | At least one gold item at top-5, but not all | Multi-part evidence aggregation |
| full_evidence | All gold items covered at top-5 | Inspect answer generation separately |
| inconsistent_stage_records | Final hit but no original hit | Inspect evaluation artifacts and labeling |
| unanswerable_false_accept | Unsupported evidence returned for abstention query | Threshold calibration on a validation split |
| unanswerable_rejected | Abstention query returned no accepted evidence | Check false rejection separately |

**Caution:** Existing gold matching may count multiple annotated spans as
mandatory and may fail to recognize equivalent evidence in other videos.
Review labels before interpreting stage rates. Prior 48/80 question sets have
already been exposed during tuning: they are **development**, not blind test
results. No new model weights, cloud processing or large benchmark jobs are
required for this analysis.

This branch is isolated because the previous session reported local changes
not yet available on GitHub (BGE 80-question runs, paired statistics, more
tests). Reconcile that work before merging.
