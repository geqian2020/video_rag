# Experiment: context expansion after reranking

Based on the existing small-to-big retrieval pipeline at commit
`2d0e6cebf5dacdc7992ec49e34522813c9e611f4`.

The original pipeline expands all RRF candidates **before** CrossEncoder
scoring. This opt-in experiment changes only ordering: rerank short original
chunks first and expand the selected evidence afterwards. It does not alter
index construction, dense/BM25 retrieval, RRF, model weights or gold labels.

## Run

Set the following environment override for the *treatment*:

```bash
RAG_RETRIEVAL__EXPAND_AFTER_RERANK=true python -m eval.run_eval --gold YOUR_DEV_GOLD.jsonl --config test/config.context.cpu.yaml --details data/post_rerank_details.jsonl --output data/post_rerank_summary.json
```

Use the same command without the environment override for the *baseline*,
saving details separately. Compare with:

```bash
python -m eval.statistics --before data/baseline_details.jsonl --after data/post_rerank_details.jsonl --output data/post_rerank_comparison.json
pytest -q tests/test_context.py
```

The flag defaults to false: the existing behavior is unchanged.

## Interpretation

- The treatment CrossEncoder sees the original chunks; the baseline sees
  expanded chunks. Candidates/results still expose expanded evidence, including
  `retrieved_start_ts` and `retrieved_end_ts` when expansion succeeds.
- `raw_recall_stage_at30` remains a measure of the original hit windows;
  `recall_stage_at30` measures the candidate-expanded windows; `rerank_at5`
  measures the final expanded evidence. Do not conflate these with answer accuracy.
- Compare the paired per-question data, regressions and token/latency costs.
  This has **not** been run on the historical 80-question model benchmark in
  this GitHub-only session. The historical 80-question labels are unreviewed,
  exposed development data, not final acceptance evidence.
- A gain in final coverage without a gain in raw recall is not a retrieval
  recall breakthrough; it can be the result of widening evidence windows.
- If results regress, leave `expand_after_rerank=false` and retain this only
  as a documented ablation.
