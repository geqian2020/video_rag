# 项目新增与升级文件目录

旧 workflow/ 保留。test/ 保存公开来源和通用评测工具；data/、模型、实际 gold 和媒体不提交。

```text
video_rag/
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── PROJECT_SOURCE.md
├── README.md
├── config.yaml
├── core/
│   ├── __init__.py
│   ├── config.py
│   ├── io.py
│   ├── media.py
│   ├── models.py
│   ├── schema.py
│   ├── sparse.py
│   ├── store.py
│   └── vocabulary.py
├── docker-compose.yml
├── docs/
│   ├── OCR_ASSISTED_ASR.md
│   ├── PROJECT_TREE.md
│   ├── REPOSITORY_REVIEW.md
│   ├── STATUS.md
│   └── VALIDATION.md
├── eval/
│   ├── __init__.py
│   ├── make_gold.py
│   ├── run_asr_eval.py
│   └── run_eval.py
├── examples/
│   ├── gold.jsonl
│   ├── transcripts.jsonl
│   └── vocabulary.json
├── ingest/
│   ├── __init__.py
│   ├── chunk.py
│   ├── convert_legacy.py
│   ├── enrich.py
│   └── index.py
├── ocr/
│   ├── __init__.py
│   ├── extract.py
│   └── vocab.py
├── requirements-asr.txt
├── requirements-dev.txt
├── requirements-ocr.txt
├── requirements.txt
├── retrieve/
│   ├── __init__.py
│   ├── rerank.py
│   └── search.py
├── serve/
│   ├── __init__.py
│   └── api.py
├── test/
│   ├── CLOUD_RUN.md
│   ├── QUESTION_PROMPT.md
│   ├── README.md
│   ├── __init__.py
│   ├── cloud_run_summary.json
│   ├── config.cpu.yaml
│   ├── download.py
│   ├── prepare_corpus.py
│   ├── requirements.txt
│   ├── run_benchmark.py
│   ├── urls.txt
│   └── videos.json
├── tests/
│   ├── test_ocr_asr.py
│   ├── test_pipeline.py
│   └── test_public_benchmark.py
├── tools/
│   ├── __init__.py
│   └── prepare_models.py
└── transcribe/
    ├── __init__.py
    └── run.py
```

运行时 data/、models/ 不提交；仓库原有 workflow/ 不在以上新增/升级源码清单中。
