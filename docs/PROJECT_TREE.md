# 新增/替换项目文件目录

workflow/ 为仓库已有内容，保留其现有文件；新增 OCR/ASR 通过 ocr 和 transcribe 入口使用。运行时数据与模型不提交。

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
├── tests/
│   ├── test_ocr_asr.py
│   └── test_pipeline.py
├── tools/
│   ├── __init__.py
│   └── prepare_models.py
└── transcribe/
    ├── __init__.py
    └── run.py
```

运行时目录 data/、models/ 不提交；仓库已有 workflow/ 不在以上新增文件清单中。
