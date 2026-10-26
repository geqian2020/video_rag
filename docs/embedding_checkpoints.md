# Semantic chunking: persistent embedding checkpoints

`python -m ingest.chunk` now checkpoints sentence embeddings automatically. The
existing `run_video_rag_prefetch_v7.sh` invocation does not need new arguments.
The default database is `chunk_embeddings.sqlite3` beside the chunk output; for
V7 this is `DATA_IN/chunks/chunk_embeddings.sqlite3`. Keep this file across runs.
It contains sentence hashes and vectors, not plaintext transcripts, and remains
local. The existing `*.sqlite*` ignore rule excludes it from Git.

Each inference batch (`models.batch_size`, currently 8) is committed as one SQLite
transaction with synchronous FULL. On restart, saved vectors are checked for
dimension, finite values, and SHA256 integrity before reuse. Only missing or
invalid sentences are encoded. A killed in-flight batch can be recomputed; a
previously committed batch is retained. A corrupt database file stops the run
with an error rather than silently deleting checkpoints.

The cache is keyed by exact sentence text and model identity. Repeated sentences
across videos share embeddings; video IDs, timestamps and original source segments
remain separate in final chunks. Editing a sentence only invalidates that text.
Changing time boundaries or chunking thresholds rebuilds chunks from reusable
sentence vectors. Changing model revision, token limit, relevant model settings,
device, or inference-library versions uses a separate cache namespace. Old
namespaces are retained. Hub models use their resolved immutable revision; local
model files are SHA256 hashed at startup so replacing local weights invalidates
the cache. Model resolution never downloads files.

The model is still loaded once per process. Grouping, token checks, and chunk
assembly run again after restart; their cached vectors require no model inference.
Batchwise inference retains the same model and chunking rules, but different
padding/batch composition may cause small floating-point differences near a
semantic threshold compared with the previous all-at-once encode call.

Progress distinguishes `reused`, `pending`, and `invalid` unique sentence counts.
`Embedding saved` is printed after a successful commit; long-running stages also
print elapsed-time heartbeats, which are not progress claims. The final chunk
JSONL is still published atomically only after all chunks pass validation.

Optional CLI arguments:

```bash
python -m ingest.chunk --input INPUT.jsonl --output OUTPUT.jsonl \
  --embedding-cache C:/temp/video_rag/chunk_embeddings.sqlite3
```

`--no-embedding-cache` explicitly opts out. No new packages are required.

## Update a Windows installation manually

Back up and replace `ingest/chunk.py`, add `ingest/embedding_cache.py`, then launch
V7 normally. No shell, model, transcript or ASR checkpoint changes are required.
Updating files does not alter an already running Python process. The old version
did not save vectors, so its previous in-memory work cannot be recovered; the
first run of this version begins accumulating checkpoints.

## Verification

```bash
python -m unittest discover -s tests -p test_embedding_checkpoints.py -v
```

These tests use deterministic fake models, including a subprocess abruptly
exiting after its first committed batch. They cover interrupted recovery, cache
hits, duplicate text, edited text, model changes, damaged rows, invalid inference,
transaction rollback, local weight replacement and unchanged chunk provenance.
Real BGE-M3 inference and Windows/OneDrive filesystem behavior still require
validation on the deployment computer.
