#!/usr/bin/env bash
# Rebuild the complete V7 runner from a previously working full V6 shell file.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
SOURCE_V6="${1:-./run_video_rag_registry_v6.sh}"
OUTPUT_V7="${2:-./run_video_rag_prefetch_v7.sh}"
[[ -f "$SOURCE_V6" ]] || { echo "V6 script not found: $SOURCE_V6" >&2; exit 2; }
python -m workflow.make_v7 "$SOURCE_V6" "$OUTPUT_V7"
bash -n "$OUTPUT_V7"
printf 'V7 generated and bash -n passed: %s\n' "$OUTPUT_V7"
printf 'Edit SOURCE_IN/DATA_IN/REPO_IN, then run: bash %s\n' "$OUTPUT_V7"
