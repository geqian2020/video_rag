#!/usr/bin/env bash
# Build and execute the tested V6 runner from the user's existing complete V5 file.
set -Eeuo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
v5="${1:-./run_video_rag_dedup_v5.sh}"
[[ -f "$v5" ]] || { echo "Missing V5 runner: $v5" >&2; exit 2; }
v6="${2:-./run_video_rag_registry_v6.sh}"
python -m workflow.make_v6 "$v5" "$v6"
bash -n "$v6"
echo "V6 generated and syntax-checked: $v6"
echo "Verify SOURCE_IN, DATA_IN, REPO_IN at the top of the generated file before executing it."
