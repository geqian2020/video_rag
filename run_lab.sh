#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# Only edit lab_input.txt. No environment/data/model/output paths to replace.
if command -v py >/dev/null 2>&1; then
  if py -3.12 -c 'pass' >/dev/null 2>&1; then
    exec py -3.12 tools/lab.py "$@"
  elif py -3.11 -c 'pass' >/dev/null 2>&1; then
    exec py -3.11 tools/lab.py "$@"
  fi
fi
for python_command in python3 python; do
  if command -v "$python_command" >/dev/null 2>&1; then
    if "$python_command" -c 'import sys; sys.exit(sys.version_info[:2] not in {(3,11),(3,12)})' >/dev/null 2>&1; then
      exec "$python_command" tools/lab.py "$@"
    fi
  fi
done
echo 'Python 3.11 or 3.12 is required. Install it with the Python launcher/PATH enabled.' >&2
exit 1
