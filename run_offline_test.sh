#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="$ROOT/lib:${LD_LIBRARY_PATH}"
exec python3 "$ROOT/offline_test.py" "$@"
