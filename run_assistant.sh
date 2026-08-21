#!/bin/bash
# Fast path (default): models preloaded, stream ASR + silence endpoint
# Legacy: ./run_assistant.sh --legacy
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="$ROOT/lib:${LD_LIBRARY_PATH}"
export PATH="$HOME/.local/bin:${PATH}"

if [ "${1:-}" = "--legacy" ]; then
  shift
  exec python3 "$ROOT/assistant_loop.py" "$@"
fi

# Prefer fast in-process loop if sherpa_onnx is importable
if python3 -c "import sherpa_onnx" 2>/dev/null; then
  exec python3 "$ROOT/assistant_fast.py" "$@"
fi

echo "[warn] sherpa_onnx not installed; falling back to legacy (slower)"
echo "       install: pip3 install --user 'sherpa-onnx==1.13.4'"
exec python3 "$ROOT/assistant_loop.py" "$@"
