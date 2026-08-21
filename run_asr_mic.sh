#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$ROOT/lib:${LD_LIBRARY_PATH}"
cd "$ROOT"
./setup_mic.sh
DEV="${1:-kws_mic}"
echo "Streaming ASR on: $DEV (Ctrl+C to stop)"
./bin/sherpa-onnx-alsa \
  --zipformer2-ctc-model=models/asr_zipformer_small_ctc_zh/model.int8.onnx \
  --tokens=models/asr_zipformer_small_ctc_zh/tokens.txt \
  --num-threads=2 \
  --provider=cpu \
  --decoding-method=greedy_search \
  "$DEV"
