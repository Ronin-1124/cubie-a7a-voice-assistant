#!/bin/bash
# Offline-ish: feed wavs through online streaming binary by decoding files
# Use sherpa-onnx (online) on wav files if supported, else measure with alsa not possible offline.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$ROOT/lib:${LD_LIBRARY_PATH}"
cd "$ROOT"
WAV="${1:-test_wavs/zh_0.wav}"
echo "ASR file test: $WAV"
# online binary accepts wavs as positional args in some builds; try sherpa-onnx
./bin/sherpa-onnx \
  --zipformer2-ctc-model=models/asr_zipformer_small_ctc_zh/model.int8.onnx \
  --tokens=models/asr_zipformer_small_ctc_zh/tokens.txt \
  --num-threads=2 \
  --provider=cpu \
  "$WAV" 2>&1 || true
