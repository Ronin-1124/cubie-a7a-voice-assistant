#!/bin/bash
# Offline wav test (demo keywords + official test clips)
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$ROOT/lib:${LD_LIBRARY_PATH}"
cd "$ROOT"
./bin/sherpa-onnx-keyword-spotter \
  --tokens=models/kws/tokens.txt \
  --encoder=models/kws/encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx \
  --decoder=models/kws/decoder-epoch-13-avg-2-chunk-8-left-64.onnx \
  --joiner=models/kws/joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx \
  --keywords-file=keywords_demo.txt \
  --num-threads=2 \
  --provider=cpu \
  test_wavs/zh_3.wav test_wavs/zh_4.wav test_wavs/zh_5.wav test_wavs/en_0.wav
