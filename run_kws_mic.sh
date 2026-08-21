#!/bin/bash
# Live mic keyword spotting on A7A
# Default device kws_mic = mono from RIGHT channel (MIC path with signal)
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
export LD_LIBRARY_PATH="$ROOT/lib:${LD_LIBRARY_PATH}"
cd "$ROOT"
./setup_mic.sh
DEV="${1:-kws_mic}"
echo "KWS listening on ALSA device: $DEV"
echo "Say clearly: 你好小瑞 / 小瑞小瑞"
echo "If no reaction: ./mic_level_test.sh   (should show AC peak >> 500)"
echo "Ctrl+C to stop"
./bin/sherpa-onnx-keyword-spotter-alsa \
  --tokens=models/kws/tokens.txt \
  --encoder=models/kws/encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx \
  --decoder=models/kws/decoder-epoch-13-avg-2-chunk-8-left-64.onnx \
  --joiner=models/kws/joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx \
  --keywords-file=keywords.txt \
  --num-threads=2 \
  --provider=cpu \
  --keywords-threshold=0.15 \
  --keywords-score=1.5 \
  --max-active-paths=4 \
  "$DEV"
