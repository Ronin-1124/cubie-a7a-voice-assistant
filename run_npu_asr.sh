#!/bin/bash
# File-based ASR on A733 NPU (zoo Zipformer encoder/decoder/joiner NBG).
# Usage: ./run_npu_asr.sh /path/to/16k_mono.wav
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
NPU_DIR="${NPU_ASR_DIR:-$HOME/npu_demos/zipformer_demo_linux_a733}"
WAV="${1:-}"

if [[ -z "$WAV" || ! -f "$WAV" ]]; then
  echo "Usage: $0 <wav>" >&2
  exit 1
fi
if [[ ! -x "$NPU_DIR/zipformer_demo_a733" ]]; then
  echo "Missing NPU zipformer demo: $NPU_DIR/zipformer_demo_a733" >&2
  exit 1
fi

# Resolve wav to absolute path (demo cwd is NPU_DIR)
case "$WAV" in
  /*) ABS="$WAV" ;;
  *) ABS="$(cd "$(dirname "$WAV")" && pwd)/$(basename "$WAV")" ;;
esac

export LD_LIBRARY_PATH="$NPU_DIR/lib:${LD_LIBRARY_PATH}"
cd "$NPU_DIR"
OUT="$(./zipformer_demo_a733 \
  -nb0 model/encoder_int16_a733.nb \
  -nb1 model/decoder_int16_a733.nb \
  -nb2 model/joiner_int16_a733.nb \
  -i "$ABS" 2>&1)" || {
  echo "$OUT" >&2
  exit 1
}
echo "$OUT" | awk '
  /Real Time Factor/ { print }
  /Zipformer output:/ {
    sub(/^.*Zipformer output:[[:space:]]*/, "")
    print "TEXT=" $0
  }
'
