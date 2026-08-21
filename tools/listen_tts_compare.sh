#!/bin/bash
# Play quality candidates. Same sentence, headphone plughw:1,0.
#   ./tools/listen_tts_compare.sh
#   ./tools/listen_tts_compare.sh "识别完成"
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
TEXT="${1:-好的，你说的是帮我打开灯。}"

echo "[compare] text=$TEXT"
echo "[compare] order: vits 8k -> matchahifi CPU 22k -> matchanpu int16 22k -> matcha vocos -> melo 44k"

play_one() {
  local eng="$1"
  echo
  echo "========== $eng =========="
  ./run_tts.sh -e "$eng" -t 2 -p -o "tts_out/compare_${eng}.wav" "$TEXT" || {
    echo "[compare] $eng FAILED"
    return 0
  }
  sleep 1
}

play_one vits
play_one matchahifi
play_one matchanpu
play_one matcha
play_one melo

echo
echo "[compare] wavs:"
ls -lh tts_out/compare_*.wav 2>/dev/null || true
echo "[compare] replay: aplay -D plughw:1,0 tts_out/compare_<engine>.wav"
