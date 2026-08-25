#!/usr/bin/env bash
# Download Matcha-baker acoustic ONNX (CPU). NPU .nb files are in prebuilt/.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="$ROOT/models"
mkdir -p "$M/tts"
cd "$M"

REL=https://github.com/k2-fsa/sherpa-onnx/releases/download

if [ ! -f tts/matcha-icefall-zh-baker/model-steps-3.onnx ]; then
  echo "get matcha-icefall-zh-baker"
  wget -q --show-progress -O matcha.tar.bz2 "$REL/tts-models/matcha-icefall-zh-baker.tar.bz2"
  tar xf matcha.tar.bz2 -C tts
  rm -f matcha.tar.bz2
fi

cp -n "$ROOT/prebuilt/vocoder/vocoder_int16_a733.nb" tts/ 2>/dev/null || true

echo "TTS acoustic ONNX in $M/tts/matcha-icefall-zh-baker/"
echo "KWS NPU .nb: copy prebuilt/kws/* to ~/npu_demos/kws_npu_demo/model/"
echo "ASR NPU: Allwinner model-zoo zipformer_demo_linux_a733"
echo "TTS NPU vocoder: models/tts/vocoder_int16_a733.nb + zoo tts_demo_a733"
