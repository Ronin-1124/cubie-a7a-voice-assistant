#!/usr/bin/env bash
# Download CPU ONNX models (not NPU .nb). Run on the board or a machine that
# can scp into ~/npu_demos/voice_assistant/models/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
M="$ROOT/models"
mkdir -p "$M/kws" "$M/asr_zipformer_small_ctc_zh" "$M/tts"
cd "$M"

dl() {
  local url="$1" name="$2"
  if [ -f "$name" ]; then
    echo "skip $name"
    return
  fi
  echo "get $name"
  wget -q --show-progress -O "$name" "$url"
}

REL=https://github.com/k2-fsa/sherpa-onnx/releases/download

# KWS
if [ ! -f kws/encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx ]; then
  dl "$REL/kws-models/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20.tar.bz2" kws.tar.bz2
  tar xf kws.tar.bz2
  src=$(find . -maxdepth 2 -type d -name 'sherpa-onnx-kws-zipformer-zh-en-3M-*' | head -1)
  cp -a "$src"/encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx \
        "$src"/decoder-epoch-13-avg-2-chunk-8-left-64.onnx \
        "$src"/joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx \
        "$src"/tokens.txt kws/
  rm -rf kws.tar.bz2 "$src"
fi

# ASR CPU (optional fallback)
if [ ! -f asr_zipformer_small_ctc_zh/model.int8.onnx ]; then
  dl "$REL/asr-models/sherpa-onnx-streaming-zipformer-small-ctc-zh-int8-2025-04-01.tar.bz2" asr.tar.bz2
  tar xf asr.tar.bz2
  src=$(find . -maxdepth 2 -type d -name 'sherpa-onnx-streaming-zipformer-small-ctc-zh*' | head -1)
  cp -a "$src"/model.int8.onnx "$src"/tokens.txt asr_zipformer_small_ctc_zh/
  rm -rf asr.tar.bz2 "$src"
fi

# TTS: VITS 8 kHz + Matcha baker
if [ ! -f tts/vits-icefall-zh-aishell3/model.onnx ]; then
  dl "$REL/tts-models/vits-icefall-zh-aishell3.tar.bz2" vits.tar.bz2
  tar xf vits.tar.bz2 -C tts
  rm -f vits.tar.bz2
fi
if [ ! -f tts/matcha-icefall-zh-baker/model-steps-3.onnx ]; then
  dl "$REL/tts-models/matcha-icefall-zh-baker.tar.bz2" matcha.tar.bz2
  tar xf matcha.tar.bz2 -C tts
  rm -f matcha.tar.bz2
fi
if [ ! -f tts/hifigan_v2.onnx ]; then
  dl "$REL/vocoder-models/hifigan_v2.onnx" tts/hifigan_v2.onnx
fi
if [ ! -f tts/vocos-22khz-univ.onnx ]; then
  dl "$REL/vocoder-models/vocos-22khz-univ.onnx" tts/vocos-22khz-univ.onnx
fi

# NPU prebuilts from this repo
cp -n "$ROOT/prebuilt/vocoder/vocoder_int16_a733.nb" tts/ 2>/dev/null || true

echo "ONNX models in $M"
echo "KWS NPU .nb: copy prebuilt/kws/* to ~/npu_demos/kws_npu_demo/model/"
echo "ASR NPU: use Allwinner model-zoo zipformer_demo_linux_a733"
echo "TTS NPU vocoder: models/tts/vocoder_int16_a733.nb + zoo tts_demo_a733"
