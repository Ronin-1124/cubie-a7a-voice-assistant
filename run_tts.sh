#!/bin/bash
# Offline TTS on A7A (sherpa-onnx)
# Default engine: vits (8 kHz, real-time). Quality options:
#   melo        MeloTTS zh_en 44.1 kHz CPU (good, slower)
#   matcha      Matcha-baker + Vocos 22.05 kHz (official pair)
#   matchahifi  Matcha-baker + HiFi-GAN v2 22.05 kHz CPU
#   matchanpu   Matcha-baker CPU + HiFi-GAN v2 NPU int16 (~59 ms vocoder)
#
# Usage:
#   ./run_tts.sh "帮我打开灯"
#   ./run_tts.sh -e vits -s 10 -t 2 -p "今天天气不错"
#   ./run_tts.sh -e matcha -p "好的，你说的是帮我打开灯。"
#   ./run_tts.sh -e matchahifi -p "识别完成"
#   ./run_tts.sh -e melo -p "你好，我是小瑞"
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export LD_LIBRARY_PATH="$ROOT/lib:${LD_LIBRARY_PATH}"

ENGINE=vits
SID=10
THREADS=2
PLAY=0
OUT=""
TEXT=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -e|--engine) ENGINE="$2"; shift 2 ;;
    -s|--sid) SID="$2"; shift 2 ;;
    -t|--threads) THREADS="$2"; shift 2 ;;
    -o|--output) OUT="$2"; shift 2 ;;
    -p|--play) PLAY=1; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      TEXT="$1"
      shift
      ;;
  esac
done

if [[ -z "$TEXT" ]]; then
  echo "Usage: $0 [-e vits|kokoro|melo|matcha|matchahifi|matchanpu|fs2npu] [-s sid] [-t threads] [-p] [-o out.wav] \"文本\"" >&2
  exit 1
fi

mkdir -p tts_out
if [[ -z "$OUT" ]]; then
  OUT="tts_out/${ENGINE}_sid${SID}_$(date +%H%M%S).wav"
fi

if [[ "$ENGINE" == "fs2npu" || "$ENGINE" == "fs2" ]]; then
  EX="$ROOT/../tts_fs2_hifigan_zh"
  if [[ ! -f "$EX/tools/tts_infer.py" ]]; then
    EX="$ROOT/tts_fs2"
  fi
  python3 "$EX/tools/tts_infer.py" \
    --text "$TEXT" \
    -o "$OUT" \
    --vocoder auto \
    ${PLAY:+--play}
  echo "[tts] done"
  exit 0
fi

if [[ "$ENGINE" == "matchanpu" ]]; then
  PY="$ROOT/tools/matcha_npu.py"
  if [[ ! -f "$PY" ]]; then
    echo "Missing $PY" >&2
    exit 1
  fi
  python3 "$PY" \
    --text "$TEXT" \
    -o "$OUT" \
    --vocoder npu \
    ${PLAY:+--play}
  echo "[tts] done"
  exit 0
fi

BIN=./bin/sherpa-onnx-offline-tts
if [[ ! -x "$BIN" ]]; then
  echo "Missing $BIN" >&2
  exit 1
fi

case "$ENGINE" in
  vits|aishell3)
    V=models/tts/vits-icefall-zh-aishell3
    set -- \
      --vits-model="$V/model.onnx" \
      --vits-lexicon="$V/lexicon.txt" \
      --vits-tokens="$V/tokens.txt" \
      --tts-rule-fsts="$V/phone.fst,$V/date.fst,$V/number.fst" \
      --num-threads="$THREADS" \
      --sid="$SID" \
      --output-filename="$OUT"
    ;;
  kokoro)
    K=models/tts/kokoro-int8-multi-lang-v1_1
    set -- \
      --kokoro-model="$K/model.int8.onnx" \
      --kokoro-voices="$K/voices.bin" \
      --kokoro-tokens="$K/tokens.txt" \
      --kokoro-data-dir="$K/espeak-ng-data" \
      --kokoro-lexicon="$K/lexicon-us-en.txt,$K/lexicon-zh.txt" \
      --tts-rule-fsts="$K/phone-zh.fst,$K/date-zh.fst,$K/number-zh.fst" \
      --num-threads="$THREADS" \
      --sid="$SID" \
      --output-filename="$OUT"
    ;;
  melo)
    M=models/tts/vits-melo-tts-zh_en
    if [[ ! -f "$M/model.onnx" ]]; then
      echo "Missing Melo model at $M/model.onnx" >&2
      exit 1
    fi
    set -- \
      --vits-model="$M/model.onnx" \
      --vits-lexicon="$M/lexicon.txt" \
      --vits-tokens="$M/tokens.txt" \
      --tts-rule-fsts="$M/phone.fst,$M/date.fst,$M/number.fst" \
      --num-threads="$THREADS" \
      --sid=0 \
      --output-filename="$OUT"
    ;;
  matcha)
    A=models/tts/matcha-icefall-zh-baker
    V=models/tts/vocos-22khz-univ.onnx
    if [[ ! -f "$A/model-steps-3.onnx" || ! -f "$V" ]]; then
      echo "Missing Matcha/Vocos under models/tts/" >&2
      exit 1
    fi
    set -- \
      --matcha-acoustic-model="$A/model-steps-3.onnx" \
      --matcha-vocoder="$V" \
      --matcha-lexicon="$A/lexicon.txt" \
      --matcha-tokens="$A/tokens.txt" \
      --tts-rule-fsts="$A/phone.fst,$A/date.fst,$A/number.fst" \
      --num-threads="$THREADS" \
      --output-filename="$OUT"
    ;;
  matchahifi)
    A=models/tts/matcha-icefall-zh-baker
    V=models/tts/hifigan_v2.onnx
    if [[ ! -f "$A/model-steps-3.onnx" || ! -f "$V" ]]; then
      echo "Missing Matcha/HiFi-GAN v2 under models/tts/" >&2
      exit 1
    fi
    set -- \
      --matcha-acoustic-model="$A/model-steps-3.onnx" \
      --matcha-vocoder="$V" \
      --matcha-lexicon="$A/lexicon.txt" \
      --matcha-tokens="$A/tokens.txt" \
      --tts-rule-fsts="$A/phone.fst,$A/date.fst,$A/number.fst" \
      --num-threads="$THREADS" \
      --output-filename="$OUT"
    ;;
  *)
    echo "engine must be vits, kokoro, melo, matcha, matchahifi, matchanpu, or fs2npu" >&2
    exit 1
    ;;
esac

echo "[tts] engine=$ENGINE sid=$SID threads=$THREADS"
echo "[tts] text=$TEXT"
echo "[tts] out=$OUT"
"$BIN" "$@" "$TEXT"

if [[ "$PLAY" -eq 1 ]]; then
  # card0=HDMI, card1=耳机 codec。裸 aplay 会走 Pulse（常被压到 ~20%）。
  amixer -c 1 cset name='HPOUT Switch' on >/dev/null 2>&1 || true
  amixer -c 1 sset 'DAC Gain' 4 >/dev/null 2>&1 || true
  amixer -c 1 sset 'HPOUT Gain' 5 >/dev/null 2>&1 || true
  pactl set-sink-mute @DEFAULT_SINK@ 0 >/dev/null 2>&1 || true
  pactl set-sink-volume @DEFAULT_SINK@ 80% >/dev/null 2>&1 || true
  aplay -D plughw:1,0 "$OUT" 2>/dev/null || aplay "$OUT"
fi

echo "[tts] done"
