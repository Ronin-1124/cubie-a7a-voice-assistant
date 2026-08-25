#!/bin/bash
# Cubie A7A: no onboard MEMS. 3.5mm HS-MIC -> MIC4 -> ADC2.
# Detect sunxi-ac101b card (this image: 0=codec, 1=HDMI).
set -e
if [ -n "${CARD:-}" ]; then
  :
else
  CARD="$(awk 'BEGIN{IGNORECASE=1} /ac101/{print $1; exit}' /proc/asound/cards 2>/dev/null || true)"
  CARD="${CARD:-0}"
fi
SRC="${MIC_SRC:-MIC4}"

mx() { amixer -c "$CARD" sset "$@" >/dev/null 2>&1 || true; }

mx HPOUT on
mx SPK off
mx LINEOUTL off
mx LINEOUTR off
mx "ADC1 ADC2 Swap" Off
mx "loopback debug" off

mx MIC1 off
mx MIC2 off
mx MIC3 off
mx MIC4 on

mx "ADC1 DATA MUX" MIC1
mx "ADC1 MUX" ADC1_DATA_MUX
mx ADC1 0
mx "ADC1 Gain" 0

mx "ADC2 PAG MUX" "$SRC"
mx "ADC2 DATA MUX" ADC2_PAG_MUX
mx "ADC2 MUX" ADC2_DATA_MUX
mx ADC2 140
mx "ADC2 Gain" 12

mx "RX1 MUX" RXL
mx "RX2 MUX" RXM1
mx "RX3 MUX" ADC2_DAT

mx "HPOUT Gain" 2
mx "DAC Gain" 0
mx DACL 140
mx DACR 140

echo "jack HS-MIC: $SRC -> ADC2  card=$CARD"
