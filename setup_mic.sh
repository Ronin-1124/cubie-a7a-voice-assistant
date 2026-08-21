#!/bin/bash
# AC101b capture on Cubie A7A.
# Working path: ADC2 ← MIC4, RIGHT channel (kws_mic ttable.0.1).
# Headset jack "HS MIC" often means TRS headphones, NOT a working mic.
# Use MIC_SRC=MIC2 only for a known-good TRRS headset mic.
set -e
CARD=1

mx() { amixer -c "$CARD" sset "$@" >/dev/null 2>&1 || true; }

mx MIC1 on
mx MIC2 on
mx MIC3 on
mx MIC4 on

mx "ADC1 DATA MUX" MIC1
mx "ADC1 MUX" ADC1_DATA_MUX
mx ADC1 180
mx "ADC1 Gain" 14

mx "ADC2 DATA MUX" ADC2_PAG_MUX
mx "ADC2 MUX" ADC2_DATA_MUX
mx ADC2 230
mx "ADC2 Gain" 24

SRC="${MIC_SRC:-MIC4}"
mx "ADC2 PAG MUX" "$SRC"

HS=$(amixer -c "$CARD" contents 2>/dev/null | awk '/name=.HS MIC Jack/{f=1} f && /values=/{print $2; exit}')
echo "ADC2 source: $SRC  (override: MIC_SRC=MIC2 ./setup_mic.sh)"
echo "Jack: HP=$(amixer -c $CARD contents 2>/dev/null | awk '/name=.HP Jack/{f=1} f && /values=/{print $2; exit}') HS_MIC=$HS"

ASOUND="${HOME}/.asoundrc"
if ! grep -q 'pcm.kws_mic' "$ASOUND" 2>/dev/null; then
  cat >> "$ASOUND" << 'ASOUND_EOF'

# --- voice_assistant KWS/ASR mics (Cubie A7A ac101b) ---
pcm.kws_mic {
    type plug
    slave {
        pcm "hw:1,0"
        channels 2
        rate 16000
        format S16_LE
    }
    ttable.0.1 1.0
}
pcm.kws_mic_left {
    type plug
    slave {
        pcm "hw:1,0"
        channels 2
        rate 16000
        format S16_LE
    }
    ttable.0.0 1.0
}
pcm.kws_mic_mix {
    type plug
    slave {
        pcm "hw:1,0"
        channels 2
        rate 16000
        format S16_LE
    }
    ttable.0.0 0.5
    ttable.0.1 0.5
}
ASOUND_EOF
  echo "Wrote pcm.kws_mic to $ASOUND"
fi
echo "setup_mic done. device=kws_mic (RIGHT / MIC4)"
