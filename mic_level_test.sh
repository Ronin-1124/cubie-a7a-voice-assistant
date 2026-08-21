#!/bin/bash
# Record 3s and print AC energy. Speak into mic while recording.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
"$ROOT/setup_mic.sh"
DEV="${1:-kws_mic}"
OUT=/tmp/mic_level_test.wav
echo "Recording 3s from $DEV — please speak now..."
arecord -D "$DEV" -f S16_LE -r 16000 -c 1 -d 3 "$OUT"
python3 - << PY
import wave,struct
w=wave.open("$OUT"); raw=w.readframes(w.getnframes())
s=struct.unpack("<"+str(len(raw)//2)+"h", raw)
mean=sum(s)/len(s); ac=[x-mean for x in s]
peak=max(abs(x) for x in ac); rms=(sum(x*x for x in ac)/len(ac))**0.5
print("file:", "$OUT")
print("DC mean: %.1f" % mean)
print("AC peak: %d (%.1f%% of full scale)" % (peak, 100*peak/32768))
print("AC rms:  %.1f" % rms)
if peak < 500:
    print(">> TOO QUIET: almost no speech energy. Check plug / MIC path / gain.")
elif peak < 3000:
    print(">> WEAK: try speaking closer or raise ADC2 Volume/Gain.")
else:
    print(">> OK: energy looks usable for KWS/ASR.")
PY
