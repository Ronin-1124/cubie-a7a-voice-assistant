#!/bin/bash
# Run inside NPU docker. Export KWS encoder/decoder/joiner int16 NBG for a733.
set -euo pipefail
export ACUITY_PATH="${ACUITY_PATH:-/usr/local/acuity_command_line_tools}"
export VIV_SDK="${VIV_SDK:-/root/Vivante_IDE/VivanteIDE5.11.0/cmdtools}"
export VIV_VX_ENABLE_GRAPH_TRANSFORM="${VIV_VX_ENABLE_GRAPH_TRANSFORM:--Dump[transform.rewrite.gather_to_stride_slice=0]}"

ROOT="$(cd "$(dirname "$0")" && pwd)"
PARTS="${*:-joiner decoder encoder}"

echo "ACUITY_PATH=$ACUITY_PATH"
echo "VIV_SDK=$VIV_SDK"
echo "parts: $PARTS"

for name in $PARTS; do
  dir="$ROOT/convert_model_${name}"
  mkdir -p "$dir/model"
  echo
  echo "========== export ${name} int16 a733 =========="
  cd "$dir"
  ./pegasus_export_ovx_nbg.sh "$name" int16 a733
  ls -lh "$dir/model" || true
done

echo
echo "========== done =========="
find "$ROOT"/convert_model_*/model -name '*.nb' -ls
