#!/usr/bin/env python3
"""Freeze KWS Zipformer chunk-8 shapes (N=1) and write Acuity NPY datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT.parent / "models" / "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
OUT = ROOT / "onnx_fixed"
DS = ROOT / "dataset"

# Runtime-probed with N=1, chunk-8, left-64 (onnxruntime).
ENC_IN = {
    "x": (1, 29, 80),
    "cached_key_0": (64, 1, 128),
    "cached_nonlin_attn_0": (1, 1, 64, 96),
    "cached_val1_0": (64, 1, 48),
    "cached_val2_0": (64, 1, 48),
    "cached_conv1_0": (1, 128, 7),
    "cached_conv2_0": (1, 128, 7),
    "cached_key_1": (32, 1, 128),
    "cached_nonlin_attn_1": (1, 1, 32, 96),
    "cached_val1_1": (32, 1, 48),
    "cached_val2_1": (32, 1, 48),
    "cached_conv1_1": (1, 128, 7),
    "cached_conv2_1": (1, 128, 7),
    "cached_key_2": (16, 1, 128),
    "cached_nonlin_attn_2": (1, 1, 16, 96),
    "cached_val1_2": (16, 1, 48),
    "cached_val2_2": (16, 1, 48),
    "cached_conv1_2": (1, 128, 7),
    "cached_conv2_2": (1, 128, 7),
    "cached_key_3": (8, 1, 256),
    "cached_nonlin_attn_3": (1, 1, 8, 96),
    "cached_val1_3": (8, 1, 96),
    "cached_val2_3": (8, 1, 96),
    "cached_conv1_3": (1, 128, 7),
    "cached_conv2_3": (1, 128, 7),
    "cached_key_4": (16, 1, 128),
    "cached_nonlin_attn_4": (1, 1, 16, 96),
    "cached_val1_4": (16, 1, 48),
    "cached_val2_4": (16, 1, 48),
    "cached_conv1_4": (1, 128, 7),
    "cached_conv2_4": (1, 128, 7),
    "cached_key_5": (32, 1, 128),
    "cached_nonlin_attn_5": (1, 1, 32, 96),
    "cached_val1_5": (32, 1, 48),
    "cached_val2_5": (32, 1, 48),
    "cached_conv1_5": (1, 128, 7),
    "cached_conv2_5": (1, 128, 7),
    "embed_states": (1, 128, 3, 19),
    "processed_lens": (1,),
}
ENC_OUT = {
    "encoder_out": (1, 4, 320),
    "new_cached_key_0": (64, 1, 128),
    "new_cached_nonlin_attn_0": (1, 1, 64, 96),
    "new_cached_val1_0": (64, 1, 48),
    "new_cached_val2_0": (64, 1, 48),
    "new_cached_conv1_0": (1, 128, 7),
    "new_cached_conv2_0": (1, 128, 7),
    "new_cached_key_1": (32, 1, 128),
    "new_cached_nonlin_attn_1": (1, 1, 32, 96),
    "new_cached_val1_1": (32, 1, 48),
    "new_cached_val2_1": (32, 1, 48),
    "new_cached_conv1_1": (1, 128, 7),
    "new_cached_conv2_1": (1, 128, 7),
    "new_cached_key_2": (16, 1, 128),
    "new_cached_nonlin_attn_2": (1, 1, 16, 96),
    "new_cached_val1_2": (16, 1, 48),
    "new_cached_val2_2": (16, 1, 48),
    "new_cached_conv1_2": (1, 128, 7),
    "new_cached_conv2_2": (1, 128, 7),
    "new_cached_key_3": (8, 1, 256),
    "new_cached_nonlin_attn_3": (1, 1, 8, 96),
    "new_cached_val1_3": (8, 1, 96),
    "new_cached_val2_3": (8, 1, 96),
    "new_cached_conv1_3": (1, 128, 7),
    "new_cached_conv2_3": (1, 128, 7),
    "new_cached_key_4": (16, 1, 128),
    "new_cached_nonlin_attn_4": (1, 1, 16, 96),
    "new_cached_val1_4": (16, 1, 48),
    "new_cached_val2_4": (16, 1, 48),
    "new_cached_conv1_4": (1, 128, 7),
    "new_cached_conv2_4": (1, 128, 7),
    "new_cached_key_5": (32, 1, 128),
    "new_cached_nonlin_attn_5": (1, 1, 32, 96),
    "new_cached_val1_5": (32, 1, 48),
    "new_cached_val2_5": (32, 1, 48),
    "new_cached_conv1_5": (1, 128, 7),
    "new_cached_conv2_5": (1, 128, 7),
    "new_embed_states": (1, 128, 3, 19),
    "new_processed_lens": (1,),
}


def set_dims(vi: onnx.ValueInfoProto, shape: tuple, etype: int | None = None) -> None:
    tt = vi.type.tensor_type
    if etype is not None:
        tt.elem_type = etype
    while len(tt.shape.dim) < len(shape):
        tt.shape.dim.add()
    while len(tt.shape.dim) > len(shape):
        tt.shape.dim.pop()
    for d, n in zip(tt.shape.dim, shape):
        d.ClearField("dim_param")
        d.dim_value = int(n)


def freeze(src: Path, dst: Path, ins: dict, outs: dict, int_inputs: set[str] | None = None) -> None:
    int_inputs = int_inputs or set()
    m = onnx.load(str(src))
    name_in = {i.name: i for i in m.graph.input}
    name_out = {o.name: o for o in m.graph.output}
    for n, sh in ins.items():
        et = TensorProto.INT64 if n in int_inputs else TensorProto.FLOAT
        set_dims(name_in[n], sh, et)
    for n, sh in outs.items():
        et = TensorProto.INT64 if n.endswith("processed_lens") else TensorProto.FLOAT
        set_dims(name_out[n], sh, et)
    try:
        m = onnx.shape_inference.infer_shapes(m)
    except Exception as e:
        print("shape_inference warning:", e)
    # re-apply after inference (Slice dims often stay symbolic)
    name_in = {i.name: i for i in m.graph.input}
    name_out = {o.name: o for o in m.graph.output}
    for n, sh in ins.items():
        et = TensorProto.INT64 if n in int_inputs else TensorProto.FLOAT
        set_dims(name_in[n], sh, et)
    for n, sh in outs.items():
        et = TensorProto.INT64 if n.endswith("processed_lens") else TensorProto.FLOAT
        set_dims(name_out[n], sh, et)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(dst))
    print("wrote", dst, "nodes", len(m.graph.node))


def write_npy() -> None:
    enc_dir = DS / "encoder"
    enc_dir.mkdir(parents=True, exist_ok=True)
    for n, sh in ENC_IN.items():
        dt = np.int64 if n == "processed_lens" else np.float32
        np.save(enc_dir / f"{n}.npy", np.zeros(sh, dtype=dt))
    dec_dir = DS / "decoder"
    dec_dir.mkdir(parents=True, exist_ok=True)
    # decoder y is token ids [1,2]
    np.save(dec_dir / "y.npy", np.zeros((1, 2), dtype=np.int64))
    (dec_dir / "dataset.txt").write_text(str(dec_dir / "y.npy") + "\n")
    joi_dir = DS / "joiner"
    joi_dir.mkdir(parents=True, exist_ok=True)
    np.save(joi_dir / "encoder_out.npy", np.zeros((1, 320), dtype=np.float32))
    np.save(joi_dir / "decoder_out.npy", np.zeros((1, 320), dtype=np.float32))
    print("dataset under", DS)


def main() -> None:
    freeze(
        SRC / "encoder-epoch-13-avg-2-chunk-8-left-64.onnx",
        OUT / "encoder.onnx",
        ENC_IN,
        ENC_OUT,
        {"processed_lens"},
    )
    from rewrite_conv_slice import rewrite as rewrite_conv_slice
    enc = onnx.load(str(OUT / "encoder.onnx"))
    n = rewrite_conv_slice(enc)
    onnx.save(enc, str(OUT / "encoder.onnx"))
    print("conv-cache slice rewrite", n, "nodes")
    freeze(
        SRC / "decoder-epoch-13-avg-2-chunk-8-left-64.onnx",
        OUT / "decoder.onnx",
        {"y": (1, 2)},
        {"decoder_out": (1, 320)},
        {"y"},
    )
    freeze(
        SRC / "joiner-epoch-13-avg-2-chunk-8-left-64.onnx",
        OUT / "joiner.onnx",
        {"encoder_out": (1, 320), "decoder_out": (1, 320)},
        {"logit": (1, 263)},
    )
    write_npy()


if __name__ == "__main__":
    main()
