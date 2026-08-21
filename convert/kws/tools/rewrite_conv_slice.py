#!/usr/bin/env python3
"""Rewrite encoder conv-cache Slice so the cut is not on the innermost dim.

VIP compiles Slice on W (axis 2 of (1,128,T)) as a 1-D memcpy(src+begin),
which inserts the next channel's old cache as a run of 7 zeros.

Fix: (1,C,T) -> transpose (1,T,C) -> slice last 7 on axis 1 -> transpose back.
The sliced 7*C block is contiguous even if NBG still does a 1-D copy.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SRC = ROOT / "onnx_fixed" / "encoder.onnx"


def _const_i64(name: str, values) -> onnx.NodeProto:
    t = numpy_helper.from_array(np.asarray(values, dtype=np.int64), name=name)
    return helper.make_node("Constant", [], [name], name=name + "_node", value=t)


def rewrite(model: onnx.ModelProto) -> int:
    g = model.graph
    by_out = {}
    consumers = {}
    for n in g.node:
        for o in n.output:
            by_out[o] = n
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    conv_outs = [o.name for o in g.output if o.name.startswith("new_cached_conv")]
    if not conv_outs:
        raise SystemExit("no new_cached_conv* graph outputs")

    # Shared slice params: last 7 on the (now) time axis 1.
    shared = [
        _const_i64("conv_slice_starts_m7", [-7]),
        _const_i64("conv_slice_ends_max", [np.iinfo(np.int64).max]),
        _const_i64("conv_slice_axes_1", [1]),
        _const_i64("conv_slice_steps_1", [1]),
    ]
    # Prepend constants once.
    for c in reversed(shared):
        g.node.insert(0, c)

    nfix = 0
    to_remove = []
    to_add = []  # (insert_after_name, nodes)
    for out_name in conv_outs:
        sl = by_out.get(out_name)
        if sl is None or sl.op_type != "Slice":
            print("skip", out_name, "producer", sl.op_type if sl else None)
            continue
        extra = [n for n in consumers.get(out_name, [])]
        if extra:
            raise SystemExit(f"{out_name} has internal consumers, not only graph output")
        data = sl.input[0]
        pfx = out_name + "_fix"
        t1 = pfx + "_t1"
        slc = pfx + "_sl"
        n_t1 = helper.make_node("Transpose", [data], [t1], name=pfx + "_transpose_in", perm=[0, 2, 1])
        n_sl = helper.make_node(
            "Slice",
            [t1, "conv_slice_starts_m7", "conv_slice_ends_max", "conv_slice_axes_1", "conv_slice_steps_1"],
            [slc],
            name=pfx + "_slice",
        )
        n_t2 = helper.make_node("Transpose", [slc], [out_name], name=pfx + "_transpose_out", perm=[0, 2, 1])
        to_remove.append(sl)
        to_add.append((data, [n_t1, n_sl, n_t2]))
        nfix += 1
        print(f"rewrite {out_name}: Slice(axis=2) <- Concat {data}  =>  T[0,2,1] / Slice(axis=1,last7) / T[0,2,1]")

    for n in to_remove:
        g.node.remove(n)

    # Insert new nodes right after the Concat they consume.
    by_out = {}
    for n in g.node:
        for o in n.output:
            by_out[o] = n
    # Insert from the end so earlier indices stay valid.
    for data, nodes in reversed(to_add):
        prod = by_out.get(data)
        if prod is None:
            # Concat is graph input — prepend near the shared constants.
            idx = 4
        else:
            idx = list(g.node).index(prod) + 1
        for i, n in enumerate(nodes):
            g.node.insert(idx + i, n)

    return nfix


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--dst", default=None, help="default: overwrite src")
    ap.add_argument("--check", action="store_true", help="ORT compare vs original")
    args = ap.parse_args()
    src = Path(args.src)
    dst = Path(args.dst) if args.dst else src
    orig = onnx.load(str(src))
    model = onnx.load(str(src))
    n = rewrite(model)
    print("rewrote", n, "conv-cache slices")
    onnx.checker.check_model(model, full_check=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(dst))
    print("wrote", dst, "nodes", len(model.graph.node))

    if args.check:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        a = ort.InferenceSession(str(src) if dst != src else orig.SerializeToString() if False else str(src), so,
                                 providers=["CPUExecutionProvider"])
        # If we overwrote src, compare by reloading orig from memory: load orig first.
        # We still have `orig` ModelProto.
        orig_path = src if dst != src else None
        if orig_path is None:
            tmp = dst.with_suffix(".orig.tmp.onnx")
            onnx.save(orig, str(tmp))
            a = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])
            b = ort.InferenceSession(str(dst), so, providers=["CPUExecutionProvider"])
        else:
            a = ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])
            b = ort.InferenceSession(str(dst), so, providers=["CPUExecutionProvider"])
        feeds = {}
        rng = np.random.default_rng(0)
        for inp in a.get_inputs():
            sh = [1 if (d is None or isinstance(d, str)) else int(d) for d in inp.shape]
            if inp.name == "processed_lens":
                feeds[inp.name] = np.zeros(sh, dtype=np.int64)
            else:
                feeds[inp.name] = rng.standard_normal(sh).astype(np.float32) * 0.1
        oa = a.run(None, feeds)
        ob = b.run(None, feeds)
        names = [o.name for o in a.get_outputs()]
        worst = 0.0
        worst_n = ""
        for n, x, y in zip(names, oa, ob):
            d = float(np.max(np.abs(x - y)))
            if d > worst:
                worst, worst_n = d, n
            if d > 1e-5 or n.startswith("new_cached_conv"):
                print(f"  {n:28s} maxabs={d:.3e} shape={x.shape}")
        print("worst", worst_n, worst)
        if orig_path is None and tmp.exists():
            tmp.unlink()
        if worst > 1e-4:
            raise SystemExit("ORT mismatch after rewrite")
        print("ORT match ok")


if __name__ == "__main__":
    main()
