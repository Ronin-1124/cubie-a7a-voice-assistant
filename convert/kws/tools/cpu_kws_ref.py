#!/usr/bin/env python3
"""CPU ONNX KWS using the same chunking + keyword trie as the NPU C++ demo."""
from __future__ import annotations

import argparse
import wave
from dataclasses import dataclass, field
from pathlib import Path

import kaldi_native_fbank as knf
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
VA = ROOT.parent
KDIR = VA / "models" / "sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20"
T, SHIFT, ENC_T, DIM, V = 29, 16, 4, 320, 263
BLANK, UNK = 0, 2

ENC_CACHE = [
    "cached_key_0", "cached_nonlin_attn_0", "cached_val1_0", "cached_val2_0",
    "cached_conv1_0", "cached_conv2_0",
    "cached_key_1", "cached_nonlin_attn_1", "cached_val1_1", "cached_val2_1",
    "cached_conv1_1", "cached_conv2_1",
    "cached_key_2", "cached_nonlin_attn_2", "cached_val1_2", "cached_val2_2",
    "cached_conv1_2", "cached_conv2_2",
    "cached_key_3", "cached_nonlin_attn_3", "cached_val1_3", "cached_val2_3",
    "cached_conv1_3", "cached_conv2_3",
    "cached_key_4", "cached_nonlin_attn_4", "cached_val1_4", "cached_val2_4",
    "cached_conv1_4", "cached_conv2_4",
    "cached_key_5", "cached_nonlin_attn_5", "cached_val1_5", "cached_val2_5",
    "cached_conv1_5", "cached_conv2_5",
    "embed_states", "processed_lens",
]


def load_wav(p: Path) -> np.ndarray:
    with wave.open(str(p), "rb") as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            pcm = pcm.reshape(-1, w.getnchannels()).mean(1)
    return pcm


def fbank(pcm: np.ndarray) -> np.ndarray:
    opts = knf.FbankOptions()
    opts.frame_opts.samp_freq = 16000
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.mel_opts.num_bins = 80
    opts.mel_opts.high_freq = -400
    ext = knf.OnlineFbank(opts)
    ext.accept_waveform(16000, pcm.tolist())
    ext.input_finished()
    n = ext.num_frames_ready
    return np.stack([ext.get_frame(i) for i in range(n)]).astype(np.float32)


def load_tokens(p: Path):
    id2t, t2id = {}, {}
    for line in p.read_text().splitlines():
        t, i = line.rsplit(" ", 1)
        i = int(i)
        id2t[i] = t
        t2id[t] = i
    return id2t, t2id


def load_kws(path: Path, t2id, default_boost=1.5, default_thr=0.25):
    kws = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        boost, thr, phrase = default_boost, default_thr, ""
        rest = line
        if "@" in rest:
            rest, phrase = rest.split("@", 1)
            phrase = phrase.strip()
        if "#" in rest:
            rest, hs = rest.split("#", 1)
            thr = float(hs.split()[0])
        if ":" in rest:
            rest, bs = rest.split(":", 1)
            boost = float(bs.split()[0])
        toks = []
        for t in rest.split():
            toks.append(t2id[t])
        kws.append((toks, phrase or "?", boost, thr))
        print(f"  kw {phrase} n={len(toks)} boost={boost} thr={thr}")
    return kws


class Trie:
    def __init__(self, kws):
        self.nodes = [{"token": -1, "level": 0, "end": False, "phrase": "", "thr": 0.25, "boost": 1.0, "next": {}}]
        for toks, phrase, boost, thr in kws:
            cur = 0
            for t in toks:
                if t not in self.nodes[cur]["next"]:
                    ni = len(self.nodes)
                    self.nodes.append({"token": t, "level": self.nodes[cur]["level"] + 1,
                                       "end": False, "phrase": "", "thr": thr, "boost": boost, "next": {}})
                    self.nodes[cur]["next"][t] = ni
                cur = self.nodes[cur]["next"][t]
            self.nodes[cur]["end"] = True
            self.nodes[cur]["phrase"] = phrase
            self.nodes[cur]["thr"] = thr

    def step(self, node, tok):
        nxt = self.nodes[node]["next"]
        if tok in nxt:
            n = nxt[tok]
            return n, self.nodes[n]["boost"]
        if node != 0 and tok in self.nodes[0]["next"]:
            n = self.nodes[0]["next"][tok]
            return n, self.nodes[n]["boost"]
        return 0, 0.0


def log_softmax(x):
    x = x - x.max()
    e = np.exp(x)
    p = e / e.sum()
    return np.log(np.maximum(p, 1e-12))


def zeros_for(sess):
    feeds = {}
    for inp in sess.get_inputs():
        sh = [1 if (d is None or isinstance(d, str)) else int(d) for d in inp.shape]
        dt = np.int64 if inp.name == "processed_lens" else np.float32
        feeds[inp.name] = np.zeros(sh, dtype=dt)
    return feeds


@dataclass
class Hyp:
    ys: list
    node: int = 0
    log_prob: float = 0.0
    tok_probs: list = field(default_factory=list)


def decode(feat, enc, dec, joi, trie, id2t, forced=True, max_paths=4, dump_prog=True):
    feeds = zeros_for(enc)
    enc_in_names = [i.name for i in enc.get_inputs()]
    enc_out_names = [o.name for o in enc.get_outputs()]
    hits = []
    nfr = feat.shape[0]
    pos = 0
    frame = 0
    beam = [Hyp(ys=[0, 0])]
    while pos + T <= nfr or (pos < nfr and pos + T > nfr):
        if pos + T > nfr:
            pad = np.zeros((T - (nfr - pos), 80), np.float32)
            chunk = np.concatenate([feat[pos:], pad], 0)
        else:
            chunk = feat[pos:pos + T]
        feeds["x"] = chunk[None, ...]
        feeds["processed_lens"] = np.array([pos], np.int64)
        outs = enc.run(None, feeds)
        enc_out = outs[0]  # [1,4,320]
        # update caches
        for i, name in enumerate(enc_out_names[1:], 1):
            src = name
            dst = src[4:] if src.startswith("new_") else src
            if dst in feeds:
                feeds[dst] = outs[i]
        for t in range(enc_out.shape[1]):
            enc_t = enc_out[:, t, :]
            cands = []
            logits_by_h = []
            for hi, h in enumerate(beam):
                y = np.array([h.ys[-2:]], np.int64)
                d_out = dec.run(None, {"y": y})[0]
                logit = joi.run(None, {"encoder_out": enc_t, "decoder_out": d_out})[0][0]
                lp = log_softmax(logit)
                logits_by_h.append(lp)
                if hi == 0:
                    b = int(lp.argmax())
                    if b != BLANK or frame < 2:
                        print(f"t={frame} top='{id2t.get(b,'?')}' p={np.exp(lp[b]):.2f}")
                # always consider blank + trie next + top acoustic
                consider = {BLANK}
                consider.update(trie.nodes[h.node]["next"].keys())
                if not forced:
                    consider.update(range(V))
                else:
                    consider.add(int(lp.argmax()))
                    # also top-3
                    consider.update(np.argpartition(-lp, 3)[:3].tolist())
                for tok in consider:
                    if tok == UNK:
                        continue
                    nn, boost = (h.node, 0.0) if tok == BLANK else trie.step(h.node, int(tok))
                    sc = h.log_prob + float(lp[tok]) + (0.0 if tok == BLANK else boost)
                    cands.append((sc, hi, int(tok), nn, boost, float(lp[tok])))
            cands.sort(key=lambda z: -z[0])
            nxt, emitted = [], False
            for sc, hi, tok, nn, boost, lpt in cands[:max_paths]:
                h = beam[hi]
                nh = Hyp(ys=list(h.ys), node=h.node, log_prob=sc, tok_probs=list(h.tok_probs))
                if tok != BLANK:
                    nh.ys.append(tok)
                    nh.node = nn
                    nh.tok_probs.append(float(np.exp(lpt)))
                nd = trie.nodes[nh.node]
                if dump_prog and nh.node > 0 and hi == 0:
                    print(f"prog t={frame} level={nd['level']} end={int(nd['end'])} tok={id2t.get(nd['token'],'?')}")
                if (not emitted) and nd["end"] and nd["level"] > 0:
                    lv = nd["level"]
                    meanp = float(np.mean(nh.tok_probs[-lv:])) if len(nh.tok_probs) >= lv else 0.0
                    if meanp >= nd["thr"]:
                        print(f"HIT '{nd['phrase']}' score={meanp:.3f} frame={frame}")
                        hits.append(nd["phrase"])
                        emitted = True
                        nh = Hyp(ys=[0, 0])
                nxt.append(nh)
            beam = nxt or [Hyp(ys=[0, 0])]
            if emitted:
                beam = [Hyp(ys=[0, 0])]
            frame += 1
        pos += SHIFT
        if pos >= nfr:
            break
    return hits


def sherpa_hits(wav, keywords, thr=0.25):
    import sherpa_onnx
    kws = sherpa_onnx.KeywordSpotter(
        tokens=str(KDIR / "tokens.txt"),
        encoder=str(KDIR / "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
        decoder=str(KDIR / "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"),
        joiner=str(KDIR / "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
        keywords_file=str(keywords),
        num_threads=2,
        sample_rate=16000,
        keywords_score=1.0,
        keywords_threshold=thr,
        provider="cpu",
    )
    pcm = load_wav(wav)
    stream = kws.create_stream()
    hits = []
    stream.accept_waveform(16000, pcm)
    stream.input_finished()
    while kws.is_ready(stream):
        kws.decode_stream(stream)
        r = kws.get_result(stream)
        if r:
            hits.append(r if isinstance(r, str) else str(r))
            kws.reset_stream(stream)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", default=str(VA / "deploy/samples/wake_nihao_xiaorui_16k.wav"))
    ap.add_argument("--keywords", default=str(VA / "convert_kws_npu/kws_demo/model/keywords_main.txt"))
    ap.add_argument("--no-forced", action="store_true")
    args = ap.parse_args()
    id2t, t2id = load_tokens(KDIR / "tokens.txt")
    kws = load_kws(Path(args.keywords), t2id)
    trie = Trie(kws)
    pcm = load_wav(Path(args.wav))
    feat = fbank(pcm)
    print("feat", feat.shape, "wav", args.wav)
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    enc = ort.InferenceSession(str(KDIR / "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"), so, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(KDIR / "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"), so, providers=["CPUExecutionProvider"])
    joi = ort.InferenceSession(str(KDIR / "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"), so, providers=["CPUExecutionProvider"])
    print("=== our decoder CPU ONNX forced=", not args.no_forced, "===")
    hits = decode(feat, enc, dec, joi, trie, id2t, forced=not args.no_forced)
    print("OURS", hits)
    print("=== sherpa KeywordSpotter ===")
    try:
        sh = sherpa_hits(Path(args.wav), Path(args.keywords))
        print("SHERPA", sh)
    except Exception as e:
        print("sherpa skip", e)


if __name__ == "__main__":
    main()
