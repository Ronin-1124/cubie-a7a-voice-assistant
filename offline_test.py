#!/usr/bin/env python3
"""Offline regression: KWS + ASR on wav samples (no microphone)."""
from __future__ import annotations

import argparse
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

ROOT = Path(__file__).resolve().parent
SR = 16000


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        assert w.getsampwidth() == 2
        nch = w.getnchannels()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        audio = audio.reshape(-1, nch).mean(axis=1)
    if rate != SR:
        x = np.linspace(0, 1, len(audio))
        y = np.linspace(0, 1, int(len(audio) * SR / rate))
        audio = np.interp(y, x, audio).astype(np.float32)
    return audio


def create_kws(keywords: Path, thr: float) -> sherpa_onnx.KeywordSpotter:
    k = ROOT / "models" / "kws"
    return sherpa_onnx.KeywordSpotter(
        tokens=str(k / "tokens.txt"),
        encoder=str(k / "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
        decoder=str(k / "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"),
        joiner=str(k / "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
        keywords_file=str(keywords),
        num_threads=2,
        sample_rate=SR,
        keywords_score=1.0,
        keywords_threshold=thr,
        provider="cpu",
    )


def create_asr() -> sherpa_onnx.OnlineRecognizer:
    a = ROOT / "models" / "asr_zipformer_small_ctc_zh"
    return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=str(a / "tokens.txt"),
        model=str(a / "model.int8.onnx"),
        num_threads=2,
        sample_rate=SR,
        enable_endpoint_detection=False,
        decoding_method="greedy_search",
        provider="cpu",
    )


def run_kws(kws, audio: np.ndarray, chunk: int = 1600) -> list[str]:
    stream = kws.create_stream()
    hits = []
    for i in range(0, len(audio), chunk):
        stream.accept_waveform(SR, audio[i : i + chunk])
        while kws.is_ready(stream):
            kws.decode_stream(stream)
            r = kws.get_result(stream)
            if r:
                hits.append(r if isinstance(r, str) else str(r))
                kws.reset_stream(stream)
    # flush
    stream.accept_waveform(SR, np.zeros(int(0.5 * SR), dtype=np.float32))
    stream.input_finished()
    while kws.is_ready(stream):
        kws.decode_stream(stream)
        r = kws.get_result(stream)
        if r:
            hits.append(r if isinstance(r, str) else str(r))
            kws.reset_stream(stream)
    return hits


def run_asr(asr, audio: np.ndarray, chunk: int = 1600) -> str:
    stream = asr.create_stream()
    t0 = time.time()
    for i in range(0, len(audio), chunk):
        stream.accept_waveform(SR, audio[i : i + chunk])
        while asr.is_ready(stream):
            asr.decode_stream(stream)
    stream.accept_waveform(SR, np.zeros(int(0.5 * SR), dtype=np.float32))
    stream.input_finished()
    while asr.is_ready(stream):
        asr.decode_stream(stream)
    dt = time.time() - t0
    text = asr.get_result(stream)
    return text, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples-dir", default=str(ROOT / "samples"))
    ap.add_argument("--keywords", default=str(ROOT / "keywords_wake.txt"))
    ap.add_argument("--keywords-threshold", type=float, default=0.25)
    ap.add_argument("--also-demo-kws", action="store_true", help="also test official demo wavs")
    args = ap.parse_args()

    samples = Path(args.samples_dir)
    keywords = Path(args.keywords)
    if not keywords.is_file():
        print("missing", keywords)
        return 1

    print("Loading models…")
    t0 = time.time()
    kws = create_kws(keywords, args.keywords_threshold)
    asr = create_asr()
    print(f"Loaded in {time.time()-t0:.2f}s\n")

    # ---- KWS cases ----
    kws_cases = [
        ("wake_nihao_16k.wav", ["你好"], True),
        ("wake_nihao_xiaorui_16k.wav", ["你好小瑞", "你好", "小瑞"], True),
        ("wake_xiaorui_16k.wav", ["小瑞"], True),
        ("wake_xiaorui_xiaorui_16k.wav", ["小瑞小瑞", "小瑞"], True),
        ("neg_silence_3s_16k.wav", [], False),
        ("neg_noise_3s_16k.wav", [], False),
        ("cmd_open_light_16k.wav", [], False),  # command alone should not wake (or may false)
    ]

    print("=" * 60)
    print("KWS tests")
    print("=" * 60)
    kws_pass = kws_fail = 0
    for name, expect_any, want_hit in kws_cases:
        path = samples / name
        if not path.is_file():
            print(f"  SKIP missing {name}")
            continue
        audio = load_wav(path)
        hits = run_kws(kws, audio)
        if want_hit:
            ok = any(h in expect_any or any(e in h for e in expect_any) for h in hits) or (
                len(hits) > 0 and any(e in "".join(hits) for e in expect_any)
            )
            # looser: any hit whose text is in expect list
            ok = any(h in expect_any for h in hits)
            if not ok and hits:
                # partial credit if any expected substring
                ok = any(any(e in h for e in expect_any) for h in hits)
        else:
            ok = len(hits) == 0
        status = "PASS" if ok else "FAIL"
        if ok:
            kws_pass += 1
        else:
            kws_fail += 1
        print(f"  [{status}] {name}")
        print(f"         hits={hits}  expect_hit={want_hit} any_of={expect_any}")

    # official demo
    if args.also_demo_kws:
        demo_kw = ROOT / "keywords_demo.txt"
        if demo_kw.is_file():
            kws2 = create_kws(demo_kw, 0.25)
            for name in ["zh_3.wav", "zh_4.wav", "zh_5.wav", "en_0.wav"]:
                p = ROOT / "test_wavs" / name
                if p.is_file():
                    hits = run_kws(kws2, load_wav(p))
                    print(f"  [demo] {name} -> {hits}")

    # ---- ASR cases ----
    asr_cases = [
        ("cmd_open_light_16k.wav", "打开灯"),
        ("cmd_weather_16k.wav", "天气"),
        ("cmd_time_16k.wav", "点"),
        ("cmd_volume_16k.wav", "音量"),
        ("cmd_mixed_16k.wav", "会议"),
        ("full_nihao_cmd_16k.wav", "灯"),
        ("full_xiaorui_weather_16k.wav", "天气"),
        ("test_wavs/0.wav", "介绍"),  # relative to ROOT
    ]

    print()
    print("=" * 60)
    print("ASR tests")
    print("=" * 60)
    asr_pass = asr_fail = 0
    for name, must_contain in asr_cases:
        if name.startswith("test_wavs"):
            path = ROOT / name
        else:
            path = samples / name
        if not path.is_file():
            print(f"  SKIP missing {name}")
            continue
        audio = load_wav(path)
        text, dt = run_asr(asr, audio)
        ok = must_contain in (text or "")
        status = "PASS" if ok else "FAIL"
        if ok:
            asr_pass += 1
        else:
            asr_fail += 1
        dur = len(audio) / SR
        print(f"  [{status}] {path.name}  {dt:.2f}s decode / {dur:.2f}s audio  RTF={dt/max(dur,1e-6):.3f}")
        print(f"         text={text!r}  need_substr={must_contain!r}")

    # ---- pipeline simulation: KWS on pipe wav, ASR on command portion ----
    print()
    print("=" * 60)
    print("Pipeline simulation (wake clip + command clip)")
    print("=" * 60)
    pipes = [
        ("wake_nihao_xiaorui_16k.wav", "cmd_open_light_16k.wav", "你好小瑞", "灯"),
        ("wake_xiaorui_xiaorui_16k.wav", "cmd_weather_16k.wav", "小瑞", "天气"),
        ("wake_nihao_16k.wav", "cmd_time_16k.wav", "你好", "点"),
    ]
    pipe_pass = pipe_fail = 0
    for wname, cname, want_kw, want_txt in pipes:
        wa, ca = samples / wname, samples / cname
        if not wa.is_file() or not ca.is_file():
            print(f"  SKIP {wname}+{cname}")
            continue
        hits = run_kws(kws, load_wav(wa))
        text, dt = run_asr(asr, load_wav(ca))
        ok_k = any(want_kw in h for h in hits) or want_kw in "".join(hits)
        # also accept exact list membership
        ok_k = ok_k or any(h == want_kw for h in hits)
        ok_a = want_txt in (text or "")
        ok = ok_k and ok_a
        status = "PASS" if ok else "FAIL"
        if ok:
            pipe_pass += 1
        else:
            pipe_fail += 1
        print(f"  [{status}] wake={wname} + cmd={cname}")
        print(f"         kws={hits}  asr={text!r} ({dt:.2f}s)")

    print()
    print("=" * 60)
    print(
        f"SUMMARY  KWS {kws_pass}/{kws_pass+kws_fail}  "
        f"ASR {asr_pass}/{asr_pass+asr_fail}  "
        f"PIPE {pipe_pass}/{pipe_pass+pipe_fail}"
    )
    print("=" * 60)
    return 0 if (kws_fail + asr_fail + pipe_fail) == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
