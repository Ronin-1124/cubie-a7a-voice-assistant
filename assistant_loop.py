#!/usr/bin/env python3
"""Cubie A7A offline voice assistant loop: KWS wake -> capture command -> ASR.

Uses prebuilt sherpa-onnx binaries (no Python bindings required).
Mic: exclusive — KWS and ASR never hold the device at the same time.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BIN = ROOT / "bin"
LIB = ROOT / "lib"

# Ensure child processes find libonnxruntime / libsherpa-onnx
os.environ["LD_LIBRARY_PATH"] = f"{LIB}:{os.environ.get('LD_LIBRARY_PATH', '')}"

JSON_RE = re.compile(r"\{.*\}")


def log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def run_setup_mic() -> None:
    script = ROOT / "setup_mic.sh"
    if script.is_file():
        subprocess.run(["bash", str(script)], check=False)


def kill_proc(proc: subprocess.Popen | None, name: str) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=1.0)
            return
        except subprocess.TimeoutExpired:
            pass
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
            return
        except subprocess.TimeoutExpired:
            pass
        proc.kill()
        proc.wait(timeout=1.0)
    except Exception as e:  # noqa: BLE001
        log(f"warn: kill {name}: {e}")


def wait_device_free(seconds: float = 0.4) -> None:
    time.sleep(seconds)


def parse_keyword_line(line: str) -> str | None:
    """Extract keyword name from spotter stdout line."""
    line = line.strip()
    if "keyword" not in line:
        return None
    m = JSON_RE.search(line)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        # partial multi-line json — rare
        return None
    kw = obj.get("keyword") or ""
    return str(kw) if kw else None


def listen_for_wake(
    device: str,
    keywords_file: Path,
    num_threads: int,
) -> str:
    """Block until a keyword is spotted; free the mic and return keyword."""
    cmd = [
        str(BIN / "sherpa-onnx-keyword-spotter-alsa"),
        f"--tokens={ROOT}/models/kws/tokens.txt",
        f"--encoder={ROOT}/models/kws/encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
        f"--decoder={ROOT}/models/kws/decoder-epoch-13-avg-2-chunk-8-left-64.onnx",
        f"--joiner={ROOT}/models/kws/joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx",
        f"--keywords-file={keywords_file}",
        f"--num-threads={num_threads}",
        "--provider=cpu",
        "--keywords-threshold=0.15",
        "--keywords-score=1.5",
        "--max-active-paths=4",
        device,
    ]
    log(f"KWS listening on {device} … (say wake word)")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(ROOT),
    )
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            # forward useful non-spam lines lightly
            if "Recording started" in line or "Use recording device" in line:
                log(line.strip())
            kw = parse_keyword_line(line)
            if kw:
                log(f"WAKE: {kw}")
                print(line.rstrip(), flush=True)
                return kw
            # still print raw keyword-ish lines
            if '"keyword"' in line:
                print(line.rstrip(), flush=True)
    finally:
        kill_proc(proc, "kws")
        wait_device_free()
    raise RuntimeError("KWS exited without keyword")


def capture_command(device: str, wav_path: Path, duration: float) -> None:
    """Record fixed-duration mono 16 kHz after wake (starts immediately)."""
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "arecord",
        "-D",
        device,
        "-f",
        "S16_LE",
        "-r",
        "16000",
        "-c",
        "1",
        "-d",
        str(int(round(duration))),
        str(wav_path),
    ]
    log(f"REC {duration:.0f}s → {wav_path.name}  (please speak command now)")
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"arecord failed: {r.stderr or r.stdout}")


def asr_file(wav_path: Path, num_threads: int) -> str:
    """Decode wav with streaming CTC model via offline-style online binary."""
    cmd = [
        str(BIN / "sherpa-onnx"),
        f"--zipformer2-ctc-model={ROOT}/models/asr_zipformer_small_ctc_zh/model.int8.onnx",
        f"--tokens={ROOT}/models/asr_zipformer_small_ctc_zh/tokens.txt",
        f"--num-threads={num_threads}",
        "--provider=cpu",
        "--decoding-method=greedy_search",
        str(wav_path),
    ]
    log("ASR decoding…")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    text = ""
    # Prefer JSON "text" field; fall back to plain line before JSON
    for line in out.splitlines():
        if '"text"' in line:
            m = JSON_RE.search(line)
            if m:
                try:
                    obj = json.loads(m.group(0))
                    if obj.get("text"):
                        text = obj["text"].strip()
                except json.JSONDecodeError:
                    pass
        # some builds print bare text line without braces
        if line.startswith("{") is False and "RTF" not in line and "threads" not in line:
            # ignore
            pass
    # Also catch lines like: 对我做了介绍...
    if not text:
        for line in out.splitlines():
            s = line.strip()
            if not s or s.startswith("/") or "Config" in s or "Recognizer" in s:
                continue
            if "RTF" in s or "threads" in s or "Elapsed" in s:
                continue
            if s.startswith("{") and "text" in s:
                continue
            # Chinese / alphanumeric command-like line
            if any("\u4e00" <= c <= "\u9fff" for c in s) or s.isascii():
                # skip English log words
                if s.split()[0:1] and s.split()[0] in {
                    "OnlineRecognizerConfig",
                    "KeywordSpotterConfig",
                    "Number",
                    "Start",
                    "Use",
                    "Current",
                    "Recording",
                    "Started",
                }:
                    continue
                if len(s) >= 2 and not s.startswith("--"):
                    text = s
    log(f"ASR done in {dt:.2f}s")
    if r.returncode != 0 and not text:
        log(f"ASR stderr tail: {out[-500:]}")
    return text


def asr_streaming_endpoint(
    device: str,
    num_threads: int,
    max_seconds: float,
    rule2_silence: float,
) -> str:
    """Live streaming ASR until endpoint or timeout (model load ~2s first)."""
    cmd = [
        str(BIN / "sherpa-onnx-alsa"),
        f"--zipformer2-ctc-model={ROOT}/models/asr_zipformer_small_ctc_zh/model.int8.onnx",
        f"--tokens={ROOT}/models/asr_zipformer_small_ctc_zh/tokens.txt",
        f"--num-threads={num_threads}",
        "--provider=cpu",
        "--decoding-method=greedy_search",
        "--enable-endpoint=true",
        f"--rule2-min-trailing-silence={rule2_silence}",
        "--rule1-min-trailing-silence=2.0",
        "--rule3-min-utterance-length=20",
        device,
    ]
    log(f"ASR streaming on {device} (endpoint silence={rule2_silence}s, max={max_seconds}s)")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(ROOT),
    )
    assert proc.stdout is not None
    last_text = ""
    got_started = False
    t_start = time.time()
    t_speak = None
    try:
        while True:
            if time.time() - t_start > max_seconds:
                log("ASR max time reached")
                break
            line = proc.stdout.readline()
            if not line and proc.poll() is not None:
                break
            if not line:
                time.sleep(0.02)
                continue
            s = line.rstrip()
            if "Started" in s or "Please speak" in s:
                got_started = True
                t_speak = time.time()
                log("ASR ready — please speak")
                continue
            if "Use recording device" in s or "sample rate" in s:
                continue
            if "OnlineRecognizerConfig" in s or "Config(" in s:
                continue
            # Display.Print style: "0: text here" or just progressive text
            m = re.match(r"^(\d+):(.*)$", s)
            if m:
                body = m.group(2).strip()
                if body:
                    last_text = body
                    print(f"  partial: {body}", flush=True)
                # heuristic: after endpoint Display often advances index and may print empty
                continue
            # bare text updates
            if s and not s.startswith("/") and "Error" not in s:
                if any("\u4e00" <= c <= "\u9fff" for c in s):
                    last_text = s
                    print(f"  partial: {s}", flush=True)
            # after we have text and some trailing silence time post last update,
            # endpoint is handled inside binary; we detect new segment by empty after text
        # give a moment
    finally:
        kill_proc(proc, "asr")
        wait_device_free()
    return last_text.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description="A7A KWS + ASR voice assistant loop")
    ap.add_argument("--device", default="kws_mic", help="ALSA device (default kws_mic)")
    ap.add_argument(
        "--keywords",
        default=str(ROOT / "keywords_wake.txt"),
        help="keywords file for wake word",
    )
    ap.add_argument(
        "--mode",
        choices=("record", "stream"),
        default="record",
        help="record: fixed window then file ASR (recommended); stream: live endpoint ASR",
    )
    ap.add_argument("--record-seconds", type=float, default=5.0, help="record mode duration")
    ap.add_argument("--stream-max-seconds", type=float, default=12.0)
    ap.add_argument("--stream-silence", type=float, default=0.9)
    ap.add_argument("--asr-threads", type=int, default=2)
    ap.add_argument("--kws-threads", type=int, default=1)
    ap.add_argument("--once", action="store_true", help="exit after one command")
    args = ap.parse_args()

    keywords = Path(args.keywords)
    if not keywords.is_file():
        # fallback
        keywords = ROOT / "keywords.txt"
    if not keywords.is_file():
        log(f"missing keywords file: {keywords}")
        return 1

    for p in [
        BIN / "sherpa-onnx-keyword-spotter-alsa",
        BIN / "sherpa-onnx",
        ROOT / "models/kws/tokens.txt",
        ROOT / "models/asr_zipformer_small_ctc_zh/model.int8.onnx",
    ]:
        if not Path(p).exists():
            log(f"missing: {p}")
            return 1

    run_setup_mic()
    log("=== Voice assistant loop ===")
    log(f"device={args.device} mode={args.mode} keywords={keywords.name}")
    log("Ctrl+C to quit")

    rounds = 0
    try:
        while True:
            try:
                kw = listen_for_wake(args.device, keywords, args.kws_threads)
            except RuntimeError as e:
                log(str(e))
                time.sleep(0.5)
                continue

            text = ""
            if args.mode == "record":
                wav = ROOT / "tmp" / "command.wav"
                try:
                    capture_command(args.device, wav, args.record_seconds)
                    text = asr_file(wav, args.asr_threads)
                except Exception as e:  # noqa: BLE001
                    log(f"capture/ASR error: {e}")
            else:
                try:
                    text = asr_streaming_endpoint(
                        args.device,
                        args.asr_threads,
                        args.stream_max_seconds,
                        args.stream_silence,
                    )
                except Exception as e:  # noqa: BLE001
                    log(f"stream ASR error: {e}")

            rounds += 1
            print("", flush=True)
            print("=" * 48, flush=True)
            print(f"  wake : {kw}", flush=True)
            print(f"  text : {text if text else '(empty)'}", flush=True)
            print("=" * 48, flush=True)
            print("", flush=True)
            # hook for later: intent / LLM / TTS
            # handle_command(text)

            if args.once:
                break
            log("back to KWS…")
    except KeyboardInterrupt:
        log("bye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
