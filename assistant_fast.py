#!/usr/bin/env python3
"""Fast voice assistant: load KWS+ASR once, stream mic, endpoint on silence.

Fixes false-wake loops:
  - short keyword 你好 uses high per-word threshold in keywords file
  - cooldown + mic drain after each session
  - ASR never ends early on empty/noise-only endpoint
"""
from __future__ import annotations

import argparse
import os
import queue
import re
import signal
import struct
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sherpa_onnx

ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 16000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def setup_mic() -> None:
    script = ROOT / "setup_mic.sh"
    if script.is_file():
        subprocess.run(
            ["bash", str(script)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


class HighPass:
    """2nd-order RBJ high-pass. Kills 50/100 Hz mains on AC101 MIC4."""

    def __init__(self, fs: int = SAMPLE_RATE, fc: float = 180.0):
        w0 = 2.0 * np.pi * fc / fs
        cosw, sinw = np.cos(w0), np.sin(w0)
        alpha = sinw / (2.0 * 0.7071)
        b0 = (1.0 + cosw) / 2.0
        b1 = -(1.0 + cosw)
        b2 = (1.0 + cosw) / 2.0
        a0 = 1.0 + alpha
        a1 = -2.0 * cosw
        a2 = 1.0 - alpha
        self.b0, self.b1, self.b2 = b0 / a0, b1 / a0, b2 / a0
        self.a1, self.a2 = a1 / a0, a2 / a0
        self.x1 = self.x2 = self.y1 = self.y2 = 0.0

    def process(self, x: np.ndarray) -> np.ndarray:
        y = np.empty_like(x)
        x1, x2, y1, y2 = self.x1, self.x2, self.y1, self.y2
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        for i, v in enumerate(x):
            o = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1 = x1, float(v)
            y2, y1 = y1, o
            y[i] = o
        self.x1, self.x2, self.y1, self.y2 = x1, x2, y1, y2
        return y


class AlsaMic:
    """Continuous mono 16 kHz capture via arecord."""

    def __init__(self, device: str):
        self.device = device
        self.proc: subprocess.Popen | None = None
        self.hp = HighPass()
        self.open()

    def open(self) -> None:
        self.close()
        self.proc = subprocess.Popen(
            [
                "arecord",
                "-D",
                self.device,
                "-f",
                "S16_LE",
                "-r",
                str(SAMPLE_RATE),
                "-c",
                "1",
                "-t",
                "raw",
                "-q",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def close(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.send_signal(signal.SIGINT)
            self.proc.wait(timeout=1)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None

    def read(self, n_samples: int = 1600) -> np.ndarray:
        assert self.proc and self.proc.stdout
        need = n_samples * 2
        buf = bytearray()
        while len(buf) < need:
            chunk = self.proc.stdout.read(need - len(buf))
            if not chunk:
                log("mic EOF, reopening…")
                time.sleep(0.2)
                self.open()
                assert self.proc and self.proc.stdout
                continue
            buf.extend(chunk)
        raw = np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0
        return self.hp.process(raw)

    def drain(self, seconds: float, chunk: int) -> None:
        """Throw away audio (cooldown / clear residual)."""
        left = int(SAMPLE_RATE * seconds)
        while left > 0:
            n = min(chunk, left)
            self.read(n)
            left -= n


def load_wav_mono16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        nch, sw, sr, nfr, _, _ = w.getparams()
        raw = w.readframes(nfr)
    if sw == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    else:
        raise RuntimeError(f"unsupported sample width {sw} in {path}")
    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)
    if sr != SAMPLE_RATE:
        # linear resample (desk test only)
        x = np.linspace(0.0, 1.0, num=len(pcm), endpoint=False)
        n_out = int(len(pcm) * SAMPLE_RATE / sr)
        pcm = np.interp(np.linspace(0.0, 1.0, num=n_out, endpoint=False), x, pcm).astype(
            np.float32
        )
    return pcm


class WavMic:
    """Feed one or more wav files as if they were the mic (quiet desk test)."""

    def __init__(self, paths: list[Path]):
        parts = [load_wav_mono16k(p) for p in paths]
        # trailing silence so ASR endpoint can fire
        parts.append(np.zeros(int(1.6 * SAMPLE_RATE), dtype=np.float32))
        self.data = np.concatenate(parts)
        self.pos = 0
        self.eof = False
        log(f"wav {len(paths)} file(s) {len(self.data)/SAMPLE_RATE:.1f}s")

    def open(self) -> None:
        pass

    def close(self) -> None:
        pass

    def read(self, n_samples: int = 1600) -> np.ndarray:
        if self.pos >= len(self.data):
            self.eof = True
            return np.zeros(n_samples, dtype=np.float32)
        take = self.data[self.pos : self.pos + n_samples]
        self.pos += n_samples
        if len(take) < n_samples:
            take = np.pad(take, (0, n_samples - len(take)))
            self.eof = True
        return take.astype(np.float32)

    def drain(self, seconds: float, chunk: int) -> None:
        left = int(SAMPLE_RATE * seconds)
        while left > 0:
            n = min(chunk, left)
            self.read(n)
            left -= n


HIT_RE = re.compile(r"^HIT '([^']+)'")


class NpuKws:
    """Streaming NPU KWS: keep NBG loaded, feed S16LE, parse HIT lines."""

    def __init__(self, npu_dir: Path, keywords: Path):
        self.npu_dir = npu_dir
        self.keywords = keywords
        self.proc: subprocess.Popen | None = None
        self.hits: queue.Queue[str] = queue.Queue()
        self.ready = threading.Event()
        self._err_tail: list[str] = []
        self.start()

    def _bin(self) -> Path:
        return self.npu_dir / "kws_npu_demo_a733"

    def start(self) -> None:
        self.stop()
        self.ready.clear()
        demo = self._bin()
        if not demo.is_file():
            raise FileNotFoundError(f"missing NPU KWS demo: {demo}")
        enc = self.npu_dir / "model" / "encoder_float_a733.nb"
        dec = self.npu_dir / "model" / "decoder_float_a733.nb"
        joi = self.npu_dir / "model" / "joiner_float_a733.nb"
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{self.npu_dir / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
        kw = str(self.keywords)
        self.proc = subprocess.Popen(
            [
                str(demo),
                "-nb0",
                str(enc),
                "-nb1",
                str(dec),
                "-nb2",
                str(joi),
                "-k",
                kw,
                "--stdin",
            ],
            cwd=str(self.npu_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
        )
        threading.Thread(target=self._read_out, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()
        if not self.ready.wait(45):
            tail = "".join(self._err_tail[-20:])
            self.stop()
            raise RuntimeError(f"NPU KWS not ready (no KWS_READY)\n{tail}")

    def _read_out(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            s = line.decode("utf-8", "replace").strip()
            if not s:
                continue
            if "KWS_READY" in s:
                self.ready.set()
            m = HIT_RE.match(s)
            if m:
                self.hits.put(m.group(1))

    def _read_err(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line = self.proc.stderr.readline()
            if not line:
                break
            s = line.decode("utf-8", "replace")
            self._err_tail.append(s)
            if len(self._err_tail) > 80:
                self._err_tail = self._err_tail[-40:]

    def feed(self, samples: np.ndarray) -> None:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("NPU KWS process not running")
        pcm = np.clip(samples, -1.0, 1.0)
        b = (pcm * 32767.0).astype(np.int16).tobytes()
        try:
            self.proc.stdin.write(b)
            self.proc.stdin.flush()
        except BrokenPipeError as e:
            raise RuntimeError("NPU KWS stdin closed") from e

    def poll_hit(self) -> str | None:
        try:
            return self.hits.get_nowait()
        except queue.Empty:
            return None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.wait(timeout=2)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None


class NpuAsr:
    """Persistent zoo zipformer NBG: framed S16LE stdin, TEXT= per utterance."""

    def __init__(self, npu_dir: Path):
        self.npu_dir = npu_dir
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue[str] = queue.Queue()
        self.ready = threading.Event()
        self._err_tail: list[str] = []
        self.start()

    def start(self) -> None:
        self.stop()
        self.ready.clear()
        demo = self.npu_dir / "zipformer_demo_a733"
        if not demo.is_file():
            raise FileNotFoundError(f"missing NPU ASR demo: {demo}")
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{self.npu_dir / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
        self.proc = subprocess.Popen(
            [
                str(demo),
                "-nb0",
                str(self.npu_dir / "model" / "encoder_int16_a733.nb"),
                "-nb1",
                str(self.npu_dir / "model" / "decoder_int16_a733.nb"),
                "-nb2",
                str(self.npu_dir / "model" / "joiner_int16_a733.nb"),
                "--stdin",
            ],
            cwd=str(self.npu_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
        )
        threading.Thread(target=self._read_out, daemon=True).start()
        threading.Thread(target=self._read_err, daemon=True).start()
        if not self.ready.wait(60):
            tail = "".join(self._err_tail[-20:])
            self.stop()
            raise RuntimeError(f"NPU ASR not ready (no ASR_READY)\n{tail}")

    def _read_out(self) -> None:
        assert self.proc and self.proc.stdout
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            s = line.decode("utf-8", "replace").strip()
            if "ASR_READY" in s:
                self.ready.set()
            if s.startswith("TEXT="):
                self.lines.put(s[5:])

    def _read_err(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line = self.proc.stderr.readline()
            if not line:
                break
            self._err_tail.append(line.decode("utf-8", "replace"))
            if len(self._err_tail) > 80:
                self._err_tail = self._err_tail[-40:]

    def feed(self, samples: np.ndarray) -> None:
        if not self.proc or not self.proc.stdin:
            raise RuntimeError("NPU ASR process not running")
        pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        try:
            self.proc.stdin.write(struct.pack("<I", len(pcm)) + pcm)
            self.proc.stdin.flush()
        except BrokenPipeError as e:
            raise RuntimeError("NPU ASR stdin closed") from e

    def end_utt(self, timeout: float = 20.0) -> str:
        if not self.proc or not self.proc.stdin:
            return ""
        try:
            self.proc.stdin.write(struct.pack("<I", 0))
            self.proc.stdin.flush()
        except BrokenPipeError:
            return ""
        try:
            return self.lines.get(timeout=timeout).strip()
        except queue.Empty:
            log("NPU ASR timeout waiting TEXT=")
            return ""

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.send_signal(signal.SIGTERM)
            self.proc.wait(timeout=3)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        self.proc = None


def ac_rms(samples: np.ndarray) -> float:
    """RMS after removing DC (this board has large DC bias)."""
    x = samples - float(np.mean(samples))
    return float(np.sqrt(np.mean(x * x)))


def create_kws(
    keywords: Path, num_threads: int, keywords_threshold: float
) -> sherpa_onnx.KeywordSpotter:
    kws_dir = ROOT / "models" / "kws"
    return sherpa_onnx.KeywordSpotter(
        tokens=str(kws_dir / "tokens.txt"),
        encoder=str(kws_dir / "encoder-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
        decoder=str(kws_dir / "decoder-epoch-13-avg-2-chunk-8-left-64.onnx"),
        joiner=str(kws_dir / "joiner-epoch-13-avg-2-chunk-8-left-64.int8.onnx"),
        keywords_file=str(keywords),
        num_threads=num_threads,
        sample_rate=SAMPLE_RATE,
        keywords_score=1.0,
        keywords_threshold=keywords_threshold,
        provider="cpu",
    )


def create_asr(num_threads: int, silence: float) -> sherpa_onnx.OnlineRecognizer:
    asr_dir = ROOT / "models" / "asr_zipformer_small_ctc_zh"
    return sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=str(asr_dir / "tokens.txt"),
        model=str(asr_dir / "model.int8.onnx"),
        num_threads=num_threads,
        sample_rate=SAMPLE_RATE,
        enable_endpoint_detection=True,
        # rule1: pure silence without decoded text — wait longer, avoid early empty end
        rule1_min_trailing_silence=3.5,
        # rule2: after some non-silence decoded, shorter silence ends utterance
        rule2_min_trailing_silence=silence,
        rule3_min_utterance_length=15.0,
        decoding_method="greedy_search",
        provider="cpu",
    )


def wait_wake(
    kws: sherpa_onnx.KeywordSpotter | None,
    npu_kws: NpuKws | None,
    mic: AlsaMic,
    chunk: int,
    min_energy: float,
) -> str:
    """Block until keyword; ignore low-energy chunks (noise / DC)."""
    stream = kws.create_stream() if kws is not None else None
    log("listening…")
    idle = 0
    while True:
        samples = mic.read(chunk)
        if getattr(mic, "eof", False):
            idle += 1
            if idle > 8:
                log("wav ended before wake")
                return ""
        else:
            idle = 0
        e = ac_rms(samples)
        if npu_kws is not None:
            hit = npu_kws.poll_hit()
            if hit:
                return hit
            if e < min_energy:
                continue
            npu_kws.feed(samples)
            hit = npu_kws.poll_hit()
            if hit:
                return hit
            continue
        # Do not feed near-silent frames into KWS (reduces false 你好)
        if e < min_energy:
            continue
        assert kws is not None and stream is not None
        stream.accept_waveform(SAMPLE_RATE, samples)
        while kws.is_ready(stream):
            kws.decode_stream(stream)
            r = kws.get_result(stream)
            if r:
                kw = r if isinstance(r, str) else str(r)
                kws.reset_stream(stream)
                return kw


def write_wav16(path: Path, samples: np.ndarray) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def record_utterance(
    mic: AlsaMic,
    chunk: int,
    max_seconds: float,
    skip_ms: int,
    min_energy: float,
    silence: float,
) -> np.ndarray:
    """Record one command: skip after wake, wait for speech, stop on trailing silence."""
    skip_samples = int(SAMPLE_RATE * skip_ms / 1000)
    skipped = 0
    while skipped < skip_samples:
        n = min(chunk, skip_samples - skipped)
        mic.read(n)
        skipped += n

    log("speak…")
    t0 = time.time()
    buf: list[np.ndarray] = []
    voiced = False
    silent_run = 0.0
    chunk_s = chunk / SAMPLE_RATE

    while time.time() - t0 < max_seconds:
        samples = mic.read(chunk)
        buf.append(samples)
        e = ac_rms(samples)
        if e >= min_energy:
            voiced = True
            silent_run = 0.0
        elif voiced:
            silent_run += chunk_s
            if silent_run >= silence:
                break

    if not buf:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(buf)


def npu_asr_file(wav: Path, npu_dir: Path) -> tuple[str, str]:
    """Run zoo zipformer NBG demo. Returns (text, rtf_line)."""
    script = ROOT / "run_npu_asr.sh"
    env = os.environ.copy()
    env["NPU_ASR_DIR"] = str(npu_dir)
    r = subprocess.run(
        ["bash", str(script), str(wav)],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    out = (r.stdout or "") + "\n" + (r.stderr or "")
    if r.returncode != 0:
        log(f"NPU ASR failed:\n{out[-800:]}")
        return "", ""
    text = ""
    rtf = ""
    for line in out.splitlines():
        if line.startswith("TEXT="):
            text = line[5:].strip()
        elif "Real Time Factor" in line:
            rtf = line.strip()
    return text, rtf


def recognize_command_npu(
    mic: AlsaMic,
    chunk: int,
    max_seconds: float,
    skip_ms: int,
    min_energy: float,
    silence: float,
    npu_asr: NpuAsr,
) -> str:
    """Record command while streaming PCM into the persistent NPU zipformer."""
    skip_samples = int(SAMPLE_RATE * skip_ms / 1000)
    skipped = 0
    while skipped < skip_samples:
        n = min(chunk, skip_samples - skipped)
        mic.read(n)
        skipped += n

    log("speak…")
    t0 = time.time()
    voiced = False
    silent_run = 0.0
    chunk_s = chunk / SAMPLE_RATE
    n_samples = 0
    while time.time() - t0 < max_seconds:
        samples = mic.read(chunk)
        npu_asr.feed(samples)
        n_samples += int(samples.size)
        e = ac_rms(samples)
        if e >= min_energy:
            voiced = True
            silent_run = 0.0
        elif voiced:
            silent_run += chunk_s
            if silent_run >= silence:
                break

    if n_samples < SAMPLE_RATE * 0.2:
        npu_asr.end_utt(timeout=5.0)
        return ""
    return npu_asr.end_utt().strip()


def recognize_command(
    asr: sherpa_onnx.OnlineRecognizer,
    mic: AlsaMic,
    chunk: int,
    max_seconds: float,
    skip_ms: int,
    min_energy: float,
) -> str:
    """Stream ASR until non-empty endpoint or max_seconds.

    Never return early on empty endpoint (noise used to look like 'speech').
    """
    stream = asr.create_stream()
    skip_samples = int(SAMPLE_RATE * skip_ms / 1000)
    skipped = 0
    while skipped < skip_samples:
        n = min(chunk, skip_samples - skipped)
        mic.read(n)
        skipped += n

    log("speak…")
    t0 = time.time()
    last_text = ""
    last_partial_print = ""
    saw_text = False

    while time.time() - t0 < max_seconds:
        samples = mic.read(chunk)
        stream.accept_waveform(SAMPLE_RATE, samples)
        while asr.is_ready(stream):
            asr.decode_stream(stream)

        text = asr.get_result(stream)
        if text and text != last_partial_print:
            last_partial_print = text
            last_text = text
            saw_text = True

        if asr.is_endpoint(stream):
            pad = np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32)
            stream.accept_waveform(SAMPLE_RATE, pad)
            stream.input_finished()
            while asr.is_ready(stream):
                asr.decode_stream(stream)
            text = (asr.get_result(stream) or last_text).strip()
            asr.reset(stream)
            if text:
                return text
            stream = asr.create_stream()
            last_text = ""
            last_partial_print = ""
            saw_text = False

    # timeout flush
    pad = np.zeros(int(0.5 * SAMPLE_RATE), dtype=np.float32)
    stream.accept_waveform(SAMPLE_RATE, pad)
    stream.input_finished()
    while asr.is_ready(stream):
        asr.decode_stream(stream)
    text = (asr.get_result(stream) or last_text).strip()
    asr.reset(stream)
    return text


def _fs2_example_dir() -> Path:
    """Host zoo example, or models copied next to the assistant on the board."""
    cands = [
        ROOT / "tts_fs2",
        ROOT.parent / "tts_fs2_hifigan_zh",
        ROOT / "models" / "tts" / "fs2_hifigan_csmsc",
    ]
    for p in cands:
        if (p / "tools" / "tts_infer.py").is_file() or (p / "tts_infer.py").is_file():
            return p
        if (p / "fastspeech2_csmsc.onnx").is_file():
            return p
    return ROOT.parent / "tts_fs2_hifigan_zh"


def speak_fs2npu(text: str, out: Path, play: bool = True) -> str | None:
    """CPU FastSpeech2-CSMSC + HiFi-GAN (ORT on host, NBG on board)."""
    ex = _fs2_example_dir()
    script = ex / "tools" / "tts_infer.py"
    if not script.is_file():
        script = ex / "tts_infer.py"
    if not script.is_file():
        log(f"fs2npu script missing under {ex}")
        return None
    model_dir = ex / "model" if (ex / "model").is_dir() else ex
    acoustic = model_dir / "fastspeech2_csmsc.onnx"
    if not acoustic.is_file():
        acoustic = ROOT / "models" / "tts" / "fs2_hifigan_csmsc" / "fastspeech2_csmsc.onnx"
    phones = ex / "frontend" / "phone_id_map.txt"
    if not phones.is_file():
        phones = ROOT / "models" / "tts" / "fs2_hifigan_csmsc" / "phone_id_map.txt"
    nb = model_dir / "hifigan_uint8_a733.nb"
    if not nb.is_file():
        nb = ROOT / "models" / "tts" / "fs2_hifigan_csmsc" / "hifigan_uint8_a733.nb"
    cmd = [
        sys.executable,
        str(script),
        "--text",
        text,
        "-o",
        str(out),
        "--acoustic",
        str(acoustic),
        "--phones",
        str(phones),
        "--nb",
        str(nb),
        "--vocoder",
        "auto",
        "--work",
        str(ROOT / "tts_out" / "fs2_work"),
    ]
    if play:
        cmd.append("--play")
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ex if (ex / "tools").is_dir() else ROOT), capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0 or not out.is_file():
        log(f"TTS fail ({dt:.1f}s)")
        return None
    log(f"TTS {dt:.1f}s")
    return str(out)


def speak_tts(
    text: str,
    engine: str = "vits",
    sid: int = 10,
    threads: int = 2,
    play: bool = True,
) -> str | None:
    """Synthesize with sherpa-onnx-offline-tts or FS2+NPU vocoder. Returns wav path."""
    if not text.strip():
        return None
    bin_tts = ROOT / "bin" / "sherpa-onnx-offline-tts"
    if not bin_tts.is_file():
        log("TTS binary missing; skip speak")
        return None
    out_dir = ROOT / "tts_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"reply_{int(time.time())}.wav"
    cmd: list[str]
    if engine in ("vits", "aishell3"):
        v = ROOT / "models" / "tts" / "vits-icefall-zh-aishell3"
        if not (v / "model.onnx").is_file():
            log(f"VITS model missing under {v}")
            return None
        cmd = [
            str(bin_tts),
            f"--vits-model={v / 'model.onnx'}",
            f"--vits-lexicon={v / 'lexicon.txt'}",
            f"--vits-tokens={v / 'tokens.txt'}",
            f"--tts-rule-fsts={v / 'phone.fst'},{v / 'date.fst'},{v / 'number.fst'}",
            f"--num-threads={threads}",
            f"--sid={sid}",
            f"--output-filename={out}",
            text,
        ]
    elif engine == "kokoro":
        k = ROOT / "models" / "tts" / "kokoro-int8-multi-lang-v1_1"
        if not (k / "model.int8.onnx").is_file():
            log(f"Kokoro model missing under {k}")
            return None
        cmd = [
            str(bin_tts),
            f"--kokoro-model={k / 'model.int8.onnx'}",
            f"--kokoro-voices={k / 'voices.bin'}",
            f"--kokoro-tokens={k / 'tokens.txt'}",
            f"--kokoro-data-dir={k / 'espeak-ng-data'}",
            f"--kokoro-lexicon={k / 'lexicon-us-en.txt'},{k / 'lexicon-zh.txt'}",
            f"--tts-rule-fsts={k / 'phone-zh.fst'},{k / 'date-zh.fst'},{k / 'number-zh.fst'}",
            f"--num-threads={threads}",
            f"--sid={sid}",
            f"--output-filename={out}",
            text,
        ]
    elif engine == "melo":
        m = ROOT / "models" / "tts" / "vits-melo-tts-zh_en"
        if not (m / "model.onnx").is_file():
            log(f"Melo model missing under {m}")
            return None
        cmd = [
            str(bin_tts),
            f"--vits-model={m / 'model.onnx'}",
            f"--vits-lexicon={m / 'lexicon.txt'}",
            f"--vits-tokens={m / 'tokens.txt'}",
            f"--tts-rule-fsts={m / 'phone.fst'},{m / 'date.fst'},{m / 'number.fst'}",
            f"--num-threads={threads}",
            "--sid=0",
            f"--output-filename={out}",
            text,
        ]
    elif engine in ("matcha", "matchahifi"):
        a = ROOT / "models" / "tts" / "matcha-icefall-zh-baker"
        voc_name = "hifigan_v2.onnx" if engine == "matchahifi" else "vocos-22khz-univ.onnx"
        v = ROOT / "models" / "tts" / voc_name
        if not (a / "model-steps-3.onnx").is_file() or not v.is_file():
            log(f"Matcha/vocoder missing: {a} {v}")
            return None
        cmd = [
            str(bin_tts),
            f"--matcha-acoustic-model={a / 'model-steps-3.onnx'}",
            f"--matcha-vocoder={v}",
            f"--matcha-lexicon={a / 'lexicon.txt'}",
            f"--matcha-tokens={a / 'tokens.txt'}",
            f"--tts-rule-fsts={a / 'phone.fst'},{a / 'date.fst'},{a / 'number.fst'}",
            f"--num-threads={threads}",
            f"--output-filename={out}",
            text,
        ]
    elif engine == "matchanpu":
        script = ROOT / "tools" / "matcha_npu.py"
        if not script.is_file():
            log(f"matcha_npu.py missing under {ROOT / 'tools'}")
            return None
        cmd = [
            sys.executable,
            str(script),
            "--text",
            text,
            "-o",
            str(out),
            "--vocoder",
            "npu",
        ]
        if play:
            cmd.append("--play")
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode != 0 or not out.is_file():
            log(f"TTS fail ({dt:.1f}s)")
            return None
        log(f"TTS {dt:.1f}s")
        return str(out)
    elif engine in ("fs2npu", "fs2"):
        return speak_fs2npu(text, out, play=play)
    else:
        log(f"unknown TTS engine: {engine}")
        return None

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = f"{ROOT / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0 or not out.is_file():
        log(f"TTS fail ({dt:.1f}s)")
        return None
    log(f"TTS {dt:.1f}s")
    if play:
        subprocess.run(
            ["amixer", "-c", "1", "cset", "name=HPOUT Switch", "on"],
            capture_output=True,
        )
        pr = subprocess.run(
            ["aplay", "-D", "plughw:1,0", str(out)],
            capture_output=True,
            text=True,
        )
        if pr.returncode != 0:
            subprocess.run(["aplay", str(out)], capture_output=True)
    return str(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fast A7A KWS+ASR (models preloaded)")
    ap.add_argument("--device", default="kws_mic")
    ap.add_argument(
        "--from-wav",
        nargs="+",
        default=None,
        metavar="WAV",
        help="desk test: feed wav(s) instead of mic (no need to speak). implies --once",
    )
    ap.add_argument("--keywords", default=str(ROOT / "keywords_wake.txt"))
    ap.add_argument("--kws-threads", type=int, default=1)
    ap.add_argument("--asr-threads", type=int, default=2)
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--silence", type=float, default=0.8)
    ap.add_argument("--max-cmd-seconds", type=float, default=8.0)
    ap.add_argument("--skip-ms", type=int, default=400, help="discard ms after wake")
    ap.add_argument(
        "--cooldown",
        type=float,
        default=1.5,
        help="seconds to ignore mic after a session (anti false-wake loop)",
    )
    ap.add_argument(
        "--min-energy",
        type=float,
        default=0.0,
        help="AC RMS gate after 50Hz HPF; 0 = feed every frame (live mic is very quiet)",
    )
    ap.add_argument(
        "--keywords-threshold",
        type=float,
        default=0.12,
        help="global KWS threshold (per-word # in file can override)",
    )
    ap.add_argument(
        "--tts",
        dest="tts",
        action="store_true",
        default=True,
        help="after ASR, synthesize and play reply (default on)",
    )
    ap.add_argument(
        "--no-tts",
        dest="tts",
        action="store_false",
        help="disable TTS playback",
    )
    ap.add_argument(
        "--tts-engine",
        default="matchanpu",
        choices=("vits", "kokoro", "melo", "matcha", "matchahifi", "matchanpu", "fs2npu"),
        help="default matchanpu=Matcha CPU + HiFi-GAN NPU; vits=8kHz CPU",
    )
    ap.add_argument("--tts-sid", type=int, default=10, help="speaker id")
    ap.add_argument("--tts-threads", type=int, default=2)
    ap.add_argument(
        "--tts-template",
        default="好的，你说的是：{text}",
        help="reply template; {text} = ASR result",
    )
    ap.add_argument(
        "--kws",
        default="npu",
        choices=("cpu", "npu"),
        help="npu=kws_npu_demo NBG (default); cpu=sherpa KeywordSpotter",
    )
    ap.add_argument(
        "--npu-kws-dir",
        default=str(Path.home() / "npu_demos" / "kws_npu_demo"),
        help="directory of kws_npu_demo_a733 + float encoder/decoder/joiner NBG",
    )
    ap.add_argument(
        "--asr",
        default="npu",
        choices=("cpu", "npu"),
        help="npu=zoo Zipformer NBG (default); cpu=sherpa Zipformer-CTC",
    )
    ap.add_argument(
        "--npu-asr-dir",
        default=str(Path.home() / "npu_demos" / "zipformer_demo_linux_a733"),
        help="directory of zipformer_demo_a733 + encoder/decoder/joiner NBG",
    )
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    keywords = Path(args.keywords)
    if not keywords.is_file():
        keywords = ROOT / "keywords.txt"
    if not keywords.is_file():
        log(f"missing keywords: {keywords}")
        return 1

    npu_dir = Path(args.npu_asr_dir)
    npu_kws_dir = Path(args.npu_kws_dir)
    if args.asr == "npu":
        demo = npu_dir / "zipformer_demo_a733"
        if not demo.is_file():
            log(f"missing NPU demo: {demo}")
            return 1
    if args.kws == "npu":
        kws_bin = npu_kws_dir / "kws_npu_demo_a733"
        if not kws_bin.is_file():
            log(f"missing NPU KWS demo: {kws_bin}")
            return 1

    wav_paths: list[Path] = []
    if args.from_wav:
        for w in args.from_wav:
            p = Path(w)
            if not p.is_file():
                p = ROOT / w
            if not p.is_file():
                log(f"missing wav: {w}")
                return 1
            wav_paths.append(p)
        args.once = True
        args.min_energy = 0.0
        args.skip_ms = min(args.skip_ms, 150)
        args.cooldown = 0.0

    if not wav_paths:
        setup_mic()
    chunk = int(SAMPLE_RATE * args.chunk_ms / 1000)

    t0 = time.time()
    kws = None
    npu_kws = None
    npu_asr = None
    if args.kws == "npu":
        npu_kws = NpuKws(npu_kws_dir, keywords)
    else:
        kws = create_kws(keywords, args.kws_threads, args.keywords_threshold)
    asr = None
    if args.asr == "cpu":
        asr = create_asr(args.asr_threads, args.silence)
    else:
        if npu_kws is None:
            npu_asr = NpuAsr(npu_dir)

    if wav_paths:
        mic = WavMic(wav_paths)
    else:
        mic = AlsaMic(args.device)
        # startup settle: drop first 0.5s
        mic.drain(0.5, chunk)

    tts_bit = args.tts_engine if args.tts else "off"
    log(
        f"kws={args.kws} asr={args.asr} tts={tts_bit}  "
        f"ready {time.time()-t0:.1f}s"
    )

    try:
        while True:
            if npu_kws is not None and not npu_kws.alive():
                log("NPU KWS died, restarting…")
                npu_kws.start()
            kw = wait_wake(kws, npu_kws, mic, chunk, args.min_energy)
            if not kw:
                log("no wake, exit")
                break
            log(f"WAKE {kw}")

            t_asr0 = time.time()
            # VIP is exclusive: drop KWS NBG before NPU ASR, reload after.
            if args.asr == "npu" and npu_kws is not None:
                npu_kws.stop()
                if npu_asr is None or not npu_asr.alive():
                    npu_asr = NpuAsr(npu_dir)
            if args.asr == "npu":
                assert npu_asr is not None
                text = recognize_command_npu(
                    mic,
                    chunk,
                    args.max_cmd_seconds,
                    args.skip_ms,
                    args.min_energy,
                    args.silence,
                    npu_asr,
                )
            else:
                text = recognize_command(
                    asr,
                    mic,
                    chunk,
                    args.max_cmd_seconds,
                    args.skip_ms,
                    args.min_energy,
                )
            t_asr1 = time.time()

            log(f"TEXT {text if text else '(empty)'}  {t_asr1 - t_asr0:.1f}s")

            if args.tts and text:
                # release mic while playing to avoid feedback into next KWS
                mic.close()
                # VIP is exclusive: Matcha NPU vocoder cannot share the NPU
                # with KWS/ASR NBGs.
                if args.tts_engine == "matchanpu":
                    if npu_kws is not None:
                        npu_kws.stop()
                    if npu_asr is not None:
                        npu_asr.stop()
                        npu_asr = None
                reply = args.tts_template.format(text=text, wake=kw)
                speak_tts(
                    reply,
                    engine=args.tts_engine,
                    sid=args.tts_sid,
                    threads=args.tts_threads,
                    play=True,
                )
                mic.open()
                mic.drain(0.3, chunk)

            if args.once:
                break

            # Anti-loop: do not immediately re-enter KWS on residual / noise
            cool = args.cooldown
            if not text:
                cool = max(cool, 2.0)
            if args.tts:
                cool = max(cool, 1.0)
            mic.drain(cool, chunk)
            need_kws_reload = npu_kws is not None and (
                (args.asr == "npu") or (args.tts and args.tts_engine == "matchanpu")
            )
            if need_kws_reload:
                if npu_asr is not None:
                    npu_asr.stop()
                    npu_asr = None
                if not npu_kws.alive():
                    npu_kws.start()
    except KeyboardInterrupt:
        log("bye")
    finally:
        if npu_kws is not None:
            npu_kws.stop()
        if npu_asr is not None:
            npu_asr.stop()
        mic.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
