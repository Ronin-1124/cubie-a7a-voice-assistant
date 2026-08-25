#!/usr/bin/env python3
"""Fast voice assistant: load KWS+ASR once, stream mic, endpoint on silence.

Fixes false-wake loops:
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

ROOT = Path(__file__).resolve().parent
SAMPLE_RATE = 16000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def codec_card() -> int:
    env = os.environ.get("ALSA_CARD")
    if env:
        return int(env)
    try:
        text = Path("/proc/asound/cards").read_text()
    except OSError:
        return 0
    for line in text.splitlines():
        if "ac101" in line.lower():
            return int(line.strip().split()[0])
    return 0


def setup_mic() -> None:
    script = ROOT / "setup_mic.sh"
    if script.is_file():
        subprocess.run(
            ["bash", str(script)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def play_wav_file(path: Path, card: int | None = None) -> None:
    """Play TTS. Mono wav is duplicated to L+R so both earcups get signal."""
    card = codec_card() if card is None else card
    play_path = path
    try:
        with wave.open(str(path), "rb") as w:
            nch, sw, sr, nfr = (
                w.getnchannels(),
                w.getsampwidth(),
                w.getframerate(),
                w.getnframes(),
            )
            data = w.readframes(nfr)
        if nch == 1 and sw == 2:
            mono = np.frombuffer(data, dtype=np.int16)
            st = np.empty(mono.size * 2, dtype=np.int16)
            st[0::2] = mono
            st[1::2] = mono
            play_path = path.with_name(path.stem + ".st.wav")
            with wave.open(str(play_path), "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(st.tobytes())
    except Exception:
        play_path = path
    subprocess.run(
        ["amixer", "-c", str(card), "sset", "HPOUT", "on"],
        capture_output=True,
    )
    pr = subprocess.run(
        ["aplay", "-q", "-D", f"plughw:{card},0", str(play_path)],
        capture_output=True,
    )
    if pr.returncode != 0:
        subprocess.run(["aplay", "-q", str(play_path)], capture_output=True)


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
    """Capture AC101B stereo PCM, keep the live jack-mic slot, apply gain.

    Direct hw:0,0 capture often blocks after the headphone jack is inserted
    because Pulse suspends the card and I2S BCLK stops. Prefer Pulse, and
    keep a silent playback running so the codec clock stays up.
    """

    def __init__(self, device: str):
        self.device = device
        self.proc: subprocess.Popen | None = None
        self._clk: subprocess.Popen | None = None
        self.hp_l = HighPass()
        self.hp_r = HighPass()
        self.gain = float(os.environ.get("MIC_GAIN", "4"))
        self.card = codec_card()
        self.open()

    def _clock_on(self) -> None:
        if self._clk is not None and self._clk.poll() is None:
            return
        subprocess.run(
            ["pactl", "suspend-source", "@DEFAULT_SOURCE@", "0"],
            capture_output=True,
        )
        subprocess.run(
            ["pactl", "suspend-sink", "@DEFAULT_SINK@", "0"],
            capture_output=True,
        )
        self._clk = subprocess.Popen(
            [
                "aplay",
                "-q",
                "-D",
                "pulse",
                "-f",
                "S16_LE",
                "-c",
                "2",
                "-r",
                "16000",
                "-t",
                "raw",
            ],
            stdin=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        z = b"\x00" * (16000 * 2 * 2 // 10)

        def _feed() -> None:
            while self._clk and self._clk.stdin:
                try:
                    self._clk.stdin.write(z)
                    self._clk.stdin.flush()
                except BrokenPipeError:
                    break

        threading.Thread(target=_feed, daemon=True).start()

    def open(self) -> None:
        self.close()
        self._clock_on()
        for dev in ("pulse", f"plughw:{self.card},0", f"hw:{self.card},0"):
            proc = subprocess.Popen(
                [
                    "arecord",
                    "-D",
                    dev,
                    "-f",
                    "S16_LE",
                    "-r",
                    str(SAMPLE_RATE),
                    "-c",
                    "2",
                    "-t",
                    "raw",
                    "-q",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            time.sleep(0.15)
            if proc.poll() is None:
                self.proc = proc
                return
            try:
                proc.kill()
            except Exception:
                pass
        raise RuntimeError("arecord failed to open pulse/plughw/hw")

    def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.send_signal(signal.SIGINT)
                self.proc.wait(timeout=1)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None
        if self._clk is not None:
            try:
                if self._clk.stdin:
                    self._clk.stdin.close()
            except Exception:
                pass
            try:
                self._clk.kill()
            except Exception:
                pass
            self._clk = None

    def read(self, n_samples: int = 1600) -> np.ndarray:
        assert self.proc and self.proc.stdout
        need = n_samples * 4
        buf = bytearray()
        while len(buf) < need:
            chunk = self.proc.stdout.read(need - len(buf))
            if not chunk:
                time.sleep(0.2)
                self.open()
                assert self.proc and self.proc.stdout
                continue
            buf.extend(chunk)
        st = np.frombuffer(bytes(buf), dtype=np.int16).astype(np.float32) / 32768.0
        left = self.hp_l.process(st[0::2])
        right = self.hp_r.process(st[1::2])
        el = float(np.sqrt(np.mean(left * left) + 1e-12))
        er = float(np.sqrt(np.mean(right * right) + 1e-12))
        x = right if er > el * 1.3 else left
        return np.clip(x * self.gain, -1.0, 1.0)

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

    def drain_hits(self) -> None:
        while self.poll_hit() is not None:
            pass

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


class NpuVocoder:
    """Persistent tts_demo_a733 --stdin: NBG stays loaded, mel chunks in, pcm out."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self._err_tail: list[str] = []
        self.n_in = 0
        self.n_out = 0
        self.start()

    def start(self) -> None:
        self.stop()
        home = Path.home()
        demo = home / "npu_demos" / "tts_demo_linux_a733" / "tts_demo_a733"
        nb = None
        for p in (
            home / "npu_demos" / "tts_demo_linux_a733" / "model" / "vocoder_int16_a733.nb",
            ROOT / "models" / "tts" / "vocoder_int16_a733.nb",
            ROOT / "prebuilt" / "vocoder" / "vocoder_int16_a733.nb",
        ):
            if p.is_file():
                nb = p
                break
        if not demo.is_file() or nb is None:
            raise FileNotFoundError(f"missing vocoder demo={demo} nb={nb}")
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = f"{demo.parent / 'lib'}:{env.get('LD_LIBRARY_PATH', '')}"
        self.proc = subprocess.Popen(
            [str(demo), "-nb", str(nb), "--stdin"],
            cwd=str(demo.parent),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            bufsize=0,
        )
        threading.Thread(target=self._read_err, daemon=True).start()
        assert self.proc.stdout
        deadline = time.time() + 45
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if not line:
                break
            s = line.decode("utf-8", "replace").strip()
            if s.startswith("VOC_READY"):
                parts = dict(
                    kv.split("=", 1) for kv in s.split()[1:] if "=" in kv
                )
                self.n_in = int(parts.get("in", "0"))
                self.n_out = int(parts.get("out", "0"))
                return
        tail = "".join(self._err_tail[-20:])
        self.stop()
        raise RuntimeError(f"NPU vocoder not ready (no VOC_READY)\n{tail}")

    def _read_err(self) -> None:
        assert self.proc and self.proc.stderr
        while True:
            line = self.proc.stderr.readline()
            if not line:
                break
            self._err_tail.append(line.decode("utf-8", "replace"))
            if len(self._err_tail) > 80:
                self._err_tail = self._err_tail[-40:]

    def run(self, mel: np.ndarray) -> np.ndarray:
        if not self.proc or not self.proc.stdin or not self.proc.stdout:
            raise RuntimeError("NPU vocoder process not running")
        flat = np.ascontiguousarray(mel, dtype=np.float32).reshape(-1)
        if self.n_in and flat.size != self.n_in:
            raise RuntimeError(f"vocoder mel size {flat.size} != {self.n_in}")
        n = struct.pack("<I", flat.size)
        try:
            self.proc.stdin.write(n + flat.tobytes())
            self.proc.stdin.flush()
        except BrokenPipeError as e:
            raise RuntimeError("NPU vocoder stdin closed") from e
        def _read_exact(n: int) -> bytes:
            buf = bytearray()
            while len(buf) < n:
                chunk = self.proc.stdout.read(n - len(buf))
                if not chunk:
                    raise RuntimeError("NPU vocoder stdout closed")
                buf.extend(chunk)
            return bytes(buf)

        hdr = _read_exact(4)
        nout = struct.unpack("<I", hdr)[0]
        raw = _read_exact(nout * 4)
        return np.frombuffer(raw, dtype=np.float32).copy()

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            if self.proc.stdin:
                self.proc.stdin.write(struct.pack("<I", 0))
                self.proc.stdin.flush()
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


def wait_wake(
    npu_kws: NpuKws,
    mic: AlsaMic,
    chunk: int,
    min_energy: float,
) -> str:
    """Block until keyword; ignore low-energy chunks (noise / DC)."""
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
        hit = npu_kws.poll_hit()
        if hit:
            return hit
        npu_kws.feed(samples)
        hit = npu_kws.poll_hit()
        if hit:
            return hit


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


    t0 = time.time()
    voiced = False
    silent_run = 0.0
    chunk_s = chunk / SAMPLE_RATE
    n_samples = 0
    while time.time() - t0 < max_seconds:
        samples = mic.read(chunk)
        npu_asr.feed(samples)
        n_samples += int(samples.size)
        if getattr(mic, "eof", False):
            break
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


def speak_tts(text: str, vocoder: NpuVocoder, play: bool = True) -> str | None:
    """Matcha-baker CPU acoustic + resident NPU vocoder."""
    if not text.strip():
        return None
    sys.path.insert(0, str(ROOT / "tools"))
    import matcha_npu as m  # noqa: E402

    t2i = m.load_tokens(m.TTS / "matcha-icefall-zh-baker" / "tokens.txt")
    word2ids = m.load_lexicon(m.TTS / "matcha-icefall-zh-baker" / "lexicon.txt", t2i)
    sentences = m.text_to_sentences(text, t2i, word2ids)
    acoustic = m.TTS / "matcha-icefall-zh-baker" / "model-steps-3.onnx"
    if not acoustic.is_file() or not sentences:
        log("TTS acoustic/g2p missing")
        return None
    t0 = time.time()
    parts: list[np.ndarray] = []
    for ids in sentences:
        mel, _ = m.run_acoustic(ids, acoustic)
        chunks, t_frames = m.pad_or_chunk(mel)
        kept = 0
        for ch in chunks:
            w = vocoder.run(ch)
            take = min(len(w), min(m.MEL_FRAMES, t_frames - kept) * m.HOP)
            parts.append(w[:take])
            kept += m.MEL_FRAMES
    wav = np.concatenate(parts) if parts else None
    if wav is None or wav.size == 0:
        log("TTS fail (empty vocoder)")
        return None
    out_dir = ROOT / "tts_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"reply_{int(time.time())}.wav"
    m.write_wav(out, wav)
    log(f"TTS {time.time() - t0:.1f}s")
    if play:
        play_wav_file(out)
    return str(out)


def main() -> int:
    ap = argparse.ArgumentParser(description="A7A NPU voice assistant (KWS+ASR+TTS)")
    ap.add_argument(
        "--from-wav",
        nargs="+",
        default=None,
        metavar="WAV",
        help="desk test: feed wav(s) instead of mic. implies --once",
    )
    ap.add_argument("--keywords", default=str(ROOT / "keywords_wake.txt"))
    ap.add_argument("--chunk-ms", type=int, default=100)
    ap.add_argument("--silence", type=float, default=0.55)
    ap.add_argument("--max-cmd-seconds", type=float, default=8.0)
    ap.add_argument("--skip-ms", type=int, default=250, help="discard ms after wake")
    ap.add_argument(
        "--cooldown",
        type=float,
        default=1.5,
        help="seconds to ignore mic after a session",
    )
    ap.add_argument(
        "--min-energy",
        type=float,
        default=0.008,
        help="AC RMS after HPF+gain; below this = silence (endpoint). Live mic idle is ~0.001",
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
        "--tts-template",
        default="好的，你说的是：{text}",
        help="reply template; {text} = ASR result",
    )
    ap.add_argument(
        "--npu-kws-dir",
        default=str(Path.home() / "npu_demos" / "kws_npu_demo"),
        help="kws_npu_demo_a733 + float encoder/decoder/joiner NBG",
    )
    ap.add_argument(
        "--npu-asr-dir",
        default=str(Path.home() / "npu_demos" / "zipformer_demo_linux_a733"),
        help="zipformer_demo_a733 + encoder/decoder/joiner NBG",
    )
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()

    keywords = Path(args.keywords)
    if not keywords.is_file():
        log(f"missing keywords: {keywords}")
        return 1

    npu_dir = Path(args.npu_asr_dir)
    npu_kws_dir = Path(args.npu_kws_dir)
    demo = npu_dir / "zipformer_demo_a733"
    kws_bin = npu_kws_dir / "kws_npu_demo_a733"
    if not demo.is_file():
        log(f"missing NPU ASR demo: {demo}")
        return 1
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
        args.skip_ms = min(args.skip_ms, 150)
        args.cooldown = 0.0

    if not wav_paths:
        setup_mic()
    chunk = int(SAMPLE_RATE * args.chunk_ms / 1000)

    t0 = time.time()
    npu_kws = NpuKws(npu_kws_dir, keywords)
    t_kws = time.time()
    npu_asr = NpuAsr(npu_dir)
    t_asr = time.time()
    npu_voc: NpuVocoder | None = None
    if args.tts:
        npu_voc = NpuVocoder()
    t_voc = time.time()
    if args.tts:
        sys.path.insert(0, str(ROOT / "tools"))
        import matcha_npu as _m  # noqa: E402

        _m.get_acoustic_session(_m.TTS / "matcha-icefall-zh-baker" / "model-steps-3.onnx")
    t1 = time.time()
    log(
        f"resident kws={t_kws - t0:.1f}s asr={t_asr - t_kws:.1f}s "
        f"voc={t_voc - t_asr:.1f}s matcha={t1 - t_voc:.1f}s  ready {t1 - t0:.1f}s"
    )

    if wav_paths:
        mic = WavMic(wav_paths)
    else:
        mic = AlsaMic("hw:0,0")
        mic.drain(0.5, chunk)

    try:
        while True:
            if not npu_kws.alive():
                log("NPU KWS died, restarting…")
                npu_kws.start()
            if not npu_asr.alive():
                log("NPU ASR died, restarting…")
                npu_asr.start()
            if npu_voc is not None and not npu_voc.alive():
                log("NPU vocoder died, restarting…")
                npu_voc.start()
            kw = wait_wake(npu_kws, mic, chunk, args.min_energy)
            if not kw:
                log("no wake, exit")
                break
            log(f"WAKE {kw}")
            npu_kws.drain_hits()

            t_asr0 = time.time()
            text = recognize_command_npu(
                mic,
                chunk,
                args.max_cmd_seconds,
                args.skip_ms,
                args.min_energy,
                args.silence,
                npu_asr,
            )
            t_asr1 = time.time()
            log(f"TEXT {text if text else '(empty)'}  {t_asr1 - t_asr0:.1f}s")
            npu_kws.drain_hits()

            if args.tts and text and npu_voc is not None:
                mic.close()
                reply = args.tts_template.format(text=text, wake=kw)
                speak_tts(reply, npu_voc, play=True)
                mic.open()
                mic.drain(0.3, chunk)
                npu_kws.drain_hits()

            if args.once:
                break

            cool = args.cooldown
            if not text:
                cool = max(cool, 2.0)
            if args.tts:
                cool = max(cool, 1.0)
            mic.drain(cool, chunk)
            npu_kws.drain_hits()
    except KeyboardInterrupt:
        log("bye")
    finally:
        npu_kws.stop()
        npu_asr.stop()
        if npu_voc is not None:
            npu_voc.stop()
        mic.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
