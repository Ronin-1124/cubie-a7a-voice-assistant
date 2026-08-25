#!/usr/bin/env python3
"""Matcha-baker CPU acoustic + zoo HiFi-GAN v2 NPU vocoder.

Mel domain matches sherpa matchahifi (22.05 kHz, hop 256, 80 bins).
The NBG is frozen to 200 frames (~2.32 s); longer text is chunked.
"""
from __future__ import annotations

import argparse
import os
import re
import struct
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TTS = ROOT / "models" / "tts"
SR = 22050
HOP = 256
N_MELS = 80
MEL_FRAMES = 200
PAD_ID = 1  # "_"
SILENCE_MEL = -11.4  # zoo vocoder calib floor
PUNCT = set("，。！？、：；,.!?:;“”\"'‘’")


def load_tokens(path: Path) -> dict[str, int]:
    t2i: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if len(parts) == 1 and parts[0].isdigit():
            t2i[" "] = int(parts[0])
        else:
            t2i[parts[0]] = int(parts[-1])
    aliases = [
        (",", "，"), (".", "。"), ("!", "！"), ("?", "？"), (":", "："),
        (";", "；"), ('"', "“"), ('"', "”"), ("'", "‘"), ("'", "’"),
    ]
    for a, b in aliases:
        if a in t2i and b not in t2i:
            t2i[b] = t2i[a]
        if b in t2i and a not in t2i:
            t2i[a] = t2i[b]
    if "、" not in t2i and "，" in t2i:
        t2i["、"] = t2i["，"]
    return t2i


def load_lexicon(path: Path, t2i: dict[str, int]) -> dict[str, list[int]]:
    word2ids: dict[str, list[int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        word = parts[0].lower()
        if word in word2ids:
            continue
        phones = parts[1:]
        if any(p not in t2i for p in phones):
            continue
        word2ids[word] = [t2i[p] for p in phones]
    return word2ids


def split_utf8(text: str) -> list[str]:
    return list(text)


def is_punct(w: str) -> bool:
    return w in PUNCT or (len(w) == 1 and not w.isalnum() and w != " ")


def phrase_match(words: list[str], lexicon: set[str], max_len: int = 10) -> list[str]:
    out: list[str] = []
    i = 0
    n = len(words)
    while i < n:
        w = ""
        ch = words[i]
        if ch and not (ch.isascii() and (ch.isalpha() or ch in ".,!?;:'\"")):
            end = min(i + max_len - 1, n - 1)
            while end > i:
                cand = "".join(words[i : end + 1])
                if cand and cand[-1].isascii() and (cand[-1].isalpha() or cand[-1] in ".,!?;:'\""):
                    end -= 1
                    continue
                if cand in lexicon:
                    w = cand
                    i = end + 1
                    break
                end -= 1
        if not w:
            w = words[i]
            i += 1
        out.append(w)
    return out


def word_to_ids(w: str, word2ids: dict[str, list[int]], t2i: dict[str, int]) -> list[int]:
    if w in word2ids:
        return list(word2ids[w])
    if w in t2i:
        return [t2i[w]]
    ids: list[int] = []
    for ch in split_utf8(w):
        if ch in word2ids:
            ids.extend(word2ids[ch])
        elif ch in t2i:
            ids.append(t2i[ch])
    return ids


def add_blank(ids: list[int], blank: int = PAD_ID) -> list[int]:
    out = [blank] * (len(ids) * 2 + 1)
    out[1::2] = ids
    return out


def text_to_sentences(text: str, t2i: dict[str, int], word2ids: dict[str, list[int]]) -> list[np.ndarray]:
    """Match sherpa CharacterLexicon: split on punctuation, add_blank per sentence."""
    s = text
    s = re.sub(r"[：、；]", "，", s)
    s = s.replace(".", "。").replace("?", "？").replace("!", "！")
    words = split_utf8(s)
    cleaned: list[str] = []
    for w in words:
        if not cleaned:
            cleaned.append(w)
            continue
        if w == " " and (cleaned[-1] == " " or is_punct(cleaned[-1])):
            continue
        if is_punct(w) and (cleaned[-1] == " " or is_punct(cleaned[-1])):
            continue
        cleaned.append(w)
    phrases = phrase_match(cleaned, set(word2ids))
    sentences: list[np.ndarray] = []
    cur: list[int] = []
    for w in phrases:
        part = word_to_ids(w, word2ids, t2i)
        if not part:
            print(f"[g2p] skip OOV {w!r}", file=sys.stderr)
            continue
        cur.extend(part)
        if is_punct(w):
            sentences.append(np.asarray(add_blank(cur, PAD_ID), dtype=np.int64))
            cur = []
    if cur:
        sentences.append(np.asarray(add_blank(cur, PAD_ID), dtype=np.int64))
    return sentences


def run_acoustic(ids: np.ndarray, onnx_path: Path, speed: float = 1.0):
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feeds = {}
    for inp in sess.get_inputs():
        name = inp.name
        if name in ("x", "tokens"):
            feeds[name] = ids.reshape(1, -1)
        elif name in ("x_length", "x_lengths"):
            feeds[name] = np.asarray([ids.size], dtype=np.int64)
        elif "noise" in name:
            feeds[name] = np.asarray([1.0], dtype=np.float32)
        elif "length" in name:
            feeds[name] = np.asarray([1.0 / max(speed, 1e-6)], dtype=np.float32)
        else:
            feeds[name] = np.asarray([1.0], dtype=np.float32)
    t0 = time.time()
    mel = sess.run(None, feeds)[0].astype(np.float32)
    dt = time.time() - t0
    if mel.ndim == 2:
        mel = mel[np.newaxis, ...]
    # want [1, 80, T]
    if mel.shape[1] != N_MELS and mel.shape[-1] == N_MELS:
        mel = np.transpose(mel, (0, 2, 1))
    return mel, dt


def pad_or_chunk(mel: np.ndarray) -> tuple[list[np.ndarray], int]:
    """Return list of [1,80,200] chunks and the frame count actually kept.

    Tiny overflow (trailing silence after punct) is truncated so we do not
    spawn a second NBG that would insert a glitch + extra pad.
    """
    if mel.ndim == 2:
        mel = mel[np.newaxis, ...]
    t = int(mel.shape[-1])
    if t > MEL_FRAMES and t <= MEL_FRAMES + 24:
        mel = mel[..., :MEL_FRAMES]
        t = MEL_FRAMES
    chunks = []
    start = 0
    while start < t:
        piece = np.full((1, N_MELS, MEL_FRAMES), SILENCE_MEL, dtype=np.float32)
        take = min(MEL_FRAMES, t - start)
        piece[..., :take] = mel[..., start : start + take]
        chunks.append(piece)
        start += MEL_FRAMES
    return chunks, t


def write_tensor_ascii(path: Path, arr: np.ndarray) -> None:
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    with path.open("w") as f:
        for v in flat:
            f.write(f"{float(v)}\n")


def load_float_bin(path: Path) -> np.ndarray:
    data = path.read_bytes()
    n = len(data) // 4
    return np.array(struct.unpack(f"<{n}f", data[: n * 4]), dtype=np.float32)


def write_wav(path: Path, wav: np.ndarray, sr: int = SR, peak: float = 0.55) -> None:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    p = float(np.max(np.abs(wav)) + 1e-12)
    # Match matchahifi loudness (~0.55) so NPU vs CPU is not a volume test.
    if p > 1e-3:
        wav = wav * (peak / p)
    wav = np.clip(wav, -1.0, 1.0)
    pcm = (wav * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def find_npu() -> tuple[Path | None, Path | None]:
    home = Path.home()
    bins = [
        home / "npu_demos" / "tts_demo_linux_a733" / "tts_demo_a733",
        ROOT.parent / "tts_npu_zh_en" / "install" / "tts_demo_linux_a733" / "tts_demo_a733",
    ]
    # int16: quiet noise floor, ~60 ms. uint8 hisses. float is slow (~3 s) and noisier on VIP.
    nb_names = [
        "vocoder_int16_a733.nb",
        "vocoder_uint8_a733.nb",
        "vocoder_float_a733.nb",
    ]
    nb_dirs = [
        home / "npu_demos" / "tts_demo_linux_a733" / "model",
        ROOT.parent / "tts_npu_zh_en" / "model",
        ROOT.parent / "tts_npu_zh_en" / "convert_model" / "vocoder" / "model",
        TTS,
    ]
    demo = next((p for p in bins if p.is_file()), None)
    nb = None
    for name in nb_names:
        for d in nb_dirs:
            p = d / name
            if p.is_file():
                nb = p
                return demo, nb
    return demo, nb


def run_vocoder_npu(mel: np.ndarray, demo: Path, nb: Path, work: Path) -> tuple[np.ndarray, float, str]:
    work.mkdir(parents=True, exist_ok=True)
    mel_path = work / "mel.tensor"
    write_tensor_ascii(mel_path, mel)
    out_bin = demo.parent / "output_0.bin"
    if out_bin.exists():
        out_bin.unlink()
    env = os.environ.copy()
    lib = demo.parent / "lib"
    env["LD_LIBRARY_PATH"] = f"{lib}:{env.get('LD_LIBRARY_PATH', '')}"
    cmd = [str(demo), "-nb", str(nb), "-i", str(mel_path), "-st", "1", "-l", "1"]
    t0 = time.time()
    r = subprocess.run(
        cmd,
        cwd=str(demo.parent),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    dt = time.time() - t0
    log = r.stdout or ""
    if r.returncode != 0 or not out_bin.exists():
        raise RuntimeError(f"NPU vocoder failed ({r.returncode}): {log[-800:]}")
    return load_float_bin(out_bin), dt, log


def _codec_card() -> int:
    try:
        text = Path("/proc/asound/cards").read_text()
    except OSError:
        return 0
    for line in text.splitlines():
        if "ac101" in line.lower():
            return int(line.strip().split()[0])
    return 0


def setup_board_audio(card: int | None = None) -> None:
    card = _codec_card() if card is None else card

    def _run(cmd):
        try:
            subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

    _run(["amixer", "-c", str(card), "sset", "HPOUT", "on"])


def play_wav(path: Path) -> None:
    card = _codec_card()
    setup_board_audio(card)
    play_path = path
    try:
        import wave as _wave

        import numpy as np

        with _wave.open(str(path), "rb") as w:
            nch, sw, sr, nfr = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            data = w.readframes(nfr)
        if nch == 1 and sw == 2:
            mono = np.frombuffer(data, dtype=np.int16)
            st = np.empty(mono.size * 2, dtype=np.int16)
            st[0::2] = mono
            st[1::2] = mono
            play_path = path.with_name(path.stem + ".st.wav")
            with _wave.open(str(play_path), "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(sr)
                w.writeframes(st.tobytes())
    except Exception:
        play_path = path
    for cmd in (
        ["aplay", "-q", "-D", f"plughw:{card},0", str(play_path)],
        ["aplay", "-q", str(play_path)],
    ):
        try:
            r = subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if r.returncode == 0:
                return
        except FileNotFoundError:
            continue
    print("[audio] playback failed; wav at", path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("-o", "--output", default="")
    ap.add_argument("--acoustic", default=str(TTS / "matcha-icefall-zh-baker" / "model-steps-3.onnx"))
    ap.add_argument("--lexicon", default=str(TTS / "matcha-icefall-zh-baker" / "lexicon.txt"))
    ap.add_argument("--tokens", default=str(TTS / "matcha-icefall-zh-baker" / "tokens.txt"))
    ap.add_argument("--nb", default="")
    ap.add_argument("--npu-bin", default="")
    ap.add_argument("--vocoder", choices=["npu"], default="npu")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--work", default=str(ROOT / "tts_out" / "matcha_npu_work"))
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    t2i = load_tokens(Path(args.tokens))
    word2ids = load_lexicon(Path(args.lexicon), t2i)
    sentences = text_to_sentences(args.text, t2i, word2ids)
    if not sentences:
        raise SystemExit("empty token ids")
    for i, ids in enumerate(sentences):
        print(f"[g2p] sent{i} tokens={ids.size} ids={ids.tolist()}")

    demo, nb = find_npu()
    if args.npu_bin:
        demo = Path(args.npu_bin)
    if args.nb:
        nb = Path(args.nb)

    wav_npu_parts: list[np.ndarray] = []
    dt_a = dt_npu = 0.0
    last_log = ""
    t_total = 0

    if not demo or not nb or not demo.is_file() or not nb.is_file():
        raise SystemExit(f"NPU vocoder missing: demo={demo} nb={nb}")

    for si, ids in enumerate(sentences):
        mel, dt = run_acoustic(ids, Path(args.acoustic), speed=args.speed)
        dt_a += dt
        print(f"[acoustic] sent{si} {dt*1000:.1f} ms  mel={tuple(mel.shape)} "
              f"mean={float(mel.mean()):.2f} min={float(mel.min()):.2f} max={float(mel.max()):.2f}")
        chunks, t_frames = pad_or_chunk(mel)
        t_total += t_frames
        print(f"[mel] sent{si} T={t_frames} nbg_chunks={len(chunks)}")

        kept = 0
        for ci, ch in enumerate(chunks):
            w, dt, log = run_vocoder_npu(ch, demo, nb, Path(args.work) / f"s{si}c{ci}")
            dt_npu += dt
            last_log = log
            take = min(len(w), min(MEL_FRAMES, t_frames - kept) * HOP)
            wav_npu_parts.append(w[:take])
            kept += MEL_FRAMES
        print(f"[vocoder npu] sent{si} {dt_npu*1000:.1f} ms cum  nb={nb.name}")

    wav = np.concatenate(wav_npu_parts) if wav_npu_parts else None
    if last_log:
        print(last_log)
    print(f"[mel] total T={t_total}")

    out = Path(args.output) if args.output else Path(args.work) / f"matchanpu_{int(time.time())}.wav"
    write_wav(out, wav)
    dur = len(wav) / SR
    rtf = (dt_a + dt_npu) / max(dur, 1e-6)
    print(f"[tts] {dur:.2f}s @{SR} RTF={rtf:.3f} peak={float(np.max(np.abs(wav))):.3f} -> {out}")

    if args.play:
        setup_board_audio()
        play_wav(out)
    print(str(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
