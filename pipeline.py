"""
AWLRI Dub Studio - processing pipeline
Tamil speech  ->  English / Hindi speech in the SAME voice, time-aligned to the original.

Stages
  1. prepare     : decode any audio/video file to mono WAV
  2. separate    : (optional) split voice from background music with Demucs
  3. transcribe  : Tamil speech-to-text with timestamps (faster-whisper large-v3)
  4. translate   : Tamil -> English / Hindi (Meta NLLB-200)
  5. voice       : clone the speaker's voice from the original (Coqui XTTS-v2)
  6. dub + sync  : speak each line, fit it into the original line's time slot
  7. mix         : place every line at its original timestamp, add background back

Set AWLRI_MOCK=1 to run the whole flow without any AI models (for testing the app).
"""
from __future__ import annotations

import gc
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

MOCK = os.environ.get("AWLRI_MOCK") == "1"
SR = 24000  # XTTS-v2 output sample rate

WHISPER_MODEL = os.environ.get("AWLRI_WHISPER_MODEL", "large-v3")
NLLB_MODEL = os.environ.get("AWLRI_NLLB_MODEL", "facebook/nllb-200-distilled-1.3B")
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"
MAX_SPEEDUP = float(os.environ.get("AWLRI_MAX_SPEEDUP", "1.35"))

LANGS = {
    "en": {"name": "English", "nllb": "eng_Latn", "xtts": "en"},
    "hi": {"name": "Hindi", "nllb": "hin_Deva", "xtts": "hi"},
}


class Cancelled(Exception):
    pass


# ----------------------------------------------------------------- utilities
def sh(*args) -> str:
    r = subprocess.run([str(a) for a in args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{Path(str(args[0])).name} failed: {r.stderr[-1200:]}")
    return r.stdout


def media_duration(path) -> float:
    out = sh("ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path)
    return float(out.strip())


def to_mono_wav(src, dst, sr: int):
    sh("ffmpeg", "-y", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", sr, dst)


def read_wav(path) -> np.ndarray:
    a, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != SR:
        raise ValueError(f"{path} is {sr} Hz, expected {SR}")
    return a


def device() -> str:
    if MOCK:
        return "cpu"
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def device_info() -> str:
    if MOCK:
        return "Test mode (no AI models)"
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return f"GPU · {p.name} · {p.total_memory / 1e9:.0f} GB"
        return "CPU only (slow)"
    except Exception:
        return "PyTorch not installed"


def free_gpu():
    gc.collect()
    if MOCK:
        return
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def trim_silence(a: np.ndarray, thresh_db: float = -40.0, pad: float = 0.03) -> np.ndarray:
    if len(a) == 0:
        return a
    win = int(0.01 * SR)
    n = len(a) // win
    if n == 0:
        return a
    frames = a[: n * win].reshape(n, win)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    loud = np.where(20 * np.log10(rms + 1e-12) > thresh_db + 20 * np.log10(rms.max() + 1e-12))[0]
    if len(loud) == 0:
        return a
    s = max(0, loud[0] * win - int(pad * SR))
    e = min(len(a), (loud[-1] + 1) * win + int(pad * SR))
    return a[s:e]


def fade(a: np.ndarray, ms: float = 15) -> np.ndarray:
    n = min(len(a) // 2, int(SR * ms / 1000))
    if n > 1:
        a = a.copy()
        r = np.linspace(0, 1, n, dtype=np.float32)
        a[:n] *= r
        a[-n:] *= r[::-1]
    return a


_RUBBERBAND = None


def _has_rubberband() -> bool:
    global _RUBBERBAND
    if _RUBBERBAND is None:
        try:
            _RUBBERBAND = " rubberband " in sh("ffmpeg", "-hide_banner", "-filters")
        except Exception:
            _RUBBERBAND = False
    return _RUBBERBAND


def time_stretch(a: np.ndarray, factor: float, tmp: Path) -> np.ndarray:
    """factor > 1 -> faster/shorter, < 1 -> slower/longer. Pitch is preserved."""
    if abs(factor - 1.0) < 0.02 or len(a) < SR // 10:
        return a
    src, dst = tmp / "_st_in.wav", tmp / "_st_out.wav"
    sf.write(src, a, SR)
    flt = f"rubberband=tempo={factor:.4f}:pitchq=quality" if _has_rubberband() else f"atempo={factor:.4f}"
    sh("ffmpeg", "-y", "-v", "error", "-i", src, "-af", flt, dst)
    return read_wav(dst)


def split_text(text: str, limit: int = 200) -> list[str]:
    """XTTS degrades on long inputs: split on sentence/clause boundaries."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return [text] if text else []
    parts = re.split(r"(?<=[.!?।,;:])\s+", text)
    out, cur = [], ""
    for p in parts:
        while len(p) > limit:  # no punctuation: cut on a space
            cut = p.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if cur:
                out.append(cur)
                cur = ""
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if len(cur) + len(p) + 1 <= limit:
            cur = f"{cur} {p}".strip()
        else:
            if cur:
                out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return [o for o in out if o]


def fmt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(segs, key, path):
    with open(path, "w", encoding="utf-8") as f:
        for i, s in enumerate(segs, 1):
            f.write(f"{i}\n{fmt_ts(s['start'])} --> {fmt_ts(s['end'])}\n{s.get(key, '')}\n\n")


# ----------------------------------------------------------------- stages
def separate(src_wav: Path, work: Path) -> tuple[Path, Path | None]:
    """Returns (vocals, background) at SR mono."""
    if MOCK:
        return src_wav, None
    out = work / "demucs"
    sh(sys.executable, "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
       "-d", device(), "-o", out, src_wav)
    stem = out / "htdemucs" / src_wav.stem
    voc, bg = work / "vocals.wav", work / "background.wav"
    to_mono_wav(stem / "vocals.wav", voc, SR)
    to_mono_wav(stem / "no_vocals.wav", bg, SR)
    return voc, bg


def transcribe(wav16: Path, total: float, prog: Callable[[float], None]) -> list[dict]:
    if MOCK:
        return _mock_segments(wav16, total)
    from faster_whisper import WhisperModel
    dev = device()
    model = WhisperModel(WHISPER_MODEL, device=dev, compute_type="float16" if dev == "cuda" else "int8")
    segs, _ = model.transcribe(
        str(wav16), language="ta", task="transcribe", beam_size=5,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 400},
        condition_on_previous_text=False,
    )
    out = []
    for s in segs:  # generator: decoding happens while iterating
        if s.text.strip():
            out.append({"start": float(s.start), "end": float(s.end), "ta": s.text.strip()})
        prog(min(1.0, s.end / max(total, 1)))
    del model
    free_gpu()
    return out


def merge_segments(segs, min_dur=1.2, max_dur=12.0, max_gap=0.35):
    """Join very short fragments to their neighbour so translation & voice have context."""
    merged: list[dict] = []
    for s in segs:
        if merged:
            p = merged[-1]
            short = (p["end"] - p["start"] < min_dur) or (s["end"] - s["start"] < min_dur)
            if short and s["start"] - p["end"] <= max_gap and s["end"] - p["start"] <= max_dur:
                p["end"] = s["end"]
                p["ta"] = f"{p['ta']} {s['ta']}"
                continue
        merged.append(dict(s))
    return merged


def translate(texts: list[str], langs: list[str], prog: Callable[[float], None]) -> dict[str, list[str]]:
    if MOCK:
        res = {l: [f"{LANGS[l]['name']} line {i + 1}: " + " ".join(["word"] * max(3, len(t) // 7))
                   for i, t in enumerate(texts)] for l in langs}
        prog(1.0)
        return res
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    dev = device()
    tok = AutoTokenizer.from_pretrained(NLLB_MODEL, src_lang="tam_Taml")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        NLLB_MODEL, torch_dtype=torch.float16 if dev == "cuda" else torch.float32).to(dev).eval()
    res: dict[str, list[str]] = {}
    bs, done, total = 12, 0, len(texts) * len(langs)
    for l in langs:
        tgt = tok.convert_tokens_to_ids(LANGS[l]["nllb"])
        out: list[str] = []
        for i in range(0, len(texts), bs):
            batch = texts[i:i + bs]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=400).to(dev)
            with torch.inference_mode():
                gen = model.generate(**enc, forced_bos_token_id=tgt, num_beams=4, max_new_tokens=320)
            out += [t.strip() for t in tok.batch_decode(gen, skip_special_tokens=True)]
            done += len(batch)
            prog(done / max(total, 1))
        res[l] = out
    del model
    free_gpu()
    return res


def pick_voice_references(vocals: np.ndarray, segs, work: Path, want: float = 25.0) -> list[str]:
    """Choose the cleanest, longest single-speaker lines as the voice sample for cloning."""
    def score(s):
        clip = vocals[int(s["start"] * SR):int(s["end"] * SR)]
        return (s["end"] - s["start"]) * float(np.sqrt((clip ** 2).mean() + 1e-12))
    cands = [s for s in segs if 3.0 <= s["end"] - s["start"] <= 12.0] or list(segs)
    cands.sort(key=score, reverse=True)
    refs, acc = [], 0.0
    for s in cands:
        clip = vocals[int(s["start"] * SR):int(s["end"] * SR)]
        if len(clip) < SR:
            continue
        p = work / f"voice_ref_{len(refs)}.wav"
        sf.write(p, fade(clip), SR)
        refs.append(str(p))
        acc += len(clip) / SR
        if acc >= want or len(refs) >= 6:
            break
    if not refs:  # extremely short input: just use the first 10 s
        p = work / "voice_ref_0.wav"
        sf.write(p, vocals[: SR * 10], SR)
        refs = [str(p)]
    return refs


class Voice:
    """Voice clone of the original speaker (XTTS-v2, supports English + Hindi)."""

    def __init__(self, refs: list[str]):
        self.model = None
        if MOCK:
            return
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from TTS.api import TTS
        self._tts = TTS(XTTS_MODEL).to(device())
        self.model = self._tts.synthesizer.tts_model
        self.gpt, self.spk = self.model.get_conditioning_latents(
            audio_path=refs, gpt_cond_len=30, max_ref_length=60)

    def say(self, text: str, lang: str, speed: float = 1.0) -> np.ndarray:
        if MOCK:
            n = max(1, int(SR * 0.055 * len(text) / speed))
            t = np.arange(n) / SR
            return (0.25 * np.sin(2 * np.pi * 180 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
        pieces = []
        for chunk in split_text(text, 200):
            r = self.model.inference(chunk, LANGS[lang]["xtts"], self.gpt, self.spk,
                                     temperature=0.65, speed=speed, enable_text_splitting=False)
            pieces += [np.asarray(r["wav"], dtype=np.float32), np.zeros(int(0.08 * SR), np.float32)]
        return np.concatenate(pieces[:-1]) if pieces else np.zeros(1, np.float32)

    def close(self):
        self.model = None
        self._tts = None
        free_gpu()


def fit_line(voice: Voice, text: str, lang: str, target: float, slot: float, tmp: Path) -> np.ndarray:
    """Speak `text` so it lasts about `target` seconds and never spills past `slot`."""
    if not text.strip():
        return np.zeros(0, np.float32)
    wav = trim_silence(voice.say(text, lang))
    dur = len(wav) / SR
    # Much too long: ask the voice model itself to talk faster (sounds more natural than stretching)
    if dur > target * 1.15:
        wav = trim_silence(voice.say(text, lang, speed=min(dur / target, 1.3)))
        dur = len(wav) / SR
    if dur > target:
        factor = min(dur / target, MAX_SPEEDUP)
        if dur / factor > slot:  # would overlap the next line: squeeze harder
            factor = min(dur / slot, 1.75)
        wav = time_stretch(wav, factor, tmp)
    elif dur < target * 0.8:  # much shorter: slow down slightly to cover the line
        wav = time_stretch(wav, max(dur / (target * 0.92), 0.88), tmp)
    # never cut words: if it still overruns, the caller nudges the next line later
    return fade(wav)


def mix(lines, total_len: int, ref_vocals: np.ndarray, background: np.ndarray | None) -> np.ndarray:
    total_len = max([total_len] + [int(st * SR) + len(w) for st, w in lines])
    track = np.zeros(total_len, np.float32)
    for start, wav in lines:
        i = int(start * SR)
        n = min(len(wav), total_len - i)
        if n > 0:
            track[i:i + n] += wav[:n]
    # loudness: match the original speech level
    def speech_rms(x):
        x = x[np.abs(x) > 1e-4]
        return float(np.sqrt((x ** 2).mean())) if len(x) else 0.0
    a, b = speech_rms(ref_vocals), speech_rms(track)
    if a > 0 and b > 0:
        track *= min(a / b, 4.0)
    if background is not None:
        n = min(len(background), total_len)
        track[:n] += background[:n]
    peak = float(np.abs(track).max() or 1.0)
    if peak > 0.98:
        track *= 0.98 / peak
    return track


# ----------------------------------------------------------------- main entry
def run(job_dir: Path, src: Path, langs: list[str], keep_background: bool,
        report: Callable[[str, float], None]) -> dict:
    """report(stage_text, overall_fraction) - may raise Cancelled."""
    work = job_dir / "work"
    work.mkdir(parents=True, exist_ok=True)

    # overall progress weights
    w = {"prep": 3, "sep": 12 if keep_background else 0, "asr": 20, "tr": 8, "tts": 57}
    tot = sum(w.values())
    base = {"prep": 0}
    base["sep"] = base["prep"] + w["prep"]
    base["asr"] = base["sep"] + w["sep"]
    base["tr"] = base["asr"] + w["asr"]
    base["tts"] = base["tr"] + w["tr"]

    def P(stage, frac, text):
        report(text, (base[stage] + w[stage] * max(0.0, min(1.0, frac))) / tot)

    P("prep", 0, "Preparing audio")
    orig = work / "original.wav"
    to_mono_wav(src, orig, SR)
    asr_in = work / "asr16k.wav"
    total = media_duration(orig)

    vocals_path, bg_path = orig, None
    if keep_background:
        P("sep", 0, "Separating voice from background music")
        vocals_path, bg_path = separate(orig, work)
        P("sep", 1, "Voice separated")
    to_mono_wav(vocals_path, asr_in, 16000)

    P("asr", 0, "Listening to Tamil speech")
    segs = transcribe(asr_in, total, lambda f: P("asr", f, f"Transcribing Tamil · {int(f * 100)}%"))
    segs = merge_segments(segs)
    if not segs:
        raise RuntimeError("No speech was detected in this file.")

    P("tr", 0, "Translating")
    tr = translate([s["ta"] for s in segs], langs, lambda f: P("tr", f, f"Translating · {int(f * 100)}%"))
    for l in langs:
        for s, t in zip(segs, tr[l]):
            s[l] = t

    vocals = read_wav(vocals_path)
    background = read_wav(bg_path) if bg_path else None
    total_len = len(read_wav(orig))

    P("tts", 0, "Learning the speaker's voice")
    refs = pick_voice_references(vocals, segs, work)
    voice = Voice(refs)

    outputs = []
    steps, done = len(segs) * len(langs), 0
    try:
        for l in langs:
            lines, cursor = [], 0.0
            for i, s in enumerate(segs):
                start = max(s["start"], cursor)  # only later than original if the previous line overran
                nxt = segs[i + 1]["start"] if i + 1 < len(segs) else total
                target = max(0.4, s["end"] - start)
                slot = max(target, nxt - start - 0.05)
                wav = fit_line(voice, s[l], l, target, slot, work)
                lines.append((start, wav))
                cursor = start + len(wav) / SR + 0.08
                done += 1
                P("tts", done / steps, f"Speaking {LANGS[l]['name']} · line {i + 1} of {len(segs)}")
            track = mix(lines, total_len, vocals, background)
            wav_p = job_dir / f"dub_{l}.wav"
            mp3_p = job_dir / f"dub_{l}.mp3"
            sf.write(wav_p, track, SR)
            sh("ffmpeg", "-y", "-v", "error", "-i", wav_p, "-codec:a", "libmp3lame", "-b:a", "192k", mp3_p)
            outputs += [
                {"file": mp3_p.name, "label": f"{LANGS[l]['name']} audio (MP3)", "lang": l, "kind": "audio"},
                {"file": wav_p.name, "label": f"{LANGS[l]['name']} audio (WAV)", "lang": l, "kind": "download"},
            ]
    finally:
        voice.close()

    # original as mp3 for the in-browser player
    sh("ffmpeg", "-y", "-v", "error", "-i", orig, "-codec:a", "libmp3lame", "-b:a", "160k", job_dir / "original.mp3")
    outputs.insert(0, {"file": "original.mp3", "label": "Original Tamil (MP3)", "lang": "ta", "kind": "audio"})

    for key in ["ta"] + langs:
        p = job_dir / f"subtitles_{key}.srt"
        write_srt(segs, key, p)
        outputs.append({"file": p.name, "label": f"{'Tamil' if key == 'ta' else LANGS[key]['name']} subtitles (SRT)",
                        "lang": key, "kind": "download"})

    (job_dir / "transcript.json").write_text(json.dumps(segs, ensure_ascii=False, indent=1), encoding="utf-8")
    report("Done", 1.0)
    return {"outputs": outputs, "segments": len(segs), "duration": total}


def _mock_segments(wav16: Path, total: float) -> list[dict]:
    segs, t, i = [], 0.4, 0
    while t + 1.5 < total:
        d = 2.0 + (i * 1.7) % 5
        e = min(t + d, total - 0.1)
        segs.append({"start": round(t, 2), "end": round(e, 2), "ta": f"தமிழ் வாக்கியம் {i + 1} " + "சொல் " * int(d * 2)})
        t, i = e + 0.6, i + 1
    return segs
