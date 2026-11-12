# AWLRI Dub Studio

Private website for the AWLRI team: upload Tamil speech (10–30 min), get it back in
**English and Hindi in the same speaker's voice**, with every line placed at the same
time as the original. Runs entirely on your own computer, using free open-source models,
so there is no per-minute cost and no audio leaves your machine.

**Powered by AWLRI**

---

## How it works

| Step | What happens | Model (free, open source) |
|---|---|---|
| 1 | Separate voice from background music (optional) | Demucs htdemucs |
| 2 | Tamil speech → Tamil text, with timestamps | Whisper large-v3 (faster-whisper) |
| 3 | Tamil text → English / Hindi text | Meta NLLB-200 1.3B |
| 4 | Learn the speaker's voice from ~25 s of the upload | Coqui XTTS-v2 |
| 5 | Speak each translated line in that voice | XTTS-v2 |
| 6 | **Sync**: fit each line into the original line's time slot (speaks faster/slower, then pitch-preserving time-stretch), place it at the original timestamp | ffmpeg |
| 7 | Mix with the background track, match loudness, export MP3 + WAV + subtitles | ffmpeg |

Outputs per file: `dub_en.mp3/.wav`, `dub_hi.mp3/.wav`, Tamil/English/Hindi `.srt` subtitles,
and a line-by-line transcript in the browser.

## Hardware

* **NVIDIA GPU with 8 GB+ VRAM recommended** (RTX 3060 12 GB or better is comfortable).
  It also runs on CPU only, but a 30-min file will take many hours.
* 16 GB RAM, ~15 GB free disk for the models (downloaded automatically on first run).
* Windows 10/11 or Ubuntu 22.04+. Python **3.10 or 3.11** recommended.

## Install (one time)

1. **Install ffmpeg** and make sure `ffmpeg -version` works in a terminal.
   Windows: `winget install Gyan.FFmpeg`  ·  Ubuntu: `sudo apt install ffmpeg`
2. **Create a Python environment** in this folder:
   ```
   python -m venv .venv
   .venv\Scripts\activate        (Windows)
   source .venv/bin/activate     (Linux)
   ```
3. **Install PyTorch with CUDA** (pick your CUDA version at https://pytorch.org/get-started/locally/), e.g.
   ```
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
4. **Install the rest:**
   ```
   pip install -r requirements.txt
   ```

## Run

1. Open `run.bat` (Windows) or `run.sh` (Linux) and **change `AWLRI_PASSWORD`**.
2. Start it: double-click `run.bat`, or `bash run.sh`.
3. Open **http://localhost:8000** and log in (user `awlri` + your password).
   Team members on the same office network open `http://<this-computer's-IP>:8000`.

The first job is slower because the models (~6 GB) download once.

**Test the website without models:** set `AWLRI_MOCK=1` in the run script. Uploads then
produce placeholder tones instead of speech, so you can check the site works before installing the AI models.

## Tips for the best result

* Clean recordings with one main speaker give the best voice match. If there are several
  speakers, the voice is learned from the clearest one, and all lines use that voice.
* Turn **"Keep background music"** off for plain speech (lectures, voice notes); it is faster.
* Translation of technical terms is the weakest step. Check the transcript view before you publish anything important.
* Settings (environment variables): `AWLRI_WHISPER_MODEL` (default `large-v3`, use `medium` on small GPUs),
  `AWLRI_NLLB_MODEL` (use `facebook/nllb-200-distilled-600M` on small GPUs), `AWLRI_MAX_MINUTES` (default 40),
  `AWLRI_MAX_SPEEDUP` (default 1.35 – how much faster a line may be squeezed to fit).

## What "sync" means here

Each sentence starts when the Tamil sentence started and ends close to when it ended.
This is **phrase-level sync**, which is the standard for voice-over and audio dubbing. It is **not lip-sync**:
the mouth shapes in a video won't match. If a translation is much longer than the original, the line
is spoken faster (up to the speed limit above). If it still doesn't fit, the next line starts a moment
later and the timing catches up at the next pause. No words are ever cut.

## Licences (important)

All parts are free, but two models are licensed for **non-commercial use only**:
* **XTTS-v2** (voice cloning): Coqui Public Model License, non-commercial
* **NLLB-200** (translation): CC-BY-NC 4.0

Internal research and education use is fine. If AWLRI ever **sells** dubbed content or offers this as a
paid service, switch those two parts to commercially licensed alternatives first.
Whisper, Demucs, faster-whisper, ffmpeg, FastAPI: permissive licences (MIT/BSD/LGPL).

Only clone voices with the speaker's consent.

## Files

```
app.py            web server, login, job queue
pipeline.py       the AI dubbing pipeline
static/index.html the website
run.sh / run.bat  start scripts (set your password here)
data/jobs/        uploads and results (delete jobs from the website)
```
