"""
AWLRI Dub Studio - private web app
Run:  uvicorn app:app --host 0.0.0.0 --port 8000
Login with AWLRI_USER / AWLRI_PASSWORD (set them in run.sh / run.bat).
"""
from __future__ import annotations

import json
import os
import queue
import secrets
import shutil
import threading
import time
import traceback
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import pipeline

BASE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("AWLRI_DATA", BASE / "data"))
JOBS_DIR = DATA / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)
MAX_MINUTES = float(os.environ.get("AWLRI_MAX_MINUTES", "40"))

USER = os.environ.get("AWLRI_USER", "awlri")
PASSWORD = os.environ.get("AWLRI_PASSWORD") or ""
if not PASSWORD:
    PASSWORD = secrets.token_urlsafe(9)
    print(f"\n  No AWLRI_PASSWORD set. Temporary login ->  user: {USER}   password: {PASSWORD}\n", flush=True)

security = HTTPBasic(realm="AWLRI Dub Studio")


def auth(c: HTTPBasicCredentials = Depends(security)):
    ok_u = secrets.compare_digest(c.username.encode(), USER.encode())
    ok_p = secrets.compare_digest(c.password.encode(), PASSWORD.encode())
    if not (ok_u and ok_p):
        raise HTTPException(401, "Wrong username or password", headers={"WWW-Authenticate": "Basic"})


app = FastAPI(title="AWLRI Dub Studio", dependencies=[Depends(auth)],
              docs_url=None, redoc_url=None, openapi_url=None)

jobs: dict[str, dict] = {}
lock = threading.Lock()
work_q: "queue.Queue[str]" = queue.Queue()


def save(job: dict):
    d = JOBS_DIR / job["id"]
    d.mkdir(exist_ok=True)
    tmp = d / "job.json.tmp"
    tmp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(d / "job.json")


def update(jid: str, **kw):
    with lock:
        job = jobs.get(jid)
        if not job:
            return
        job.update(kw)
        save(job)


def public(job: dict) -> dict:
    return {k: v for k, v in job.items() if k not in ("input",)}


# ----------------------------------------------------------------- worker
def worker():
    while True:
        jid = work_q.get()
        job = jobs.get(jid)
        if not job or job["status"] != "queued":
            continue
        update(jid, status="running", started=time.time(), stage="Starting", progress=0.0)
        last = [0.0]

        def report(stage: str, frac: float):
            if jobs.get(jid, {}).get("cancel"):
                raise pipeline.Cancelled()
            now = time.time()
            if now - last[0] > 0.7 or frac >= 1.0:  # don't hammer the disk
                last[0] = now
                update(jid, stage=stage, progress=round(frac, 4))

        try:
            res = pipeline.run(JOBS_DIR / jid, JOBS_DIR / jid / job["input"], job["langs"],
                               job["keep_background"], report)
            update(jid, status="done", stage="Done", progress=1.0, finished=time.time(),
                   outputs=res["outputs"], segments=res["segments"])
            shutil.rmtree(JOBS_DIR / jid / "work", ignore_errors=True)
        except pipeline.Cancelled:
            update(jid, status="cancelled", stage="Cancelled", finished=time.time())
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            update(jid, status="failed", stage="Failed", error=str(e)[:600], finished=time.time())


def load_existing():
    for f in sorted(JOBS_DIR.glob("*/job.json")):
        try:
            job = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if job.get("status") == "running":
            job.update(status="failed", stage="Failed", error="Server was restarted while this was processing.")
            save(job)
        jobs[job["id"]] = job
    for job in sorted(jobs.values(), key=lambda j: j["created"]):
        if job["status"] == "queued":
            work_q.put(job["id"])


load_existing()
threading.Thread(target=worker, daemon=True).start()


# ----------------------------------------------------------------- routes
@app.get("/", response_class=HTMLResponse)
def index():
    return (BASE / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/config")
def config():
    return {"max_minutes": MAX_MINUTES, "device": pipeline.device_info(),
            "languages": {k: v["name"] for k, v in pipeline.LANGS.items()}}


@app.post("/api/jobs")
def create_job(file: UploadFile = File(...), langs: str = Form("en,hi"), keep_background: bool = Form(True)):
    chosen = [l for l in dict.fromkeys(langs.split(",")) if l in pipeline.LANGS]
    if not chosen:
        raise HTTPException(400, "Choose at least one output language.")
    jid = uuid.uuid4().hex[:12]
    d = JOBS_DIR / jid
    d.mkdir(parents=True)
    ext = Path(file.filename or "audio").suffix.lower()[:8] or ".bin"
    dst = d / f"input{ext}"
    with open(dst, "wb") as f:
        shutil.copyfileobj(file.file, f, 1 << 20)
    try:
        dur = pipeline.media_duration(dst)
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, "This file could not be read as audio. Try MP3, WAV, M4A, OGG or MP4.")
    if dur > MAX_MINUTES * 60:
        shutil.rmtree(d, ignore_errors=True)
        raise HTTPException(400, f"Audio is {dur / 60:.1f} min - the limit is {MAX_MINUTES:.0f} min.")
    job = {"id": jid, "name": Path(file.filename or "audio").name, "input": dst.name,
           "created": time.time(), "duration": dur, "langs": chosen, "keep_background": keep_background,
           "status": "queued", "stage": "Waiting in queue", "progress": 0.0, "outputs": []}
    with lock:
        jobs[jid] = job
        save(job)
    work_q.put(jid)
    return public(job)


@app.get("/api/jobs")
def list_jobs():
    return [public(j) for j in sorted(jobs.values(), key=lambda j: j["created"], reverse=True)]


def get(jid: str) -> dict:
    job = jobs.get(jid)
    if not job:
        raise HTTPException(404, "Not found")
    return job


@app.get("/api/jobs/{jid}")
def job_detail(jid: str):
    return public(get(jid))


@app.get("/api/jobs/{jid}/transcript")
def transcript(jid: str):
    p = JOBS_DIR / get(jid)["id"] / "transcript.json"
    if not p.exists():
        raise HTTPException(404, "Transcript not ready")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/api/jobs/{jid}/files/{name}")
def job_file(jid: str, name: str, download: bool = False):
    job = get(jid)
    allowed = {o["file"] for o in job.get("outputs", [])}
    if name not in allowed:
        raise HTTPException(404, "Not found")
    p = JOBS_DIR / job["id"] / name
    stem = Path(job["name"]).stem
    return FileResponse(p, filename=f"{stem}_{name}" if download else None)


@app.post("/api/jobs/{jid}/cancel")
def cancel(jid: str):
    job = get(jid)
    if job["status"] == "queued":
        update(jid, status="cancelled", stage="Cancelled")
    elif job["status"] == "running":
        update(jid, cancel=True, stage="Cancelling…")
    return public(jobs[jid])


@app.delete("/api/jobs/{jid}")
def delete(jid: str):
    job = get(jid)
    if job["status"] == "running":
        raise HTTPException(409, "Cancel it first.")
    with lock:
        jobs.pop(jid, None)
    shutil.rmtree(JOBS_DIR / job["id"], ignore_errors=True)
    return {"ok": True}
