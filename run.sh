#!/usr/bin/env bash
# AWLRI Dub Studio - start the server (Linux / macOS)
cd "$(dirname "$0")"
[ -d .venv ] && source .venv/bin/activate

export AWLRI_USER="awlri"
export AWLRI_PASSWORD="change-this-password"   # <-- set your team password here
export COQUI_TOS_AGREED=1
# export AWLRI_MOCK=1                           # test the website without AI models

echo "AWLRI Dub Studio -> http://localhost:8000  (other computers: http://<this-PC-IP>:8000)"
exec uvicorn app:app --host 0.0.0.0 --port 8000
