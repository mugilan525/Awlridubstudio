@echo off
REM AWLRI Dub Studio - start the server (Windows)
cd /d "%~dp0"
if exist .venv\Scripts\activate.bat call .venv\Scripts\activate.bat

set AWLRI_USER=awlri
set AWLRI_PASSWORD=change-this-password
set COQUI_TOS_AGREED=1
REM set AWLRI_MOCK=1

echo AWLRI Dub Studio -^> http://localhost:8000   (other computers: http://THIS-PC-IP:8000)
uvicorn app:app --host 0.0.0.0 --port 8000
pause
