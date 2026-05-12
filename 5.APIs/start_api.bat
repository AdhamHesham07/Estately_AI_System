@echo off
echo ==========================================
echo Starting Real Estate AI System API...
echo ==========================================

:: Move to the parent directory (AI_System root) where the .venv is
cd /d "%~dp0\.."

echo Activating Virtual Environment and Starting Server...
call .\.venv\Scripts\python.exe -m uvicorn 5.APIs.main:app --host 127.0.0.1 --port 8000 --reload

pause
