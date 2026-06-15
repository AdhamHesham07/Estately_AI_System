@echo off
echo ==========================================
echo Starting Real Estate AI System API...
echo ==========================================

:: Move to the APIs directory
cd /d "%~dp0"

echo Activating Server...
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

pause
