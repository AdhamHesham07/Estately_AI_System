@echo off
echo ==========================================
echo Restarting Real Estate AI System API...
echo ==========================================

:: Move to the APIs directory
cd /d "%~dp0"

echo Stopping existing FastAPI servers on port 8000...
FOR /F "tokens=5" %%T IN ('netstat -a -n -o ^| findstr :8000') DO (
    taskkill /F /PID %%T >nul 2>&1
)

echo Starting API server with updated code...
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

pause
