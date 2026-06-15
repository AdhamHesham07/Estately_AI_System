@echo off
echo Stopping existing FastAPI servers on port 8000...
FOR /F "tokens=5" %%T IN ('netstat -a -n -o ^| findstr :8000') DO (
    taskkill /F /PID %%T >nul 2>&1
)

echo Starting new FastAPI server with updated code...
python 5.APIs/main.py
pause
