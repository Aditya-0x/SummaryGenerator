@echo off
title Prism Studio Launcher
echo ==========================================
echo       Starting Prism Studio API
echo ==========================================

REM Check if virtual environment exists
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found in .venv\
    echo Please make sure the project is set up correctly.
    pause
    exit /b
)

echo [INFO] Starting Uvicorn server in the background...
REM Start uvicorn in a new command window so you can see the logs and close it when done
start "Prism Studio Server" cmd /k ".\.venv\Scripts\python.exe -m uvicorn mlplo.api:app --host 127.0.0.1 --port 8000"

echo [INFO] Waiting for server to initialize (this takes a moment to load the model)...
timeout /t 15 /nobreak > NUL

echo [INFO] Opening Prism Studio in your default web browser...
start http://127.0.0.1:8000

echo [SUCCESS] Launcher completed. You can close this window.
timeout /t 2 > NUL
exit
