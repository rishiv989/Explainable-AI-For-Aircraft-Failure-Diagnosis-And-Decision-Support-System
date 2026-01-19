@echo off
echo ==========================================
echo       Starting Cortex XAI System
echo ==========================================

:: Start Backend in a new window
echo Starting Backend (FastAPI)...
start "XAI Backend" cmd /k "uvicorn api.main:app --reload --port 8000"

:: Wait a moment for backend to initialize
timeout /t 3 /nobreak >nul

:: Start Frontend in a new window
echo Starting Frontend (React)...
cd frontend
start "XAI Frontend" cmd /k "npm run dev"

echo.
echo ==========================================
echo System Running!
echo Frontend: http://localhost:5173
echo Backend:  http://localhost:8000
echo ==========================================
echo.
pause
