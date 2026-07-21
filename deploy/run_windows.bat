@echo off
REM Binance alert unified server (Windows)
REM HOW TO USE:
REM   1) Replace the two values below with your own token and chat id.
REM   2) Double-click this file, or run it from a command prompt.
REM Closing the window stops the server. For 24/7, see DEPLOY.md (Task Scheduler).

set TELEGRAM_BOT_TOKEN=PASTE_YOUR_BOT_TOKEN_HERE
set TELEGRAM_CHAT_ID=PASTE_YOUR_CHAT_ID_HERE

cd /d "%~dp0.."

REM Use the virtualenv python if present, otherwise the system python.
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

echo Starting server. Press Ctrl+C in this window to stop.
%PYTHON% serve.py
pause
