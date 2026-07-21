@echo off
REM ── 바이낸스 알림 통합 서버 (Windows) ─────────────────────────────
REM 사용법:
REM   1) 아래 두 줄의 값을 본인 것으로 바꾸세요.
REM   2) 이 파일을 더블클릭하거나 명령창에서 실행하세요.
REM 창을 닫으면 서버도 멈춥니다. 24시간 운영은 DEPLOY.md의 작업 스케줄러 안내 참고.

set TELEGRAM_BOT_TOKEN=여기에_봇토큰_붙여넣기
set TELEGRAM_CHAT_ID=여기에_챗아이디_붙여넣기

cd /d "%~dp0\.."

REM 가상환경이 있으면 사용, 없으면 시스템 파이썬 사용
if exist ".venv\Scripts\python.exe" (
    set PYTHON=.venv\Scripts\python.exe
) else (
    set PYTHON=python
)

echo 서버를 시작합니다. 종료하려면 이 창에서 Ctrl+C 를 누르세요.
%PYTHON% serve.py
pause
