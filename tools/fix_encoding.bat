@echo off
echo [fix_encoding] starting...
cd /d "%~dp0\.."
if not exist .venv\Scripts\python.exe (
  echo [fix_encoding] ERROR: missing .venv
  exit /b 1
)
.venv\Scripts\python.exe tools\fix_enc.py
if errorlevel 1 (
  echo [fix_encoding] ERROR: fix_enc.py failed
  exit /b 1
)
echo [fix_encoding] complete
