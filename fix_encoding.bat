@echo off
cd /d "%~dp0"
echo [fix_encoding] project: %CD%
if not exist ".venv\Scripts\python.exe" (
  echo [fix_encoding] ERROR: missing .venv — run: python -m venv .venv
  exit /b 1
)
".venv\Scripts\python.exe" tools\fix_enc.py
if errorlevel 1 exit /b 1
for /d /r src %%i in (__pycache__) do rmdir /s /q "%%i" 2>nul
for /d /r src %%i in (__pycache__) do rmdir /s /q "%%i" 2>nul
for /d /r app %%i in (__pycache__) do rmdir /s /q "%%i" 2>nul
echo [fix_encoding] OK — now run: run_app.ps1
exit /b 0
