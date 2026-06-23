@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    call ".venv\Scripts\activate.bat"
    pip install -r requirements.txt
) else (
    call ".venv\Scripts\activate.bat"
    pip install -q -r requirements.txt
)

if exist "tools\fix_enc.py" (
    echo [run_app] Fixing UTF-16 / null-byte files if needed...
    ".venv\Scripts\python.exe" "tools\fix_enc.py"
)

set "PYTHONPATH=src"
echo [run_app] Starting Streamlit...
".venv\Scripts\streamlit.exe" run "app\streamlit_app.py"

endlocal
