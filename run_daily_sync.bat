@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  python -m venv .venv
  call .venv\Scripts\activate.bat
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
set PYTHONPATH=src
if exist tools\fix_encoding.bat call tools\fix_encoding.bat
echo Running daily sync...
python -m sports_prop_edge.sync_main %*
if %ERRORLEVEL% NEQ 0 (
  echo Sync finished with errors. See data\cache\last_sync_report.json
  exit /b %ERRORLEVEL%
)
echo Sync OK. Live history: data\live\history_merged.csv
