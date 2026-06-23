@echo off
cd /d "%~dp0"
set PYTHONPATH=src
if not exist .venv\Scripts\python.exe (
  echo Create a venv and install requirements first: pip install -r requirements.txt
  exit /b 1
)
.venv\Scripts\python.exe -m sports_prop_edge.deployment.server
