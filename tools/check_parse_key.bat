@echo off
cd /d "%~dp0.."
set PYTHONPATH=src
call .venv\Scripts\activate.bat
python tools\check_parse_key.py
