@echo off
cd /d "%~dp0.."
set PYTHONPATH=src
call .venv\Scripts\activate.bat
python tools\parse_debug.py > data\cache\parse_debug_stdout.txt 2>&1
