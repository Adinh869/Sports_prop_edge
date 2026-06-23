@echo off
cd /d "%~dp0.."
set PYTHONPATH=src
call .venv\Scripts\activate.bat
python tools\auto_grade_pending.py > data\cache\auto_grade_stdout.txt 2>&1
