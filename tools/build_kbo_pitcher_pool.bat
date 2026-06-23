@echo off
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe (
  python -m venv .venv
  call .venv\Scripts\activate.bat
  pip install -r requirements.txt
) else (
  call .venv\Scripts\activate.bat
)
set PYTHONPATH=src
set KBO_PITCHER_REBUILD=1
if exist tools\fix_encoding.bat call tools\fix_encoding.bat
if not exist .env (
  echo.
  echo Create .env from .env.example and set PARSE_API_KEY for player name lookup.
  echo.
)
echo Building KBO pitcher pool (Oct -^> today)...
python -m sports_prop_edge.sync_main --board-role pitcher --props data\props\tonight_props.csv
exit /b %ERRORLEVEL%
