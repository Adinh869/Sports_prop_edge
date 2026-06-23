@echo off
setlocal
cd /d "%~dp0.."
echo [repair] project: %CD%
if not exist ".venv\Scripts\python.exe" (
  echo [repair] ERROR: .venv\Scripts\python.exe not found
  exit /b 1
)
echo [repair] running fix_enc.py...
".venv\Scripts\python.exe" tools\fix_enc.py
if errorlevel 1 (
  echo [repair] fix_enc.py failed
  exit /b 1
)
echo [repair] testing imports...
".venv\Scripts\python.exe" -c "import sys; sys.path.insert(0,'src'); from sports_prop_edge.strategy.payouts import PayoutProfile; from sports_prop_edge.strategy.card_builder import CardRules; print('imports ok')"
if errorlevel 1 (
  echo [repair] import test FAILED
  exit /b 1
)
echo [repair] all good - start app with:
echo   .\.venv\Scripts\streamlit.exe run app\streamlit_app.py
endlocal
