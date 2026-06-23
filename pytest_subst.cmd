@echo off
cmd /v:on /c "subst Z: C:\Users\ALEX'S~1\Desktop\SPORTS~2 && Z: && cd \ && .venv\Scripts\python.exe -m pytest tests\test_calibration.py tests\test_matchup_factors.py tests\test_baseball_projections.py tests\test_probability_ledger.py tests\test_sgp_full_board.py -q & set PYTEST_EC=!ERRORLEVEL! & subst Z: /d & exit /b !PYTEST_EC!"
