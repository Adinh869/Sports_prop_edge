@echo off
subst X: "C:\Users\Alex's Liar\Desktop\sports_prop_edge" 2>nul
X:\.venv\Scripts\python.exe -m pytest X:\tests\test_pick_workflow.py X:\tests\test_math.py -q
echo EXIT:%ERRORLEVEL%
