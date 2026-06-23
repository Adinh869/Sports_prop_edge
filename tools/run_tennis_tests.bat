@echo off
subst X: "C:\Users\Alex's Liar\Desktop\sports_prop_edge" 2>nul
X:\.venv\Scripts\python.exe -m pytest X:\tests\test_tennis_client.py X:\tests\test_prizepicks.py::test_normalize_stat_type_tennis X:\tests\test_prizepicks.py::test_league_to_game_title X:\tests\test_prop_filters.py::test_is_modelable_prop_tennis_break_points -q
