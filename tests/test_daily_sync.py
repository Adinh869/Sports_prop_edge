import pandas as pd

from sports_prop_edge.data.daily_sync import merge_history


def test_merge_history_dedupes():
    existing = pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "game_title": "KBO",
                "player": "lee jung-hoo",
                "team": "kiwoom",
                "opponent": "lg",
                "hits": 1,
            }
        ]
    )
    new_rows = pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "game_title": "KBO",
                "player": "lee jung-hoo",
                "team": "kiwoom",
                "opponent": "lg",
                "hits": 2,
            },
            {
                "date": "2026-06-02",
                "game_title": "KBO",
                "player": "lee jung-hoo",
                "team": "kiwoom",
                "opponent": "ssg",
                "hits": 1,
            },
        ]
    )
    out = merge_history(existing, new_rows)
    assert len(out) == 2
    june1 = out[out["date"] == pd.Timestamp("2026-06-01")].iloc[0]
    assert june1["hits"] == 2
