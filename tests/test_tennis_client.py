from sports_prop_edge.integrations.tennis_client import count_break_points_won


def test_count_break_points_won_basic():
    pointbypoint = [
        {
            "player_served": "First Player",
            "points": [
                {"score": "0 - 15", "break_point": "Second Player"},
                {"score": "0 - 30", "break_point": "Second Player"},
                {"score": "0 - 40", "break_point": "Second Player"},
                {"score": "0 - 0", "break_point": None},
            ],
        }
    ]
    assert count_break_points_won(pointbypoint, "second") == 2
    assert count_break_points_won(pointbypoint, "first") == 0


def test_count_break_points_won_no_data():
    assert count_break_points_won([], "first") == 0
    assert count_break_points_won(None, "second") == 0
