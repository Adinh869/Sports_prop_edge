from sports_prop_edge.integrations import soccer_client as sc


def test_parse_player_stat_block_full_match():
    block = {
        "games": {"minutes": 90},
        "goals": {"total": 2, "assists": 1, "saves": 0},
        "shots": {"total": 5, "on": 3},
        "passes": {"total": 42},
        "tackles": {"total": 2},
    }
    parsed = sc._parse_player_stat_block(block)
    assert parsed is not None
    assert parsed["goals"] == 2.0
    assert parsed["assists"] == 1.0
    assert parsed["shots"] == 5.0
    assert parsed["shots_on_target"] == 3.0
    assert parsed["passes"] == 42.0
    assert parsed["tackles"] == 2.0


def test_parse_player_stat_block_dnp():
    assert sc._parse_player_stat_block({"games": {"minutes": 0}}) is None


def test_search_soccer_player_key_uses_cache(tmp_path, monkeypatch):
    cache = tmp_path / "soccer_player_keys.json"
    cache.write_text('{"erling haaland": "123"}', encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise AssertionError("API should not be called when cache hits")

    monkeypatch.setattr(sc, "_api_get", _boom)
    player_id, name = sc.search_soccer_player_key("Erling Haaland", cache_path=cache)
    assert player_id == "123"
    assert name == "Erling Haaland"


def test_fetch_soccer_player_log_builds_rows(tmp_path, monkeypatch):
    cache = tmp_path / "soccer_player_keys.json"
    fixture_cache = tmp_path / "soccer_fixture_stats.json"

    monkeypatch.setattr(sc, "search_soccer_player_key", lambda name, **_: ("99", name))

    def _fake_fixtures(player_id, *, max_fixtures):
        assert player_id == "99"
        return [
            {
                "fixture": {"id": 1001, "date": "2026-05-10T15:00:00+00:00"},
                "teams": {
                    "home": {"id": 10, "name": "Manchester City"},
                    "away": {"id": 20, "name": "Arsenal"},
                },
            }
        ]

    monkeypatch.setattr(sc, "_fetch_recent_fixtures", _fake_fixtures)

    def _fake_fixture_stats(fixture_id, player_id, *, fixture_cache_path):
        assert fixture_id == "1001"
        assert player_id == "99"
        return (
            {
                "minutes": 90.0,
                "games": 1.0,
                "goals": 1.0,
                "assists": 0.0,
                "shots": 4.0,
                "shots_on_target": 2.0,
                "passes": 30.0,
                "tackles": 1.0,
                "saves": 0.0,
            },
            "10",
            "Manchester City",
        )

    monkeypatch.setattr(sc, "_fixture_player_stats", _fake_fixture_stats)

    df = sc.fetch_soccer_player_log(
        "Erling Haaland",
        cache_path=cache,
        fixture_cache_path=fixture_cache,
    )
    assert len(df) == 1
    assert df.iloc[0]["game_title"] == "SOCCER"
    assert df.iloc[0]["goals"] == 1.0
    assert df.iloc[0]["opponent"] == "arsenal"
    assert df.iloc[0]["minutes"] == 90.0


def test_fetch_soccer_player_log_requires_api_key(monkeypatch):
    monkeypatch.delenv("API_FOOTBALL_KEY", raising=False)
    monkeypatch.delenv("API_SPORTS_KEY", raising=False)
    try:
        sc._api_key()
    except ValueError as exc:
        assert "API_FOOTBALL_KEY" in str(exc)
    else:
        raise AssertionError("expected missing API key error")
