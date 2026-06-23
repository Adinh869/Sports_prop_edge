import json

from sports_prop_edge.integrations.mykbo_scraper.cache import (
    get_mykbo_cache,
    load_player_id_map,
    reset_mykbo_cache,
    save_player_id_entry,
)
from sports_prop_edge.integrations.mykbo_scraper.cache_stats import CacheStatistics
from sports_prop_edge.integrations.mykbo_scraper.games import (
    build_game_record,
    extract_player_links,
    fetch_game_record,
    merge_player_index,
)
from sports_prop_edge.integrations.mykbo_scraper.http import MyKBOHttpClient, is_cloudflare_challenge
from sports_prop_edge.integrations.mykbo_scraper.resolve import resolve_kbo_player
from sports_prop_edge.integrations.mykbo_scraper.search import parse_search_payload, search_players, title_to_name


def test_title_to_name():
    assert title_to_name("Takeda Shota (타케다) - SP - #23") == "Takeda Shota"


def test_parse_search_payload():
    payload = {
        "results": {
            "SSG Landers": {
                "results": [{"id": 2987, "title": "Takeda Shota (타케다) - SP - #23"}],
            }
        }
    }
    matches = parse_search_payload(payload)
    assert matches[0]["id"] == "2987"
    assert matches[0]["name"] == "Takeda Shota"


def test_player_id_map_persists(tmp_path):
    save_player_id_entry(
        tmp_path,
        "shota takeda",
        mykbo_id="2987",
        matched_name="Takeda Shota",
        method="json_search",
    )
    mapping = load_player_id_map(tmp_path)
    assert mapping["shota takeda"]["mykbo_id"] == "2987"


def test_resolve_uses_id_map_without_http(tmp_path):
    save_player_id_entry(
        tmp_path,
        "an woo-jin",
        mykbo_id="1234",
        matched_name="An Woo-jin",
        method="id_map",
    )
    row = resolve_kbo_player(tmp_path, "an woo-jin", ensure_game_index=False)
    assert row.mykbo_id == "1234"
    assert row.method == "id_map"


def test_extract_player_links_from_html():
    html = '<a href="/players/2987-Takeda">Takeda Shota</a>'
    links = extract_player_links(html)
    assert links[0]["id"] == "2987"
    assert "Takeda" in links[0]["name"]


def test_merge_player_index_last_name_key():
    index = merge_player_index({}, [{"id": "99", "name": "Shota Takeda"}], game_id="1")
    assert "shota takeda" in index or any(v["id"] == "99" for v in index.values())


def test_cloudflare_detection():
    assert is_cloudflare_challenge("Enable JavaScript and cookies to continue")
    assert not is_cloudflare_challenge('{"results": {}}')


def test_cache_statistics_counters():
    stats = CacheStatistics()
    stats.record_hit(2, avoided_http=True)
    stats.record_miss(3)
    assert stats.cache_hits == 1
    assert stats.cache_misses == 1
    assert stats.requests_avoided == 1
    assert stats.hits_by_level[2] == 1
    assert stats.misses_by_level[3] == 1


def test_search_results_cached(tmp_path, monkeypatch):
    reset_mykbo_cache(tmp_path)
    calls = {"n": 0}

    class FakeResponse:
        def json(self):
            return {
                "results": {
                    "Team": {
                        "results": [{"id": 1, "title": "Test Player - SP"}],
                    }
                }
            }

    class FakeClient:
        def get(self, *args, **kwargs):
            calls["n"] += 1
            return FakeResponse()

    monkeypatch.setattr(
        "sports_prop_edge.integrations.mykbo_scraper.search.get_client",
        lambda: FakeClient(),
    )

    first = search_players("test player", root=tmp_path)
    second = search_players("test player", root=tmp_path)
    assert len(first) == 1
    assert calls["n"] == 1
    assert second == first
    stats = get_mykbo_cache(tmp_path).stats
    assert stats.cache_hits >= 1
    assert stats.requests_avoided >= 1


def test_game_record_cached(tmp_path, monkeypatch):
    reset_mykbo_cache(tmp_path)
    html = '<a href="/players/99">Test Pitcher</a>'
    calls = {"n": 0}

    class FakeResponse:
        text = html

    class FakeClient:
        def get(self, *args, **kwargs):
            calls["n"] += 1
            return FakeResponse()

    monkeypatch.setattr(
        "sports_prop_edge.integrations.mykbo_scraper.games.get_client",
        lambda: FakeClient(),
    )

    record1, hit1 = fetch_game_record("123", root=tmp_path)
    record2, hit2 = fetch_game_record("123", root=tmp_path)
    assert calls["n"] == 1
    assert not hit1
    assert hit2
    assert record1["game_id"] == "123"
    assert record2["game_id"] == "123"


def test_daily_pool_cache(tmp_path):
    reset_mykbo_cache(tmp_path)
    import pandas as pd

    from sports_prop_edge.data.kbo_pitcher_pool import load_kbo_pitcher_pool, save_kbo_pitcher_pool

    pool = pd.DataFrame([{"player": "a", "outs": 15}])
    save_kbo_pitcher_pool(pool, tmp_path, slate_date="2026-06-14")
    loaded = load_kbo_pitcher_pool(tmp_path, slate_date="2026-06-14")
    assert len(loaded) == 1
    assert loaded.iloc[0]["player"] == "a"
    stats = get_mykbo_cache(tmp_path).stats
    assert stats.hits_by_level.get(4, 0) >= 1
