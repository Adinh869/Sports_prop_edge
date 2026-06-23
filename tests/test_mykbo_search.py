import json

from sports_prop_edge.integrations.kbo_client import (
    _mykbo_title_to_name,
    resolve_mykbo_player_id_html,
    search_mykbo_players_json,
)


def test_mykbo_title_to_name():
    assert _mykbo_title_to_name("Takeda Shota (타케다) - SP - #23") == "Takeda Shota"


def test_search_mykbo_players_json_parses_payload(monkeypatch):
    from sports_prop_edge.integrations.mykbo_scraper.search import search_players

    payload = {
        "results": {
            "SSG Landers": {
                "name": "SSG Landers",
                "results": [
                    {
                        "active": True,
                        "id": 2987,
                        "title": "Takeda Shota (타케다) - SP - #23",
                        "url": "/players/2987-Takeda-Shota-SSG-Landers",
                    }
                ],
            }
        }
    }

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    monkeypatch.setattr(
        "sports_prop_edge.integrations.mykbo_scraper.http.MyKBOHttpClient.get",
        lambda self, *args, **kwargs: FakeResponse(),
    )
    monkeypatch.setattr("sports_prop_edge.integrations.mykbo_scraper.http.time.sleep", lambda *_: None)

    matches = search_players("shota takeda")
    assert len(matches) == 1
    assert matches[0]["id"] == "2987"
    assert matches[0]["name"] == "Takeda Shota"


def test_resolve_mykbo_player_id_from_json_search(monkeypatch):
    payload = {
        "results": {
            "Kiwoom Heroes": {
                "results": [
                    {
                        "id": 1234,
                        "title": "An Woo-jin (안우진) - SP - #19",
                    }
                ],
            }
        }
    }

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return payload

    monkeypatch.setattr(
        "sports_prop_edge.integrations.mykbo_scraper.http.MyKBOHttpClient.get",
        lambda self, *args, **kwargs: FakeResponse(),
    )
    monkeypatch.setattr("sports_prop_edge.integrations.mykbo_scraper.http.time.sleep", lambda *_: None)

    assert resolve_mykbo_player_id_html("an woo-jin") == "1234"
