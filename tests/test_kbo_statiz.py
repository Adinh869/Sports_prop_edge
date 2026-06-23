from sports_prop_edge.integrations.kbo_client import (
    resolve_statiz_player_id,
    search_statiz_players,
)


def test_search_statiz_players_parses_links(monkeypatch):
    html = """
    <html><body>
      <a href="/player/?m=view&s=10123">Lee Jung-hoo</a>
      <a href="https://statiz.sporki.com/player/?m=view&s=10456">Lee Jung Woo</a>
    </body></html>
    """

    class FakeResponse:
        text = html

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "sports_prop_edge.integrations.kbo_client.requests.get",
        lambda *args, **kwargs: FakeResponse(),
    )

    matches = search_statiz_players("lee jung-hoo")
    assert len(matches) == 2
    assert matches[0]["id"] == "10123"
    assert matches[0]["name"] == "Lee Jung-hoo"


def test_resolve_statiz_player_id_uses_cache():
    cache: dict[str, str] = {"lee jung-hoo": "99999"}
    assert resolve_statiz_player_id("lee jung-hoo", None, id_cache=cache) == "99999"


def test_resolve_statiz_player_id_prefers_exact_name_match(monkeypatch):
    html = """
    <a href="/player/?m=view&s=111">Choi Jeong-hee</a>
    <a href="/player/?m=view&s=222">Choi Jeong Hee</a>
    """

    class FakeResponse:
        text = html

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "sports_prop_edge.integrations.kbo_client.requests.get",
        lambda *args, **kwargs: FakeResponse(),
    )

    player_id = resolve_statiz_player_id("choi jeong-hee", None, id_cache={})
    assert player_id == "111"
