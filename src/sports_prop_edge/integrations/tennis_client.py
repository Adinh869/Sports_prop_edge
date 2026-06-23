"""Tennis match logs via api.api-tennis.com (Break Points Won and related props)."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import requests

from sports_prop_edge.integrations.name_utils import fuzzy_best_match, normalize_lookup_name

GAME_TITLE = "TENNIS"
API_BASE = "https://api.api-tennis.com/tennis/"
DEFAULT_LOOKBACK_DAYS = 365
REQUEST_PAUSE_SEC = 0.35

PlayerSide = Literal["first", "second"]
_SIDE_LABEL = {"first": "First Player", "second": "Second Player"}


def _api_key() -> str:
    for env_name in ("API_TENNIS_KEY", "API_SPORTS_KEY", "API_FOOTBALL_KEY"):
        val = str(os.getenv(env_name, "")).strip()
        if val:
            return val
    raise ValueError(
        "Tennis sync needs API_TENNIS_KEY in .env (free key at https://api-tennis.com). "
        "API_SPORTS_KEY also accepted if you use API-Sports tennis."
    )


def _api_get(method: str, **params: Any) -> dict[str, Any]:
    key = _api_key()
    payload = {"method": method, "APIkey": key, **params}
    time.sleep(REQUEST_PAUSE_SEC)
    response = requests.get(API_BASE, params=payload, timeout=45)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError(f"Tennis API {method}: unexpected response type")
    if str(data.get("error", "0")) == "1":
        errs = data.get("result")
        msg = errs if isinstance(errs, str) else json.dumps(errs, default=str)[:240]
        raise ValueError(f"Tennis API {method} error: {msg}")
    return data


def _player_side_label(side: PlayerSide) -> str:
    return _SIDE_LABEL[side]


def _side_from_label(label: str) -> PlayerSide | None:
    text = str(label or "").strip().lower()
    if "first" in text:
        return "first"
    if "second" in text:
        return "second"
    return None


def _parse_tennis_point_score(score: str) -> tuple[int, int] | None:
    """Parse '30 - 15' into comparable point indices."""
    text = str(score or "").strip().upper()
    if not text or text in {"-", "0 - 0"}:
        return None
    parts = re.split(r"\s*-\s*", text)
    if len(parts) != 2:
        return None

    def _idx(token: str) -> int | None:
        token = token.strip().upper()
        if token in {"0", "15", "30", "40"}:
            return {"0": 0, "15": 1, "30": 2, "40": 3}[token]
        if token == "AD":
            return 4
        return None

    a = _idx(parts[0])
    b = _idx(parts[1])
    if a is None or b is None:
        return None
    return a, b


def _point_winner(before: str, after: str) -> PlayerSide | None:
    b = _parse_tennis_point_score(before)
    a = _parse_tennis_point_score(after)
    if not b or not a:
        return None
    if a[0] > b[0]:
        return "first"
    if a[1] > b[1]:
        return "second"
    return None


def count_break_points_won(pointbypoint: list[dict[str, Any]] | None, side: PlayerSide) -> int:
    """Count converted break points for one player from api-tennis pointbypoint data."""
    holder_label = _player_side_label(side)
    won = 0
    for game in pointbypoint or []:
        if not isinstance(game, dict):
            continue
        server_side = _side_from_label(str(game.get("player_served", "")))
        if server_side is None:
            continue
        returner: PlayerSide = "second" if server_side == "first" else "first"
        points = game.get("points") or []
        prev_score = "0 - 0"
        for pt in points:
            if not isinstance(pt, dict):
                continue
            score = str(pt.get("score", "")).strip()
            bp = pt.get("break_point")
            if bp and str(bp).strip() == holder_label and returner == side:
                winner = _point_winner(prev_score, score)
                if winner == side:
                    won += 1
            if score:
                prev_score = score
    return won


def _names_match(target: str, candidate: str) -> bool:
    a = normalize_lookup_name(target)
    b = normalize_lookup_name(candidate)
    if not a or not b:
        return False
    if a == b:
        return True
    a_parts = a.split()
    b_parts = b.split()
    if len(a_parts) >= 2 and len(b_parts) >= 2:
        if a_parts[-1] == b_parts[-1] and a_parts[0][0] == b_parts[0][0]:
            return True
    ranked = fuzzy_best_match(target, [candidate], min_score=0.86)
    return bool(ranked)


def _iter_fixture_names(event: dict[str, Any]) -> list[tuple[PlayerSide, str, str]]:
    rows: list[tuple[PlayerSide, str, str]] = []
    first_name = str(event.get("event_first_player", "")).strip()
    first_key = str(event.get("first_player_key", "")).strip()
    second_name = str(event.get("event_second_player", "")).strip()
    second_key = str(event.get("second_player_key", "")).strip()
    if first_name and first_key:
        rows.append(("first", first_name, first_key))
    if second_name and second_key:
        rows.append(("second", second_name, second_key))
    return rows


def _load_player_key_cache(cache_path: Path) -> dict[str, str]:
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k).strip(): str(v).strip() for k, v in data.items() if str(k).strip() and str(v).strip()}


def _save_player_key_cache(cache_path: Path, mapping: dict[str, str]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")


def search_tennis_player_key(
    name: str,
    *,
    cache_path: Path | None = None,
    lookback_days: int = 120,
) -> tuple[str, str]:
    """Resolve PrizePicks name -> api-tennis player_key by scanning recent fixtures."""
    key = normalize_lookup_name(name)
    if not key:
        raise ValueError("Tennis: empty player name")

    cache_file = cache_path or Path("data/cache/tennis_player_keys.json")
    cached = _load_player_key_cache(cache_file)
    if key in cached:
        return cached[key], name

    end = date.today()
    start = end - timedelta(days=max(7, lookback_days))
    data = _api_get(
        "get_fixtures",
        date_start=start.isoformat(),
        date_stop=end.isoformat(),
    )
    events = data.get("result") or []
    if not isinstance(events, list):
        events = []

    candidates: list[tuple[str, str]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        for _side, pname, pkey in _iter_fixture_names(event):
            if _names_match(name, pname):
                candidates.append((pkey, pname))

    if not candidates:
        raise ValueError(f"Tennis: no player_key found for {name!r} in last {lookback_days} days")

    # Prefer longest exact surname match via fuzzy ranking on display names.
    unique: dict[str, str] = {}
    for pkey, pname in candidates:
        unique.setdefault(pkey, pname)
    names = list(unique.values())
    ranked = fuzzy_best_match(name, names, min_score=0.80)
    if ranked:
        pick_name = ranked[0][0]
        for pkey, pname in unique.items():
            if pname == pick_name:
                cached[key] = pkey
                _save_player_key_cache(cache_file, cached)
                return pkey, pick_name

    pkey, pname = next(iter(unique.items()))
    cached[key] = pkey
    _save_player_key_cache(cache_file, cached)
    return pkey, pname


def _fetch_fixtures_for_player(
    player_key: str,
    *,
    date_start: date,
    date_stop: date,
) -> list[dict[str, Any]]:
    data = _api_get(
        "get_fixtures",
        date_start=date_start.isoformat(),
        date_stop=date_stop.isoformat(),
        player_key=str(player_key),
    )
    events = data.get("result") or []
    return [e for e in events if isinstance(e, dict)]


def _event_to_row(
    event: dict[str, Any],
    *,
    player_key: str,
    canonical_name: str,
) -> dict[str, Any] | None:
    status = str(event.get("event_status", "")).strip().lower()
    if status not in {"finished", "after overtime"} and str(event.get("event_winner", "")).strip() == "":
        if str(event.get("event_final_result", "")).strip() in {"", "-"}:
            return None

    side: PlayerSide | None = None
    opponent = ""
    for s, pname, pkey in _iter_fixture_names(event):
        if str(pkey) == str(player_key):
            side = s
        elif _names_match(canonical_name, pname) or str(pkey) == str(player_key):
            side = s
    if side is None:
        return None

    opp_side: PlayerSide = "second" if side == "first" else "first"
    for s, pname, _pkey in _iter_fixture_names(event):
        if s == opp_side:
            opponent = normalize_lookup_name(pname)
            break

    event_date = str(event.get("event_date", "")).strip()[:10]
    if not event_date:
        return None

    bp_won = count_break_points_won(event.get("pointbypoint"), side)
    return {
        "date": event_date,
        "game_title": GAME_TITLE,
        "player": normalize_lookup_name(canonical_name),
        "team": normalize_lookup_name(canonical_name),
        "opponent": opponent or "unknown",
        "games": 1,
        "break_points_won": float(bp_won),
    }


def default_tennis_lookback_days() -> int:
    return DEFAULT_LOOKBACK_DAYS


def fetch_tennis_player_log(
    player_name: str,
    *,
    lookback_days: int | None = None,
    cache_path: Path | None = None,
) -> pd.DataFrame:
    """Pull finished match logs for a tennis player (break_points_won per match)."""
    days = int(lookback_days or default_tennis_lookback_days())
    player_key, canonical = search_tennis_player_key(
        player_name,
        cache_path=cache_path,
        lookback_days=min(days, 120),
    )

    end = date.today()
    start = end - timedelta(days=days)
    events = _fetch_fixtures_for_player(player_key, date_start=start, date_stop=end)

    # Finished matches may omit pointbypoint unless fetched individually.
    rows: list[dict[str, Any]] = []
    seen_events: set[str] = set()
    for event in events:
        ekey = str(event.get("event_key", "")).strip()
        row = _event_to_row(event, player_key=player_key, canonical_name=canonical)
        if row:
            rows.append(row)
            if ekey:
                seen_events.add(ekey)
            continue
        if not ekey or ekey in seen_events:
            continue
        try:
            detail = _api_get("get_fixtures", match_key=ekey)
            detail_events = detail.get("result") or []
            if isinstance(detail_events, list) and detail_events:
                detail_event = detail_events[0]
                if isinstance(detail_event, dict):
                    row = _event_to_row(detail_event, player_key=player_key, canonical_name=canonical)
                    if row:
                        rows.append(row)
                        seen_events.add(ekey)
        except Exception:
            continue

    if not rows:
        raise ValueError(f"Tennis: no finished match logs for {player_name!r}")

    out = pd.DataFrame(rows)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"])
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    keys = ["date", "game_title", "player", "team", "opponent"]
    return out.drop_duplicates(subset=keys, keep="last").sort_values("date").reset_index(drop=True)
