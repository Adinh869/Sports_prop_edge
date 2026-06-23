"""WNBA slate context: injuries, lineups, expected minutes (free APIs)."""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from sports_prop_edge.integrations.name_utils import normalize_lookup_name

_CACHE_TTL_HOURS = 2
_REQUEST_PAUSE = 0.35
_WNBA_LEAGUE_ID = "10"
_ESPN_INJURIES_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/injuries"

_DROP_STATUSES = frozenset({"out", "out for season", "suspended", "inactive"})
_DOUBTFUL_STATUSES = frozenset({"doubtful", "questionable", "gtd", "game time decision"})


def _cache_fresh(path: Path, ttl_hours: float = _CACHE_TTL_HOURS) -> bool:
    if not path.exists():
        return False
    age_h = (datetime.now(timezone.utc).timestamp() - path.stat().st_mtime) / 3600
    return age_h < ttl_hours


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _team_key(value: str) -> str:
    return str(value or "").strip().lower()


def _event_date(row: pd.Series) -> str:
    raw = row.get("event_time") or row.get("date")
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return date.today().isoformat()
    parsed = pd.to_datetime(raw, errors="coerce")
    if pd.isna(parsed):
        return date.today().isoformat()
    return parsed.date().isoformat()


def fetch_espn_wnba_injury_status(cache_dir: Path) -> dict[str, str]:
    """Player key -> injury status (Out, Doubtful, etc.) from ESPN public API."""
    cache_path = cache_dir / "wnba_espn_injuries.json"
    if _cache_fresh(cache_path, ttl_hours=2):
        data = _read_json(cache_path)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}

    mapping: dict[str, str] = {}
    try:
        time.sleep(_REQUEST_PAUSE)
        response = requests.get(_ESPN_INJURIES_URL, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException:
        return mapping

    for team_block in payload.get("injuries") or []:
        if not isinstance(team_block, dict):
            continue
        for entry in team_block.get("injuries") or []:
            if not isinstance(entry, dict):
                continue
            athlete = entry.get("athlete") if isinstance(entry.get("athlete"), dict) else {}
            name = str(athlete.get("displayName", "")).strip()
            status = str(entry.get("status", "")).strip()
            if name and status:
                mapping[normalize_lookup_name(name)] = status

    _write_json(cache_path, mapping)
    return mapping


def fetch_wnba_scoreboard(game_date: str, cache_dir: Path) -> list[dict[str, Any]]:
    cache_path = cache_dir / f"wnba_scoreboard_{game_date}.json"
    if _cache_fresh(cache_path, ttl_hours=1):
        data = _read_json(cache_path)
        if isinstance(data, list):
            return data

    try:
        from nba_api.stats.endpoints import scoreboardv3
        from nba_api.stats.library.parameters import LeagueID
    except ImportError:
        return []

    time.sleep(0.5)
    try:
        board = scoreboardv3.ScoreboardV3(
            game_date=game_date,
            league_id=LeagueID.wnba,
            timeout=45,
        )
        header = board.game_header.get_data_frame()
    except Exception:
        return []

    games: list[dict[str, Any]] = []
    if header is None or header.empty:
        _write_json(cache_path, games)
        return games

    line_score = board.line_score.get_data_frame()
    for _, row in header.iterrows():
        gid = str(row.get("gameId", "")).strip()
        if not gid:
            continue
        teams: dict[str, str] = {}
        if line_score is not None and not line_score.empty and "gameId" in line_score.columns:
            subset = line_score[line_score["gameId"].astype(str) == gid]
            for _, team_row in subset.iterrows():
                tri = str(team_row.get("teamTricode", "")).strip().lower()
                if tri:
                    teams[tri] = tri
        games.append(
            {
                "gameId": gid,
                "gameStatus": int(row.get("gameStatus", 0) or 0),
                "teams": teams,
            }
        )

    _write_json(cache_path, games)
    return games


def _game_for_team(games: list[dict[str, Any]], team: str, opponent: str) -> dict[str, Any] | None:
    team_k = _team_key(team)
    opp_k = _team_key(opponent)
    for game in games:
        teams = set(game.get("teams") or {})
        if team_k and team_k in teams:
            if not opp_k or opp_k in teams:
                return game
    return None


def fetch_game_inactive_player_ids(game_id: str, cache_dir: Path) -> set[int]:
    cache_path = cache_dir / f"wnba_inactive_{game_id}.json"
    if _cache_fresh(cache_path, ttl_hours=1):
        data = _read_json(cache_path)
        if isinstance(data, list):
            return {int(x) for x in data}

    ids: set[int] = set()
    try:
        from nba_api.stats.endpoints import boxscoresummaryv3
    except ImportError:
        return ids

    time.sleep(0.5)
    try:
        summary = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id, timeout=45)
        frame = summary.inactive_players.get_data_frame()
    except Exception:
        _write_json(cache_path, list(ids))
        return ids

    if frame is not None and not frame.empty:
        id_col = "personId" if "personId" in frame.columns else "PLAYER_ID"
        if id_col in frame.columns:
            for val in frame[id_col].dropna().tolist():
                try:
                    ids.add(int(val))
                except (TypeError, ValueError):
                    pass

    _write_json(cache_path, list(sorted(ids)))
    return ids


def fetch_game_starter_ids(game_id: str, cache_dir: Path) -> set[int]:
    """Starter player ids when box score is available (pre-tip or live)."""
    cache_path = cache_dir / f"wnba_starters_{game_id}.json"
    if _cache_fresh(cache_path, ttl_hours=1):
        data = _read_json(cache_path)
        if isinstance(data, list):
            return {int(x) for x in data}

    ids: set[int] = set()
    try:
        from nba_api.stats.endpoints import boxscoretraditionalv3
    except ImportError:
        return ids

    time.sleep(0.5)
    try:
        box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=45)
        frame = box.player_stats.get_data_frame()
    except Exception:
        _write_json(cache_path, list(ids))
        return ids

    if frame is not None and not frame.empty:
        pos_col = "position" if "position" in frame.columns else "START_POSITION"
        id_col = "personId" if "personId" in frame.columns else "PLAYER_ID"
        if id_col in frame.columns and pos_col in frame.columns:
            for _, row in frame.iterrows():
                pos = str(row.get(pos_col, "")).strip()
                if not pos or pos.upper() in {"", "N/A", "NONE"}:
                    continue
                try:
                    ids.add(int(row[id_col]))
                except (TypeError, ValueError):
                    pass

    _write_json(cache_path, list(sorted(ids)))
    return ids


def projected_starters_from_history(
    history: pd.DataFrame,
    team: str,
    *,
    top_n: int = 5,
) -> set[str]:
    """Top-N players by recent average minutes for a team (projected rotation)."""
    if history.empty or "team" not in history.columns:
        return set()
    team_k = _team_key(team)
    rows = history[history["team"].astype(str).str.lower() == team_k]
    if rows.empty or "minutes" not in rows.columns:
        return set()
    rows = rows.copy()
    rows["player_key"] = rows["player"].astype(str).map(normalize_lookup_name)
    rows["minutes"] = pd.to_numeric(rows["minutes"], errors="coerce")
    if "date" in rows.columns:
        rows = rows.sort_values("date")
    recent = rows.groupby("player_key").tail(5)
    avg = recent.groupby("player_key")["minutes"].mean().dropna()
    if avg.empty:
        return set()
    return set(avg.sort_values(ascending=False).head(top_n).index.astype(str))


def recent_minutes_average(history: pd.DataFrame, player: str, *, default: float = 28.0) -> float:
    if history.empty:
        return default
    key = normalize_lookup_name(player)
    rows = history[history["player"].astype(str).map(normalize_lookup_name) == key]
    if rows.empty or "minutes" not in rows.columns:
        return default
    mins = pd.to_numeric(rows.sort_values("date").tail(5)["minutes"], errors="coerce").dropna()
    if mins.empty:
        return default
    return float(max(8.0, min(40.0, mins.mean())))


def injury_should_drop(status: str) -> bool:
    return str(status or "").strip().lower() in _DROP_STATUSES


def injury_minutes_factor(status: str) -> float:
    text = str(status or "").strip().lower()
    if injury_should_drop(text):
        return 0.0
    if text in _DOUBTFUL_STATUSES:
        return 0.78
    return 1.0


def load_wnba_history(root: Path) -> pd.DataFrame:
    candidates = [
        root / "data" / "live" / "history_merged.csv",
        root / "data" / "live" / "wnba_history.csv",
    ]
    frames: list[pd.DataFrame] = []
    for path in candidates:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        if "game_title" in df.columns:
            df = df[df["game_title"].astype(str).str.upper() == "WNBA"]
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["date", "player", "team", "opponent"], keep="last"
    )
