"""KBO logs: MyKBO HTML scrape (default), optional Parse API, optional Statiz."""

# Keep this file UTF-8 (UTF-16 causes: SyntaxError null bytes).

from __future__ import annotations

import io
import json
import os
import re
import time
from datetime import date, timedelta
from pathlib import Path
from collections.abc import Callable
from typing import Any
from urllib.parse import quote_plus, urljoin

import pandas as pd
import requests

from sports_prop_edge.integrations.mykbo_client import (
    _batter_row_from_box_score,
    _name_matches,
    _pitcher_row_from_box_score,
)
from sports_prop_edge.integrations.name_utils import fuzzy_best_match, names_match, normalize_lookup_name

GAME_TITLE = "KBO"
KBO_DEFAULT_SEASON_YEARS: tuple[int, ...] = (2025, 2026)
MYKBO_BASE = "https://mykbostats.com"
STATIZ_BASE = "https://statiz.sporki.com/player/"
STATIZ_ALT = "https://www.statiz.co.kr/player/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

KBO_COLUMN_ALIASES = {
    "date": {"date", "game_date", "일자", "날짜"},
    "player": {"player", "name"},
    "team": {"team", "tm", "팀"},
    "opponent": {"opponent", "opp", "상대"},
    "plate_appearances": {"pa", "plate_appearances", "타석"},
    "hits": {"h", "hits", "안타"},
    "runs": {"r", "runs", "득점"},
    "rbis": {"rbi", "rbis", "타점"},
    "strikeouts": {"so", "k", "strikeouts", "삼진"},
    "total_bases": {"tb", "total_bases", "루타"},
    "walks": {"bb", "walks", "볼넷"},
    "stolen_bases": {"sb", "stolen_bases", "도루"},
}

GAME_LINK_RE = re.compile(
    r'(?:href=["\'])?(?:https?://mykbostats\.com)?/games/(\d+)',
    re.IGNORECASE,
)
MYKBO_PLAYER_LINK_RE = re.compile(
    r'(?:href=["\'])?(?:https?://mykbostats\.com)?/players/(\d+)',
    re.IGNORECASE,
)
FINAL_RE = re.compile(r"\bFinal\b", re.IGNORECASE)
FINAL_SCORE_RE = re.compile(
    r"([A-Za-z][A-Za-z .'-]+?)\s+(?:[LW]:\s*[^<\d]+?)?\s*(\d+)\s*:\s*(\d+)\s+Final\s+([A-Za-z][A-Za-z .'-]+)",
    re.IGNORECASE,
)
TITLE_RE = re.compile(r"^#\s*(.+?)\s+(\d+)\s*:\s*(\d+)\s+(.+?)\s*$", re.MULTILINE)
DATE_HEADING_RE = re.compile(
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\s+"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+"
    r"(\d{1,2}),\s+(\d{4})",
    re.IGNORECASE,
)
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
STATIZ_LINK_RE = re.compile(
    r'href=["\']?(?:https?://statiz\.sporki\.com)?/player/\?m=view&s=(\d+)["\']?[^>]*>([^<]+)<',
    re.IGNORECASE,
)
LINEUP_NAME_RE = re.compile(
    r"(?:^|[\n>])\s*(?:\d+\s+|↳\s*)([A-Za-z][A-Za-z .'-]+?)(?:#\d+)?",
    re.MULTILINE,
)
FIELD_POSITIONS = {"p", "c", "1b", "2b", "3b", "ss", "lf", "cf", "rf", "dh", "ph", "pr"}

# Common watchlist typos / romanization variants -> MyKBO spellings.
KBO_WATCHLIST_ALIASES: dict[str, list[str]] = {
    "choi jeong-hee": ["choi jeong"],
    "choi jung-hee": ["choi jeong"],
    "lee jung-hoo": [],  # MLB (Giants) since 2024 — not in KBO box scores.
}


def _kbo_search_terms(watchlist_name: str) -> list[str]:
    key = watchlist_name.strip().lower()
    if not key:
        return []
    terms = [key]
    for alias in KBO_WATCHLIST_ALIASES.get(key, []):
        alias = alias.strip().lower()
        if alias and alias not in terms:
            terms.append(alias)
    return terms


def _matches_kbo_watchlist(watchlist_name: str, candidate: str) -> bool:
    return any(
        names_match(term, candidate, min_fuzzy=0.78) or _name_matches(term, candidate)
        for term in _kbo_search_terms(watchlist_name)
    )


def _resolve_kbo_watchlist_name(watchlist_names: set[str], row_player: str) -> str | None:
    lowered = row_player.strip().lower()
    if lowered in watchlist_names:
        return lowered
    for watch_name in watchlist_names:
        if _matches_kbo_watchlist(watch_name, row_player):
            return watch_name
    return None


def _pick_column(columns: list[str], aliases: set[str]) -> str | None:
    lowered = {c.lower().strip(): c for c in columns}
    for alias in aliases:
        if alias in lowered:
            return lowered[alias]
    return None


def normalize_kbo_export(raw: pd.DataFrame, default_player: str | None = None) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    cols = list(raw.columns)
    mapping = {c: _pick_column(cols, a) for c, a in KBO_COLUMN_ALIASES.items()}
    mapping = {k: v for k, v in mapping.items() if v}
    if "date" not in mapping:
        raise ValueError("KBO table needs a date column")

    rows: list[dict[str, Any]] = []
    for _, r in raw.iterrows():
        player_val = default_player
        if "player" in mapping:
            player_val = str(r[mapping["player"]])
        if not player_val:
            raise ValueError("KBO rows need player name")
        row: dict[str, Any] = {
            "date": pd.to_datetime(r[mapping["date"]], errors="coerce"),
            "game_title": GAME_TITLE,
            "player": str(player_val).strip().lower(),
            "team": str(r[mapping["team"]]).strip().lower() if "team" in mapping else "unknown",
            "opponent": str(r[mapping["opponent"]]).strip().lower() if "opponent" in mapping else "unknown",
            "minutes": 1,
            "games": 1,
        }
        row["plate_appearances"] = float(r[mapping["plate_appearances"]]) if "plate_appearances" in mapping else 4.0
        for stat in ("hits", "runs", "rbis", "strikeouts", "total_bases", "walks", "stolen_bases"):
            row[stat] = float(r[mapping[stat]] or 0) if stat in mapping else 0.0
        rows.append(row)
    return pd.DataFrame(rows).dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def load_kbo_history_csv(path: str | Path, default_player: str | None = None) -> pd.DataFrame:
    return normalize_kbo_export(pd.read_csv(path), default_player=default_player)


def _http_get(url: str, *, pause: float = 0.3) -> str:
    response = requests.get(url, timeout=45, headers=HEADERS)
    response.raise_for_status()
    time.sleep(pause)
    return response.text


def _mykbo_get(path: str) -> str:
    url = path if path.startswith("http") else urljoin(MYKBO_BASE, path)
    return _http_get(url)


def _parse_game_date(html: str, fallback: date) -> str:
    match = DATE_HEADING_RE.search(html)
    if match:
        month = MONTHS[match.group(1).lower()]
        return date(int(match.group(3)), month, int(match.group(2))).isoformat()
    return fallback.isoformat()


def _parse_teams(html: str) -> tuple[str, str] | None:
    match = TITLE_RE.search(html)
    if match:
        return match.group(1).strip().lower(), match.group(4).strip().lower()
    final = FINAL_SCORE_RE.search(html)
    if final:
        return final.group(1).strip().lower(), final.group(4).strip().lower()
    plain = re.search(
        r"(?:^|>)\s*([A-Za-z][A-Za-z .'-]+?)\s+(\d+)\s*:\s*(\d+)\s+([A-Za-z][A-Za-z .'-]+?)\s*(?:Final|<)",
        html,
        re.MULTILINE,
    )
    if plain:
        return plain.group(1).strip().lower(), plain.group(4).strip().lower()
    vs = re.search(
        r"(?:#\s*|<h1[^>]*>\s*)([A-Za-z][A-Za-z .'-]+?)\s+vs\.?\s+([A-Za-z][A-Za-z .'-]+)",
        html,
        re.IGNORECASE,
    )
    if vs:
        return vs.group(1).strip().lower(), vs.group(2).strip().lower()
    score_line = re.search(
        r"([A-Za-z][A-Za-z .'-]+?)\s+(\d+)\s*:\s*(\d+)\s+([A-Za-z][A-Za-z .'-]+)",
        html,
        re.IGNORECASE,
    )
    if score_line:
        return score_line.group(1).strip().lower(), score_line.group(4).strip().lower()
    title_m = re.search(r"<title>([^<]+)</title>", html, re.IGNORECASE)
    if title_m:
        title_text = title_m.group(1)
        score = re.search(
            r"([A-Za-z][A-Za-z .'-]+?)\s+(\d+)\s*:\s*(\d+)\s+([A-Za-z][A-Za-z .'-]+)",
            title_text,
            re.IGNORECASE,
        )
        if score:
            return score.group(1).strip().lower(), score.group(4).strip().lower()
        vs_t = re.search(
            r"([A-Za-z][A-Za-z .'-]+?)\s+vs\.?\s+([A-Za-z][A-Za-z .'-]+)",
            title_text,
            re.IGNORECASE,
        )
        if vs_t:
            away = vs_t.group(1).strip().lower()
            home = re.sub(r"\s*[-|].*$", "", vs_t.group(2)).strip().lower()
            return away, home
    headers = re.findall(r"####\s*#\d+\s+([A-Za-z][A-Za-z .'-]+)", html)
    if len(headers) >= 2:
        return headers[0].strip().lower(), headers[1].strip().lower()
    try:
        from bs4 import BeautifulSoup

        for tag in ("h1", "h2"):
            for node in BeautifulSoup(html, "lxml").find_all(tag):
                text = node.get_text(" ", strip=True)
                m = re.search(
                    r"([A-Za-z][A-Za-z .'-]+?)\s+(\d+)\s*:\s*(\d+)\s+([A-Za-z][A-Za-z .'-]+)",
                    text,
                    re.IGNORECASE,
                )
                if m:
                    return m.group(1).strip().lower(), m.group(4).strip().lower()
                m = re.search(
                    r"([A-Za-z][A-Za-z .'-]+?)\s+vs\.?\s+([A-Za-z][A-Za-z .'-]+)",
                    text,
                    re.IGNORECASE,
                )
                if m:
                    return m.group(1).strip().lower(), m.group(2).strip().lower()
    except Exception:
        pass
    return None


def _player_from_cell(cell: Any) -> str:
    text = str(cell).strip()
    if not text or text.lower() == "nan":
        return ""
    text = re.sub(r"#\d+.*$", "", text).strip()
    return re.sub(r"↳.*$", "", text).strip()


def _clean_mykbo_box_score_name(name: str) -> str:
    """MyKBO box scores often prefix lineup slot: '5 Díaz' -> 'diaz'."""
    text = normalize_lookup_name(name)
    text = re.sub(r"^\d+\s+", "", text)
    text = re.sub(r"^[.\s]+", "", text)
    return text.strip()


def _kbo_name_parts(name: str) -> list[str]:
    return [p for p in name.replace("-", " ").split() if p]


def _pp_matches_scraped_kbo(pp_name: str, scraped_name: str) -> bool:
    pp = normalize_lookup_name(pp_name)
    scraped = _clean_mykbo_box_score_name(scraped_name)
    if not pp or not scraped:
        return False
    if names_match(pp, scraped, min_fuzzy=0.72):
        return True
    pp_parts = _kbo_name_parts(pp)
    sc_parts = _kbo_name_parts(scraped)
    if pp_parts and sc_parts and pp_parts[-1] == sc_parts[-1]:
        return True
    if len(sc_parts) == 1 and pp_parts and sc_parts[0] == pp_parts[-1]:
        return True
    # Shortened box-score names: "hilliard" -> "sam hilliard"
    if len(sc_parts) == 1 and sc_parts[0] in pp_parts:
        return True
    return bool(fuzzy_best_match(pp, [scraped], min_score=0.55))


def _looks_like_position(val: str) -> bool:
    token = val.strip().lower()
    return token in FIELD_POSITIONS or (len(token) <= 3 and token.isalpha())


def _extract_batting_name_blocks(html: str, n_tables: int = 2) -> list[list[str]]:
    """MyKBO lists batter names above each box-score table (not in the table)."""
    section = html
    if re.search(r"batting", html, re.I):
        section = re.split(r"batting", html, maxsplit=1, flags=re.I)[1]
    if re.search(r"pitching", section, re.I):
        section = re.split(r"pitching", section, maxsplit=1, flags=re.I)[0]

    table_matches = list(re.finditer(r"<table\b", section, re.I))
    blocks: list[list[str]] = []
    for i, match in enumerate(table_matches[:n_tables]):
        start = table_matches[i - 1].end() if i > 0 else 0
        prefix_text = re.sub(r"<[^>]+>", "\n", section[start : match.start()])
        names: list[str] = []
        for m in LINEUP_NAME_RE.finditer(prefix_text):
            name = m.group(1).strip()
            if name.lower() in FIELD_POSITIONS:
                continue
            names.append(name)
        blocks.append(names)
    while len(blocks) < n_tables:
        blocks.append([])
    return blocks[:n_tables]


def _table_to_batters(
    table: pd.DataFrame,
    team: str,
    opponent: str,
    game_date: str,
    *,
    lineup_names: list[str] | None = None,
) -> list[dict]:
    if table.empty:
        return []
    cols = {str(c).strip().lower(): c for c in table.columns}
    if "ab" not in cols and "h" not in cols:
        return []
    rows: list[dict] = []
    name_col = table.columns[0]
    for i, (_, r) in enumerate(table.iterrows()):
        name = _player_from_cell(r[name_col])
        if lineup_names and (not name or _looks_like_position(name) or name.startswith(".")):
            if i < len(lineup_names):
                name = lineup_names[i]
        if not name or name.lower() in {"totals", "total", "team"}:
            continue
        if _looks_like_position(name):
            continue
        name = _clean_mykbo_box_score_name(name)
        if not name:
            continue
        ab = float(pd.to_numeric(r.get(cols.get("ab", ""), 0), errors="coerce") or 0)
        h = float(pd.to_numeric(r.get(cols.get("h", ""), 0), errors="coerce") or 0)
        bb = float(pd.to_numeric(r.get(cols.get("bb", ""), 0), errors="coerce") or 0)
        so = float(pd.to_numeric(r.get(cols.get("so", ""), 0), errors="coerce") or 0)
        pa = ab + bb + float(pd.to_numeric(r.get(cols.get("hbp", ""), 0), errors="coerce") or 0)
        batter = {
            "player": name,
            "ab": ab, "h": h, "bb": bb, "so": so, "pa": pa,
            "r": pd.to_numeric(r.get(cols.get("r", ""), 0), errors="coerce"),
            "rbi": pd.to_numeric(r.get(cols.get("rbi", ""), 0), errors="coerce"),
            "hr": pd.to_numeric(r.get(cols.get("hr", ""), 0), errors="coerce"),
        }
        row = _batter_row_from_box_score(batter, game_date, team, opponent)
        if row:
            rows.append(row)
    return rows


def _is_game_box_score_table(table: pd.DataFrame) -> bool:
    if table.empty:
        return False
    cols = {str(c).strip().lower() for c in table.columns}
    return "ab" in cols and bool(cols & {"h", "rbi", "r"})


def _parse_batting_tables(html: str, away: str, home: str, game_date: str) -> list[dict]:
    tables = _statiz_tables_from_html(html)
    batting = [t for t in tables if _is_game_box_score_table(t)]
    if not batting:
        return []
    name_blocks = _extract_batting_name_blocks(html, n_tables=max(len(batting), 2))
    rows: list[dict] = []
    if len(batting) == 1:
        rows.extend(
            _table_to_batters(
                batting[0],
                away,
                home,
                game_date,
                lineup_names=name_blocks[0] if name_blocks else None,
            )
        )
        return rows
    away_names = name_blocks[0] if name_blocks else []
    home_names = name_blocks[1] if len(name_blocks) > 1 else []
    rows.extend(_table_to_batters(batting[0], away, home, game_date, lineup_names=away_names))
    rows.extend(_table_to_batters(batting[1], home, away, game_date, lineup_names=home_names))
    for idx, table in enumerate(batting[2:], start=2):
        lineup = name_blocks[idx] if idx < len(name_blocks) else None
        rows.extend(_table_to_batters(table, away, home, game_date, lineup_names=lineup))
    return rows


def _is_pitching_box_score_table(table: pd.DataFrame) -> bool:
    if table.empty:
        return False
    cols = {str(c).strip().lower() for c in table.columns}
    if "ab" in cols:
        return False
    return "ip" in cols and bool(cols & {"so", "k", "h", "bb", "er", "r"})


def _extract_pitching_name_blocks(html: str, n_tables: int = 2) -> list[list[str]]:
    """Pitcher names appear above each pitching box-score table."""
    if not re.search(r"pitching", html, re.I):
        return [[] for _ in range(n_tables)]
    section = re.split(r"pitching", html, maxsplit=1, flags=re.I)[1]
    table_matches = list(re.finditer(r"<table\b", section, re.I))
    blocks: list[list[str]] = []
    for i, match in enumerate(table_matches[:n_tables]):
        start = table_matches[i - 1].end() if i > 0 else 0
        prefix_text = re.sub(r"<[^>]+>", "\n", section[start : match.start()])
        names: list[str] = []
        for m in LINEUP_NAME_RE.finditer(prefix_text):
            name = m.group(1).strip()
            if name.lower() in FIELD_POSITIONS:
                continue
            names.append(name)
        blocks.append(names)
    while len(blocks) < n_tables:
        blocks.append([])
    return blocks[:n_tables]


def _table_to_pitchers(
    table: pd.DataFrame,
    team: str,
    opponent: str,
    game_date: str,
    *,
    lineup_names: list[str] | None = None,
) -> list[dict]:
    if table.empty:
        return []
    cols = {str(c).strip().lower(): c for c in table.columns}
    if "ip" not in cols:
        return []
    rows: list[dict] = []
    name_col = table.columns[0]
    for i, (_, r) in enumerate(table.iterrows()):
        name = _player_from_cell(r[name_col])
        if lineup_names and (not name or _looks_like_position(name) or name.startswith(".")):
            if i < len(lineup_names):
                name = lineup_names[i]
        if not name or name.lower() in {"totals", "total", "team"}:
            continue
        if _looks_like_position(name):
            continue
        name = _clean_mykbo_box_score_name(name)
        if not name:
            continue
        pitcher = {
            "player": name,
            "ip": r.get(cols.get("ip", ""), 0),
            "h": pd.to_numeric(r.get(cols.get("h", ""), 0), errors="coerce"),
            "r": pd.to_numeric(r.get(cols.get("r", ""), 0), errors="coerce"),
            "er": pd.to_numeric(r.get(cols.get("er", ""), 0), errors="coerce"),
            "bb": pd.to_numeric(r.get(cols.get("bb", ""), 0), errors="coerce"),
            "so": pd.to_numeric(
                r.get(cols.get("so", cols.get("k", "")), 0),
                errors="coerce",
            ),
        }
        row = _pitcher_row_from_box_score(pitcher, game_date, team, opponent)
        if row:
            rows.append(row)
    return rows


def _parse_pitching_tables(html: str, away: str, home: str, game_date: str) -> list[dict]:
    tables = _statiz_tables_from_html(html)
    pitching = [t for t in tables if _is_pitching_box_score_table(t)]
    if not pitching:
        return []
    name_blocks = _extract_pitching_name_blocks(html, n_tables=max(len(pitching), 2))
    rows: list[dict] = []
    if len(pitching) == 1:
        rows.extend(
            _table_to_pitchers(
                pitching[0],
                away,
                home,
                game_date,
                lineup_names=name_blocks[0] if name_blocks else None,
            )
        )
        return rows
    away_names = name_blocks[0] if name_blocks else []
    home_names = name_blocks[1] if len(name_blocks) > 1 else []
    rows.extend(_table_to_pitchers(pitching[0], away, home, game_date, lineup_names=away_names))
    rows.extend(_table_to_pitchers(pitching[1], home, away, game_date, lineup_names=home_names))
    for idx, table in enumerate(pitching[2:], start=2):
        lineup = name_blocks[idx] if idx < len(name_blocks) else None
        rows.extend(_table_to_pitchers(table, away, home, game_date, lineup_names=lineup))
    return rows


def fetch_mykbo_player_page_game_log(
    player_name: str,
    mykbo_player_id: str,
    *,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Fallback: scrape per-game log table from a MyKBO player profile page."""
    pid = str(mykbo_player_id).strip()
    if not pid:
        return pd.DataFrame()
    html = _mykbo_get(f"/players/{pid}")
    tables = pd.read_html(io.StringIO(html))
    game_table: pd.DataFrame | None = None
    for table in tables:
        cols = {str(c).strip().lower() for c in table.columns}
        if cols & {"date", "ab", "h"} == {"date", "ab", "h"} or ("date" in cols and "ab" in cols):
            game_table = table
            break
    if game_table is None or game_table.empty:
        return pd.DataFrame()

    colmap = {str(c).strip().lower(): c for c in game_table.columns}
    cutoff = date.today() - timedelta(days=lookback_days)
    rows: list[dict[str, Any]] = []
    for _, r in game_table.iterrows():
        raw_date = r.get(colmap.get("date", ""), "")
        parsed = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(parsed):
            continue
        gdate = parsed.date()
        if gdate < cutoff:
            continue
        opp_raw = str(r.get(colmap.get("opp", colmap.get("opponent", "")), "unknown"))
        opp = re.sub(r"^vs\.|^@+", "", opp_raw, flags=re.I).strip().lower() or "unknown"
        batter = {
            "player": player_name,
            "ab": r.get(colmap.get("ab", ""), 0),
            "h": r.get(colmap.get("h", ""), 0),
            "bb": r.get(colmap.get("bb", ""), 0),
            "so": r.get(colmap.get("so", colmap.get("k", "")), 0),
            "r": r.get(colmap.get("r", ""), 0),
            "rbi": r.get(colmap.get("rbi", ""), 0),
            "hr": r.get(colmap.get("hr", ""), 0),
            "2b": r.get(colmap.get("2b", ""), 0),
            "3b": r.get(colmap.get("3b", ""), 0),
        }
        row = _batter_row_from_box_score(batter, gdate.isoformat(), "unknown", opp)
        if row:
            row["player"] = player_name.strip().lower()
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).dropna(subset=["date"]).drop_duplicates(
        subset=["date", "player", "team", "opponent"], keep="last"
    )
    return out.sort_values("date").reset_index(drop=True)


def kbo_season_date_window(
    season_years: tuple[int, ...] = KBO_DEFAULT_SEASON_YEARS,
) -> tuple[date, date]:
    """Inclusive date range covering KBO regular seasons (opens ~late March)."""
    years = tuple(sorted({int(y) for y in season_years if int(y) > 0}))
    if not years:
        years = KBO_DEFAULT_SEASON_YEARS
    return date(years[0], 3, 1), date.today()


def kbo_pitcher_window_since_october(
    today: date | None = None,
    *,
    since_month: int = 10,
    since_day: int = 1,
) -> tuple[date, date]:
    """Oct 1 → yesterday (playoffs + current season through latest completed games)."""
    anchor = today or date.today()
    end = anchor - timedelta(days=1)
    start_year = anchor.year if anchor.month >= since_month else anchor.year - 1
    start = date(start_year, since_month, since_day)
    if start > end:
        start = date(end.year, since_month, since_day)
        if start > end:
            start = end
    return start, end


def load_cached_kbo_game_ids(root: Path | None) -> list[str]:
    if not root:
        return []
    path = root / "data" / "cache" / "sync_state.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [str(g).strip() for g in data.get("kbo_game_ids", []) if str(g).strip()]
    except Exception:
        return []


def _is_pitching_game_log_table(table: pd.DataFrame) -> bool:
    if table.empty:
        return False
    cols = {str(c).strip().lower() for c in table.columns}
    if "ab" in cols:
        return False
    return "date" in cols and ("ip" in cols or "inn" in cols)


def _pitching_log_from_game_table(
    game_table: pd.DataFrame,
    player_name: str,
    *,
    start: date,
    end: date,
) -> pd.DataFrame:
    if game_table is None or game_table.empty:
        return pd.DataFrame()
    colmap = {str(c).strip().lower(): c for c in game_table.columns}
    team_col = colmap.get("team") or colmap.get("tm")
    rows: list[dict[str, Any]] = []
    for _, r in game_table.iterrows():
        raw_date = r.get(colmap.get("date", ""), "")
        parsed = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(parsed):
            continue
        gdate = parsed.date()
        if gdate < start or gdate > end:
            continue
        opp_raw = str(r.get(colmap.get("opp", colmap.get("opponent", "")), "unknown"))
        opp = re.sub(r"^vs\.|^@+", "", opp_raw, flags=re.I).strip().lower() or "unknown"
        team_raw = str(r.get(team_col, "unknown")) if team_col else "unknown"
        team = re.sub(r"^vs\.|^@+", "", team_raw, flags=re.I).strip().lower() or "unknown"
        pitcher = {
            "player": player_name,
            "ip": r.get(colmap.get("ip", colmap.get("inn", "")), 0),
            "h": r.get(colmap.get("h", ""), 0),
            "bb": r.get(colmap.get("bb", ""), 0),
            "so": r.get(colmap.get("so", colmap.get("k", "")), 0),
            "r": r.get(colmap.get("r", ""), 0),
            "er": r.get(colmap.get("er", ""), 0),
        }
        row = _pitcher_row_from_box_score(pitcher, gdate.isoformat(), team, opp)
        if row:
            row["player"] = normalize_lookup_name(player_name)
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows).dropna(subset=["date"]).drop_duplicates(
        subset=["date", "player", "team", "opponent"], keep="last"
    )
    return out.sort_values("date").reset_index(drop=True)


def fetch_kbo_statiz_pitching_log(
    statiz_player_id: str,
    player_name: str,
    *,
    season_years: tuple[int, ...] = KBO_DEFAULT_SEASON_YEARS,
) -> pd.DataFrame:
    """Per-game pitching log from a Statiz player profile."""
    pid = _clean_player_id(statiz_player_id)
    if not pid:
        return pd.DataFrame()
    win_start, win_end = kbo_season_date_window(season_years)
    end = min(win_end, date.today() - timedelta(days=1))
    last_err: Exception | None = None
    for base in (STATIZ_BASE, STATIZ_ALT):
        try:
            html = _http_get(f"{base}?m=view&s={pid}")
            tables = _statiz_tables_from_html(html)
            game_table: pd.DataFrame | None = None
            for table in tables:
                if _is_pitching_game_log_table(table):
                    game_table = table
                    break
            if game_table is None or game_table.empty:
                raise ValueError("No pitching game log table")
            return _pitching_log_from_game_table(
                game_table,
                player_name,
                start=win_start,
                end=end,
            )
        except Exception as exc:
            last_err = exc
    raise ValueError(f"Statiz pitching log failed for id={pid}: {last_err}")


def fetch_mykbo_player_page_pitching_log(
    player_name: str,
    mykbo_player_id: str,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    season_years: tuple[int, ...] = KBO_DEFAULT_SEASON_YEARS,
) -> pd.DataFrame:
    """Per-game pitching log from a MyKBO player profile (2025+ seasons on one page)."""
    pid = str(mykbo_player_id).strip()
    if not pid:
        return pd.DataFrame()
    start, end = start_date, end_date
    if start is None or end is None:
        win_start, win_end = kbo_season_date_window(season_years)
        start = start or win_start
        end = end or win_end

    html = _mykbo_get(f"/players/{pid}")
    tables = pd.read_html(io.StringIO(html))
    game_table: pd.DataFrame | None = None
    for table in tables:
        if _is_pitching_game_log_table(table):
            game_table = table
            break
    if game_table is None or game_table.empty:
        return pd.DataFrame()
    return _pitching_log_from_game_table(game_table, player_name, start=start, end=end)


def fetch_mykbo_player_pitching_log(
    player_name: str,
    *,
    mykbo_player_id: str | None = None,
    season_years: tuple[int, ...] = KBO_DEFAULT_SEASON_YEARS,
    start_date: date | None = None,
    end_date: date | None = None,
    api_key: str | None = None,
) -> pd.DataFrame:
    """Pitching game logs for one player (MyKBO page scrape first, Parse fallback)."""
    if start_date is not None and end_date is not None:
        start, end = start_date, end_date
    else:
        start, end = kbo_pitcher_window_since_october()
    last_err: Exception | None = None
    try:
        pid = str(mykbo_player_id or resolve_mykbo_player_id_html(player_name)).strip()
        log = fetch_mykbo_player_page_pitching_log(
            player_name,
            pid,
            start_date=start,
            end_date=end,
            season_years=season_years,
        )
        if not log.empty:
            return log
    except Exception as exc:
        last_err = exc

    if last_err:
        raise ValueError(
            f"MyKBO pitching log empty for {player_name!r} ({start}..{end}): {last_err}"
        )
    return pd.DataFrame()


def _extract_mykbo_game_ids(html: str, default_date: str, seen: set[str]) -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for gid in GAME_LINK_RE.findall(html):
        if gid in seen:
            continue
        seen.add(gid)
        found.append((gid, default_date))
    return found


def _seed_game_ids_from_schedule_pages() -> list[int]:
    """Highest /games/{id} links visible on MyKBO schedule (usually ~1 week)."""
    seeds: list[int] = []
    for path in ("/schedule", "/"):
        try:
            html = _mykbo_get(path)
        except Exception:
            continue
        for gid in GAME_LINK_RE.findall(html):
            if str(gid).isdigit():
                seeds.append(int(gid))
    return sorted(set(seeds))


def list_mykbo_final_game_ids_by_scan(
    start: date,
    end: date,
    *,
    max_games: int = 500,
    extra_seeds: list[int] | None = None,
) -> list[tuple[str, str]]:
    """Walk sequential /games/{id} downward — MyKBO schedule HTML lacks deep history."""
    seeds = _seed_game_ids_from_schedule_pages()
    if extra_seeds:
        seeds = sorted(set(seeds + [int(s) for s in extra_seeds if str(s).isdigit()]))
    if not seeds:
        return []
    high = max(seeds)
    low_seed = min(seeds)
    low = max(1, min(low_seed, high) - max(1, max_games))
    out: list[tuple[str, str]] = []
    below_start_streak = 0
    for gid in range(high, low - 1, -1):
        try:
            html = _mykbo_get(f"/games/{gid}")
        except Exception:
            continue
        if not FINAL_RE.search(html):
            continue
        gdate = _parse_game_date(html, date.today())
        try:
            gd = date.fromisoformat(gdate)
        except ValueError:
            continue
        if gd > end:
            continue
        if gd < start:
            below_start_streak += 1
            if below_start_streak >= 36:
                break
            continue
        below_start_streak = 0
        out.append((str(gid), gdate))
    return out


def list_mykbo_final_game_ids(
    start: date,
    end: date,
    *,
    require_batting: bool = False,
    extra_game_ids: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Collect /games/{id} links from schedule pages (no player IDs required)."""
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    pages: list[tuple[str, str]] = []
    cursor = start
    while cursor <= end:
        iso = cursor.isoformat()
        pages.append((f"/schedule/{iso}", iso))
        pages.append((f"/schedule/{cursor.year}/{cursor.month:02d}/{cursor.day:02d}", iso))
        cursor += timedelta(days=1)
    month_cursor = date(start.year, start.month, 1)
    while month_cursor <= end:
        pages.append((f"/schedule/{month_cursor.year}/{month_cursor.month:02d}", start.isoformat()))
        if month_cursor.month == 12:
            month_cursor = date(month_cursor.year + 1, 1, 1)
        else:
            month_cursor = date(month_cursor.year, month_cursor.month + 1, 1)
    pages.extend([("/", end.isoformat()), ("/schedule", end.isoformat())])

    for path, default_date in pages:
        try:
            html = _mykbo_get(path)
        except Exception:
            continue
        out.extend(_extract_mykbo_game_ids(html, default_date, seen))

    if not require_batting:
        return out

    verified: list[tuple[str, str]] = []
    checked: set[str] = set()
    for gid, fallback in out:
        if gid in checked:
            continue
        checked.add(gid)
        try:
            html = _mykbo_get(f"/games/{gid}")
        except Exception:
            continue
        if not FINAL_RE.search(html):
            continue
        gdate = _parse_game_date(html, date.fromisoformat(fallback))
        verified.append((gid, gdate))

    extra_ints = [int(g) for g in (extra_game_ids or []) if str(g).isdigit()]
    scan_budget = max(800, (end - start).days * 8 + 80)
    scanned = list_mykbo_final_game_ids_by_scan(
        start, end, max_games=scan_budget, extra_seeds=extra_ints
    )
    merged: dict[str, str] = {gid: gdate for gid, gdate in verified}
    for gid, gdate in scanned:
        merged.setdefault(gid, gdate)
    for gid in extra_game_ids or []:
        gid = str(gid).strip()
        if not gid or gid in merged:
            continue
        try:
            html = _mykbo_get(f"/games/{gid}")
        except Exception:
            continue
        if not FINAL_RE.search(html):
            continue
        gdate = _parse_game_date(html, date.today())
        try:
            gd = date.fromisoformat(gdate)
        except ValueError:
            continue
        if start <= gd <= end:
            merged[gid] = gdate
    return sorted(merged.items(), key=lambda item: item[1])


def fetch_mykbo_game_batting_rows(game_id: str, *, game_date: date | None = None) -> list[dict]:
    html = _mykbo_get(f"/games/{game_id}")
    teams = _parse_teams(html)
    away, home = teams if teams else ("away", "home")
    gdate = _parse_game_date(html, game_date or date.today())
    return _parse_batting_tables(html, away, home, gdate)


def fetch_mykbo_game_pitching_rows(game_id: str, *, game_date: date | None = None) -> list[dict]:
    html = _mykbo_get(f"/games/{game_id}")
    teams = _parse_teams(html)
    away, home = teams if teams else ("away", "home")
    gdate = _parse_game_date(html, game_date or date.today())
    return _parse_pitching_tables(html, away, home, gdate)


def fetch_kbo_scrape_daily_box_scores(
    watchlist_names: set[str] | list[str],
    *,
    lookback_days: int = 3,
    fetched_game_ids: set[str] | None = None,
    role: str = "hitter",
) -> tuple[pd.DataFrame, set[str]]:
    names = {n.strip().lower() for n in watchlist_names if str(n).strip()}
    if not names:
        return pd.DataFrame(), fetched_game_ids or set()
    already = set(fetched_game_ids or set())
    end = date.today()
    start = end - timedelta(days=max(lookback_days, 1))
    rows: list[dict] = []
    used_ids: set[str] = set(already)
    fetch_rows = (
        fetch_mykbo_game_pitching_rows
        if str(role or "hitter").strip().lower() == "pitcher"
        else fetch_mykbo_game_batting_rows
    )
    for gid, gdate in list_mykbo_final_game_ids(start, end):
        if gid in already:
            continue
        try:
            game_rows = fetch_rows(gid, game_date=date.fromisoformat(gdate))
        except Exception:
            continue
        used_ids.add(gid)
        for row in game_rows:
            watch_name = _resolve_kbo_watchlist_name(names, row["player"])
            if watch_name:
                if row["player"].strip().lower() != watch_name:
                    row = dict(row)
                    row["player"] = watch_name
                rows.append(row)
    if not rows:
        return pd.DataFrame(), used_ids
    out = pd.DataFrame(rows).dropna(subset=["date"]).drop_duplicates(
        subset=["date", "player", "team", "opponent"], keep="last"
    )
    return out.sort_values("date").reset_index(drop=True), used_ids


def _scrape_mykbo_batting_rows(lookback_days: int) -> list[dict]:
    # Completed games only; today's slate usually has lineups but no AB/H/RBI yet.
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=max(lookback_days - 1, 1))
    rows: list[dict] = []
    for gid, gdate in list_mykbo_final_game_ids(start, end, require_batting=True):
        try:
            rows.extend(fetch_mykbo_game_batting_rows(gid, game_date=date.fromisoformat(gdate)))
        except Exception:
            continue
    return rows


def _scrape_mykbo_pitching_rows(
    *,
    lookback_days: int | None = None,
    season_years: tuple[int, ...] | None = KBO_DEFAULT_SEASON_YEARS,
    extra_game_ids: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    on_game_progress: Callable[[int, int, str], None] | None = None,
) -> list[dict]:
    if start_date is not None and end_date is not None:
        start, end = start_date, end_date
    elif season_years:
        start, end = kbo_season_date_window(season_years)
        end = min(end, date.today() - timedelta(days=1))
    else:
        end = date.today() - timedelta(days=1)
        start = end - timedelta(days=max((lookback_days or 120) - 1, 1))
    rows: list[dict] = []
    game_list = list_mykbo_final_game_ids(
        start,
        end,
        require_batting=True,
        extra_game_ids=extra_game_ids,
    )
    total_games = len(game_list)
    for idx, (gid, gdate) in enumerate(game_list, start=1):
        if on_game_progress:
            on_game_progress(idx, total_games, gid)
        try:
            gd = date.fromisoformat(gdate) if gdate else None
            rows.extend(fetch_mykbo_game_pitching_rows(gid, game_date=gd))
        except Exception:
            continue
    return rows


def scrape_all_mykbo_pitching_logs(
    *,
    lookback_days: int | None = None,
    season_years: tuple[int, ...] | None = KBO_DEFAULT_SEASON_YEARS,
    extra_game_ids: list[str] | None = None,
    start_date: date | None = None,
    end_date: date | None = None,
    on_game_progress: Callable[[int, int, str], None] | None = None,
) -> pd.DataFrame:
    """All pitching lines from final MyKBO games (bulk box scores; shallow on old dates)."""
    raw_rows = _scrape_mykbo_pitching_rows(
        lookback_days=lookback_days,
        season_years=season_years,
        extra_game_ids=extra_game_ids,
        start_date=start_date,
        end_date=end_date,
        on_game_progress=on_game_progress,
    )
    if not raw_rows:
        return pd.DataFrame()
    out = pd.DataFrame(raw_rows).dropna(subset=["date"]).drop_duplicates(
        subset=["date", "player", "team", "opponent"], keep="last"
    )
    return out.sort_values(["date", "player"]).reset_index(drop=True)


def _map_pp_names_to_scraped(
    targets: set[str],
    scraped_names: list[str],
) -> dict[str, str]:
    """Map scraped box-score names -> PrizePicks canonical name."""
    mapping: dict[str, str] = {}
    scraped_unique = sorted({_clean_mykbo_box_score_name(n) for n in scraped_names if n})
    scraped_unique = [s for s in scraped_unique if s]
    for target in targets:
        if not target:
            continue
        ranked = fuzzy_best_match(target, scraped_unique, min_score=0.72)
        if ranked:
            mapping[ranked[0][0]] = target
            continue
        for scraped in scraped_unique:
            if _pp_matches_scraped_kbo(target, scraped):
                mapping[scraped] = target
                break
    return mapping


def sync_kbo_players_via_mykbo_scrape(
    players: list[str],
    *,
    lookback_days: int = 120,
    role: str = "hitter",
) -> tuple[pd.DataFrame, list[str]]:
    """Scrape MyKBO box scores; returns (logs, players still missing).

    role: hitter (batting tables) or pitcher (pitching tables).
    """
    names = {normalize_lookup_name(p) for p in players if str(p).strip()}
    if not names:
        return pd.DataFrame(), []

    role_l = str(role or "hitter").strip().lower()
    if role_l == "pitcher":
        raw_rows = _scrape_mykbo_pitching_rows(lookback_days=lookback_days, season_years=None)
    else:
        raw_rows = _scrape_mykbo_batting_rows(lookback_days)
    if not raw_rows:
        return pd.DataFrame(), sorted(names)

    rows: list[dict] = []
    for row in raw_rows:
        scraped = _clean_mykbo_box_score_name(str(row.get("player", "")))
        if not scraped:
            continue
        pp_name = _resolve_kbo_watchlist_name(names, scraped)
        if not pp_name:
            for target in names:
                if _pp_matches_scraped_kbo(target, scraped):
                    pp_name = target
                    break
        if not pp_name:
            continue
        out_row = dict(row)
        out_row["player"] = pp_name
        rows.append(out_row)

    found = {r["player"].strip().lower() for r in rows}
    missing = sorted(n for n in names if n not in found)
    if not rows:
        return pd.DataFrame(), missing
    out = pd.DataFrame(rows).dropna(subset=["date"]).drop_duplicates(
        subset=["date", "player", "team", "opponent"], keep="last"
    )
    return out.sort_values(["player", "date"]).reset_index(drop=True), missing


def _clean_player_id(val: str | None) -> str:
    s = str(val or "").strip()
    if s.endswith(".0") and s[:-2].isdigit():
        return s[:-2]
    return s


def _statiz_search_queries(player_name: str) -> list[str]:
    """PP romanization variants for Statiz search."""
    base = normalize_lookup_name(player_name)
    queries: list[str] = []

    def add(q: str) -> None:
        q = normalize_lookup_name(q)
        if q and q not in queries:
            queries.append(q)

    add(base)
    add(base.replace("-", " "))
    add(base.replace("-", ""))
    parts = base.replace("-", " ").split()
    if len(parts) >= 2:
        add(parts[-1])
        add(parts[0])
        add(f"{parts[-1]} {parts[0]}")
        if len(parts) >= 3:
            add(" ".join(parts[1:]))
    if base.startswith("koo "):
        add(base.replace("koo", "ku", 1))
    return queries


def _parse_statiz_search_html(html: str) -> list[dict[str, str]]:
    seen: set[str] = set()
    out: list[dict[str, str]] = []

    def add(pid: str, name: str) -> None:
        pid = _clean_player_id(pid)
        name = re.sub(r"\s+", " ", str(name or "")).strip()
        if not pid or not name or pid in seen:
            return
        seen.add(pid)
        out.append({"id": pid, "name": name})

    for pid, name in STATIZ_LINK_RE.findall(html):
        add(pid, name)
    for m in re.finditer(
        r'href=["\'][^"\']*(?:player/\?m=view&s=|player/)(\d+)["\'][^>]*>([^<]+)<',
        html,
        re.IGNORECASE,
    ):
        add(m.group(1), m.group(2))

    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        for a in soup.find_all("a", href=True):
            href = str(a["href"])
            m = re.search(r"[?&]s=(\d+)|/player/(\d+)", href)
            if m:
                add(m.group(1) or m.group(2), a.get_text(" ", strip=True))
    except Exception:
        pass
    return out


def search_statiz_players(query: str) -> list[dict[str, str]]:
    q = quote_plus(query.strip())
    urls = [
        f"https://statiz.sporki.com/search/?m=player&s={q}",
        f"https://www.statiz.co.kr/player/?m=search&s={q}",
        f"{STATIZ_BASE}?m=search&s={q}",
        f"{STATIZ_ALT}?m=search&s={q}",
    ]
    for url in urls:
        try:
            html = _http_get(url, pause=0.2)
            matches = _parse_statiz_search_html(html)
            if matches:
                return matches
        except Exception:
            continue
    return []


def _mykbo_title_to_name(title: str) -> str:
    """'Takeda Shota (타케다) - SP - #23' -> 'Takeda Shota'."""
    text = re.sub(r"\s+", " ", str(title or "").strip())
    if " (" in text:
        text = text.split(" (", 1)[0].strip()
    if " - " in text:
        text = text.split(" - ", 1)[0].strip()
    return text


def search_mykbo_players_json(query: str) -> list[dict[str, str]]:
    """MyKBO native JSON search: GET /players/search?q=…"""
    q = quote_plus(query.strip())
    if not q:
        return []
    url = f"{MYKBO_BASE}/players/search?q={q}"
    session = requests.Session()
    session.get(f"{MYKBO_BASE}/", timeout=30, headers={**HEADERS, "Accept": "text/html,*/*"})
    response = session.get(
        url,
        timeout=30,
        headers={
            **HEADERS,
            "Accept": "application/json,text/html,*/*",
            "Referer": f"{MYKBO_BASE}/players",
            "X-Requested-With": "XMLHttpRequest",
        },
    )
    response.raise_for_status()
    time.sleep(0.35)
    payload = response.json()
    if not isinstance(payload, dict):
        return []

    seen: set[str] = set()
    out: list[dict[str, str]] = []

    def add(pid: str, name: str) -> None:
        pid = _clean_player_id(pid)
        name = _mykbo_title_to_name(name)
        if not pid or not name or pid in seen:
            return
        seen.add(pid)
        out.append({"id": pid, "name": name})

    teams = payload.get("results")
    if isinstance(teams, dict):
        for block in teams.values():
            if not isinstance(block, dict):
                continue
            for item in block.get("results") or []:
                if not isinstance(item, dict):
                    continue
                add(str(item.get("id", "")), str(item.get("title", "")))
    return out


def search_mykbo_players_html(query: str) -> list[dict[str, str]]:
    """Find MyKBO /players/{id} via JSON search and HTML fallbacks (no Parse)."""
    from sports_prop_edge.integrations.mykbo_scraper.search import search_players

    try:
        matches = search_players(query)
        if matches:
            return matches
    except Exception:
        pass

    seen: set[str] = set()
    out: list[dict[str, str]] = []
    q = quote_plus(query.strip())
    paths = [
        f"/players/search?q={q}",
        f"/players?search={q}",
        f"/players?query={q}",
        f"/player-search?query={q}",
        f"/search?query={q}",
    ]

    def add(pid: str, name: str) -> None:
        pid = _clean_player_id(pid)
        name = re.sub(r"\s+", " ", str(name or "")).strip()
        if not pid or not name or pid in seen:
            return
        seen.add(pid)
        out.append({"id": pid, "name": name})

    for path in paths:
        try:
            if path.startswith("/players/search?"):
                try:
                    from sports_prop_edge.integrations.mykbo_scraper.search import search_players

                    matches = search_players(query)
                    if matches:
                        return matches
                except Exception:
                    pass
                continue
            html = _mykbo_get(path)
        except Exception:
            continue
        for pid in MYKBO_PLAYER_LINK_RE.findall(html):
            add(pid, query)
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")
            for a in soup.select('a[href*="/players/"]'):
                href = str(a.get("href", ""))
                m = re.search(r"/players/(\d+)", href)
                if m:
                    add(m.group(1), a.get_text(" ", strip=True))
        except Exception:
            pass
        if out:
            return out
    return out


def resolve_mykbo_player_id_html(player_name: str) -> str:
    matches = search_mykbo_players_html(player_name)
    if not matches:
        raise ValueError(f"MyKBO HTML: no player found for {player_name!r}")
    names = [m["name"] for m in matches]
    ranked = fuzzy_best_match(player_name, names, min_score=0.75)
    if ranked:
        pick = ranked[0][0]
        for m in matches:
            if m["name"] == pick:
                return m["id"]
    for m in matches:
        if names_match(player_name, m["name"], min_fuzzy=0.75):
            return m["id"]
    return matches[0]["id"]


def search_statiz_players_fuzzy(player_name: str) -> list[dict[str, str]]:
    """Try several romanization queries; merge unique Statiz hits."""
    seen: set[str] = set()
    merged: list[dict[str, str]] = []
    for query in _statiz_search_queries(player_name):
        for match in search_statiz_players(query):
            pid = _clean_player_id(match["id"])
            if not pid or pid in seen:
                continue
            seen.add(pid)
            merged.append({**match, "id": pid})
        if merged:
            break
    return merged


def resolve_statiz_player_id(player_name: str, statiz_player_id: str | None = None, *, id_cache: dict[str, str] | None = None) -> str:
    cache = id_cache if id_cache is not None else {}
    key = normalize_lookup_name(player_name)
    clean_id = _clean_player_id(statiz_player_id)
    if clean_id:
        cache[key] = clean_id
        return clean_id
    if key in cache:
        return cache[key]
    matches = search_statiz_players_fuzzy(player_name)
    if not matches:
        raise ValueError(f"Statiz: no player found for {player_name!r}")
    names = [m["name"] for m in matches]
    ranked = fuzzy_best_match(player_name, names, min_score=0.78)
    if ranked:
        pick_name = ranked[0][0]
        for match in matches:
            if match["name"] == pick_name:
                cache[key] = match["id"]
                return match["id"]
    for match in matches:
        if names_match(player_name, match["name"], min_fuzzy=0.78):
            cache[key] = match["id"]
            return match["id"]
    cache[key] = matches[0]["id"]
    return matches[0]["id"]


def _statiz_tables_from_html(html: str) -> list[pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        tables = []
    if tables:
        return tables
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "lxml")
        for table in soup.find_all("table"):
            try:
                parsed = pd.read_html(io.StringIO(str(table)))[0]
                tables.append(parsed)
            except Exception:
                continue
    except Exception:
        pass
    return tables


def fetch_kbo_statiz_game_log(statiz_player_id: str, player_name: str) -> pd.DataFrame:
    last_err: Exception | None = None
    pid = _clean_player_id(statiz_player_id)
    for base in (STATIZ_BASE, STATIZ_ALT):
        try:
            html = _http_get(f"{base}?m=view&s={pid}")
            tables = _statiz_tables_from_html(html)
            if not tables:
                raise ValueError("No tables found")
            game_log = tables[-1]
            for table in tables:
                cols = {str(c).lower() for c in table.columns}
                if cols & KBO_COLUMN_ALIASES["date"] and cols & KBO_COLUMN_ALIASES["hits"]:
                    game_log = table
                    break
            return normalize_kbo_export(game_log, default_player=player_name)
        except Exception as exc:
            last_err = exc
    raise ValueError(f"Statiz fetch failed for id={pid}: {last_err}")


def sync_kbo_players_via_mykbo_pages(
    players: list[str],
    *,
    errors: list[str] | None = None,
    pause_seconds: float = 0.35,
) -> pd.DataFrame:
    """Resolve MyKBO player pages and scrape per-game tables."""
    frames: list[pd.DataFrame] = []
    for name in players:
        try:
            pid = resolve_mykbo_player_id_html(name)
            log = fetch_mykbo_player_page_game_log(name, pid, lookback_days=120)
            if not log.empty:
                frames.append(log)
        except Exception as exc:
            if errors is not None:
                errors.append(f"KBO MyKBO page {name!r}: {exc}")
        time.sleep(pause_seconds)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sync_kbo_players_via_parse_api(
    players: list[str],
    *,
    errors: list[str] | None = None,
) -> pd.DataFrame:
    if not os.getenv("PARSE_API_KEY"):
        return pd.DataFrame()
    from sports_prop_edge.integrations.mykbo_client import fetch_mykbo_player_game_log

    frames: list[pd.DataFrame] = []
    for name in players:
        try:
            log = fetch_mykbo_player_game_log(name)
            if not log.empty:
                frames.append(log)
        except Exception as exc:
            if errors is not None:
                errors.append(f"KBO Parse API {name!r}: {exc}")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def sync_kbo_players_via_statiz(
    players: list[str],
    *,
    watchlist: pd.DataFrame | None = None,
    id_cache: dict[str, str] | None = None,
    pause_seconds: float = 0.35,
    errors: list[str] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cache = id_cache if id_cache is not None else {}
    wl = watchlist if watchlist is not None else pd.DataFrame()
    for name in players:
        try:
            statiz_id = None
            if not wl.empty and "statiz_player_id" in wl.columns:
                rows = wl[(wl["sport"] == "KBO") & (wl["player"] == name.strip().lower())]
                if not rows.empty:
                    statiz_id = _clean_player_id(rows.iloc[0].get("statiz_player_id"))
            player_id = resolve_statiz_player_id(name, statiz_id, id_cache=cache)
            log = fetch_kbo_statiz_game_log(player_id, name)
            if not log.empty:
                frames.append(log)
            elif errors is not None:
                errors.append(f"KBO Statiz: empty log for {name!r} (id={player_id})")
        except Exception as exc:
            if errors is not None:
                errors.append(f"KBO Statiz {name!r}: {exc}")
        time.sleep(pause_seconds)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_kbo_player_log(
    player_name: str,
    csv_path: str | Path | None = None,
    statiz_player_id: str | None = None,
    *,
    source: str = "auto",
    mykbo_player_id: str | None = None,
    parse_api_key: str | None = None,
) -> pd.DataFrame:
    src = source.strip().lower()
    has_parse = bool(parse_api_key or os.getenv("PARSE_API_KEY"))

    if src == "csv" or csv_path:
        return load_kbo_history_csv(csv_path, default_player=player_name)  # type: ignore[arg-type]
    if src == "statiz":
        pid = resolve_statiz_player_id(player_name, statiz_player_id)
        return fetch_kbo_statiz_game_log(pid, player_name)
    if src == "mykbo" or (src == "auto" and has_parse):
        from sports_prop_edge.integrations.mykbo_client import fetch_mykbo_player_game_log
        return fetch_mykbo_player_game_log(player_name, mykbo_player_id=mykbo_player_id, api_key=parse_api_key)
    if src == "scrape" or src == "auto":
        df, _missing = sync_kbo_players_via_mykbo_scrape([player_name], lookback_days=120)
        return df
    if statiz_player_id:
        return fetch_kbo_statiz_game_log(statiz_player_id, player_name)
    raise ValueError(f"Unsupported KBO source: {source}")
