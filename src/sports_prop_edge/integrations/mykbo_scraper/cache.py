"""Multi-level persistent cache for MyKBO scraper.

Level 2: player_ids.json (resolved players + search results)
Level 3: games/{game_id}.json
Level 4: pools/{date}.json
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sports_prop_edge.integrations.mykbo_scraper.cache_stats import CacheStatistics
from sports_prop_edge.integrations.name_utils import normalize_lookup_name

PLAYER_IDS_FILE = "player_ids.json"
PLAYER_INDEX_FILE = "player_index.json"
GAMES_DIR = "games"
POOLS_DIR = "pools"

_cache_registry: dict[str, MyKBOCache] = {}


def cache_root(root: Path) -> Path:
    path = root / "data" / "cache" / "mykbo"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _safe_id(value: str) -> str:
    return re.sub(r"[^\w\-]", "", str(value))


class MyKBOCache:
    """Levels 2–4 disk cache with statistics."""

    def __init__(self, root: Path, *, stats: CacheStatistics | None = None) -> None:
        self.root = root
        self.base = cache_root(root)
        self.stats = stats or CacheStatistics()

    def _player_ids_path(self) -> Path:
        return self.base / PLAYER_IDS_FILE

    def _load_player_ids_doc(self) -> dict[str, Any]:
        raw = _read_json(self._player_ids_path())
        if not isinstance(raw, dict):
            return {"players": {}, "searches": {}}
        if "players" in raw or "searches" in raw:
            players = raw.get("players") if isinstance(raw.get("players"), dict) else {}
            searches = raw.get("searches") if isinstance(raw.get("searches"), dict) else {}
            return {"players": players, "searches": searches}
        return {"players": raw, "searches": {}}

    def _save_player_ids_doc(self, doc: dict[str, Any]) -> None:
        _write_json(self._player_ids_path(), doc)

    def get_player_entry(self, props_key: str) -> dict[str, str] | None:
        key = normalize_lookup_name(props_key)
        doc = self._load_player_ids_doc()
        entry = doc["players"].get(key)
        if not isinstance(entry, dict) or not (entry.get("mykbo_id") or entry.get("statiz_id")):
            return None
        self.stats.record_hit(2, avoided_http=True)
        return {k: str(v) for k, v in entry.items()}

    def save_player_entry(
        self,
        props_key: str,
        *,
        mykbo_id: str = "",
        statiz_id: str = "",
        matched_name: str = "",
        method: str = "",
    ) -> None:
        key = normalize_lookup_name(props_key)
        doc = self._load_player_ids_doc()
        doc["players"][key] = {
            "mykbo_id": str(mykbo_id or "").strip(),
            "statiz_id": str(statiz_id or "").strip(),
            "matched_name": str(matched_name or "").strip(),
            "method": str(method or "").strip(),
            "cached_at": _utc_now(),
        }
        self._save_player_ids_doc(doc)

    def get_search_results(self, query: str) -> list[dict[str, str]] | None:
        key = normalize_lookup_name(query)
        doc = self._load_player_ids_doc()
        block = doc["searches"].get(key)
        if not isinstance(block, dict):
            return None
        matches = block.get("matches")
        if not isinstance(matches, list) or not matches:
            return None
        self.stats.record_hit(2, avoided_http=True)
        return [dict(m) for m in matches if isinstance(m, dict)]

    def save_search_results(self, query: str, matches: list[dict[str, str]]) -> None:
        key = normalize_lookup_name(query)
        doc = self._load_player_ids_doc()
        doc["searches"][key] = {
            "matches": matches,
            "cached_at": _utc_now(),
        }
        self._save_player_ids_doc(doc)

    def game_path(self, game_id: str) -> Path:
        return self.base / GAMES_DIR / f"{_safe_id(game_id)}.json"

    def get_game(self, game_id: str) -> dict[str, Any] | None:
        gid = _safe_id(game_id)
        if not gid:
            return None
        legacy_html = self.base / GAMES_DIR / f"{gid}.html"
        record = _read_json(self.game_path(gid))
        if isinstance(record, dict) and record.get("game_id"):
            self.stats.record_hit(3, avoided_http=True)
            return record
        if legacy_html.exists():
            try:
                html = legacy_html.read_text(encoding="utf-8")
                from sports_prop_edge.integrations.mykbo_scraper.games import build_game_record

                record = build_game_record(html, gid)
                self.save_game(gid, record)
                self.stats.record_hit(3, avoided_http=True)
                return record
            except OSError:
                pass
        return None

    def save_game(self, game_id: str, record: dict[str, Any]) -> Path:
        gid = _safe_id(game_id)
        record = dict(record)
        record["game_id"] = gid
        record["cached_at"] = _utc_now()
        path = self.game_path(gid)
        _write_json(path, record)
        return path

    def pool_path(self, slate_date: str) -> Path:
        return self.base / POOLS_DIR / f"{slate_date}.json"

    def get_daily_pool(self, slate_date: str | None = None) -> dict[str, Any] | None:
        day = slate_date or date.today().isoformat()
        record = _read_json(self.pool_path(day))
        if not isinstance(record, dict) or not record.get("rows"):
            return None
        self.stats.record_hit(4, avoided_http=True)
        return record

    def save_daily_pool(self, pool: pd.DataFrame, slate_date: str | None = None) -> Path:
        day = slate_date or date.today().isoformat()
        rows = pool.to_dict(orient="records") if not pool.empty else []
        record = {
            "slate_date": day,
            "cached_at": _utc_now(),
            "row_count": len(rows),
            "rows": rows,
        }
        path = self.pool_path(day)
        _write_json(path, record)
        return path

    def load_daily_pool_df(self, slate_date: str | None = None) -> pd.DataFrame:
        record = self.get_daily_pool(slate_date)
        if not record:
            self.stats.record_miss(4)
            return pd.DataFrame()
        rows = record.get("rows") or []
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def load_player_index(self) -> dict[str, dict[str, str]]:
        raw = _read_json(self.base / PLAYER_INDEX_FILE)
        return raw if isinstance(raw, dict) else {}

    def save_player_index(self, index: dict[str, dict[str, str]]) -> None:
        _write_json(self.base / PLAYER_INDEX_FILE, index)


def get_mykbo_cache(root: Path) -> MyKBOCache:
    key = str(root.resolve())
    if key not in _cache_registry:
        _cache_registry[key] = MyKBOCache(root)
    return _cache_registry[key]


def reset_mykbo_cache(root: Path | None = None) -> None:
    if root is None:
        _cache_registry.clear()
        return
    _cache_registry.pop(str(root.resolve()), None)


def load_player_id_map(root: Path) -> dict[str, dict[str, str]]:
    return get_mykbo_cache(root)._load_player_ids_doc()["players"]  # type: ignore[return-value]


def save_player_id_entry(
    root: Path,
    props_key: str,
    *,
    mykbo_id: str = "",
    statiz_id: str = "",
    matched_name: str = "",
    method: str = "",
) -> None:
    get_mykbo_cache(root).save_player_entry(
        props_key,
        mykbo_id=mykbo_id,
        statiz_id=statiz_id,
        matched_name=matched_name,
        method=method,
    )


def load_player_index(root: Path) -> dict[str, dict[str, str]]:
    return get_mykbo_cache(root).load_player_index()


def save_player_index(root: Path, index: dict[str, dict[str, str]]) -> None:
    get_mykbo_cache(root).save_player_index(index)


def read_game_html(root: Path, game_id: str) -> str | None:
    record = get_mykbo_cache(root).get_game(game_id)
    if isinstance(record, dict) and record.get("html"):
        return str(record["html"])
    legacy = cache_root(root) / GAMES_DIR / f"{_safe_id(game_id)}.html"
    if legacy.exists():
        try:
            return legacy.read_text(encoding="utf-8")
        except OSError:
            return None
    return None


def write_game_html(root: Path, game_id: str, html: str) -> Path:
    from sports_prop_edge.integrations.mykbo_scraper.games import build_game_record

    cache = get_mykbo_cache(root)
    record = build_game_record(html, game_id)
    return cache.save_game(game_id, record)
