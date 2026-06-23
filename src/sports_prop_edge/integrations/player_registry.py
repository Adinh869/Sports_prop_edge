"""Persistent player ID cache + manual aliases (mirrors esports player_aliases.json)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from sports_prop_edge.integrations.name_utils import normalize_lookup_name


@dataclass
class PlayerRecord:
    sport: str
    canonical_name: str
    statiz_player_id: str = ""
    mykbo_player_id: str = ""
    nba_player_id: str = ""
    nfl_gsis_id: str = ""
    mlb_player_id: str = ""
    nhl_player_id: str = ""
    wnba_player_id: str = ""
    resolved_source_name: str = ""
    match_method: str = ""
    confidence: float = 0.0
    team_hint: str = ""

    @property
    def key(self) -> str:
        return f"{self.sport.upper()}|{normalize_lookup_name(self.canonical_name)}"


def _registry_path(root: Path) -> Path:
    return root / "data" / "cache" / "player_registry.json"


def _aliases_path(root: Path) -> Path:
    return root / "data" / "config" / "player_aliases.json"


def _aliases_suggested_path(root: Path) -> Path:
    return root / "data" / "config" / "player_aliases_suggested.json"


def load_registry(root: Path) -> dict[str, dict]:
    path = _registry_path(root)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw if isinstance(raw, dict) else {}


def save_registry(root: Path, registry: dict[str, dict]) -> None:
    path = _registry_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")


def load_aliases(root: Path) -> dict[str, str]:
    """Keys: 'SPORT|pp_name' or plain pp_name -> canonical history name."""
    merged: dict[str, str] = {}
    for fname in (_aliases_path(root), _aliases_suggested_path(root)):
        if not fname.exists():
            continue
        try:
            raw = json.loads(fname.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(raw, dict):
            continue
        for k, v in raw.items():
            if not k or not v or str(k).startswith("_"):
                continue
            merged[normalize_lookup_name(str(k))] = normalize_lookup_name(str(v))
    return merged


def alias_for(root: Path, sport: str, player_name: str) -> str:
    aliases = load_aliases(root)
    sport_key = sport.upper()
    name = normalize_lookup_name(player_name)
    for key in (f"{sport_key}|{name}", name):
        if key in aliases:
            return aliases[key]
        compact = key.split("|")[-1] if "|" in key else key
        if compact in aliases:
            return aliases[compact]
    return name


def get_record(root: Path, sport: str, player_name: str) -> PlayerRecord | None:
    registry = load_registry(root)
    key = f"{sport.upper()}|{normalize_lookup_name(player_name)}"
    raw = registry.get(key)
    if not raw:
        return None
    fields = {
        k: v
        for k, v in raw.items()
        if k in PlayerRecord.__dataclass_fields__ and k not in {"sport", "canonical_name"}
    }
    return PlayerRecord(
        sport=sport.upper(),
        canonical_name=normalize_lookup_name(player_name),
        **fields,
    )


def upsert_record(root: Path, record: PlayerRecord) -> None:
    registry = load_registry(root)
    registry[record.key] = asdict(record)
    save_registry(root, registry)


def save_suggested_aliases(root: Path, pairs: dict[str, str]) -> None:
    path = _aliases_suggested_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing = {}
    if not isinstance(existing, dict):
        existing = {}
    existing.update(pairs)
    path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
