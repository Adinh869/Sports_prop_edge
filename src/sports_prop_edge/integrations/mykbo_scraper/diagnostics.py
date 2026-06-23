"""Sync diagnostics for MyKBO player resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PlayerMatchRow:
    props_name: str
    pp_team: str
    mykbo_id: str = ""
    statiz_id: str = ""
    matched_name: str = ""
    method: str = "unmatched"
    has_pitching_log: bool = False
    history_rows: int = 0
    error: str = ""


@dataclass
class SyncDiagnostics:
    player_matches: list[PlayerMatchRow] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    requests_avoided: int = 0
    hits_by_level: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})
    misses_by_level: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})
    cache_hits_player: int = 0
    cache_hits_game: int = 0
    search_requests: int = 0
    game_requests: int = 0
    cloudflare_failures: int = 0
    index_entries: int = 0

    def absorb_cache_stats(self, stats: Any) -> None:
        self.cache_hits += stats.cache_hits
        self.cache_misses += stats.cache_misses
        self.requests_avoided += stats.requests_avoided
        for level, count in stats.hits_by_level.items():
            self.hits_by_level[level] = self.hits_by_level.get(level, 0) + count
        for level, count in stats.misses_by_level.items():
            self.misses_by_level[level] = self.misses_by_level.get(level, 0) + count
        self.cache_hits_player = self.hits_by_level.get(2, 0)
        self.cache_hits_game = self.hits_by_level.get(3, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "player_matches": [row.__dict__ for row in self.player_matches],
            "unmatched": list(self.unmatched),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "requests_avoided": self.requests_avoided,
            "hits_by_level": dict(self.hits_by_level),
            "misses_by_level": dict(self.misses_by_level),
            "cache_hits_player": self.cache_hits_player,
            "cache_hits_game": self.cache_hits_game,
            "search_requests": self.search_requests,
            "game_requests": self.game_requests,
            "cloudflare_failures": self.cloudflare_failures,
            "index_entries": self.index_entries,
        }
