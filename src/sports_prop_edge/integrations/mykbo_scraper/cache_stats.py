"""Cache hit/miss statistics for MyKBO scraper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CacheStatistics:
    """Tracks multi-level cache effectiveness."""

    cache_hits: int = 0
    cache_misses: int = 0
    requests_avoided: int = 0
    hits_by_level: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})
    misses_by_level: dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0, 4: 0})

    def record_hit(self, level: int, *, avoided_http: bool = True) -> None:
        self.cache_hits += 1
        self.hits_by_level[level] = self.hits_by_level.get(level, 0) + 1
        if avoided_http:
            self.requests_avoided += 1

    def record_miss(self, level: int) -> None:
        self.cache_misses += 1
        self.misses_by_level[level] = self.misses_by_level.get(level, 0) + 1

    def merge(self, other: CacheStatistics) -> None:
        self.cache_hits += other.cache_hits
        self.cache_misses += other.cache_misses
        self.requests_avoided += other.requests_avoided
        for level, count in other.hits_by_level.items():
            self.hits_by_level[level] = self.hits_by_level.get(level, 0) + count
        for level, count in other.misses_by_level.items():
            self.misses_by_level[level] = self.misses_by_level.get(level, 0) + count

    def to_dict(self) -> dict[str, Any]:
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "requests_avoided": self.requests_avoided,
            "hits_by_level": dict(self.hits_by_level),
            "misses_by_level": dict(self.misses_by_level),
        }
