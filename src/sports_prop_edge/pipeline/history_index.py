"""Pre-indexed player game logs for fast projection lookups."""

from __future__ import annotations

import pandas as pd

from sports_prop_edge.integrations.name_utils import normalize_lookup_name


class HistoryIndex:
    """Maps (player, sport) → sorted game-log slice; avoids per-prop full-table scans."""

    def __init__(self, history: pd.DataFrame):
        self._by_player_sport: dict[tuple[str, str], pd.DataFrame] = {}
        self._by_player_team_sport: dict[tuple[str, str, str], pd.DataFrame] = {}
        self._enriched_cache: dict[tuple[str, str, str], pd.DataFrame] = {}
        if history is None or history.empty:
            return

        work = history.copy()
        work["_player_key"] = work["player"].astype(str).map(normalize_lookup_name)
        if "game_title" in work.columns:
            work["_sport"] = work["game_title"].astype(str).str.upper().str.strip()
        else:
            work["_sport"] = ""

        if "date" in work.columns:
            work = work.sort_values("date")

        for (player_key, sport), grp in work.groupby(["_player_key", "_sport"], sort=False):
            clean = grp.drop(columns=["_player_key", "_sport"], errors="ignore")
            self._by_player_sport[(player_key, sport)] = clean

        if "team" in work.columns:
            work["_team_key"] = work["team"].astype(str).str.lower().str.strip()
            for (player_key, sport, team_key), grp in work.groupby(
                ["_player_key", "_sport", "_team_key"], sort=False
            ):
                if not team_key:
                    continue
                clean = grp.drop(columns=["_player_key", "_sport", "_team_key"], errors="ignore")
                self._by_player_team_sport[(player_key, sport, team_key)] = clean

    def slice(
        self,
        player: str,
        game_title: str | None = None,
        team: str | None = None,
    ) -> pd.DataFrame:
        player_key = normalize_lookup_name(player)
        sport = str(game_title or "").upper().strip()

        if team:
            team_key = str(team).lower().strip()
            team_rows = self._by_player_team_sport.get((player_key, sport, team_key))
            if team_rows is not None and not team_rows.empty:
                return team_rows.copy()

        rows = self._by_player_sport.get((player_key, sport))
        if rows is not None and not rows.empty:
            return rows.copy()

        if sport:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []
        for (pk, _sport), grp in self._by_player_sport.items():
            if pk == player_key:
                frames.append(grp)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("date") if "date" in frames[0].columns else pd.concat(frames, ignore_index=True)

    def enriched_slice(
        self,
        player: str,
        game_title: str | None = None,
        team: str | None = None,
    ) -> pd.DataFrame:
        """Cached history slice with sport-specific derived stat columns."""
        player_key = normalize_lookup_name(player)
        sport = str(game_title or "").upper().strip()
        team_key = str(team or "").lower().strip()
        cache_key = (player_key, sport, team_key)
        cached = self._enriched_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        rows = self.slice(player, game_title, team)
        if not rows.empty and sport in {"NBA", "WNBA", "CBB"}:
            from sports_prop_edge.models.projections import enrich_basketball_history_rows

            rows = enrich_basketball_history_rows(rows)
        elif not rows.empty and sport in {"KBO", "MLB"}:
            from sports_prop_edge.models.projections import enrich_baseball_history_rows

            rows = enrich_baseball_history_rows(rows)
        self._enriched_cache[cache_key] = rows
        return rows.copy()
