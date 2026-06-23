"""Projection models for traditional sports player props."""

from __future__ import annotations

from sports_prop_edge.integrations.name_utils import normalize_lookup_name

from dataclasses import dataclass

import pandas as pd

from sports_prop_edge.data.loaders import BASEBALL_MARKETS, NBA_MARKETS, NFL_MARKETS, SOCCER_MARKETS, TENNIS_MARKETS

MARKET_TO_HISTORY_COL = {
    "points": "points",
    "rebounds": "rebounds",
    "assists": "assists",
    "threes": "threes",
    "steals": "steals",
    "blocks": "blocks",
    "turnovers": "turnovers",
    "pra": "pra",
    "pts_rebs": "pts_rebs",
    "pts_asts": "pts_asts",
    "rebs_asts": "rebs_asts",
    "hits": "hits",
    "runs": "runs",
    "rbis": "rbis",
    "strikeouts": "strikeouts",
    "total_bases": "total_bases",
    "walks": "walks",
    "stolen_bases": "stolen_bases",
    "singles": "singles",
    "doubles": "doubles",
    "hits_runs_rbis": "hits_runs_rbis",
    "home_runs": "home_runs",
    "passing_yards": "passing_yards",
    "rushing_yards": "rushing_yards",
    "receiving_yards": "receiving_yards",
    "receptions": "receptions",
    "passing_tds": "passing_tds",
    "rushing_tds": "rushing_tds",
    "receiving_tds": "receiving_tds",
    "fantasy_points": "fantasy_points",
    "pitcher_strikeouts": "pitcher_strikeouts",
    "hits_allowed": "hits_allowed",
    "pitcher_outs": "outs_pitched",
    "outs_pitched": "outs_pitched",
    "earned_runs": "earned_runs",
    "runs_allowed": "runs",
    "walks_allowed": "walks",
    "break_points_won": "break_points_won",
    "aces": "aces",
    "games_won": "games_won",
    "double_faults": "double_faults",
    "goals": "goals",
    "shots": "shots",
    "shots_on_target": "shots_on_target",
    "passes": "passes",
    "tackles": "tackles",
    "saves": "saves",
}

SPORT_DEFAULT_VOLUME = {
    "NBA": ("per_minute", "expected_minutes", 32.0),
    "WNBA": ("per_minute", "expected_minutes", 32.0),
    "KBO": ("per_plate_appearance", "expected_plate_appearances", 4.0),
    "MLB": ("per_plate_appearance", "expected_plate_appearances", 4.0),
    "NFL": ("per_game", "expected_games", 1.0),
    "TENNIS": ("per_game", "expected_games", 1.0),
    "SOCCER": ("per_minute", "expected_minutes", 90.0),
}

PER_GAME_BASEBALL_STATS = {
    "hits_runs_rbis",
    "fantasy_points",
    "walks",
    "runs_allowed",
}

# Pitcher counting stats: rate per out × blended expected outs (MLB/KBO game logs).
PITCHER_VOLUME_STATS = frozenset(
    {
        "pitcher_strikeouts",
        "hits_allowed",
        "outs_pitched",
        "earned_runs",
        "innings_pitched",
        "walks_allowed",
    }
)


def _numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def enrich_baseball_history_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Derive composite baseball stats when game logs only have counting columns."""
    if rows.empty:
        return rows

    work = rows.copy()
    hits = _numeric_series(work, "hits")
    runs = _numeric_series(work, "runs")
    rbis = _numeric_series(work, "rbis")
    singles = _numeric_series(work, "singles")
    doubles = _numeric_series(work, "doubles")
    walks = _numeric_series(work, "walks")
    stolen_bases = _numeric_series(work, "stolen_bases")
    total_bases = _numeric_series(work, "total_bases")

    if "hits_runs_rbis" not in work.columns or work["hits_runs_rbis"].isna().all():
        work["hits_runs_rbis"] = hits + runs + rbis

    needs_fantasy = "fantasy_points" not in work.columns or work["fantasy_points"].isna().all()
    if needs_fantasy:
        home_runs = _numeric_series(work, "home_runs")
        if home_runs.sum() == 0 and total_bases.sum() > 0:
            home_runs = ((total_bases - singles - 2 * doubles) / 4).clip(lower=0).round()
        triples = (hits - singles - doubles - home_runs).clip(lower=0)
        # PrizePicks hitter fantasy: 1B=3, 2B=5, 3B=8, HR=10, R=2, RBI=2, BB=2, SB=5
        work["fantasy_points"] = (
            singles * 3
            + doubles * 5
            + triples * 8
            + home_runs * 10
            + runs * 2
            + rbis * 2
            + walks * 2
            + stolen_bases * 5
        )

    return work


def enrich_basketball_history_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Derive PRA combo stats from points / rebounds / assists game logs."""
    if rows.empty:
        return rows

    work = rows.copy()
    points = _numeric_series(work, "points")
    rebounds = _numeric_series(work, "rebounds")
    assists = _numeric_series(work, "assists")

    if "pra" not in work.columns or work["pra"].isna().all():
        work["pra"] = points + rebounds + assists
    work["pts_rebs"] = points + rebounds
    work["pts_asts"] = points + assists
    work["rebs_asts"] = rebounds + assists
    return work


@dataclass
class ProjectionConfig:
    recent_events: int = 10
    baseline_events: int = 30
    recent_weight: float = 0.65
    default_expected_minutes: float = 32.0
    default_expected_plate_appearances: float = 4.0
    default_expected_games: float = 1.0
    default_expected_outs: float = 17.0
    min_events_for_grade: int = 5


class SportPropProjector:
    """Rolling rate projection for NBA, KBO/MLB, and NFL count stats."""

    def __init__(self, config: ProjectionConfig | None = None):
        self.config = config or ProjectionConfig()
        if not 0 <= self.config.recent_weight <= 1:
            raise ValueError("recent_weight must be between 0 and 1")

    @staticmethod
    def _sport_for_market(market: str, game_title: str | None = None) -> str:
        market = market.lower()
        if game_title:
            sport = str(game_title).upper()
            if sport in SPORT_DEFAULT_VOLUME:
                return sport
        if market in NBA_MARKETS:
            return "NBA"
        if market in BASEBALL_MARKETS:
            return "KBO"
        if market in NFL_MARKETS:
            return "NFL"
        if market in TENNIS_MARKETS:
            return "TENNIS"
        if market in SOCCER_MARKETS:
            return "SOCCER"
        return str(game_title or "NBA").upper()

    @staticmethod
    def _filtered_history(
        history: pd.DataFrame,
        player: str,
        game_title: str | None = None,
        team: str | None = None,
    ) -> pd.DataFrame:
        player_key = normalize_lookup_name(player)
        hist_players = history["player"].astype(str).map(normalize_lookup_name)
        rows = history[hist_players == player_key].sort_values("date")
        if game_title and "game_title" in rows.columns:
            game_rows = rows[rows["game_title"].str.upper() == str(game_title).upper()]
            if not game_rows.empty:
                rows = game_rows
        if team:
            team_rows = rows[rows["team"].str.lower() == str(team).lower()]
            if not team_rows.empty:
                rows = team_rows
        return rows

    @staticmethod
    def _pitcher_outs_series(rows: pd.DataFrame) -> pd.Series:
        if "outs_pitched" in rows.columns:
            outs = pd.to_numeric(rows["outs_pitched"], errors="coerce")
            if outs.notna().any():
                return outs.fillna(0.0)
        if "innings_pitched" in rows.columns:
            ip = pd.to_numeric(rows["innings_pitched"], errors="coerce").fillna(0.0)
            whole = ip.astype(int)
            thirds = ((ip - whole) * 3).round().astype(int).clip(0, 2)
            return (whole * 3 + thirds).astype(float)
        return pd.Series(0.0, index=rows.index, dtype=float)

    def _blended_pitcher_outs(self, rows: pd.DataFrame) -> float:
        """Recent + baseline outs blend for MLB/KBO starters (mirrors NBA minutes)."""
        if rows.empty:
            return self.config.default_expected_outs
        outs = self._pitcher_outs_series(rows)
        outs = outs[outs > 0]
        if outs.empty:
            return self.config.default_expected_outs
        recent = outs.tail(self.config.recent_events).mean()
        baseline = outs.tail(self.config.baseline_events).mean()
        if pd.isna(recent) and pd.isna(baseline):
            blended = self.config.default_expected_outs
        elif pd.isna(recent):
            blended = float(baseline)
        elif pd.isna(baseline):
            blended = float(recent)
        else:
            blended = (
                self.config.recent_weight * float(recent)
                + (1 - self.config.recent_weight) * float(baseline)
            )
        return float(max(9.0, min(27.0, blended)))

    def _is_pitching_log(self, rows: pd.DataFrame) -> bool:
        outs = self._pitcher_outs_series(rows)
        if outs.sum() <= 0:
            return False
        if "plate_appearances" in rows.columns:
            pa = pd.to_numeric(rows["plate_appearances"], errors="coerce").fillna(0)
            return float(pa.sum()) == 0.0
        return True

    def _rate(self, rows: pd.DataFrame, stat_col: str, sport: str) -> tuple[float | None, str]:
        if rows.empty or stat_col not in rows.columns:
            return None, "missing"

        pitcher_stat = stat_col in PITCHER_VOLUME_STATS or (
            stat_col in {"walks", "runs"} and self._is_pitching_log(rows)
        )
        if sport in {"MLB", "KBO"} and pitcher_stat:
            outs = self._pitcher_outs_series(rows)
            valid = outs > 0
            if valid.any():
                stat_vals = pd.to_numeric(rows[stat_col], errors="coerce").fillna(0.0)
                out_sum = float(outs[valid].sum())
                if out_sum > 0:
                    return float(stat_vals[valid].sum() / out_sum), "per_out"

        if stat_col in PER_GAME_BASEBALL_STATS:
            vals = pd.to_numeric(rows[stat_col], errors="coerce").dropna()
            if not vals.empty:
                return float(vals.mean()), "per_game"

        stat_sum = rows[stat_col].dropna().sum()
        preferred_basis, _, _ = SPORT_DEFAULT_VOLUME.get(sport, ("per_game", "expected_games", 1.0))

        if preferred_basis == "per_minute" and "minutes" in rows.columns and rows["minutes"].dropna().sum() > 0:
            return float(stat_sum / rows["minutes"].dropna().sum()), "per_minute"
        if (
            preferred_basis == "per_plate_appearance"
            and "plate_appearances" in rows.columns
            and rows["plate_appearances"].dropna().sum() > 0
        ):
            return float(stat_sum / rows["plate_appearances"].dropna().sum()), "per_plate_appearance"
        if "minutes" in rows.columns and rows["minutes"].dropna().sum() > 0:
            return float(stat_sum / rows["minutes"].dropna().sum()), "per_minute"
        if "plate_appearances" in rows.columns and rows["plate_appearances"].dropna().sum() > 0:
            return float(stat_sum / rows["plate_appearances"].dropna().sum()), "per_plate_appearance"
        if "games" in rows.columns and rows["games"].dropna().sum() > 0:
            return float(stat_sum / rows["games"].dropna().sum()), "per_game"
        return float(rows[stat_col].dropna().mean()), "per_event"

    def _expected_volume(self, row: pd.Series, rate_basis: str, sport: str) -> float:
        _, volume_col, default = SPORT_DEFAULT_VOLUME.get(sport, ("per_game", "expected_games", 1.0))
        if rate_basis == "per_minute":
            value = row.get("expected_minutes", self.config.default_expected_minutes)
            return float(value) if pd.notna(value) else self.config.default_expected_minutes
        if rate_basis == "per_plate_appearance":
            value = row.get("expected_plate_appearances", self.config.default_expected_plate_appearances)
            return float(value) if pd.notna(value) else self.config.default_expected_plate_appearances
        if rate_basis == "per_game":
            value = row.get("expected_games", self.config.default_expected_games)
            return float(value) if pd.notna(value) else self.config.default_expected_games
        if rate_basis == "per_out":
            for key in ("expected_outs", "expected_outs_pitched"):
                value = row.get(key)
                if value is not None and pd.notna(value):
                    return float(value)
            return self.config.default_expected_outs
        value = row.get(volume_col, default)
        return float(value) if pd.notna(value) else default

    def _blended_minutes(self, rows: pd.DataFrame) -> float:
        """Recent + baseline minutes blend (NBA/WNBA) when prop row has no override."""
        if rows.empty or "minutes" not in rows.columns:
            return self.config.default_expected_minutes
        mins = pd.to_numeric(rows["minutes"], errors="coerce").dropna()
        if mins.empty:
            return self.config.default_expected_minutes
        recent = mins.tail(self.config.recent_events).mean()
        baseline = mins.tail(self.config.baseline_events).mean()
        if pd.isna(recent) and pd.isna(baseline):
            blended = self.config.default_expected_minutes
        elif pd.isna(recent):
            blended = float(baseline)
        elif pd.isna(baseline):
            blended = float(recent)
        else:
            blended = (
                self.config.recent_weight * float(recent)
                + (1 - self.config.recent_weight) * float(baseline)
            )
        return float(max(8.0, min(44.0, blended)))

    def _blended_soccer_minutes(self, rows: pd.DataFrame) -> float:
        if rows.empty or "minutes" not in rows.columns:
            return 90.0
        mins = pd.to_numeric(rows["minutes"], errors="coerce").dropna()
        if mins.empty:
            return 90.0
        recent = mins.tail(self.config.recent_events).mean()
        baseline = mins.tail(self.config.baseline_events).mean()
        if pd.isna(recent) and pd.isna(baseline):
            blended = 90.0
        elif pd.isna(recent):
            blended = float(baseline)
        elif pd.isna(baseline):
            blended = float(recent)
        else:
            blended = (
                self.config.recent_weight * float(recent)
                + (1 - self.config.recent_weight) * float(baseline)
            )
        return float(max(45.0, min(120.0, blended)))

    def _blended_plate_appearances(self, rows: pd.DataFrame) -> float:
        """Recent + baseline PA blend for MLB/KBO hitters."""
        if rows.empty or "plate_appearances" not in rows.columns:
            return self.config.default_expected_plate_appearances
        pa = pd.to_numeric(rows["plate_appearances"], errors="coerce").dropna()
        pa = pa[pa > 0]
        if pa.empty:
            return self.config.default_expected_plate_appearances
        recent = pa.tail(self.config.recent_events).mean()
        baseline = pa.tail(self.config.baseline_events).mean()
        if pd.isna(recent) and pd.isna(baseline):
            blended = self.config.default_expected_plate_appearances
        elif pd.isna(recent):
            blended = float(baseline)
        elif pd.isna(baseline):
            blended = float(recent)
        else:
            blended = (
                self.config.recent_weight * float(recent)
                + (1 - self.config.recent_weight) * float(baseline)
            )
        return float(max(2.0, min(6.0, blended)))

    @staticmethod
    def _naive_timestamp(value: object) -> pd.Timestamp:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            return ts
        return ts.tz_convert(None)

    def _rest_volume_factor(
        self,
        rows: pd.DataFrame,
        sport: str,
        event_time: object | None = None,
    ) -> float:
        """Back-to-back / extra rest multiplier from game-log dates."""
        if sport not in {"NBA", "WNBA", "CBB", "NFL", "MLB", "KBO"}:
            return 1.0
        if rows.empty or "date" not in rows.columns:
            return 1.0

        if event_time is not None and not pd.isna(event_time):
            last_date = self._naive_timestamp(
                pd.to_datetime(rows["date"], errors="coerce", utc=True).dropna().max()
            )
            event_date = self._naive_timestamp(event_time)
            if pd.notna(last_date) and pd.notna(event_date):
                days_rest = (event_date.normalize() - last_date.normalize()).days
                if days_rest <= 1:
                    return 0.94
                if days_rest >= 3:
                    return 1.02
                return 1.0

        dates = pd.to_datetime(rows["date"], errors="coerce", utc=True).dropna()
        dates = dates.map(lambda ts: ts.tz_convert(None) if pd.notna(ts) else ts).sort_values()
        if len(dates) < 2:
            return 1.0
        gap_days = (dates.iloc[-1] - dates.iloc[-2]).days
        if gap_days <= 1:
            return 0.94
        if gap_days >= 3:
            return 1.02
        return 1.0

    def _default_volume_from_rows(self, rows: pd.DataFrame, basis: str, sport: str) -> float:
        if basis == "per_minute" and sport in {"NBA", "WNBA", "CBB"}:
            return self._blended_minutes(rows)
        if basis == "per_minute" and sport == "SOCCER":
            return self._blended_soccer_minutes(rows)
        if basis == "per_out" and sport in {"MLB", "KBO"}:
            return self._blended_pitcher_outs(rows)
        if basis == "per_plate_appearance" and sport in {"MLB", "KBO"}:
            return self._blended_plate_appearances(rows)
        _, _, default = SPORT_DEFAULT_VOLUME.get(sport, ("per_game", "expected_games", 1.0))
        return float(default)

    def _last_game_timestamp(self, rows: pd.DataFrame) -> pd.Timestamp | pd.NaT:
        if rows.empty or "date" not in rows.columns:
            return pd.NaT
        return self._naive_timestamp(pd.to_datetime(rows["date"], errors="coerce", utc=True).dropna().max())

    def _group_rate_fields_from_rows(
        self,
        rows: pd.DataFrame,
        *,
        market: str,
        sport: str,
        player: str,
    ) -> dict[str, float | str | int | None | pd.Timestamp]:
        stat_col = MARKET_TO_HISTORY_COL.get(str(market).lower(), str(market).lower())
        if rows.empty or stat_col not in rows.columns:
            return {
                "player": player,
                "market": market,
                "sport": sport,
                "stat_col": stat_col,
                "projected_mean": None,
                "recent_rate": None,
                "baseline_rate": None,
                "rate_basis": "missing",
                "events_used": int(len(rows)),
                "expected_volume": None,
                "blended_rate": None,
                "_default_volume": None,
                "_last_game_date": self._last_game_timestamp(rows),
                "_rest_gap_factor": 1.0,
            }

        recent = rows.tail(self.config.recent_events)
        baseline = rows.tail(self.config.baseline_events)
        recent_rate, recent_basis = self._rate(recent, stat_col, sport)
        baseline_rate, baseline_basis = self._rate(baseline, stat_col, sport)
        basis = recent_basis if recent_rate is not None else baseline_basis

        if recent_rate is None and baseline_rate is None:
            blended_rate = None
        elif recent_rate is None:
            blended_rate = baseline_rate
        elif baseline_rate is None:
            blended_rate = recent_rate
        else:
            blended_rate = (
                self.config.recent_weight * recent_rate
                + (1 - self.config.recent_weight) * baseline_rate
            )

        default_volume = (
            self._default_volume_from_rows(rows, basis, sport) if blended_rate is not None else None
        )
        rest_gap_factor = self._rest_volume_factor(rows, sport, None) if not rows.empty else 1.0
        return {
            "player": player,
            "market": market,
            "sport": sport,
            "stat_col": stat_col,
            "recent_rate": recent_rate,
            "baseline_rate": baseline_rate,
            "rate_basis": basis,
            "events_used": int(len(rows)),
            "blended_rate": blended_rate,
            "_default_volume": default_volume,
            "_last_game_date": self._last_game_timestamp(rows),
            "_rest_gap_factor": rest_gap_factor,
        }

    def _expected_volume_from_prop(
        self,
        prop: pd.Series,
        *,
        basis: str,
        sport: str,
        default_volume: float | None,
    ) -> float | None:
        if default_volume is None:
            return None
        if basis == "per_minute" and sport in {"NBA", "WNBA", "CBB"}:
            explicit_min = prop.get("expected_minutes")
            if explicit_min is None or pd.isna(explicit_min):
                return float(default_volume)
            return self._expected_volume(prop, basis, sport)
        if basis == "per_minute" and sport == "SOCCER":
            explicit_min = prop.get("expected_minutes")
            if explicit_min is None or pd.isna(explicit_min):
                return float(default_volume)
            return self._expected_volume(prop, basis, sport)
        if basis == "per_out" and sport in {"MLB", "KBO"}:
            explicit_outs = prop.get("expected_outs")
            if explicit_outs is None or pd.isna(explicit_outs):
                explicit_outs = prop.get("expected_outs_pitched")
            if explicit_outs is None or pd.isna(explicit_outs):
                return float(default_volume)
            return self._expected_volume(prop, basis, sport)
        if basis == "per_plate_appearance" and sport in {"MLB", "KBO"}:
            explicit_pa = prop.get("expected_plate_appearances")
            if explicit_pa is None or pd.isna(explicit_pa):
                return float(default_volume)
            return self._expected_volume(prop, basis, sport)
        return self._expected_volume(prop, basis, sport)

    def _rest_volume_factor_from_dates(
        self,
        sport: str,
        last_game_date: object,
        event_time: object | None,
    ) -> float:
        if sport not in {"NBA", "WNBA", "CBB", "NFL", "MLB", "KBO"}:
            return 1.0
        if pd.isna(last_game_date):
            return 1.0
        if event_time is not None and not pd.isna(event_time):
            event_date = self._naive_timestamp(event_time)
            last_date = self._naive_timestamp(last_game_date)
            if pd.notna(last_date) and pd.notna(event_date):
                days_rest = (event_date.normalize() - last_date.normalize()).days
                if days_rest <= 1:
                    return 0.94
                if days_rest >= 3:
                    return 1.02
                return 1.0
        return 1.0

    def _finalize_grouped_projections(self, work: pd.DataFrame) -> pd.DataFrame:
        out = work.copy()
        has_rate = out["blended_rate"].notna()
        out["expected_volume"] = pd.NA
        out["projected_mean"] = pd.NA
        out["rest_adjustment"] = pd.NA

        if not has_rate.any():
            return out

        vol = pd.to_numeric(out["_default_volume"], errors="coerce")
        sport = out["sport"].astype(str)
        basis = out["rate_basis"].astype(str)

        if "expected_minutes" in out.columns:
            explicit_min = pd.to_numeric(out["expected_minutes"], errors="coerce")
            nba_mask = has_rate & basis.eq("per_minute") & sport.isin(["NBA", "WNBA", "CBB"])
            vol = vol.mask(nba_mask & explicit_min.notna(), explicit_min)
            soccer_mask = has_rate & basis.eq("per_minute") & sport.eq("SOCCER")
            vol = vol.mask(soccer_mask & explicit_min.notna(), explicit_min)

        if "expected_outs" in out.columns or "expected_outs_pitched" in out.columns:
            explicit_outs = pd.to_numeric(out.get("expected_outs"), errors="coerce")
            pitched_outs = pd.to_numeric(out.get("expected_outs_pitched"), errors="coerce")
            combined_outs = explicit_outs.fillna(pitched_outs)
            out_mask = has_rate & basis.eq("per_out") & sport.isin(["MLB", "KBO"])
            vol = vol.mask(out_mask & combined_outs.notna(), combined_outs)

        if "expected_plate_appearances" in out.columns:
            explicit_pa = pd.to_numeric(out["expected_plate_appearances"], errors="coerce")
            pa_mask = has_rate & basis.eq("per_plate_appearance") & sport.isin(["MLB", "KBO"])
            vol = vol.mask(pa_mask & explicit_pa.notna(), explicit_pa)

        per_game_mask = has_rate & basis.eq("per_game")
        if per_game_mask.any() and "expected_games" in out.columns:
            explicit_games = pd.to_numeric(out["expected_games"], errors="coerce")
            vol = vol.mask(per_game_mask & explicit_games.notna(), explicit_games)

        rest = pd.to_numeric(out.get("_rest_gap_factor", 1.0), errors="coerce").fillna(1.0)
        eligible_sports = sport.isin(["NBA", "WNBA", "CBB", "NFL", "MLB", "KBO"])
        if "event_time" in out.columns:
            event_dt = pd.to_datetime(out["event_time"], errors="coerce", utc=True).dt.tz_convert(None)
            last_dt = pd.to_datetime(out["_last_game_date"], errors="coerce")
            has_event = has_rate & eligible_sports & event_dt.notna() & last_dt.notna()
            if has_event.any():
                days_rest = (event_dt.dt.normalize() - last_dt.dt.normalize()).dt.days
                rest_event = pd.Series(1.0, index=out.index)
                rest_event.loc[has_event & (days_rest <= 1)] = 0.94
                rest_event.loc[has_event & (days_rest >= 3)] = 1.02
                rest = rest_event.where(has_event, rest)

        vol = vol * rest

        def _adj_series(col: str) -> pd.Series:
            if col in out.columns:
                return pd.to_numeric(out[col], errors="coerce").fillna(1.0)
            return pd.Series(1.0, index=out.index, dtype=float)

        opponent_adj = _adj_series("opponent_adjustment")
        pace_adj = _adj_series("pace_adjustment")
        home_adj = _adj_series("home_adjustment")
        weather_adj = _adj_series("weather_adjustment")

        out.loc[has_rate, "expected_volume"] = vol[has_rate]
        out.loc[has_rate, "rest_adjustment"] = rest[has_rate]
        out.loc[has_rate, "projected_mean"] = (
            out.loc[has_rate, "blended_rate"].astype(float)
            * vol[has_rate].astype(float)
            * opponent_adj[has_rate]
            * pace_adj[has_rate]
            * home_adj[has_rate]
            * weather_adj[has_rate]
        )
        return out

    def _projection_group_keys(self, props: pd.DataFrame) -> pd.DataFrame:
        work = props.copy()
        work["_pk"] = work["player"].astype(str).map(normalize_lookup_name)
        work["_sport"] = work["game_title"].astype(str).str.upper().str.strip()
        work["_market"] = work["market"].astype(str).str.lower().str.strip()
        work["_team"] = work.get("team", pd.Series("", index=work.index)).fillna("").astype(str).str.lower().str.strip()
        return work

    def project_player(
        self,
        history: pd.DataFrame,
        player: str,
        market: str,
        game_title: str | None = None,
        team: str | None = None,
        prop_row: pd.Series | None = None,
        *,
        history_index: object | None = None,
    ) -> dict[str, float | str | int | None]:
        sport = self._sport_for_market(market, game_title)
        stat_col = MARKET_TO_HISTORY_COL.get(str(market).lower(), str(market).lower())
        if history_index is not None:
            rows = history_index.enriched_slice(player, game_title, team)
        else:
            rows = self._filtered_history(history, player=player, game_title=game_title, team=team)
            if not rows.empty and sport in {"NBA", "WNBA", "CBB"}:
                rows = enrich_basketball_history_rows(rows)
            if not rows.empty and sport in {"KBO", "MLB"}:
                rows = enrich_baseball_history_rows(rows)

        group_fields = self._group_rate_fields_from_rows(
            rows, market=market, sport=sport, player=player
        )
        if group_fields.get("blended_rate") is None:
            return {
                "player": player,
                "market": market,
                "sport": sport,
                "stat_col": stat_col,
                "projected_mean": None,
                "recent_rate": group_fields.get("recent_rate"),
                "baseline_rate": group_fields.get("baseline_rate"),
                "rate_basis": group_fields.get("rate_basis", "missing"),
                "events_used": group_fields.get("events_used", int(len(rows))),
                "expected_volume": None,
            }

        source_row = prop_row if prop_row is not None else pd.Series()
        vol = self._expected_volume_from_prop(
            source_row,
            basis=str(group_fields["rate_basis"]),
            sport=sport,
            default_volume=group_fields.get("_default_volume"),
        )
        rest_adj = self._rest_volume_factor_from_dates(
            sport, group_fields.get("_last_game_date"), source_row.get("event_time")
        )
        event_time = source_row.get("event_time")
        if event_time is None or pd.isna(event_time):
            rest_adj = float(group_fields.get("_rest_gap_factor", 1.0))
        if vol is None:
            return {
                "player": player,
                "market": market,
                "sport": sport,
                "stat_col": stat_col,
                "projected_mean": None,
                "recent_rate": group_fields["recent_rate"],
                "baseline_rate": group_fields["baseline_rate"],
                "rate_basis": group_fields["rate_basis"],
                "events_used": group_fields["events_used"],
                "expected_volume": None,
            }

        expected_volume = float(vol) * rest_adj
        opponent_adj = float(source_row.get("opponent_adjustment", 1.0) or 1.0)
        pace_adj = float(source_row.get("pace_adjustment", 1.0) or 1.0)
        home_adj = float(source_row.get("home_adjustment", 1.0) or 1.0)
        weather_adj = float(source_row.get("weather_adjustment", 1.0) or 1.0)
        projected_mean = (
            float(group_fields["blended_rate"])
            * expected_volume
            * opponent_adj
            * pace_adj
            * home_adj
            * weather_adj
        )
        return {
            "player": player,
            "market": market,
            "sport": sport,
            "stat_col": stat_col,
            "projected_mean": projected_mean,
            "recent_rate": group_fields["recent_rate"],
            "baseline_rate": group_fields["baseline_rate"],
            "rate_basis": group_fields["rate_basis"],
            "events_used": group_fields["events_used"],
            "expected_volume": expected_volume,
            "rest_adjustment": rest_adj,
        }

    def project_props(
        self,
        props: pd.DataFrame,
        history: pd.DataFrame,
        history_index: object | None = None,
    ) -> pd.DataFrame:
        if props is None or props.empty:
            return pd.DataFrame()

        index = history_index
        if index is None and history is not None and not history.empty:
            from sports_prop_edge.pipeline.history_index import HistoryIndex

            index = HistoryIndex(history)

        work = self._projection_group_keys(props)
        group_cols = ["_pk", "_sport", "_market", "_team"]
        unique_groups = work[group_cols].drop_duplicates()

        group_records: list[dict] = []
        for pk, sport_key, market, team in unique_groups.itertuples(index=False, name=None):
            resolved_sport = self._sport_for_market(str(market), str(sport_key))
            if index is not None:
                rows = index.enriched_slice(str(pk), str(sport_key), str(team) or None)
            else:
                rows = self._filtered_history(
                    history,
                    player=str(pk),
                    game_title=str(sport_key),
                    team=str(team) or None,
                )
                if not rows.empty and resolved_sport in {"NBA", "WNBA", "CBB"}:
                    rows = enrich_basketball_history_rows(rows)
                if not rows.empty and resolved_sport in {"KBO", "MLB"}:
                    rows = enrich_baseball_history_rows(rows)

            fields = self._group_rate_fields_from_rows(
                rows,
                market=str(market),
                sport=resolved_sport,
                player=str(pk),
            )
            group_records.append(
                {
                    "_pk": pk,
                    "_sport": sport_key,
                    "_market": market,
                    "_team": team,
                    "sport": fields.get("sport"),
                    "stat_col": fields.get("stat_col"),
                    "recent_rate": fields.get("recent_rate"),
                    "baseline_rate": fields.get("baseline_rate"),
                    "rate_basis": fields.get("rate_basis"),
                    "events_used": fields.get("events_used"),
                    "blended_rate": fields.get("blended_rate"),
                    "_default_volume": fields.get("_default_volume"),
                    "_last_game_date": fields.get("_last_game_date"),
                    "_rest_gap_factor": fields.get("_rest_gap_factor", 1.0),
                }
            )

        merged = work.merge(pd.DataFrame(group_records), on=group_cols, how="left")
        finalized = self._finalize_grouped_projections(merged)
        drop_cols = [
            "_pk",
            "_sport",
            "_market",
            "_team",
            "blended_rate",
            "_default_volume",
            "_last_game_date",
            "_rest_gap_factor",
        ]
        return finalized.drop(columns=[c for c in drop_cols if c in finalized.columns])
