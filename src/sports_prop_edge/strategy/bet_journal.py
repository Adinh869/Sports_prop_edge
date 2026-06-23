"""User bet journal for traditional sports picks (all leagues)."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from sports_prop_edge.data.prop_filters import PITCHER_MARKETS, is_modelable_prop
from sports_prop_edge.integrations.name_utils import normalize_lookup_name
from sports_prop_edge.strategy.sgp_math import OFFICIAL_PAIR_BREAKEVEN

OFFICIAL_TIERS = frozenset({"STRONG", "PLAYABLE"})
STRICT_OFFICIAL_TIERS = frozenset({"STRONG"})
RESEARCH_TIERS = frozenset({"RESEARCH"})

STAKE_TIERS = ("official", "paper", "research")
BET_FORMATS = ("single", "parlay_2leg")
STATUSES = ("pending", "graded")

JOURNAL_COLUMNS = [
    "bet_id",
    "date_added",
    "slate_date",
    "bet_format",
    "sport",
    "stake_tier",
    "card",
    "matchup",
    "player",
    "team",
    "opponent",
    "market",
    "line",
    "side",
    "player2",
    "team2",
    "opponent2",
    "market2",
    "line2",
    "side2",
    "pick_tier",
    "model_probability",
    "leg1_model_probability",
    "leg2_model_probability",
    "joint_probability_method",
    "joint_probability_assumes_independence",
    "dfs_edge",
    "projected_mean",
    "status",
    "result",
    "leg1_result",
    "leg2_result",
    "actual_stat_1",
    "actual_stat_2",
    "profit_units",
    "notes",
    "pick_key",
    "source_panel",
    "ledger_synced",
]


def journal_path(root: Path | None = None) -> Path:
    base = root or Path(__file__).resolve().parents[3]
    return base / "data" / "user_bet_journal.csv"


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _today_iso() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def _norm_side(value: Any) -> str:
    return str(value or "").strip().lower()


def _norm_result(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"WIN", "LOSS", "PUSH", "REFUND"}:
        return text
    if text in {"W", "HIT"}:
        return "WIN"
    if text in {"L", "MISS"}:
        return "LOSS"
    return ""


def _line_key(line: object) -> str:
    try:
        return f"{round(float(line), 2):.2f}"
    except (TypeError, ValueError):
        return str(line or "").strip()


def pick_key_single_leg(row: dict[str, Any] | pd.Series, slate_date: str | None = None) -> str:
    data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    slate = str(slate_date or data.get("slate_date") or _today_iso())
    return "||".join(
        [
            slate,
            str(data.get("sport") or data.get("game_title", "")).strip().upper(),
            str(data.get("player", "")).strip().lower(),
            str(data.get("market", "")).strip().lower(),
            _norm_side(data.get("side", "")),
            _line_key(data.get("line", "")),
        ]
    )


def pick_key_parlay(leg_a: dict[str, Any], leg_b: dict[str, Any], slate_date: str | None = None) -> str:
    slate = str(slate_date or _today_iso())
    keys = sorted([pick_key_single_leg(leg_a, slate), pick_key_single_leg(leg_b, slate)])
    return f"parlay||{keys[0]}||{keys[1]}"


JOURNAL_STRING_COLS = (
    "bet_format",
    "sport",
    "stake_tier",
    "card",
    "matchup",
    "player",
    "team",
    "opponent",
    "market",
    "side",
    "player2",
    "team2",
    "opponent2",
    "market2",
    "side2",
    "pick_tier",
    "joint_probability_method",
    "status",
    "result",
    "leg1_result",
    "leg2_result",
    "notes",
    "pick_key",
    "source_panel",
)


def _coerce_journal_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in JOURNAL_STRING_COLS:
        if col in out.columns:
            out[col] = out[col].fillna("").astype(object).map(
                lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)
            )
    for col in (
        "line",
        "line2",
        "model_probability",
        "leg1_model_probability",
        "leg2_model_probability",
        "dfs_edge",
        "projected_mean",
        "actual_stat_1",
        "actual_stat_2",
        "profit_units",
    ):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "joint_probability_assumes_independence" in out.columns:
        out["joint_probability_assumes_independence"] = (
            out["joint_probability_assumes_independence"]
            .astype(str)
            .str.lower()
            .isin({"1", "true", "yes"})
        )
    if "ledger_synced" in out.columns:
        out["ledger_synced"] = out["ledger_synced"].astype(str).str.lower().isin({"1", "true", "yes"})
    return out


def journal_has_pick_key(pick_key: str, root: Path | None = None) -> bool:
    if not pick_key:
        return False
    journal = load_journal(root)
    if journal.empty:
        return False
    return pick_key in journal["pick_key"].astype(str).tolist()


def load_journal(root: Path | None = None) -> pd.DataFrame:
    path = journal_path(root)
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    if df.empty and len(df.columns) == 0:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    for col in JOURNAL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return _coerce_journal_string_columns(df[JOURNAL_COLUMNS].copy())


class JournalBatchWriter:
    """Stage many journal rows in memory, then save the CSV once."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = root
        self.journal = load_journal(root)
        self._keys: set[str] = set()
        if not self.journal.empty and "pick_key" in self.journal.columns:
            self._keys = set(self.journal["pick_key"].astype(str).tolist())
        self._pending: list[dict[str, Any]] = []

    def has_pick_key(self, pick_key: str) -> bool:
        return bool(pick_key) and pick_key in self._keys

    def stage(self, entry: dict[str, Any]) -> None:
        pk = str(entry.get("pick_key", ""))
        if pk:
            self._keys.add(pk)
        self._pending.append(entry)

    def flush(self) -> None:
        if not self._pending:
            return
        self.journal = pd.concat(
            [self.journal, pd.DataFrame(self._pending)],
            ignore_index=True,
        )
        save_journal(self.journal, self.root)
        self._pending.clear()


def save_journal(df: pd.DataFrame, root: Path | None = None) -> None:
    path = journal_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in JOURNAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    out[JOURNAL_COLUMNS].to_csv(path, index=False)


def grade_leg(side: str, line: float, actual: float | None) -> str | None:
    if actual is None or pd.isna(actual):
        return None
    side_clean = _norm_side(side)
    try:
        line_f = float(line)
        actual_f = float(actual)
    except (TypeError, ValueError):
        return None
    if side_clean == "over":
        if actual_f > line_f:
            return "WIN"
        if actual_f < line_f:
            return "LOSS"
        return "PUSH"
    if side_clean == "under":
        if actual_f < line_f:
            return "WIN"
        if actual_f > line_f:
            return "LOSS"
        return "PUSH"
    return None


def grade_parlay_result(leg_results: list[str | None]) -> str | None:
    settled = [r for r in leg_results if r]
    if not settled:
        return None
    if any(r == "LOSS" for r in settled):
        return "LOSS"
    if all(r == "WIN" for r in settled):
        return "WIN"
    if any(r == "PUSH" for r in settled):
        return "PUSH"
    return None


def independent_joint_probability(leg1_p: float | None, leg2_p: float | None) -> float | None:
    if leg1_p is None or leg2_p is None:
        return None
    return float(leg1_p) * float(leg2_p)


def default_profit_units(result: str, *, bet_format: str = "single") -> float:
    res = _norm_result(result)
    if res == "WIN":
        return 2.0 if bet_format == "parlay_2leg" else 1.0
    if res == "LOSS":
        return -1.0
    if res == "PUSH":
        return 0.0
    return 0.0


def add_bet(
    *,
    stake_tier: str,
    bet_format: str,
    sport: str,
    card: str,
    matchup: str,
    player: str,
    team: str,
    opponent: str,
    market: str,
    line: float,
    side: str,
    player2: str = "",
    team2: str = "",
    opponent2: str = "",
    market2: str = "",
    line2: float | None = None,
    side2: str = "",
    pick_tier: str = "",
    model_probability: float | None = None,
    leg1_model_probability: float | None = None,
    leg2_model_probability: float | None = None,
    joint_probability_method: str = "",
    joint_probability_assumes_independence: bool = False,
    dfs_edge: float | None = None,
    projected_mean: float | None = None,
    slate_date: str | None = None,
    notes: str = "",
    pick_key: str = "",
    source_panel: str = "picks_tab",
    skip_duplicate: bool = True,
    root: Path | None = None,
    batch: JournalBatchWriter | None = None,
) -> dict[str, Any] | None:
    tier = str(stake_tier).strip().lower()
    if tier not in STAKE_TIERS:
        raise ValueError(f"stake_tier must be one of {STAKE_TIERS}")
    fmt = str(bet_format).strip().lower()
    if fmt not in BET_FORMATS:
        raise ValueError(f"bet_format must be one of {BET_FORMATS}")

    slate = slate_date or _today_iso()
    if not pick_key:
        if fmt == "parlay_2leg" and player2:
            pick_key = pick_key_parlay(
                {"player": player, "side": side, "line": line, "market": market, "sport": sport, "slate_date": slate},
                {"player": player2, "side": side2, "line": line2, "market": market2, "sport": sport, "slate_date": slate},
                slate,
            )
        else:
            pick_key = pick_key_single_leg(
                {"player": player, "side": side, "line": line, "market": market, "sport": sport, "slate_date": slate},
                slate,
            )
    if skip_duplicate:
        if batch is not None:
            if batch.has_pick_key(pick_key):
                return None
        elif journal_has_pick_key(pick_key, root):
            return None

    leg1_p = pd.to_numeric(leg1_model_probability, errors="coerce")
    leg2_p = pd.to_numeric(leg2_model_probability, errors="coerce")
    if pd.isna(leg1_p) and model_probability is not None and fmt == "single":
        leg1_p = pd.to_numeric(model_probability, errors="coerce")
    joint_p = pd.to_numeric(model_probability, errors="coerce")
    method = str(joint_probability_method or "").strip()
    assumes_indep = bool(joint_probability_assumes_independence)
    if fmt == "single":
        if pd.isna(joint_p) and pd.notna(leg1_p):
            joint_p = float(leg1_p)
        method = method or "single_leg"
        assumes_indep = False
    elif fmt == "parlay_2leg":
        if pd.isna(joint_p) and pd.notna(leg1_p) and pd.notna(leg2_p):
            joint_p = independent_joint_probability(float(leg1_p), float(leg2_p))
        method = method or "independent_product"
        assumes_indep = True

    entry = {
        "bet_id": uuid.uuid4().hex[:12],
        "date_added": _now_iso(),
        "slate_date": slate,
        "bet_format": fmt,
        "sport": str(sport).strip().upper(),
        "stake_tier": tier,
        "card": str(card).strip(),
        "matchup": str(matchup).strip(),
        "player": str(player).strip(),
        "team": str(team).strip(),
        "opponent": str(opponent).strip(),
        "market": str(market).strip(),
        "line": float(line or 0),
        "side": _norm_side(side),
        "player2": str(player2).strip(),
        "team2": str(team2).strip(),
        "opponent2": str(opponent2).strip(),
        "market2": str(market2).strip(),
        "line2": float(line2) if line2 is not None and str(line2).strip() != "" else pd.NA,
        "side2": _norm_side(side2),
        "pick_tier": str(pick_tier).strip().upper(),
        "model_probability": float(joint_p) if pd.notna(joint_p) else pd.NA,
        "leg1_model_probability": float(leg1_p) if pd.notna(leg1_p) else pd.NA,
        "leg2_model_probability": float(leg2_p) if pd.notna(leg2_p) else pd.NA,
        "joint_probability_method": method,
        "joint_probability_assumes_independence": assumes_indep,
        "dfs_edge": dfs_edge,
        "projected_mean": projected_mean,
        "status": "pending",
        "result": "",
        "leg1_result": "",
        "leg2_result": "",
        "actual_stat_1": pd.NA,
        "actual_stat_2": pd.NA,
        "profit_units": pd.NA,
        "notes": str(notes).strip(),
        "pick_key": pick_key,
        "source_panel": str(source_panel).strip(),
        "ledger_synced": False,
    }

    if batch is not None:
        batch.stage(entry)
        return entry

    journal = load_journal(root)
    journal = pd.concat([journal, pd.DataFrame([entry])], ignore_index=True)
    save_journal(journal, root)
    return entry


def add_pick_from_row(
    row: dict[str, Any] | pd.Series,
    *,
    stake_tier: str = "official",
    slate_date: str | None = None,
    notes: str = "",
    source_panel: str = "picks_tab",
    skip_duplicate: bool = True,
    root: Path | None = None,
    batch: JournalBatchWriter | None = None,
) -> dict[str, Any] | None:
    data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    sport = str(data.get("sport") or data.get("game_title", "")).strip().upper()
    prob = pd.to_numeric(data.get("model_probability"), errors="coerce")
    edge = pd.to_numeric(data.get("dfs_edge"), errors="coerce")
    proj = pd.to_numeric(data.get("projected_mean"), errors="coerce")
    card = f"{data.get('player', '')} {str(data.get('side', '')).upper()} {data.get('line', '')} {data.get('market', '')}"
    matchup = f"{data.get('team', '')} vs {data.get('opponent', '')}".strip(" vs ")

    leg_prob = float(prob) if pd.notna(prob) else None
    return add_bet(
        stake_tier=stake_tier,
        bet_format="single",
        sport=sport,
        card=card,
        matchup=matchup,
        player=str(data.get("player", "")),
        team=str(data.get("team", "")),
        opponent=str(data.get("opponent", "")),
        market=str(data.get("market", "")),
        line=float(data.get("line", 0) or 0),
        side=str(data.get("side", "")),
        pick_tier=str(data.get("pick_tier", "")),
        model_probability=leg_prob,
        leg1_model_probability=leg_prob,
        joint_probability_method="single_leg",
        joint_probability_assumes_independence=False,
        dfs_edge=float(edge) if pd.notna(edge) else None,
        projected_mean=float(proj) if pd.notna(proj) else None,
        slate_date=slate_date,
        notes=notes or f"Queued from pick sheet ({data.get('pick_tier', '')})",
        source_panel=source_panel,
        skip_duplicate=skip_duplicate,
        root=root,
        batch=batch,
    )


def add_sgp_row(
    row: dict[str, Any] | pd.Series,
    *,
    stake_tier: str = "official",
    slate_date: str | None = None,
    source_panel: str = "sgp_tab",
    root: Path | None = None,
    batch: JournalBatchWriter | None = None,
) -> dict[str, Any] | None:
    data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    prob = pd.to_numeric(data.get("pair_hit_probability"), errors="coerce")
    leg1_p = pd.to_numeric(data.get("leg1_model_probability"), errors="coerce")
    leg2_p = pd.to_numeric(data.get("leg2_model_probability"), errors="coerce")
    edge = pd.to_numeric(data.get("avg_edge"), errors="coerce")
    return add_bet(
        stake_tier=stake_tier,
        bet_format="parlay_2leg",
        sport=str(data.get("sport", "")).strip().upper(),
        card=str(data.get("card", "")),
        matchup=str(data.get("matchup", "")),
        player=str(data.get("leg1_player", "")),
        team=str(data.get("leg1_team", "")),
        opponent=str(data.get("leg1_opponent", "")),
        market=str(data.get("leg1_market", "")),
        line=float(data.get("leg1_line", 0) or 0),
        side=str(data.get("leg1_side", "")),
        player2=str(data.get("leg2_player", "")),
        team2=str(data.get("leg2_team", "")),
        opponent2=str(data.get("leg2_opponent", "")),
        market2=str(data.get("leg2_market", "")),
        line2=float(data.get("leg2_line", 0) or 0),
        side2=str(data.get("leg2_side", "")),
        pick_tier=f"{data.get('leg1_tier', '')}/{data.get('leg2_tier', '')}",
        model_probability=float(prob) if pd.notna(prob) else None,
        leg1_model_probability=float(leg1_p) if pd.notna(leg1_p) else None,
        leg2_model_probability=float(leg2_p) if pd.notna(leg2_p) else None,
        joint_probability_method="independent_product",
        joint_probability_assumes_independence=True,
        dfs_edge=float(edge) if pd.notna(edge) else None,
        slate_date=slate_date,
        notes="Queued from SGP pairs",
        source_panel=source_panel,
        root=root,
        batch=batch,
    )


def add_power_card_row(
    card_row: dict[str, Any] | pd.Series,
    legs_df: pd.DataFrame,
    *,
    stake_tier: str = "official",
    slate_date: str | None = None,
    source_panel: str = "power_play_tab",
    root: Path | None = None,
    batch: JournalBatchWriter | None = None,
) -> dict[str, Any] | None:
    data = card_row.to_dict() if hasattr(card_row, "to_dict") else dict(card_row)
    indexes = data.get("leg_indexes") or []
    if len(indexes) < 2:
        return None
    leg_a = legs_df.iloc[int(indexes[0])]
    leg_b = legs_df.iloc[int(indexes[1])]
    sports = {str(leg_a.get("game_title", "")).upper(), str(leg_b.get("game_title", "")).upper()}
    sport = sports.pop() if len(sports) == 1 else "MULTI"
    prob = pd.to_numeric(data.get("power_hit_probability"), errors="coerce")
    leg1_p = pd.to_numeric(leg_a.get("model_probability"), errors="coerce")
    leg2_p = pd.to_numeric(leg_b.get("model_probability"), errors="coerce")
    edge = pd.to_numeric(data.get("avg_edge"), errors="coerce")
    return add_bet(
        stake_tier=stake_tier,
        bet_format="parlay_2leg",
        sport=sport,
        card=str(data.get("card", "")),
        matchup=" | ".join(
            sorted(
                {
                    f"{leg_a.get('team', '')} vs {leg_a.get('opponent', '')}".strip(" vs "),
                    f"{leg_b.get('team', '')} vs {leg_b.get('opponent', '')}".strip(" vs "),
                }
            )
        ),
        player=str(leg_a.get("player", "")),
        team=str(leg_a.get("team", "")),
        opponent=str(leg_a.get("opponent", "")),
        market=str(leg_a.get("market", "")),
        line=float(leg_a.get("line", 0) or 0),
        side=str(leg_a.get("side", "")),
        player2=str(leg_b.get("player", "")),
        team2=str(leg_b.get("team", "")),
        opponent2=str(leg_b.get("opponent", "")),
        market2=str(leg_b.get("market", "")),
        line2=float(leg_b.get("line", 0) or 0),
        side2=str(leg_b.get("side", "")),
        pick_tier="POWER",
        model_probability=float(prob) if pd.notna(prob) else None,
        leg1_model_probability=float(leg1_p) if pd.notna(leg1_p) else None,
        leg2_model_probability=float(leg2_p) if pd.notna(leg2_p) else None,
        joint_probability_method="independent_product",
        joint_probability_assumes_independence=True,
        dfs_edge=float(edge) if pd.notna(edge) else None,
        slate_date=slate_date,
        notes="Queued from power play card",
        source_panel=source_panel,
        root=root,
        batch=batch,
    )


def queue_pick_sheet_rows(
    pick_sheet: pd.DataFrame,
    *,
    stake_tier: str = "official",
    include_research: bool = False,
    tiers: frozenset[str] | set[str] | None = None,
    source_panel: str = "pick_sheet",
    root: Path | None = None,
    batch: JournalBatchWriter | None = None,
) -> tuple[int, int]:
    if pick_sheet is None or pick_sheet.empty:
        return 0, 0

    if tiers is None:
        allowed = set(OFFICIAL_TIERS)
        if include_research:
            allowed |= RESEARCH_TIERS
    else:
        allowed = {str(t).upper() for t in tiers}

    work = pick_sheet.copy()
    if "pick_tier" in work.columns:
        work = work[work["pick_tier"].astype(str).str.upper().isin(allowed)].copy()

    added = skipped = 0
    for _, row in work.iterrows():
        if add_pick_from_row(
            row,
            stake_tier=stake_tier,
            source_panel=source_panel,
            root=root,
            batch=batch,
        ):
            added += 1
        else:
            skipped += 1
    return added, skipped


def queue_pick_sheet_selection(
    pick_sheet: pd.DataFrame,
    selected_labels: list[str],
    *,
    stake_tier: str = "paper",
    source_panel: str = "paper_manual",
    root: Path | None = None,
) -> tuple[int, int]:
    if pick_sheet is None or pick_sheet.empty or not selected_labels:
        return 0, 0
    selected = set(selected_labels)
    added = skipped = 0
    for _, row in pick_sheet.iterrows():
        if format_single_pick_label(row) not in selected:
            continue
        if add_pick_from_row(row, stake_tier=stake_tier, source_panel=source_panel, root=root):
            added += 1
        else:
            skipped += 1
    return added, skipped


def is_official_sgp_row(row: pd.Series) -> bool:
    t1 = str(row.get("leg1_tier", "")).upper()
    t2 = str(row.get("leg2_tier", "")).upper()
    return t1 in OFFICIAL_TIERS and t2 in OFFICIAL_TIERS


def _official_baseball_sgp_ok(row: pd.Series) -> bool:
    """MLB/KBO official SGP: pitcher + hitter, pitcher leg must be STRONG."""
    sport = str(row.get("sport", "")).upper()
    if sport not in {"MLB", "KBO"}:
        return True
    if int(row.get("pair_priority", 0) or 0) < 1:
        return False
    m1 = str(row.get("leg1_market", "")).lower()
    m2 = str(row.get("leg2_market", "")).lower()
    t1 = str(row.get("leg1_tier", "")).upper()
    t2 = str(row.get("leg2_tier", "")).upper()
    if m1 in PITCHER_MARKETS:
        return t1 == "STRONG"
    if m2 in PITCHER_MARKETS:
        return t2 == "STRONG"
    return False


def filter_official_singles(pick_sheet: pd.DataFrame) -> pd.DataFrame:
    """STRONG singles only, above PrizePicks-style breakeven bar (all sports)."""
    if pick_sheet is None or pick_sheet.empty:
        return pd.DataFrame()
    from sports_prop_edge.strategy.pick_workflow import pick_best_market_per_player

    out = pick_sheet[pick_sheet["pick_tier"].astype(str).str.upper().isin(STRICT_OFFICIAL_TIERS)].copy()
    if "dfs_edge" in out.columns:
        out = out[out["dfs_edge"].fillna(-1) >= OFFICIAL_SINGLE_MIN_EDGE]
    if "model_probability" in out.columns:
        out = out[out["model_probability"].fillna(0) >= OFFICIAL_SINGLE_MIN_PROB]
    out = pick_best_market_per_player(out)
    sort_cols = [c for c in ("dfs_edge", "model_probability") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False)
    return out.head(AUTO_QUEUE_MAX_SINGLES).reset_index(drop=True)


def _official_wnba_sgp_ok(row: pd.Series) -> bool:
    """WNBA official SGP: both legs STRONG (stricter than other sports)."""
    if str(row.get("sport", "")).upper() != "WNBA":
        return True
    t1 = str(row.get("leg1_tier", "")).upper()
    t2 = str(row.get("leg2_tier", "")).upper()
    return t1 == "STRONG" and t2 == "STRONG"


def filter_official_sgp_pairs(sgp_df: pd.DataFrame) -> pd.DataFrame:
    if sgp_df is None or sgp_df.empty:
        return pd.DataFrame()
    out = sgp_df[sgp_df.apply(is_official_sgp_row, axis=1)].copy()
    if "same_team" in out.columns:
        out = out[~out["same_team"].fillna(False)]
    if "min_edge" in out.columns:
        out = out[out["min_edge"].fillna(-1) >= OFFICIAL_SGP_MIN_MIN_EDGE]
    if "leg1_model_probability" in out.columns and "leg2_model_probability" in out.columns:
        out = out[
            (out["leg1_model_probability"].fillna(0) >= OFFICIAL_POWER_MIN_PROB)
            & (out["leg2_model_probability"].fillna(0) >= OFFICIAL_POWER_MIN_PROB)
        ]
    strong_leg = out["leg1_tier"].astype(str).str.upper().eq("STRONG") | out[
        "leg2_tier"
    ].astype(str).str.upper().eq("STRONG")
    out = out[strong_leg]
    if not out.empty:
        out = out[out.apply(_official_baseball_sgp_ok, axis=1)]
        out = out[out.apply(_official_wnba_sgp_ok, axis=1)]
    if "pair_hit_probability" in out.columns:
        out = out[
            pd.to_numeric(out["pair_hit_probability"], errors="coerce").fillna(0)
            >= OFFICIAL_PAIR_BREAKEVEN
        ]
    return out.reset_index(drop=True)


def paper_parlay_discipline_warnings(sgp_df: pd.DataFrame, selected_cards: list[str]) -> list[str]:
    """Warn when a paper parlay includes PASS/RESEARCH legs or mixed quality."""
    if sgp_df is None or sgp_df.empty or not selected_cards:
        return []
    chosen = set(str(c) for c in selected_cards)
    warnings: list[str] = []
    for _, row in sgp_df.iterrows():
        if str(row.get("card", "")) not in chosen:
            continue
        for n in (1, 2):
            tier = str(row.get(f"leg{n}_tier", "")).upper()
            player = row.get(f"leg{n}_player", "")
            if tier in {"PASS", ""}:
                warnings.append(f"{player}: tier PASS — model would not play this leg")
            elif tier == "RESEARCH":
                warnings.append(f"{player}: RESEARCH only — thin sample / below auto-play bar")
        t1 = str(row.get("leg1_tier", "")).upper()
        t2 = str(row.get("leg2_tier", "")).upper()
        if "STRONG" in {t1, t2} and "PASS" in {t1, t2}:
            warnings.append("STRONG + PASS pairing — your journal losses often look like this")
    return warnings


def filter_official_power_cards(cards_df: pd.DataFrame, legs_df: pd.DataFrame) -> pd.DataFrame:
    if cards_df is None or cards_df.empty or legs_df is None or legs_df.empty:
        return pd.DataFrame()
    out = cards_df[cards_df.apply(_card_is_official, axis=1, legs_df=legs_df)].copy()
    if out.empty:
        return out
    both_strong = out.apply(
        lambda card: _card_tiers(card, legs_df) == STRICT_OFFICIAL_TIERS,
        axis=1,
    )
    out = out[both_strong]
    if "min_edge" in out.columns:
        out = out[out["min_edge"].fillna(-1) >= OFFICIAL_POWER_MIN_EDGE]
    if "min_probability" in out.columns:
        out = out[out["min_probability"].fillna(0) >= OFFICIAL_POWER_MIN_PROB]
    sort_col = "card_ev_per_dollar" if "card_ev_per_dollar" in out.columns else "avg_edge"
    if sort_col in out.columns:
        out = out.sort_values(sort_col, ascending=False)
    return out.head(AUTO_QUEUE_MAX_POWER).reset_index(drop=True)


def _sgp_is_research(row: pd.Series) -> bool:
    t1 = str(row.get("leg1_tier", "")).upper()
    t2 = str(row.get("leg2_tier", "")).upper()
    return t1 in RESEARCH_TIERS or t2 in RESEARCH_TIERS


def format_single_pick_label(row: pd.Series | dict) -> str:
    data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    sport = str(data.get("sport") or data.get("game_title", "")).strip().upper()
    return (
        f"{sport} | {data.get('player', '')} {str(data.get('side', '')).upper()} "
        f"{data.get('line', '')} {data.get('market', '')} [{data.get('pick_tier', '')}]"
    )


def _card_tiers(card_row: pd.Series, legs_df: pd.DataFrame) -> set[str]:
    indexes = card_row.get("leg_indexes") or []
    tiers: set[str] = set()
    for idx in indexes:
        tiers.add(str(legs_df.iloc[int(idx)].get("pick_tier", "")).upper())
    return tiers


def _card_is_official(card_row: pd.Series, legs_df: pd.DataFrame) -> bool:
    tiers = _card_tiers(card_row, legs_df)
    return bool(tiers) and tiers <= set(OFFICIAL_TIERS)


def _card_is_research(card_row: pd.Series, legs_df: pd.DataFrame) -> bool:
    return bool(_card_tiers(card_row, legs_df) & RESEARCH_TIERS)


sgp_has_research_leg = _sgp_is_research
card_has_research_leg = _card_is_research


def prop_leg_key(
    sport: Any,
    player: Any,
    market: Any,
    line: Any,
    side: Any,
) -> tuple[str, str, str, float, str]:
    return (
        str(sport or "").strip().upper(),
        normalize_lookup_name(str(player or "")),
        str(market or "").strip().lower(),
        round(float(line), 2),
        _norm_side(side),
    )


def build_board_leg_keys(props_board: pd.DataFrame) -> frozenset[tuple[str, str, str, float, str]]:
    """Valid (sport, player, market, line, side) legs from the filtered props board."""
    if props_board is None or props_board.empty:
        return frozenset()
    keys: set[tuple[str, str, str, float, str]] = set()
    sport_col = (
        props_board["game_title"].astype(str)
        if "game_title" in props_board.columns
        else pd.Series("", index=props_board.index)
    )
    stat_col = (
        props_board["stat_type"].astype(str)
        if "stat_type" in props_board.columns
        else pd.Series("", index=props_board.index)
    )
    for idx, row in props_board.iterrows():
        sport = sport_col.loc[idx]
        stat_type = stat_col.loc[idx]
        if not is_modelable_prop(stat_type, row.get("market", ""), sport):
            continue
        keys.add(
            prop_leg_key(
                sport,
                row.get("player", ""),
                row.get("market", ""),
                row.get("line", 0),
                row.get("side", ""),
            )
        )
    return frozenset(keys)


def journal_bet_on_board(
    row: pd.Series | dict[str, Any],
    board_keys: frozenset[tuple[str, str, str, float, str]],
) -> bool:
    if not board_keys:
        return True
    data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    fmt = str(data.get("bet_format", "single")).strip().lower()
    k1 = prop_leg_key(
        data.get("sport", ""),
        data.get("player", ""),
        data.get("market", ""),
        data.get("line", 0),
        data.get("side", ""),
    )
    if fmt != "parlay_2leg":
        return k1 in board_keys
    k2 = prop_leg_key(
        data.get("sport", ""),
        data.get("player2", ""),
        data.get("market2", ""),
        data.get("line2", 0),
        data.get("side2", ""),
    )
    return k1 in board_keys and k2 in board_keys


def find_off_board_journal_bets(
    journal: pd.DataFrame,
    board_keys: frozenset[tuple[str, str, str, float, str]],
    *,
    stake_tier: str | None = "official",
    source_panel: str | None = "auto_official",
    status: str = "pending",
) -> pd.DataFrame:
    """Journal rows that do not match any leg on the current modelable props board."""
    if journal is None or journal.empty or not board_keys:
        return pd.DataFrame(columns=journal.columns if journal is not None else JOURNAL_COLUMNS)
    work = journal.copy()
    if status:
        work = work[work["status"].astype(str).str.lower() == status.lower()]
    if stake_tier:
        work = work[work["stake_tier"].astype(str).str.lower() == stake_tier.lower()]
    if source_panel:
        work = work[work["source_panel"].astype(str) == source_panel]
    if work.empty:
        return work
    on_board = work.apply(lambda row: journal_bet_on_board(row, board_keys), axis=1)
    return work[~on_board.fillna(False)].reset_index(drop=True)


AUTO_QUEUE_MAX_SINGLES = 6
AUTO_QUEUE_MAX_SGPS = 3
AUTO_QUEUE_MAX_POWER = 2
OFFICIAL_SGP_MIN_MIN_EDGE = 0.03
OFFICIAL_SINGLE_MIN_EDGE = 0.05
OFFICIAL_SINGLE_MIN_PROB = 0.60
OFFICIAL_POWER_MIN_EDGE = 0.04
OFFICIAL_POWER_MIN_PROB = 0.58


MANUAL_JOURNAL_SOURCE_PANELS = frozenset({"paper_manual", "paper_sgp"})


def delete_pending_auto_official_bets(root: Path | None = None) -> int:
    """Remove pending official rows logged by the app (not paper slips you chose manually)."""
    journal = load_journal(root)
    if journal.empty:
        return 0
    pending = journal["status"].astype(str).str.lower() == "pending"
    official = journal["stake_tier"].astype(str).str.lower() == "official"
    if "source_panel" in journal.columns:
        manual = journal["source_panel"].astype(str).isin(MANUAL_JOURNAL_SOURCE_PANELS)
        mask = pending & official & ~manual
    else:
        mask = pending & official
    ids = journal.loc[mask, "bet_id"].astype(str).tolist()
    return delete_bets(ids, root=root)


def purge_off_board_auto_bets(
    root: Path | None = None,
    *,
    props_board: pd.DataFrame | None = None,
    props_path: Path | None = None,
) -> int:
    """Delete pending auto-official journal rows not on the current props board."""
    base = root or Path(__file__).resolve().parents[3]
    board = props_board
    if board is None:
        from sports_prop_edge.data.loaders import load_props

        path = props_path or (base / "data" / "props" / "tonight_props.csv")
        if not path.exists():
            return 0
        board = load_props(path)
    keys = build_board_leg_keys(board)
    if not keys:
        return 0
    stale = find_off_board_journal_bets(load_journal(root), keys)
    if stale.empty:
        return 0
    return delete_bets(stale["bet_id"].astype(str).tolist(), root=root)


def filter_pick_sheet_to_board(
    pick_sheet: pd.DataFrame,
    board_keys: frozenset[tuple[str, str, str, float, str]],
) -> pd.DataFrame:
    if pick_sheet is None or pick_sheet.empty or not board_keys:
        return pick_sheet
    keep: list[bool] = []
    for _, row in pick_sheet.iterrows():
        sport = row.get("game_title", row.get("sport", ""))
        key = prop_leg_key(sport, row.get("player"), row.get("market"), row.get("line"), row.get("side"))
        keep.append(key in board_keys)
    return pick_sheet[pd.Series(keep, index=pick_sheet.index)].reset_index(drop=True)


def filter_sgp_pairs_to_board(
    sgp_pairs: pd.DataFrame,
    board_keys: frozenset[tuple[str, str, str, float, str]],
) -> pd.DataFrame:
    if sgp_pairs is None or sgp_pairs.empty or not board_keys:
        return sgp_pairs
    keep: list[bool] = []
    for _, row in sgp_pairs.iterrows():
        sport = row.get("sport", row.get("game_title", ""))
        k1 = prop_leg_key(
            sport,
            row.get("leg1_player"),
            row.get("leg1_market"),
            row.get("leg1_line"),
            row.get("leg1_side"),
        )
        k2 = prop_leg_key(
            sport,
            row.get("leg2_player"),
            row.get("leg2_market"),
            row.get("leg2_line"),
            row.get("leg2_side"),
        )
        keep.append(k1 in board_keys and k2 in board_keys)
    return sgp_pairs[pd.Series(keep, index=sgp_pairs.index)].reset_index(drop=True)


def board_fingerprint(scored_best: pd.DataFrame) -> str:
    if scored_best is None or scored_best.empty:
        return ""
    cols = [c for c in ["game_title", "player", "market", "line", "side", "pick_tier"] if c in scored_best.columns]
    blob = scored_best[cols].astype(str).sort_values(cols).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _cap_pick_sheet(sheet: pd.DataFrame, limit: int) -> pd.DataFrame:
    if sheet is None or sheet.empty or limit <= 0:
        return sheet
    work = sheet.copy()
    tier_rank = {"STRONG": 0, "PLAYABLE": 1, "RESEARCH": 2, "PASS": 3}
    if "pick_tier" in work.columns:
        work["_tier_rank"] = work["pick_tier"].astype(str).str.upper().map(tier_rank).fillna(9)
    else:
        work["_tier_rank"] = 9
    sort_cols = ["_tier_rank"]
    if "dfs_edge" in work.columns:
        sort_cols.append("dfs_edge")
    if "model_probability" in work.columns:
        sort_cols.append("model_probability")
    ascending = [True] + [False] * (len(sort_cols) - 1)
    return work.sort_values(sort_cols, ascending=ascending).head(limit).drop(columns=["_tier_rank"])


def _cap_sgp_pairs(sgp: pd.DataFrame, limit: int) -> pd.DataFrame:
    if sgp is None or sgp.empty or limit <= 0:
        return sgp
    sort_cols = [c for c in ("avg_edge", "pair_hit_probability") if c in sgp.columns]
    if not sort_cols:
        return sgp.head(limit)
    return sgp.sort_values(sort_cols, ascending=False).head(limit)


def _cap_power_cards(cards: pd.DataFrame, limit: int) -> pd.DataFrame:
    if cards is None or cards.empty or limit <= 0:
        return cards
    sort_col = "power_hit_probability" if "power_hit_probability" in cards.columns else "avg_edge"
    if sort_col not in cards.columns:
        return cards.head(limit)
    return cards.sort_values(sort_col, ascending=False).head(limit)


def auto_queue_board_to_journal(
    *,
    pick_sheet: pd.DataFrame,
    sgp_pairs: pd.DataFrame,
    power_cards: pd.DataFrame,
    power_pool: pd.DataFrame,
    queue_official: bool,
    queue_paper: bool,
    root: Path | None = None,
    props_board: pd.DataFrame | None = None,
) -> dict[str, tuple[int, int]]:
    """Queue official (STRONG/PLAYABLE) and paper (RESEARCH) plays; skips duplicates."""
    from sports_prop_edge.strategy.pick_workflow import pick_best_market_per_player

    totals: dict[str, tuple[int, int]] = {}
    batch = JournalBatchWriter(root) if queue_official or queue_paper else None
    board_keys = build_board_leg_keys(props_board) if props_board is not None else frozenset()
    if not pick_sheet.empty:
        pick_sheet = pick_best_market_per_player(pick_sheet.copy())
    singles = filter_pick_sheet_to_board(pick_sheet, board_keys) if not pick_sheet.empty else pick_sheet
    sgps = filter_sgp_pairs_to_board(sgp_pairs, board_keys) if sgp_pairs is not None and not sgp_pairs.empty else sgp_pairs

    if queue_official:
        singles = filter_official_singles(singles)
        sgps = _cap_sgp_pairs(sgps, AUTO_QUEUE_MAX_SGPS)

    if queue_official and not singles.empty:
        totals["official_singles"] = queue_pick_sheet_rows(
            singles,
            stake_tier="official",
            tiers=STRICT_OFFICIAL_TIERS,
            source_panel="auto_official",
            root=root,
            batch=batch,
        )

    if queue_paper and not singles.empty:
        totals["paper_singles"] = queue_pick_sheet_rows(
            singles,
            stake_tier="paper",
            tiers=RESEARCH_TIERS,
            source_panel="auto_paper",
            root=root,
            batch=batch,
        )

    if queue_official and sgps is not None and not sgps.empty:
        official_sgp = filter_official_sgp_pairs(sgps)
        cards = official_sgp["card"].astype(str).tolist()
        totals["official_sgp"] = queue_sgp_rows(
            official_sgp,
            cards,
            stake_tier="official",
            source_panel="auto_official",
            root=root,
            batch=batch,
        ) if cards else (0, 0)

    if queue_paper and sgps is not None and not sgps.empty:
        research_sgp = sgps[sgps.apply(_sgp_is_research, axis=1)]
        cards = research_sgp["card"].astype(str).tolist()
        totals["paper_sgp"] = queue_sgp_rows(
            sgps,
            cards,
            stake_tier="paper",
            source_panel="auto_paper",
            root=root,
            batch=batch,
        ) if cards else (0, 0)

    if queue_official and power_cards is not None and not power_cards.empty and not power_pool.empty:
        official_cards = filter_official_power_cards(power_cards, power_pool)
        if board_keys:
            official_cards = official_cards[
                official_cards.apply(
                    lambda card: all(
                        prop_leg_key(
                            power_pool.iloc[int(i)].get("game_title", ""),
                            power_pool.iloc[int(i)].get("player", ""),
                            power_pool.iloc[int(i)].get("market", ""),
                            power_pool.iloc[int(i)].get("line", 0),
                            power_pool.iloc[int(i)].get("side", ""),
                        )
                        in board_keys
                        for i in (card.get("leg_indexes") or [])
                    ),
                    axis=1,
                )
            ]
        cards = official_cards["card"].astype(str).tolist()
        totals["official_power"] = queue_power_card_rows(
            power_cards,
            power_pool,
            cards,
            stake_tier="official",
            source_panel="auto_official",
            root=root,
            batch=batch,
        ) if cards else (0, 0)

    if queue_paper and power_cards is not None and not power_cards.empty and not power_pool.empty:
        research_cards = power_cards[power_cards.apply(_card_is_research, axis=1, legs_df=power_pool)]
        cards = research_cards["card"].astype(str).tolist()
        totals["paper_power"] = queue_power_card_rows(
            power_cards,
            power_pool,
            cards,
            stake_tier="paper",
            root=root,
            batch=batch,
        ) if cards else (0, 0)

    if batch is not None:
        batch.flush()

    return totals


def format_auto_queue_summary(totals: dict[str, tuple[int, int]]) -> str:
    if not totals:
        return ""
    bits: list[str] = []
    for key, (added, skipped) in totals.items():
        if added or skipped:
            label = key.replace("_", " ")
            bits.append(f"{label}: +{added} new, {skipped} dup")
    return " · ".join(bits)


def queue_sgp_rows(
    sgp_df: pd.DataFrame,
    selected_cards: list[str],
    *,
    stake_tier: str = "official",
    source_panel: str = "sgp_tab",
    root: Path | None = None,
    batch: JournalBatchWriter | None = None,
) -> tuple[int, int]:
    if sgp_df is None or sgp_df.empty or not selected_cards:
        return 0, 0
    added = skipped = 0
    for _, row in sgp_df.iterrows():
        if str(row.get("card", "")) not in selected_cards:
            continue
        if add_sgp_row(
            row,
            stake_tier=stake_tier,
            source_panel=source_panel,
            root=root,
            batch=batch,
        ):
            added += 1
        else:
            skipped += 1
    return added, skipped


def queue_power_card_rows(
    cards_df: pd.DataFrame,
    legs_df: pd.DataFrame,
    selected_cards: list[str],
    *,
    stake_tier: str = "official",
    source_panel: str = "power_play_tab",
    root: Path | None = None,
    batch: JournalBatchWriter | None = None,
) -> tuple[int, int]:
    if cards_df is None or cards_df.empty or not selected_cards:
        return 0, 0
    added = skipped = 0
    for _, row in cards_df.iterrows():
        if str(row.get("card", "")) not in selected_cards:
            continue
        if add_power_card_row(
            row,
            legs_df,
            stake_tier=stake_tier,
            source_panel=source_panel,
            root=root,
            batch=batch,
        ):
            added += 1
        else:
            skipped += 1
    return added, skipped


def format_journal_label(row: pd.Series | dict) -> str:
    data = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    leg2 = (
        f" + {data.get('player2', '')} {str(data.get('side2', '')).upper()} "
        f"{data.get('line2', '')} {data.get('market2', '')}"
        if str(data.get("player2", "")).strip()
        else ""
    )
    return (
        f"{data.get('bet_id', '')} | {str(data.get('status', '')).upper()} | "
        f"{data.get('stake_tier', '')} | {data.get('sport', '')} | "
        f"{data.get('player', '')} {str(data.get('side', '')).upper()} "
        f"{data.get('line', '')} {data.get('market', '')}{leg2}"
    )


def delete_bets(bet_ids: list[str] | str, root: Path | None = None) -> int:
    """Remove bets from the journal and drop any matching probability-ledger rows."""
    if isinstance(bet_ids, str):
        ids = {bet_ids.strip()} if bet_ids.strip() else set()
    else:
        ids = {str(x).strip() for x in bet_ids if str(x).strip()}
    if not ids:
        return 0

    journal = load_journal(root)
    if journal.empty:
        return 0
    before = len(journal)
    journal = journal[~journal["bet_id"].astype(str).isin(ids)].reset_index(drop=True)
    removed = before - len(journal)
    if removed:
        save_journal(journal, root)
        from sports_prop_edge.strategy.probability_ledger import remove_ledger_entries_for_bets

        remove_ledger_entries_for_bets(ids, root=root)
    return removed


def grade_bet(
    bet_id: str,
    *,
    result: str | None = None,
    actual_stat_1: float | None = None,
    actual_stat_2: float | None = None,
    profit_units: float | None = None,
    notes: str | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    journal = load_journal(root)
    if journal.empty:
        raise ValueError("No bets in journal.")
    mask = journal["bet_id"].astype(str) == str(bet_id)
    if not mask.any():
        raise ValueError(f"Unknown bet_id: {bet_id}")

    idx = journal.index[mask][0]
    row = journal.loc[idx].to_dict()
    fmt = str(row.get("bet_format", "single")).lower()

    leg1 = grade_leg(row.get("side", ""), row.get("line", 0), actual_stat_1)
    leg2 = None
    if fmt == "parlay_2leg" and str(row.get("player2", "")).strip():
        leg2 = grade_leg(row.get("side2", ""), row.get("line2", 0), actual_stat_2)

    auto_result = grade_parlay_result([leg1, leg2]) if fmt == "parlay_2leg" else leg1
    final_result = _norm_result(result) or auto_result
    if not final_result:
        raise ValueError("Could not determine result. Enter WIN/LOSS or provide actual stat(s).")

    if profit_units is None:
        profit_units = default_profit_units(final_result, bet_format=fmt)

    journal.at[idx, "status"] = "graded"
    journal.at[idx, "result"] = final_result
    journal.at[idx, "leg1_result"] = leg1 or ""
    journal.at[idx, "leg2_result"] = leg2 or ""
    journal.at[idx, "actual_stat_1"] = actual_stat_1 if actual_stat_1 is not None else pd.NA
    journal.at[idx, "actual_stat_2"] = actual_stat_2 if actual_stat_2 is not None else pd.NA
    journal.at[idx, "profit_units"] = profit_units
    if notes is not None:
        journal.at[idx, "notes"] = str(notes).strip()

    save_journal(journal, root)
    updated = journal.loc[idx].to_dict()
    from sports_prop_edge.strategy.probability_ledger import sync_bet_to_ledger

    sync_bet_to_ledger(updated, root=root)
    return updated


def summarize_journal(journal: pd.DataFrame) -> dict[str, int | float]:
    if journal.empty:
        return {"total": 0, "pending": 0, "graded": 0, "wins": 0, "losses": 0, "profit_units": 0.0}
    pending = int((journal["status"].astype(str).str.lower() == "pending").sum())
    graded = journal[journal["status"].astype(str).str.lower() == "graded"]
    wins = int((graded["result"].astype(str).str.upper() == "WIN").sum())
    losses = int((graded["result"].astype(str).str.upper() == "LOSS").sum())
    profit = pd.to_numeric(graded.get("profit_units"), errors="coerce").fillna(0).sum()
    return {
        "total": int(len(journal)),
        "pending": pending,
        "graded": int(len(graded)),
        "wins": wins,
        "losses": losses,
        "profit_units": float(profit),
    }


def summarize_journal_breakdown(journal: pd.DataFrame) -> pd.DataFrame:
    """P&L by sport and bet format for graded bets."""
    if journal is None or journal.empty:
        return pd.DataFrame(columns=["sport", "bet_format", "bets", "wins", "losses", "profit_units"])
    graded = journal[journal["status"].astype(str).str.lower() == "graded"].copy()
    if graded.empty:
        return pd.DataFrame(columns=["sport", "bet_format", "bets", "wins", "losses", "profit_units"])
    graded["profit_units"] = pd.to_numeric(graded["profit_units"], errors="coerce").fillna(0)
    graded["win"] = graded["result"].astype(str).str.upper() == "WIN"
    grouped = (
        graded.groupby(["sport", "bet_format"], dropna=False)
        .agg(
            bets=("win", "count"),
            wins=("win", "sum"),
            losses=("win", lambda s: int((~s).sum())),
            profit_units=("profit_units", "sum"),
        )
        .reset_index()
    )
    return grouped.sort_values("profit_units", ascending=False)
