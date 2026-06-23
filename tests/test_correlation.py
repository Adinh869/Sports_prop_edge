"""Tests for parlay-only correlation factor model."""

from __future__ import annotations

import pandas as pd
import pytest

from sports_prop_edge.strategy.card_builder import CardRules, build_cards
from sports_prop_edge.strategy.correlation import (
    adjusted_pair_probability,
    build_correlation_context,
    card_joint_correlation_factor,
    pairwise_correlation_factor,
    structural_pair_correlation_factor,
)
from sports_prop_edge.strategy.payouts import default_profiles


def _leg(**kwargs) -> pd.Series:
    base = {
        "game_title": "NBA",
        "event_time": "2026-06-10",
        "player": "p1",
        "team": "bos",
        "opponent": "nyk",
        "market": "points",
        "line": 24.5,
        "side": "over",
        "model_probability": 0.60,
        "dfs_edge": 0.04,
    }
    base.update(kwargs)
    return pd.Series(base)


def test_scoring_module_does_not_use_correlation():
    from pathlib import Path

    scoring_path = Path(__file__).resolve().parents[1] / "src" / "sports_prop_edge" / "strategy" / "scoring.py"
    source = scoring_path.read_text(encoding="utf-8")
    assert "correlation" not in source


def test_same_game_basketball_market_factor_below_one():
    a = _leg(player="a", market="points", side="over")
    b = _leg(player="b", market="assists", side="over", team="nyk", opponent="bos")
    ctx = build_correlation_context(a, b)
    assert ctx.same_game
    assert not ctx.same_team
    rho = structural_pair_correlation_factor(a, b, ctx=ctx)
    assert 0.75 < rho < 1.0


def test_nfl_same_team_more_correlated_than_cross_team():
    qb = _leg(game_title="NFL", market="passing_yards", team="dal", opponent="nyg", player="qb")
    wr = _leg(
        game_title="NFL",
        market="receiving_yards",
        team="dal",
        opponent="nyg",
        player="wr",
        side="over",
    )
    rb = _leg(
        game_title="NFL",
        market="rushing_yards",
        team="nyg",
        opponent="dal",
        player="rb",
        side="over",
    )
    same_team = pairwise_correlation_factor(qb, wr)
    cross_team = pairwise_correlation_factor(qb, rb)
    assert same_team < cross_team


def test_mixed_direction_pulls_toward_independence():
    a = _leg(market="points", side="over")
    b = _leg(player="b", market="rebounds", side="under", team="nyk", opponent="bos")
    same_dir = pairwise_correlation_factor(
        a, _leg(player="b", market="rebounds", side="over", team="nyk", opponent="bos")
    )
    mixed = pairwise_correlation_factor(a, b)
    assert mixed > same_dir


def test_adjusted_pair_probability_scales_independence():
    a = _leg(model_probability=0.62)
    b = _leg(player="b", market="rebounds", model_probability=0.58, team="nyk", opponent="bos")
    joint, factor = adjusted_pair_probability("NBA", a, b, same_team=False)
    assert factor < 1.0
    assert joint == pytest.approx(0.62 * 0.58 * factor)


def test_cross_game_power_card_factor_near_one():
    legs = pd.DataFrame(
        [
            _leg(game_title="NBA", event_time="t1", player="a", team="bos", opponent="nyk"),
            _leg(
                game_title="NBA",
                event_time="t2",
                player="b",
                team="lal",
                opponent="gsw",
                market="rebounds",
            ),
        ]
    )
    assert card_joint_correlation_factor(legs) == pytest.approx(1.0, abs=0.02)


def test_build_cards_applies_correlation_to_power_hit_only():
    profile = default_profiles()[1]
    scored = pd.DataFrame(
        [
            {
                **dict(_leg(game_title="NBA", event_time="t1", player="a")),
                "recommendation": "PLAY",
                "dfs_edge": 0.05,
                "model_probability": 0.62,
            },
            {
                **dict(
                    _leg(
                        game_title="NBA",
                        event_time="t1",
                        player="b",
                        market="rebounds",
                        team="nyk",
                        opponent="bos",
                    )
                ),
                "recommendation": "PLAY",
                "dfs_edge": 0.04,
                "model_probability": 0.58,
            },
        ]
    )
    cards = build_cards(
        scored,
        profile,
        CardRules(legs=2, min_edge=0.02, min_probability=0.50, max_per_event=2),
    )
    assert not cards.empty
    indep = 0.62 * 0.58
    assert float(cards.iloc[0]["power_hit_probability"]) < indep
    assert "correlation_factor" in cards.columns


def _parlay_row(
    *,
    sport: str,
    m1: str,
    m2: str,
    joint_p: float,
    win: bool,
    slate_date: str = "2026-06-01",
) -> dict:
    return {
        "ledger_key": f"{sport}-{m1}-{m2}-{slate_date}-{int(win)}",
        "bet_id": f"{sport}-{m1}-{m2}-{slate_date}",
        "bet_format": "parlay_2leg",
        "sport": sport,
        "market1": m1,
        "market2": m2,
        "leg1_model_probability": 0.62,
        "leg2_model_probability": 0.58,
        "model_probability_raw": joint_p,
        "joint_model_probability": joint_p,
        "joint_probability_method": "pair_hit_probability",
        "model_probability_source": "journal",
        "result": "WIN" if win else "LOSS",
        "slate_date": slate_date,
        "date_graded": f"{slate_date} 12:00:00",
    }


def _write_parlay_ledger(path, rows: list[dict]) -> None:
    from sports_prop_edge.strategy.probability_ledger import LEDGER_COLUMNS

    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    for col in LEDGER_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    text_cols = {
        "ledger_key",
        "bet_id",
        "date_graded",
        "slate_date",
        "sport",
        "bet_format",
        "result",
        "joint_probability_method",
        "model_probability_source",
        "market1",
        "market2",
    }
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype("string")
    df[LEDGER_COLUMNS].to_csv(path, index=False)


def test_empirical_blend_alpha_grows_with_sample_size():
    from sports_prop_edge.strategy.correlation import CorrelationCalibrationConfig, empirical_blend_alpha

    cfg = CorrelationCalibrationConfig(min_samples=4, full_weight_samples=40)
    assert empirical_blend_alpha(3, config=cfg) == 0.0
    assert empirical_blend_alpha(4, config=cfg) == 0.0
    assert empirical_blend_alpha(22, config=cfg) == pytest.approx(0.5, abs=0.01)
    assert empirical_blend_alpha(40, config=cfg) == pytest.approx(1.0)


def test_build_empirical_correlation_table_from_ledger(tmp_path):
    from sports_prop_edge.strategy.correlation import (
        CorrelationCalibrationConfig,
        build_empirical_correlation_table,
    )

    ledger_path = tmp_path / "data" / "pick_results_ledger.csv"
    rows = []
    for i, win in enumerate([True, False, False, True, False, False]):
        rows.append(
            _parlay_row(
                sport="NBA",
                m1="points",
                m2="assists",
                joint_p=0.40,
                win=win,
                slate_date=f"2026-06-{i + 1:02d}",
            )
        )
    _write_parlay_ledger(ledger_path, rows)

    cfg = CorrelationCalibrationConfig(window_days=None, min_samples=4, full_weight_samples=40)
    table = build_empirical_correlation_table(tmp_path, config=cfg)
    key = ("NBA", "assists", "points")
    assert key in table
    stats = table[key]
    assert stats.sample_size == 6
    assert stats.observed_hit_rate == pytest.approx(2 / 6)
    assert stats.expected_hit_rate == pytest.approx(0.40)
    assert stats.correction_factor == pytest.approx((2 / 6) / 0.40, rel=0.02)
    assert stats.alpha > 0.0


def test_pairwise_uses_higher_empirical_weight_with_more_samples(tmp_path):
    from sports_prop_edge.strategy.correlation import (
        CorrelationCalibrationConfig,
        build_empirical_correlation_table,
        structural_pair_correlation_factor,
    )

    a = _leg(player="a", market="points")
    b = _leg(player="b", market="assists", team="nyk", opponent="bos")

    structural = structural_pair_correlation_factor(a, b)

    ledger_path = tmp_path / "data" / "pick_results_ledger.csv"
    rows = [
        _parlay_row(sport="NBA", m1="points", m2="assists", joint_p=0.40, win=True, slate_date="2026-06-01"),
        _parlay_row(sport="NBA", m1="points", m2="assists", joint_p=0.40, win=False, slate_date="2026-06-02"),
        _parlay_row(sport="NBA", m1="points", m2="assists", joint_p=0.40, win=False, slate_date="2026-06-03"),
        _parlay_row(sport="NBA", m1="points", m2="assists", joint_p=0.40, win=False, slate_date="2026-06-04"),
    ]
    _write_parlay_ledger(ledger_path, rows)
    small_cfg = CorrelationCalibrationConfig(window_days=None, min_samples=4, full_weight_samples=8)
    small_table = build_empirical_correlation_table(tmp_path, config=small_cfg)
    rho_small = pairwise_correlation_factor(a, b, empirical_table=small_table)

    pattern = [True, False, False, False]
    rows.extend(
        _parlay_row(
            sport="NBA",
            m1="points",
            m2="assists",
            joint_p=0.40,
            win=pattern[i],
            slate_date=f"2026-05-{d:02d}",
        )
        for i, d in enumerate(range(5, 9))
    )
    _write_parlay_ledger(ledger_path, rows)
    large_table = build_empirical_correlation_table(tmp_path, config=small_cfg)
    rho_large = pairwise_correlation_factor(a, b, empirical_table=large_table)

    stats = large_table[("NBA", "assists", "points")]
    emp_factor = stats.correction_factor
    assert small_table[("NBA", "assists", "points")].alpha == 0.0
    assert stats.base_alpha == 1.0
    assert stats.regime == "stable"
    assert stats.alpha == 1.0
    assert rho_small == pytest.approx(structural)
    assert rho_large == pytest.approx(emp_factor)
    assert rho_large != pytest.approx(rho_small)


def test_rolling_window_excludes_old_parlays(tmp_path):
    from sports_prop_edge.strategy.correlation import CorrelationCalibrationConfig, build_empirical_correlation_table

    ledger_path = tmp_path / "data" / "pick_results_ledger.csv"
    rows = [
        _parlay_row(sport="NBA", m1="points", m2="rebounds", joint_p=0.32, win=True, slate_date="2020-01-01"),
        _parlay_row(sport="NBA", m1="points", m2="rebounds", joint_p=0.32, win=True, slate_date="2020-01-02"),
        _parlay_row(sport="NBA", m1="points", m2="rebounds", joint_p=0.32, win=False, slate_date="2020-01-03"),
        _parlay_row(sport="NBA", m1="points", m2="rebounds", joint_p=0.32, win=False, slate_date="2020-01-04"),
    ]
    _write_parlay_ledger(ledger_path, rows)
    cfg = CorrelationCalibrationConfig(window_days=30, min_samples=4)
    assert build_empirical_correlation_table(tmp_path, config=cfg) == {}


def test_detect_pair_regime_volatile_on_hot_streak(tmp_path):
    from sports_prop_edge.strategy.correlation import (
        CorrelationCalibrationConfig,
        build_empirical_correlation_table,
        structural_pair_correlation_factor,
    )

    ledger_path = tmp_path / "data" / "pick_results_ledger.csv"
    rows = []
    for d in range(1, 9):
        rows.append(
            _parlay_row(
                sport="NBA",
                m1="points",
                m2="assists",
                joint_p=0.35,
                win=False,
                slate_date=f"2026-05-{d:02d}",
            )
        )
    for d in range(9, 17):
        rows.append(
            _parlay_row(
                sport="NBA",
                m1="points",
                m2="assists",
                joint_p=0.35,
                win=True,
                slate_date=f"2026-05-{d:02d}",
            )
        )
    _write_parlay_ledger(ledger_path, rows)

    cfg = CorrelationCalibrationConfig(
        window_days=None,
        min_samples=4,
        full_weight_samples=12,
        regime_recent_bets=8,
        regime_prior_min_bets=4,
    )
    table = build_empirical_correlation_table(tmp_path, config=cfg)
    stats = table[("NBA", "assists", "points")]
    assert stats.regime == "volatile"
    assert stats.alpha < stats.base_alpha
    assert stats.regime_alpha_scale == pytest.approx(cfg.regime_volatile_alpha_scale)

    a = _leg(player="a", market="points")
    b = _leg(player="b", market="assists", team="nyk", opponent="bos")
    structural = structural_pair_correlation_factor(a, b)
    rho = pairwise_correlation_factor(a, b, empirical_table=table)
    assert abs(rho - structural) < abs(rho - stats.correction_factor)


def test_detect_pair_regime_warming_and_cooling():
    from sports_prop_edge.strategy.correlation import CorrelationCalibrationConfig, detect_pair_regime

    cfg = CorrelationCalibrationConfig(
        regime_recent_bets=8,
        regime_prior_min_bets=8,
        regime_shift_threshold=0.10,
        regime_volatile_threshold=0.25,
    )
    base = {
        "sport": "NBA",
        "market_a": "assists",
        "market_b": "points",
        "expected": 0.40,
    }
    prior_rows = pd.DataFrame(
        [
            {**base, "win": i <= 5, "ref_date": pd.Timestamp(f"2026-05-{i:02d}")}
            for i in range(1, 13)
        ]
    )
    warming_recent = pd.DataFrame(
        [{**base, "win": i <= 4, "ref_date": pd.Timestamp(f"2026-06-{i:02d}")} for i in range(1, 9)]
    )
    cooling_recent = pd.DataFrame(
        [{**base, "win": i <= 3, "ref_date": pd.Timestamp(f"2026-06-{i:02d}")} for i in range(1, 9)]
    )

    warm_regime, warm_scale, _, _ = detect_pair_regime(
        pd.concat([prior_rows, warming_recent], ignore_index=True), config=cfg
    )
    cool_regime, cool_scale, _, _ = detect_pair_regime(
        pd.concat([prior_rows, cooling_recent], ignore_index=True), config=cfg
    )
    assert warm_regime == "warming"
    assert cool_regime == "cooling"
    assert warm_scale == pytest.approx(cfg.regime_drift_alpha_scale)
    assert cool_scale == pytest.approx(cfg.regime_drift_alpha_scale)
