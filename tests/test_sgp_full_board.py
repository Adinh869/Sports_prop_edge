from pathlib import Path

import pandas as pd

from sports_prop_edge.data.props_pipeline import score_baseball_sgp_pool, score_full_board_sgp_pool
from sports_prop_edge.strategy.leg_pool import LegPoolSettings
from sports_prop_edge.strategy.payouts import default_profiles


def test_score_full_board_sgp_pool_empty():
    profile = default_profiles()[1]
    pool = LegPoolSettings.balanced()
    assert score_full_board_sgp_pool(
        Path("."), pd.DataFrame(), payout_profile=profile, leg_pool=pool
    ).empty


def test_score_baseball_alias_empty():
    profile = default_profiles()[1]
    pool = LegPoolSettings.balanced()
    assert score_baseball_sgp_pool(
        Path("."), pd.DataFrame(), payout_profile=profile, leg_pool=pool
    ).empty
