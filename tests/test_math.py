from sports_prop_edge.models.distributions import poisson_prob_over, poisson_prob_under
from sports_prop_edge.strategy.payouts import PayoutProfile
from sports_prop_edge.utils.odds import american_to_decimal, ev_per_dollar


def test_american_to_decimal():
    assert round(american_to_decimal(100), 2) == 2.00
    assert round(american_to_decimal(-110), 3) == 1.909


def test_ev_per_dollar():
    assert round(ev_per_dollar(2.0, 0.55), 3) == 0.100


def test_poisson_probs_sum_approximately_for_half_line():
    over = poisson_prob_over(15.5, 16.2)
    under = poisson_prob_under(15.5, 16.2)
    assert abs((over + under) - 1) < 1e-9


def test_power_breakeven():
    profile = PayoutProfile("2-pick 3x", 2, {2: 3.0})
    assert round(profile.breakeven_leg_probability(), 3) == 0.577
