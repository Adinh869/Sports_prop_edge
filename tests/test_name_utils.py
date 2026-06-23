from sports_prop_edge.integrations.name_utils import (
    fuzzy_best_match,
    is_combo_player,
    names_match,
    normalize_lookup_name,
)


def test_combo_player_detected():
    assert is_combo_player("kim keon-woo + kim yun sik")
    assert not is_combo_player("choi jeong")


def test_names_match_hyphen_variants():
    assert names_match("choi jeong", "Choi Jeong")
    assert names_match("koo ja-wook", "Koo Ja Wook")


def test_fuzzy_best_match():
    ranked = fuzzy_best_match("jaylen brown", ["Jayson Tatum", "Jaylen Brown"], min_score=0.80)
    assert ranked[0][0] == "Jaylen Brown"
