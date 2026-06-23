"""Command-line runner for batch scoring and card building."""

from __future__ import annotations

import argparse
from pathlib import Path

from sports_prop_edge.data.loaders import load_history, load_props
from sports_prop_edge.models.projections import ProjectionConfig, SportPropProjector
from sports_prop_edge.strategy.card_builder import CardRules, build_cards
from sports_prop_edge.strategy.payouts import profile_by_name
from sports_prop_edge.strategy.scoring import score_props


def main() -> None:
    parser = argparse.ArgumentParser(description="Score sports DFS/pick'em props.")
    parser.add_argument("--props", required=True, help="Path to props CSV")
    parser.add_argument("--history", required=True, help="Path to historical stats CSV")
    parser.add_argument("--profile", default="2-pick power example: 3x", help="Payout profile name")
    parser.add_argument("--distribution", default="poisson", choices=["poisson", "negative_binomial"])
    parser.add_argument("--out", default="scored_props.csv", help="Output scored props CSV path")
    parser.add_argument("--cards-out", default="candidate_cards.csv", help="Output card CSV path")
    parser.add_argument("--build-cards", action="store_true", help="Also build candidate pick'em cards")
    parser.add_argument("--min-edge", type=float, default=0.02, help="Minimum edge for card legs")
    parser.add_argument("--min-prob", type=float, default=0.50, help="Minimum model probability for card legs")
    args = parser.parse_args()

    props = load_props(args.props)
    history = load_history(args.history)
    projector = SportPropProjector(ProjectionConfig())
    projected = projector.project_props(props, history)
    profile = profile_by_name(args.profile)
    scored = score_props(projected, profile, distribution=args.distribution)

    out = Path(args.out)
    scored.to_csv(out, index=False)
    print(f"Wrote {len(scored)} scored props to {out}")

    if args.build_cards:
        cards = build_cards(
            scored,
            profile,
            CardRules(legs=profile.legs, min_edge=args.min_edge, min_probability=args.min_prob),
        )
        cards_out = Path(args.cards_out)
        if "leg_indexes" in cards.columns:
            cards = cards.drop(columns=["leg_indexes"])
        cards.to_csv(cards_out, index=False)
        print(f"Wrote {len(cards)} candidate cards to {cards_out}")


if __name__ == "__main__":
    main()
