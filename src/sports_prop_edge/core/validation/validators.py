"""Strict validators for pipeline input contracts."""

from __future__ import annotations

import math
from typing import Any

from sports_prop_edge.core.validation.exceptions import SchemaError, ValidationError
from sports_prop_edge.core.validation.schemas import LedgerEntry, PropInput, SGPInput

_REQUIRED_PROP_FIELDS = ("player", "sport", "market", "line")
_REQUIRED_SGP_FIELDS = (
    "leg1_player",
    "leg2_player",
    "sport",
    "correlation_factor",
    "pair_hit_probability",
)
_REQUIRED_LEDGER_FIELDS = ("sport", "market_a", "market_b", "result", "predicted_prob")
_VALID_LEDGER_RESULTS = frozenset({"WIN", "LOSS"})


def _require_text(value: Any, field: str) -> str:
    if value is None:
        raise ValidationError(f"{field} is required")
    text = str(value).strip()
    if not text:
        raise ValidationError(f"{field} must be a non-empty string")
    return text


def _require_finite_number(value: Any, field: str) -> float:
    if value is None:
        raise ValidationError(f"{field} is required")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field} must be a number") from exc
    if not math.isfinite(number):
        raise ValidationError(f"{field} must be finite")
    return number


def validate_probability(p: float, *, field: str = "probability") -> float:
    """Ensure probability is in [0, 1]."""
    value = _require_finite_number(p, field)
    if value < 0.0 or value > 1.0:
        raise ValidationError(f"{field} must be between 0 and 1 inclusive, got {value}")
    return value


def validate_line(line: float) -> float:
    """Ensure prop line is strictly positive."""
    value = _require_finite_number(line, "line")
    if value <= 0.0:
        raise ValidationError(f"line must be > 0, got {value}")
    return value


def _coerce_prop(prop: PropInput | dict[str, Any]) -> PropInput:
    if isinstance(prop, PropInput):
        return prop
    if not isinstance(prop, dict):
        raise SchemaError(f"PropInput must be PropInput or dict, got {type(prop).__name__}")
    missing = [name for name in _REQUIRED_PROP_FIELDS if name not in prop]
    if missing:
        raise SchemaError(f"PropInput missing fields: {', '.join(missing)}")
    try:
        return PropInput(
            player=str(prop["player"]),
            sport=str(prop["sport"]),
            market=str(prop["market"]),
            line=float(prop["line"]),
            odds=None if prop.get("odds") is None else float(prop["odds"]),
        )
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"PropInput schema coercion failed: {exc}") from exc


def _coerce_sgp(sgp: SGPInput | dict[str, Any]) -> SGPInput:
    if isinstance(sgp, SGPInput):
        return sgp
    if not isinstance(sgp, dict):
        raise SchemaError(f"SGPInput must be SGPInput or dict, got {type(sgp).__name__}")
    missing = [name for name in _REQUIRED_SGP_FIELDS if name not in sgp]
    if missing:
        raise SchemaError(f"SGPInput missing fields: {', '.join(missing)}")
    try:
        return SGPInput(
            leg1_player=str(sgp["leg1_player"]),
            leg2_player=str(sgp["leg2_player"]),
            sport=str(sgp["sport"]),
            correlation_factor=float(sgp["correlation_factor"]),
            pair_hit_probability=float(sgp["pair_hit_probability"]),
        )
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"SGPInput schema coercion failed: {exc}") from exc


def _coerce_ledger(entry: LedgerEntry | dict[str, Any]) -> LedgerEntry:
    if isinstance(entry, LedgerEntry):
        return entry
    if not isinstance(entry, dict):
        raise SchemaError(f"LedgerEntry must be LedgerEntry or dict, got {type(entry).__name__}")
    missing = [name for name in _REQUIRED_LEDGER_FIELDS if name not in entry]
    if missing:
        raise SchemaError(f"LedgerEntry missing fields: {', '.join(missing)}")
    try:
        return LedgerEntry(
            sport=str(entry["sport"]),
            market_a=str(entry["market_a"]),
            market_b=str(entry["market_b"]),
            result=str(entry["result"]).upper(),  # type: ignore[arg-type]
            predicted_prob=float(entry["predicted_prob"]),
        )
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"LedgerEntry schema coercion failed: {exc}") from exc


def validate_prop(prop: PropInput | dict[str, Any]) -> PropInput:
    """Validate a single prop input; raise ValidationError on failure."""
    item = _coerce_prop(prop)
    player = _require_text(item.player, "player")
    sport = _require_text(item.sport, "sport")
    market = _require_text(item.market, "market")
    line = validate_line(item.line)
    odds = None
    if item.odds is not None:
        odds = _require_finite_number(item.odds, "odds")
        if odds <= 0.0:
            raise ValidationError(f"odds must be > 0 when provided, got {odds}")
    return PropInput(player=player, sport=sport, market=market, line=line, odds=odds)


def validate_sgp(sgp: SGPInput | dict[str, Any]) -> SGPInput:
    """Validate a single SGP input; raise ValidationError on failure."""
    item = _coerce_sgp(sgp)
    leg1 = _require_text(item.leg1_player, "leg1_player")
    leg2 = _require_text(item.leg2_player, "leg2_player")
    sport = _require_text(item.sport, "sport")
    if leg1.lower() == leg2.lower():
        raise ValidationError("leg1_player and leg2_player must be different players")
    correlation_factor = _require_finite_number(item.correlation_factor, "correlation_factor")
    if correlation_factor <= 0.0:
        raise ValidationError(f"correlation_factor must be > 0, got {correlation_factor}")
    pair_hit_probability = validate_probability(item.pair_hit_probability, field="pair_hit_probability")
    return SGPInput(
        leg1_player=leg1,
        leg2_player=leg2,
        sport=sport,
        correlation_factor=correlation_factor,
        pair_hit_probability=pair_hit_probability,
    )


def validate_ledger(entry: LedgerEntry | dict[str, Any]) -> LedgerEntry:
    """Validate a ledger entry contract; raise ValidationError on failure."""
    item = _coerce_ledger(entry)
    sport = _require_text(item.sport, "sport")
    market_a = _require_text(item.market_a, "market_a")
    market_b = _require_text(item.market_b, "market_b")
    result = str(item.result).upper()
    if result not in _VALID_LEDGER_RESULTS:
        raise ValidationError(f"result must be WIN or LOSS, got {item.result!r}")
    predicted_prob = validate_probability(item.predicted_prob, field="predicted_prob")
    return LedgerEntry(
        sport=sport,
        market_a=market_a,
        market_b=market_b,
        result=result,  # type: ignore[arg-type]
        predicted_prob=predicted_prob,
    )
