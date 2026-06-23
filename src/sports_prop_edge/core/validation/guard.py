"""Resilient safety gates that filter invalid inputs without crashing the pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypeVar

from sports_prop_edge.core.validation.exceptions import SafetyError, SchemaError, ValidationError
from sports_prop_edge.core.validation.schemas import PropInput, SGPInput
from sports_prop_edge.core.validation.validators import validate_prop, validate_sgp

T = TypeVar("T")


def _filter_valid(
    items: Iterable[T],
    validator,
    *,
    label: str,
    max_invalid_ratio: float | None = None,
) -> list:
    """Return only entries that pass validation; drop invalid silently."""
    valid: list = []
    invalid_count = 0
    total = 0

    for item in items:
        total += 1
        try:
            valid.append(validator(item))
        except (ValidationError, SchemaError):
            invalid_count += 1
            continue

    if (
        max_invalid_ratio is not None
        and total > 0
        and (invalid_count / total) > max_invalid_ratio
    ):
        raise SafetyError(
            f"{label}: invalid input ratio {invalid_count}/{total} exceeds "
            f"allowed {max_invalid_ratio:.0%}"
        )

    return valid


def safe_validate_props(
    props: Iterable[PropInput | dict[str, Any]],
    *,
    max_invalid_ratio: float | None = None,
) -> list[PropInput]:
    """Return only props that pass strict validation; drop invalid entries."""
    return _filter_valid(
        props,
        validate_prop,
        label="props",
        max_invalid_ratio=max_invalid_ratio,
    )


def safe_validate_sgps(
    sgps: Iterable[SGPInput | dict[str, Any]],
    *,
    max_invalid_ratio: float | None = None,
) -> list[SGPInput]:
    """Return only SGPs that pass strict validation; drop invalid entries."""
    return _filter_valid(
        sgps,
        validate_sgp,
        label="sgps",
        max_invalid_ratio=max_invalid_ratio,
    )
