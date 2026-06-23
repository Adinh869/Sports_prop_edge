"""Normalize probability fields on pick_results_ledger rows (sports)."""

from __future__ import annotations

import pandas as pd

PROB_COLUMNS = [
    "model_probability",
    "model_probability_raw",
    "leg1_model_probability",
    "leg2_model_probability",
    "joint_model_probability",
]


def normalize_probability_value(value) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null", "n/a"}:
            return None
        text = text.replace("%", "")
        try:
            value = float(text)
        except ValueError:
            return None
    else:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
    if pd.isna(value):
        return None
    if value > 1.5:
        value = value / 100.0
    if 0.0 <= value <= 1.0:
        return value
    return None


def enrich_ledger_probabilities(df: pd.DataFrame) -> pd.DataFrame:
    """Fill model_probability_raw; keep model_probability equal to raw (no auto-tightening)."""
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()
    for col in PROB_COLUMNS + ["model_probability_source", "joint_probability_method"]:
        if col not in out.columns:
            out[col] = pd.NA

    for idx, row in out.iterrows():
        raw = normalize_probability_value(row.get("model_probability_raw"))
        if raw is None:
            raw = normalize_probability_value(row.get("joint_model_probability"))
        if raw is None:
            raw = normalize_probability_value(row.get("model_probability"))

        if raw is None:
            continue

        out.at[idx, "model_probability_raw"] = raw
        out.at[idx, "model_probability"] = raw
        if pd.isna(out.at[idx, "model_probability_source"]):
            method = str(row.get("joint_probability_method", "") or "").strip()
            out.at[idx, "model_probability_source"] = method or "model_probability_raw"

    return out
