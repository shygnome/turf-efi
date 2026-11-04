"""Bin-based expected xT risk model for line-breaking passes.

Expected risk = mean xT_gain per [zone × pressure_binary × game_state_bucket]
cell, computed from all LBPs in the dataset.
Residual = observed xT_gain − expected.

pressure_col is optional: when omitted the model bins by zone × game_state only
(backward-compatible with datasets extracted before pressure was added).
"""

from __future__ import annotations

import pandas as pd

__all__ = [
    "bucket_game_state",
    "build_expected_xt",
    "add_residuals",
]


def bucket_game_state(score_diff: int) -> str:
    """Map a score_diff integer to a 5-bucket game-state label."""
    if score_diff <= -2:
        return "losing_2+"
    if score_diff == -1:
        return "losing_1"
    if score_diff == 0:
        return "level"
    if score_diff == 1:
        return "winning_1"
    return "winning_2+"


def build_expected_xt(
    lbp_df: pd.DataFrame,
    zone_col: str = "zone_thirds",
    pressure_col: str | None = None,
) -> dict[tuple[str, ...], float]:
    """Return mean xT_gain per bin cell.

    Bins by (zone, game_state_bucket) when pressure_col is None, or by
    (zone, pressure_value, game_state_bucket) when pressure_col is supplied.

    lbp_df must contain columns: *zone_col*, score_diff, actual_xt_gain,
    and *pressure_col* if provided.
    """
    df = lbp_df.copy()
    df["_gs"] = df["score_diff"].apply(bucket_game_state)
    if pressure_col is None:
        group_cols = [zone_col, "_gs"]
    else:
        group_cols = [zone_col, pressure_col, "_gs"]
    means = df.groupby(group_cols)["actual_xt_gain"].mean()
    return {(k if isinstance(k, tuple) else (k,)): float(v) for k, v in means.items()}


def add_residuals(
    features_df: pd.DataFrame,
    expected: dict[tuple[str, ...], float],
    zone_col: str = "zone_thirds",
    pressure_col: str | None = None,
) -> pd.DataFrame:
    """Add game_state_bucket, xt_expected, and xt_residual columns.

    Uses the same binning key as build_expected_xt (zone × game_state, or
    zone × pressure × game_state when pressure_col is supplied).
    Cells absent from *expected* receive NaN for xt_expected/xt_residual.
    Does not mutate the input DataFrame.
    """
    df = features_df.copy()
    df["game_state_bucket"] = df["score_diff"].apply(bucket_game_state)

    if pressure_col is None:
        df["xt_expected"] = df.apply(
            lambda r: expected.get((r[zone_col], r["game_state_bucket"]), float("nan")),
            axis=1,
        )
    else:
        df["xt_expected"] = df.apply(
            lambda r: expected.get(
                (r[zone_col], r[pressure_col], r["game_state_bucket"]), float("nan")
            ),
            axis=1,
        )

    df["xt_residual"] = df["actual_xt_gain"] - df["xt_expected"]
    return df
