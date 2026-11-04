"""Tests for turf.efi.risk — bin-based expected xT risk model."""

from __future__ import annotations

import pandas as pd
import pytest

from turf.efi.risk import add_residuals, bucket_game_state, build_expected_xt

# ── bucket_game_state ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("score_diff", "expected"),
    [
        (-7, "losing_2+"),
        (-3, "losing_2+"),
        (-2, "losing_2+"),
        (-1, "losing_1"),
        (0, "level"),
        (1, "winning_1"),
        (2, "winning_2+"),
        (5, "winning_2+"),
        (7, "winning_2+"),
    ],
)
def test_bucket_game_state(score_diff: int, expected: str) -> None:
    assert bucket_game_state(score_diff) == expected


# ── build_expected_xt ─────────────────────────────────────────────────────────


@pytest.fixture()
def lbp_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_thirds": ["t_def", "t_def", "t_mid", "t_mid", "t_att"],
            "score_diff": [0, 0, -1, 1, 0],
            "actual_xt_gain": [0.01, 0.03, 0.02, 0.04, 0.10],
        }
    )


def test_build_expected_xt_keys(lbp_df: pd.DataFrame) -> None:
    expected = build_expected_xt(lbp_df)
    assert ("t_def", "level") in expected
    assert ("t_mid", "losing_1") in expected
    assert ("t_mid", "winning_1") in expected
    assert ("t_att", "level") in expected


def test_build_expected_xt_values(lbp_df: pd.DataFrame) -> None:
    expected = build_expected_xt(lbp_df)
    assert expected[("t_def", "level")] == pytest.approx(0.02)  # mean(0.01, 0.03)
    assert expected[("t_mid", "losing_1")] == pytest.approx(0.02)
    assert expected[("t_mid", "winning_1")] == pytest.approx(0.04)
    assert expected[("t_att", "level")] == pytest.approx(0.10)


def test_build_expected_xt_custom_zone_col(lbp_df: pd.DataFrame) -> None:
    df = lbp_df.rename(columns={"zone_thirds": "zone_van_gaal"})
    expected = build_expected_xt(df, zone_col="zone_van_gaal")
    assert ("t_def", "level") in expected


# ── add_residuals ─────────────────────────────────────────────────────────────


@pytest.fixture()
def features_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_thirds": ["t_def", "t_mid", "t_att", "t_def"],
            "score_diff": [0, -1, 0, 1],
            "actual_xt_gain": [0.03, 0.02, 0.10, 0.05],
        }
    )


def test_add_residuals_columns(features_df: pd.DataFrame, lbp_df: pd.DataFrame) -> None:
    expected = build_expected_xt(lbp_df)
    result = add_residuals(features_df, expected)
    assert "game_state_bucket" in result.columns
    assert "xt_expected" in result.columns
    assert "xt_residual" in result.columns


def test_add_residuals_values(features_df: pd.DataFrame, lbp_df: pd.DataFrame) -> None:
    expected = build_expected_xt(lbp_df)
    result = add_residuals(features_df, expected)
    # t_def / level: expected = 0.02, observed = 0.03 → residual = 0.01
    row = result[(result["zone_thirds"] == "t_def") & (result["score_diff"] == 0)].iloc[
        0
    ]
    assert row["xt_expected"] == pytest.approx(0.02)
    assert row["xt_residual"] == pytest.approx(0.01)


def test_add_residuals_unseen_cell_is_nan(features_df: pd.DataFrame) -> None:
    # Pass in an expected dict missing the t_def/winning_1 cell
    sparse_expected: dict[tuple[str, str], float] = {("t_mid", "level"): 0.05}
    result = add_residuals(features_df, sparse_expected)
    nan_rows = result[result["xt_expected"].isna()]
    assert len(nan_rows) > 0


def test_add_residuals_does_not_mutate_input(features_df: pd.DataFrame) -> None:
    original_cols = list(features_df.columns)
    expected: dict[tuple[str, str], float] = {("t_def", "level"): 0.02}
    add_residuals(features_df, expected)
    assert list(features_df.columns) == original_cols


# ── pressure binning ──────────────────────────────────────────────────────────


@pytest.fixture()
def lbp_df_with_pressure() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_thirds": ["t_def", "t_def", "t_def", "t_mid"],
            "under_pressure": [True, True, False, False],
            "score_diff": [0, 0, 0, 0],
            "actual_xt_gain": [0.01, 0.03, 0.10, 0.05],
        }
    )


def test_build_expected_xt_with_pressure_keys(
    lbp_df_with_pressure: pd.DataFrame,
) -> None:
    expected = build_expected_xt(lbp_df_with_pressure, pressure_col="under_pressure")
    assert ("t_def", True, "level") in expected
    assert ("t_def", False, "level") in expected
    assert ("t_mid", False, "level") in expected


def test_build_expected_xt_with_pressure_values(
    lbp_df_with_pressure: pd.DataFrame,
) -> None:
    expected = build_expected_xt(lbp_df_with_pressure, pressure_col="under_pressure")
    # t_def / True / level: mean(0.01, 0.03) = 0.02
    assert expected[("t_def", True, "level")] == pytest.approx(0.02)
    # t_def / False / level: 0.10
    assert expected[("t_def", False, "level")] == pytest.approx(0.10)


def test_add_residuals_with_pressure(
    lbp_df_with_pressure: pd.DataFrame,
) -> None:
    expected = build_expected_xt(lbp_df_with_pressure, pressure_col="under_pressure")
    features = pd.DataFrame(
        {
            "zone_thirds": ["t_def", "t_def"],
            "under_pressure": [True, False],
            "score_diff": [0, 0],
            "actual_xt_gain": [0.05, 0.12],
        }
    )
    result = add_residuals(features, expected, pressure_col="under_pressure")
    # t_def / True / level → expected 0.02; residual = 0.05 - 0.02 = 0.03
    assert result.iloc[0]["xt_expected"] == pytest.approx(0.02)
    assert result.iloc[0]["xt_residual"] == pytest.approx(0.03)
    # t_def / False / level → expected 0.10; residual = 0.12 - 0.10 = 0.02
    assert result.iloc[1]["xt_expected"] == pytest.approx(0.10)
    assert result.iloc[1]["xt_residual"] == pytest.approx(0.02)
