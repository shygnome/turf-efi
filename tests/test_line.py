from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from turf.efi.line import (
    analyze_lines,
    detect_lines_frame,
    find_goalkeeper,
    fix_player_assignments,
    infer_attacking_direction,
    smooth_line_assignments,
    vote_line_count,
)

# ── fixtures ──────────────────────────────────────────────────────────────────

# GK in _event_df (player 3) is at x ≈ +43 → defends right, attacks left
_EVENT_ATK_DIR = -1


def _away_df(**players: float) -> pd.DataFrame:
    """Single-row DataFrame with Away_N_x columns from keyword args."""
    return pd.DataFrame([{f"Away_{n}_x": v for n, v in players.items()}])


def _event_df() -> pd.DataFrame:
    """Five-frame event with GK (player 3) and two clear defensive lines.

    GK at x ≈ +43 → defends right goal → attacking_direction = -1.
    """
    rows = []
    for f in range(5):
        row: dict[str, object] = {"frame": f, "Period": 1, "Time [s]": f * 0.033}
        row["Away_3_x"] = 43.0 + f * 0.01
        row["Away_3_y"] = 0.0
        for pid, x, y in [
            ("2", 8.0, -12.0),
            ("11", 9.0, -5.0),
            ("6", 18.0, -6.0),
            ("7", 19.0, 6.0),
            ("9", 25.0, -4.0),
            ("10", 26.0, 4.0),
        ]:
            row[f"Away_{pid}_x"] = x + f * 0.1
            row[f"Away_{pid}_y"] = float(y)
        rows.append(row)
    return pd.DataFrame(rows)


# ── find_goalkeeper ───────────────────────────────────────────────────────────


class TestFindGoalkeeper:
    def test_defending_right_extreme_positive_x(self) -> None:
        # Team defends right goal (GK at max positive x) → attacking_direction=-1
        df = _away_df(**{"2": 10.0, "3": 44.0, "4": 18.0})
        assert find_goalkeeper(df, "Away", attacking_direction=-1) == "3"

    def test_majority_vote_overrides_bad_first_frame(self) -> None:
        # Frame 0: tracking swap — player 2 appears deep (wrong)
        # Frames 1-3: player 3 is consistently the deepest (correct GK)
        rows = [
            {"Away_2_x": 50.0, "Away_3_x": 20.0},
            {"Away_2_x": 10.0, "Away_3_x": 44.0},
            {"Away_2_x": 10.0, "Away_3_x": 44.0},
            {"Away_2_x": 10.0, "Away_3_x": 44.0},
        ]
        # GK (player 3) at x=44 → defends right → attacking_direction=-1
        result = find_goalkeeper(pd.DataFrame(rows), "Away", attacking_direction=-1)
        assert result == "3"

    def test_minority_bad_frames_do_not_flip_result(self) -> None:
        # 1 bad frame out of 5 should not change the result
        rows = [
            {"Away_1_x": 44.0, "Away_5_x": 10.0},  # correct
            {"Away_1_x": 44.0, "Away_5_x": 10.0},  # correct
            {"Away_1_x": 10.0, "Away_5_x": 44.0},  # bad tracking frame
            {"Away_1_x": 44.0, "Away_5_x": 10.0},  # correct
            {"Away_1_x": 44.0, "Away_5_x": 10.0},  # correct
        ]
        # GK (player 1) at x=44 → defends right → attacking_direction=-1
        result = find_goalkeeper(pd.DataFrame(rows), "Away", attacking_direction=-1)
        assert result == "1"

    def test_defending_left_extreme_negative_x(self) -> None:
        # Team defends left goal (GK at min negative x) → attacking_direction=+1
        df = _away_df(**{"2": -5.0, "3": -43.0, "4": -12.0})
        assert find_goalkeeper(df, "Away", attacking_direction=1) == "3"

    def test_direction_aware_outfielder_pushed_into_opponent_half_right(self) -> None:
        # Team defends RIGHT (positive x). Real GK at +40, but outfielder
        # (player 16) pushed all the way to -42. abs().idxmax() picks 16 (wrong).
        # attacking_direction=-1 → use idxmax → correctly picks player 1 (GK at +40).
        rows = [
            {
                "Away_1_x": 40.0,   # GK — defending right goal
                "Away_16_x": -42.0,  # outfielder pushed far forward
                "Away_2_x": 15.0, "Away_3_x": 10.0, "Away_4_x": 5.0,
            }
        ] * 5
        result = find_goalkeeper(pd.DataFrame(rows), "Away", attacking_direction=-1)
        assert result == "1"

    def test_direction_aware_outfielder_pushed_into_opponent_half_left(self) -> None:
        # Symmetric: team defends LEFT. GK at -40, outfielder at +42.
        # attacking_direction=+1 → use idxmin → correctly picks player 1 (GK at -40).
        rows = [
            {
                "Away_1_x": -40.0,  # GK — defending left goal
                "Away_16_x": 42.0,   # outfielder pushed far forward
                "Away_2_x": -15.0, "Away_3_x": -10.0, "Away_4_x": -5.0,
            }
        ] * 5
        assert find_goalkeeper(pd.DataFrame(rows), "Away", attacking_direction=1) == "1"


# ── infer_attacking_direction ─────────────────────────────────────────────────


class TestInferAttackingDirection:
    def test_gk_at_positive_x_returns_minus_one(self) -> None:
        # GK at max positive x → team attacks toward negative x
        df = _away_df(**{"1": 43.0, "2": 10.0, "3": 5.0})
        assert infer_attacking_direction(df, "Away") == -1

    def test_gk_at_negative_x_returns_plus_one(self) -> None:
        # GK at min negative x → team attacks toward positive x
        df = _away_df(**{"1": -43.0, "2": -10.0, "3": -5.0})
        assert infer_attacking_direction(df, "Away") == 1

    def test_uses_first_non_nan_frame(self) -> None:
        # First frame is all NaN; second frame has valid positions
        rows = [
            {"Away_1_x": float("nan"), "Away_2_x": float("nan")},
            {"Away_1_x": -42.0, "Away_2_x": 5.0},
        ]
        assert infer_attacking_direction(pd.DataFrame(rows), "Away") == 1

    def test_no_matching_columns_raises(self) -> None:
        df = pd.DataFrame([{"Home_1_x": 10.0}])
        with pytest.raises(ValueError):
            infer_attacking_direction(df, "Away")

    def test_all_nan_raises(self) -> None:
        df = pd.DataFrame([{"Away_1_x": float("nan"), "Away_2_x": float("nan")}])
        with pytest.raises(ValueError):
            infer_attacking_direction(df, "Away")


# ── detect_lines_frame ────────────────────────────────────────────────────────


class TestDetectLinesFrame:
    def test_empty_returns_empty(self) -> None:
        assert detect_lines_frame({}) == {}

    def test_three_clear_groups_assigns_correct_lines(self) -> None:
        positions = {
            "2": 8.0,
            "11": 9.0,
            "6": 18.0,
            "7": 19.0,
            "9": 27.0,
            "10": 28.0,
        }
        result = detect_lines_frame(positions)
        assert result["2"] == result["11"] == 1
        assert result["6"] == result["7"] == 2
        assert result["9"] == result["10"] == 3

    def test_lines_ordered_deepest_first(self) -> None:
        positions = {
            "2": 28.0,
            "11": 29.0,
            "6": 10.0,
            "7": 11.0,
        }
        result = detect_lines_frame(positions)
        assert result["6"] < result["2"]

    def test_never_more_than_four_lines(self) -> None:
        positions = {str(i): float(i * 3) for i in range(1, 11)}
        assert len(set(detect_lines_frame(positions).values())) <= 4

    def test_at_least_two_lines_when_enough_players(self) -> None:
        positions = {str(i): float(i) for i in range(1, 9)}
        assert len(set(detect_lines_frame(positions).values())) >= 2

    def test_singleton_merged_all_lines_min_two_players(self) -> None:
        positions = {
            "2": 8.0,
            "11": 9.0,
            "6": 18.0,
            "7": 19.0,
            "8": 19.5,
            "5": 22.5,  # isolated between groups
            "9": 30.0,
            "10": 31.0,
        }
        result = detect_lines_frame(positions)
        counts = Counter(result.values())
        assert all(c >= 2 for c in counts.values())

    def test_all_players_assigned(self) -> None:
        positions = {"2": 8.0, "11": 9.0, "6": 18.0, "7": 19.0}
        result = detect_lines_frame(positions)
        assert set(result.keys()) == set(positions.keys())

    def test_fallback_preserves_min_lines_when_merge_would_collapse(self) -> None:
        # 3 tightly packed + 1 outlier: Ward k=2 gives 3+1,
        # naive merge would collapse to 1 line
        positions = {"2": 1.0, "11": 1.1, "6": 1.2, "7": 10.0}
        result = detect_lines_frame(positions)
        assert len(set(result.values())) >= 2

    def test_min_line_gap_merges_close_adjacent_lines(self) -> None:
        # 3 initial lines: [10,11], [12,13], [25,26]
        # first two are only 2 m apart → should merge with gap=5.0
        positions = {
            "2": 10.0,
            "11": 11.0,
            "6": 12.0,
            "7": 13.0,
            "9": 25.0,
            "10": 26.0,
        }
        result = detect_lines_frame(positions, min_line_gap=5.0)
        assert len(set(result.values())) == 2

    def test_min_line_gap_keeps_well_separated_lines(self) -> None:
        # 10 m gap between clusters — gap=5.0 should not merge them
        positions = {"2": 5.0, "11": 6.0, "6": 16.0, "7": 17.0}
        result = detect_lines_frame(positions, min_line_gap=5.0)
        assert len(set(result.values())) == 2

    def test_min_line_gap_stops_at_min_lines(self) -> None:
        # Even with a huge threshold, we never go below min_lines
        positions = {"2": 1.0, "11": 2.0, "6": 3.0, "7": 4.0}
        result = detect_lines_frame(positions, min_line_gap=100.0, min_lines=2)
        assert len(set(result.values())) >= 2

    def test_min_line_gap_zero_is_no_op(self) -> None:
        positions = {
            "2": 10.0,
            "11": 11.0,
            "6": 12.0,
            "7": 13.0,
        }
        assert detect_lines_frame(positions, min_line_gap=0.0) == detect_lines_frame(
            positions
        )


# ── analyze_lines ─────────────────────────────────────────────────────────────


class TestAnalyzeLines:
    def test_adds_line_column_for_each_player(self) -> None:
        df = _event_df()
        result = analyze_lines(df, "Away", _EVENT_ATK_DIR)
        for pid in ["2", "3", "6", "7", "9", "10", "11"]:
            assert f"Away_{pid}_line" in result.columns

    def test_adds_line_count_column(self) -> None:
        result = analyze_lines(_event_df(), "Away", _EVENT_ATK_DIR)
        assert "line_count" in result.columns

    def test_gk_always_line_zero(self) -> None:
        result = analyze_lines(_event_df(), "Away", _EVENT_ATK_DIR)
        assert (result["Away_3_line"] == 0).all()

    def test_outfield_lines_between_one_and_four(self) -> None:
        result = analyze_lines(_event_df(), "Away", _EVENT_ATK_DIR)
        for pid in ["2", "6", "7", "9", "10", "11"]:
            vals = result[f"Away_{pid}_line"].dropna()
            assert vals.between(1, 4).all(), f"Player {pid} out-of-range line values"

    def test_absent_player_x_gives_nan_line(self) -> None:
        df = _event_df()
        df["Away_14_x"] = np.nan
        df["Away_14_y"] = np.nan
        result = analyze_lines(df, "Away", _EVENT_ATK_DIR)
        assert result["Away_14_line"].isna().all()

    def test_partially_present_player_nan_for_missing_frames(self) -> None:
        df = _event_df()
        df["Away_14_x"] = np.nan
        df["Away_14_y"] = np.nan
        df.loc[0:1, "Away_14_x"] = [12.0, 12.1]
        result = analyze_lines(df, "Away", _EVENT_ATK_DIR)
        assert not pd.isna(result.loc[0, "Away_14_line"])
        assert pd.isna(result.loc[2, "Away_14_line"])

    def test_original_columns_preserved(self) -> None:
        df = _event_df()
        result = analyze_lines(df, "Away", _EVENT_ATK_DIR)
        for col in df.columns:
            assert col in result.columns

    def test_row_count_unchanged(self) -> None:
        df = _event_df()
        assert len(analyze_lines(df, "Away", _EVENT_ATK_DIR)) == len(df)

    def test_min_line_gap_propagated_to_frame_detection(self) -> None:
        # 3 close lines in event data: [10,11], [12,13], [25,26]
        # with gap=5.0 the first two merge → line_count should be 2
        rows = []
        for f in range(3):
            row: dict[str, object] = {"frame": f, "Period": 1, "Time [s]": f * 0.033}
            row["Away_3_x"] = 43.0
            row["Away_3_y"] = 0.0
            for pid, x, y in [
                ("2", 10.0, 0.0),
                ("11", 11.0, 0.0),
                ("6", 12.0, 0.0),
                ("7", 13.0, 0.0),
                ("9", 25.0, 0.0),
                ("10", 26.0, 0.0),
            ]:
                row[f"Away_{pid}_x"] = float(x)
                row[f"Away_{pid}_y"] = float(y)
            rows.append(row)
        df = pd.DataFrame(rows)
        result = analyze_lines(df, "Away", _EVENT_ATK_DIR, min_line_gap=5.0)
        assert (result["line_count"] == 2).all()

    def test_absent_gk_gets_nan_line(self) -> None:
        df = _event_df()
        df.loc[2, "Away_3_x"] = np.nan
        result = analyze_lines(df, "Away", _EVENT_ATK_DIR)
        assert pd.isna(result.loc[2, "Away_3_line"])
        assert result.loc[0, "Away_3_line"] == 0


# ── smooth_line_assignments ───────────────────────────────────────────────────


class TestSmoothLineAssignments:
    def test_stable_assignments_unchanged(self) -> None:
        df = _event_df()
        lined = analyze_lines(df, "Away", _EVENT_ATK_DIR)
        result = smooth_line_assignments(lined, "Away", window=3)
        for pid in ["2", "6", "7", "9", "10", "11"]:
            col = f"Away_{pid}_line"
            assert result[col].tolist() == lined[col].tolist()

    def test_single_frame_flicker_corrected(self) -> None:
        df = _event_df()
        lined = analyze_lines(df, "Away", _EVENT_ATK_DIR)
        # Inject a one-frame flip on player 2 (usually line 1)
        original_val = float(lined.at[2, "Away_2_line"])
        flipped_val = 3.0 if original_val != 3.0 else 1.0
        lined.at[2, "Away_2_line"] = flipped_val
        result = smooth_line_assignments(lined, "Away", window=5)
        # Middle frame should revert to majority (original) value
        assert int(result.at[2, "Away_2_line"]) == int(original_val)

    def test_all_nan_stays_nan(self) -> None:
        df = _event_df()
        lined = analyze_lines(df, "Away", _EVENT_ATK_DIR)
        lined["Away_14_line"] = np.nan
        result = smooth_line_assignments(lined, "Away")
        assert result["Away_14_line"].isna().all()

    def test_non_line_columns_unchanged(self) -> None:
        df = _event_df()
        lined = analyze_lines(df, "Away", _EVENT_ATK_DIR)
        original_x = lined["Away_2_x"].tolist()
        result = smooth_line_assignments(lined, "Away")
        assert result["Away_2_x"].tolist() == original_x

    def test_returns_copy_not_mutation(self) -> None:
        df = _event_df()
        lined = analyze_lines(df, "Away", _EVENT_ATK_DIR)
        original = lined["Away_2_line"].tolist()
        smooth_line_assignments(lined, "Away")
        assert lined["Away_2_line"].tolist() == original

    def test_gk_line_zero_preserved(self) -> None:
        df = _event_df()
        lined = analyze_lines(df, "Away", _EVENT_ATK_DIR)
        result = smooth_line_assignments(lined, "Away")
        assert (result["Away_3_line"].dropna() == 0).all()

    def test_nan_positions_not_filled_by_smoothing(self) -> None:
        df = _event_df()
        lined = analyze_lines(df, "Away", _EVENT_ATK_DIR)
        # Inject NaN for player 2 at frame 2 (simulating mid-clip absence)
        lined.at[2, "Away_2_line"] = float("nan")
        result = smooth_line_assignments(lined, "Away", window=5)
        # Smoothing must not invent a value where the original was NaN
        assert pd.isna(result.at[2, "Away_2_line"])


# ── analyze_lines force_n_lines ───────────────────────────────────────────────


class TestAnalyzeLinesForceNLines:
    def test_force_two_lines_gives_consistent_count(self) -> None:
        df = _event_df()
        result = analyze_lines(df, "Away", _EVENT_ATK_DIR, force_n_lines=2)
        assert (result["line_count"] == 2).all()

    def test_force_three_lines_gives_consistent_count(self) -> None:
        df = _event_df()
        result = analyze_lines(df, "Away", _EVENT_ATK_DIR, force_n_lines=3)
        assert (result["line_count"] == 3).all()

    def test_force_n_lines_none_matches_default(self) -> None:
        df = _event_df()
        pd.testing.assert_frame_equal(
            analyze_lines(df, "Away", _EVENT_ATK_DIR),
            analyze_lines(df, "Away", _EVENT_ATK_DIR, force_n_lines=None),
        )

    def test_force_n_lines_outfield_players_still_assigned(self) -> None:
        df = _event_df()
        result = analyze_lines(df, "Away", _EVENT_ATK_DIR, force_n_lines=2)
        for pid in ["2", "6", "7", "9", "10", "11"]:
            vals = result[f"Away_{pid}_line"].dropna()
            assert vals.between(1, 4).all()


# ── vote_line_count ───────────────────────────────────────────────────────────


class TestVoteLineCount:
    def test_returns_most_common_count(self) -> None:
        df = pd.DataFrame({"line_count": [3, 3, 3, 2, 3, 4, 3]})
        assert vote_line_count(df) == 3

    def test_ignores_zero_counts(self) -> None:
        df = pd.DataFrame({"line_count": [0, 3, 3, 0, 3]})
        assert vote_line_count(df) == 3

    def test_single_non_zero_frame(self) -> None:
        df = pd.DataFrame({"line_count": [0, 0, 2]})
        assert vote_line_count(df) == 2

    def test_all_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            vote_line_count(pd.DataFrame({"line_count": [0, 0, 0]}))

    def test_two_line_count_from_real_event(self) -> None:
        df = _event_df()
        lined = analyze_lines(df, "Away", _EVENT_ATK_DIR)
        k = vote_line_count(lined)
        assert 1 <= k <= 4


# ── fix_player_assignments ────────────────────────────────────────────────────


class TestFixPlayerAssignments:
    def test_minority_flip_replaced_by_majority(self) -> None:
        # Player 2 is line 2 for 4 frames, line 1 for 1 frame → locked to 2
        df = pd.DataFrame({"Away_2_line": [2.0, 2.0, 1.0, 2.0, 2.0]})
        result = fix_player_assignments(df, "Away")
        assert (result["Away_2_line"] == 2.0).all()

    def test_gk_line_zero_unchanged(self) -> None:
        df = pd.DataFrame(
            {
                "Away_3_line": [0.0, 0.0, 0.0],
                "Away_2_line": [2.0, 1.0, 2.0],
            }
        )
        result = fix_player_assignments(df, "Away")
        assert (result["Away_3_line"] == 0.0).all()

    def test_nan_stays_nan(self) -> None:
        df = pd.DataFrame({"Away_2_line": [2.0, float("nan"), 2.0, 2.0, float("nan")]})
        result = fix_player_assignments(df, "Away")
        assert pd.isna(result.loc[1, "Away_2_line"])
        assert pd.isna(result.loc[4, "Away_2_line"])

    def test_returns_copy_not_mutation(self) -> None:
        df = pd.DataFrame({"Away_2_line": [2.0, 1.0, 2.0]})
        original = df["Away_2_line"].tolist()
        fix_player_assignments(df, "Away")
        assert df["Away_2_line"].tolist() == original

    def test_consistent_player_unchanged(self) -> None:
        df = pd.DataFrame({"Away_2_line": [3.0, 3.0, 3.0, 3.0]})
        result = fix_player_assignments(df, "Away")
        assert (result["Away_2_line"] == 3.0).all()

    def test_non_line_columns_unchanged(self) -> None:
        df = pd.DataFrame(
            {
                "Away_2_x": [10.0, 11.0, 12.0],
                "Away_2_line": [2.0, 1.0, 2.0],
            }
        )
        result = fix_player_assignments(df, "Away")
        assert result["Away_2_x"].tolist() == [10.0, 11.0, 12.0]

    def test_each_player_has_single_line_after_fix(self) -> None:
        df = _event_df()
        lined = analyze_lines(df, "Away", _EVENT_ATK_DIR, force_n_lines=2)
        fixed = fix_player_assignments(lined, "Away")
        line_cols = [
            c for c in fixed.columns if c.startswith("Away_") and c.endswith("_line")
        ]
        for col in line_cols:
            vals = fixed[col].dropna()
            outfield = vals[vals > 0]
            if not outfield.empty:
                assert outfield.nunique() == 1, f"{col} still has multiple values"

    def test_multi_frame_oscillation_resolved(self) -> None:
        # Player oscillates 4 frames in line 1, then 4 frames in line 2, then 3 in 1
        # Majority = line 1 (7 vs 4)
        df = pd.DataFrame(
            {"Away_2_line": [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0]}
        )
        result = fix_player_assignments(df, "Away")
        assert (result["Away_2_line"] == 1.0).all()
