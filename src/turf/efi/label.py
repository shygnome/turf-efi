from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def detect_attack_direction(lines_df: pd.DataFrame, defending_team: str) -> int:
    """
    Return +1 or -1: the x-direction of attack (toward the defending GK's goal).

    Uses the median x of the defending GK across all frames. GK is identified
    by the player column whose _line value is 0. If GK x > 0, the defending goal
    is on the positive side, so the attack runs in the +x direction.
    """
    line_cols = [
        c
        for c in lines_df.columns
        if c.startswith(defending_team + "_") and c.endswith("_line")
    ]
    for col in line_cols:
        if (lines_df[col] == 0).any():
            player_id = col[: -len("_line")].rsplit("_", 1)[1]
            gk_x_col = f"{defending_team}_{player_id}_x"
            if gk_x_col not in lines_df.columns:
                continue
            median_x: float = float(lines_df[gk_x_col].median())
            return 1 if median_x > 0 else -1
    raise ValueError(
        f"No GK (line=0) found for team '{defending_team}' in lines DataFrame"
    )


def compute_conceptual_line_xs(
    frame_row: pd.Series,
    defending_team: str,
    attack_dir: int,
) -> list[float]:
    """
    Return x-positions of conceptual lines [L1_x, L2_x, ..., Ln_x] from a single
    frame row of lines.csv, sorted from most-advanced (L1) to deepest (Ln).

    L1 = first line encountered in the attack direction (the pressing forwards).
    Conceptual order is determined by sorting LEAK line means by ascending norm
    (x * attack_dir), so L1 always has the smallest norm regardless of direction.

    GK (line=0) is excluded.
    """
    line_cols = [
        c
        for c in frame_row.index
        if c.startswith(defending_team + "_") and c.endswith("_line")
    ]

    line_xs: dict[int, list[float]] = {}
    for col in line_cols:
        line_val = frame_row[col]
        if pd.isna(line_val) or int(line_val) == 0:
            continue
        leak_line = int(line_val)
        player_id = col[: -len("_line")].rsplit("_", 1)[1]
        x_col = f"{defending_team}_{player_id}_x"
        if x_col not in frame_row.index:
            continue
        x_val = frame_row[x_col]
        if pd.isna(x_val):
            continue
        line_xs.setdefault(leak_line, []).append(float(x_val))

    if not line_xs:
        return []

    line_means = {k: sum(v) / len(v) for k, v in line_xs.items()}
    # Sort by ascending norm: smallest norm = most advanced (first encountered)
    return sorted(line_means.values(), key=lambda x: x * attack_dir)


def assign_zone(
    ball_x: float,
    conceptual_line_xs: list[float],
    attack_dir: int,
) -> str:
    """
    Map a ball x-position to a zone name given conceptual line positions.

    conceptual_line_xs: [L1_x, L2_x, ..., Ln_x] in conceptual order
                        (L1 = most advanced press, already sorted by ascending norm).
    attack_dir: +1 or -1.

    Zones:
      L1-zone   — ball has not yet beaten the most advanced press
      L2-zone   — ball has beaten L1 but not L2
      ...
      Ln-zone   — ball has beaten L(n-1) but not Ln
      danger-zone — ball has beaten all outfield lines (between last line and GK)
    """
    norm_ball = ball_x * attack_dir
    for i, line_x in enumerate(conceptual_line_xs):
        if norm_ball < line_x * attack_dir:
            return f"L{i + 1}-zone"
    return "danger-zone"


def _zone_index(zone: str, n_lines: int) -> int:
    """Convert zone name to a comparable integer index.

    L1-zone=0, L2-zone=1, ..., Ln-zone=n-1, danger-zone=n.
    """
    if zone == "danger-zone":
        return n_lines
    return int(zone[1 : zone.index("-")]) - 1


def detect_line_break(
    start_ball_x: float,
    end_ball_x: float,
    start_line_xs: list[float],
    end_line_xs: list[float],
    attack_dir: int,
) -> dict[str, Any]:
    """
    Determine whether a pass broke through one or more defensive unit lines.

    start_line_xs: conceptual line x-positions at the start frame of the event.
    end_line_xs:   conceptual line x-positions at the end frame of the event.
    Both lists must be in conceptual order (L1 first, as returned by
    compute_conceptual_line_xs).

    Returns a dict with:
      is_line_breaking  — True if the ball advanced through at least one line
      lines_broken_count — number of conceptual lines crossed (0 if not line breaking)
      lines_broken       — list of conceptual line numbers crossed, e.g. [1, 2]
    """
    start_zone = assign_zone(start_ball_x, start_line_xs, attack_dir)
    end_zone = assign_zone(end_ball_x, end_line_xs, attack_dir)

    start_idx = _zone_index(start_zone, len(start_line_xs))
    end_idx = _zone_index(end_zone, len(end_line_xs))

    if end_idx <= start_idx:
        return {"is_line_breaking": False, "lines_broken_count": 0, "lines_broken": []}

    lines_broken = list(range(start_idx + 1, end_idx + 1))
    return {
        "is_line_breaking": True,
        "lines_broken_count": len(lines_broken),
        "lines_broken": lines_broken,
    }


def get_conceptual_line_player_positions(
    frame_row: pd.Series[Any],
    defending_team: str,
    attack_dir: int,
) -> dict[int, list[tuple[float, float]]]:
    """Return {conceptual_line_idx (1-based) -> [(x, y), ...]} for a single frame row.

    Conceptual ordering matches compute_conceptual_line_xs: L1 = most advanced press.
    GK (line=0) is excluded.
    """
    line_xy: dict[int, list[tuple[float, float]]] = {}
    line_cols = [
        c
        for c in frame_row.index
        if c.startswith(defending_team + "_") and c.endswith("_line")
    ]
    for col in line_cols:
        line_val = frame_row[col]
        if pd.isna(line_val) or int(line_val) == 0:
            continue
        leak_line = int(line_val)
        player_id = col[: -len("_line")].rsplit("_", 1)[1]
        x_col = f"{defending_team}_{player_id}_x"
        y_col = f"{defending_team}_{player_id}_y"
        if x_col not in frame_row.index or y_col not in frame_row.index:
            continue
        x_val = frame_row[x_col]
        y_val = frame_row[y_col]
        if pd.isna(x_val) or pd.isna(y_val):
            continue
        line_xy.setdefault(leak_line, []).append((float(x_val), float(y_val)))

    if not line_xy:
        return {}

    leak_line_means = {k: sum(x for x, y in v) / len(v) for k, v in line_xy.items()}
    sorted_leaks = sorted(
        leak_line_means.keys(), key=lambda k: leak_line_means[k] * attack_dir
    )
    return {i + 1: line_xy[leak] for i, leak in enumerate(sorted_leaks)}


def find_line_crossing_frame(
    lines_df: pd.DataFrame,
    defending_team: str,
    attack_dir: int,
    conceptual_line_idx: int,
) -> int:
    """Return the first frame index where ball_x crosses conceptual line idx's mean x.

    If the ball starts already past the line, returns 0.
    If the ball never crosses the line, returns the last frame index.
    """
    for i, (_, row) in enumerate(lines_df.iterrows()):
        positions = get_conceptual_line_player_positions(
            row, defending_team, attack_dir
        )
        if conceptual_line_idx not in positions:
            continue
        line_xs = [x for x, y in positions[conceptual_line_idx]]
        if not line_xs:
            continue
        line_mean_x = sum(line_xs) / len(line_xs)
        ball_x = float(row["ball_x"])
        if ball_x * attack_dir >= line_mean_x * attack_dir:
            return i
    return len(lines_df) - 1


def classify_pass_direction(ball_y: float, player_ys: list[float]) -> str:
    """Return "Through" if ball_y is within the y-extent of player_ys, else "Around"."""
    if not player_ys:
        return "Around"
    return "Through" if min(player_ys) <= ball_y <= max(player_ys) else "Around"


def is_point_in_convex_hull(
    point: tuple[float, float],
    positions: list[tuple[float, float]],
) -> bool:
    """Return True if point is inside or on the convex hull of positions.

    Requires at least 3 positions. Returns False for degenerate inputs.
    """
    if len(positions) < 3:
        return False
    from scipy.spatial import Delaunay  # type: ignore[import-untyped]

    try:
        hull = Delaunay(np.array(positions, dtype=np.float64))
        return bool(hull.find_simplex(np.array(point, dtype=np.float64)) >= 0)
    except Exception:
        return False


def compute_direction_labels(
    lines_df: pd.DataFrame,
    defending_team: str,
    attack_dir: int,
    lines_broken: list[int],
) -> list[str]:
    """Return "Through" or "Around" for each broken conceptual line."""
    result: list[str] = []
    for line_idx in lines_broken:
        frame_i = find_line_crossing_frame(
            lines_df, defending_team, attack_dir, line_idx
        )
        row = lines_df.iloc[frame_i]
        positions = get_conceptual_line_player_positions(
            row, defending_team, attack_dir
        )
        player_ys = [y for x, y in positions.get(line_idx, [])]
        ball_y = float(row["ball_y"])
        result.append(classify_pass_direction(ball_y, player_ys))
    return result


def compute_all_adjacent_hull_vertices(
    lines_df: pd.DataFrame,
    defending_team: str,
    attack_dir: int,
) -> list[list[tuple[float, float]]]:
    """Return convex hull vertices for each adjacent conceptual line pair.

    For K conceptual lines returns up to K-1 hulls: (L1,L2), (L2,L3), ...
    Pairs with fewer than 3 combined players are skipped.
    """
    from scipy.spatial import ConvexHull

    last_row = lines_df.iloc[-1]
    positions = get_conceptual_line_player_positions(
        last_row, defending_team, attack_dir
    )
    n_lines = max(positions.keys(), default=0)
    hulls: list[list[tuple[float, float]]] = []
    for i in range(1, n_lines):
        pair: list[tuple[float, float]] = list(positions.get(i, [])) + list(
            positions.get(i + 1, [])
        )
        if len(pair) < 3:
            continue
        try:
            pts = np.array(pair, dtype=np.float64)
            hull = ConvexHull(pts)
            hulls.append([(float(pts[j, 0]), float(pts[j, 1])) for j in hull.vertices])
        except Exception:
            pass
    return hulls


def compute_location_label(
    lines_df: pd.DataFrame,
    defending_team: str,
    attack_dir: int,
    end_ball_x: float,
    end_ball_y: float,
) -> str | None:
    """Return "Inside" if ball end position falls inside any adjacent-pair convex hull.

    Returns None when fewer than 2 conceptual lines exist (no pair can be formed).
    """
    last_row = lines_df.iloc[-1]
    positions = get_conceptual_line_player_positions(
        last_row, defending_team, attack_dir
    )
    n_lines = max(positions.keys(), default=0)
    if n_lines < 2:
        return None

    endpoint = (end_ball_x, end_ball_y)
    for i in range(1, n_lines):
        pair: list[tuple[float, float]] = list(positions.get(i, [])) + list(
            positions.get(i + 1, [])
        )
        if is_point_in_convex_hull(endpoint, pair):
            return "Inside"
    return "Outside"


def compute_team_hull_vertices(
    lines_df: pd.DataFrame,
    defending_team: str,
    attack_dir: int,
) -> list[list[tuple[float, float]]]:
    """Return convex hull vertices for each adjacent conceptual line pair.

    Returns K-1 hulls for K lines: (L1,L2), (L2,L3), ...
    Returns an empty list when no valid hull can be formed.
    """
    return compute_all_adjacent_hull_vertices(lines_df, defending_team, attack_dir)


def compute_location_hull_vertices(
    lines_df: pd.DataFrame,
    defending_team: str,
    attack_dir: int,
    end_line_xs: list[float],
    end_ball_x: float,
) -> list[tuple[float, float]] | None:
    """Return convex hull polygon vertices for the adjacent line pair at the end zone.

    Same zone selection as compute_location_label. Returns None for L1-zone,
    danger-zone, or when fewer than 3 players form the pair.
    """
    end_zone = assign_zone(end_ball_x, end_line_xs, attack_dir)
    if end_zone == "L1-zone" or end_zone == "danger-zone":
        return None

    zone_idx = _zone_index(end_zone, len(end_line_xs))
    line_a_idx = zone_idx
    line_b_idx = zone_idx + 1

    last_row = lines_df.iloc[-1]
    positions = get_conceptual_line_player_positions(
        last_row, defending_team, attack_dir
    )
    pair_positions: list[tuple[float, float]] = []
    pair_positions.extend(positions.get(line_a_idx, []))
    pair_positions.extend(positions.get(line_b_idx, []))

    if len(pair_positions) < 3:
        return None

    from scipy.spatial import ConvexHull

    try:
        pts = np.array(pair_positions, dtype=np.float64)
        hull = ConvexHull(pts)
        return [(float(pts[i, 0]), float(pts[i, 1])) for i in hull.vertices]
    except Exception:
        return None


def label_event(event_dir: Path, defending_team: str) -> dict[str, Any]:
    """
    Compute line-break labels for a single pass event.

    Reads lines.csv from event_dir. Uses the first frame for start line positions
    and ball position, and the last frame for end line positions and ball position.

    Returns a dict suitable for merging into metadata.
    """
    lines_path = event_dir / "lines.csv"
    lines_df = pd.read_csv(lines_path)

    attack_dir = detect_attack_direction(lines_df, defending_team)

    first_row = lines_df.iloc[0]
    last_row = lines_df.iloc[-1]

    start_line_xs = compute_conceptual_line_xs(first_row, defending_team, attack_dir)
    end_line_xs = compute_conceptual_line_xs(last_row, defending_team, attack_dir)

    start_ball_x = float(first_row["ball_x"])
    end_ball_x = float(last_row["ball_x"])
    end_ball_y = float(last_row["ball_y"])

    result = detect_line_break(
        start_ball_x, end_ball_x, start_line_xs, end_line_xs, attack_dir
    )

    if result["is_line_breaking"]:
        result["direction_per_line"] = compute_direction_labels(
            lines_df, defending_team, attack_dir, result["lines_broken"]
        )
        result["location_after_break"] = compute_location_label(
            lines_df, defending_team, attack_dir, end_ball_x, end_ball_y
        )
    else:
        result["direction_per_line"] = []
        result["location_after_break"] = None

    return result


def label_all_passes(
    output_dir: Path,
    dataset_id: str,
    match_id: str,
) -> pd.DataFrame:
    """
    Label all pass events for a match and return the labeled DataFrame.

    Reads from output_dir/<dataset_id>/<match_id>/pass/metadata.csv and the
    per-event lines.csv files produced by `turf analyze leak extract-line`.

    Writes labeled_metadata.csv next to metadata.csv and returns the DataFrame.
    """
    pass_dir = output_dir / dataset_id / match_id / "pass"
    meta_path = pass_dir / "metadata.csv"
    metadata = pd.read_csv(meta_path)

    label_rows: list[dict[str, Any]] = []
    for _, row in metadata.iterrows():
        event_idx = int(row["event_idx"])
        attacking_team = str(row["team"])
        defending_team = "Away" if attacking_team == "Home" else "Home"

        event_dir = pass_dir / str(event_idx)
        lines_path = event_dir / "lines.csv"
        _skip_row: dict[str, Any] = {
            "event_idx": event_idx,
            "is_line_breaking": None,
            "lines_broken_count": None,
            "lines_broken": None,
            "direction_per_line": None,
            "location_after_break": None,
        }

        if not lines_path.exists():
            label_rows.append(_skip_row)
            continue

        try:
            labels = label_event(event_dir, defending_team)
        except (ValueError, KeyError):
            label_rows.append(_skip_row)
            continue

        label_rows.append(
            {
                "event_idx": event_idx,
                "is_line_breaking": labels["is_line_breaking"],
                "lines_broken_count": labels["lines_broken_count"],
                "lines_broken": str(labels["lines_broken"]),
                "direction_per_line": str(labels["direction_per_line"]),
                "location_after_break": labels["location_after_break"],
            }
        )

    labels_df = pd.DataFrame(label_rows)
    labeled = metadata.merge(labels_df, on="event_idx", how="left")
    labeled.to_csv(pass_dir / "labeled_metadata.csv", index=False)
    return labeled


def compute_pass_stats(df: pd.DataFrame) -> dict[str, Any]:
    """
    Compute summary statistics from a labeled_metadata DataFrame.

    Rows where is_line_breaking is NaN/None are treated as skipped (no lines.csv).
    by_team and by_period totals count only labeled (non-skipped) rows.
    """
    total = len(df)
    labeled_mask = df["is_line_breaking"].notna()
    n_labeled = int(labeled_mask.sum())
    n_skipped = total - n_labeled

    labeled = df[labeled_mask]
    n_breaking = int(labeled["is_line_breaking"].eq(True).sum())
    n_not_breaking = n_labeled - n_breaking

    breaking = labeled[labeled["is_line_breaking"].eq(True)]
    lines_broken_dist: dict[int, int] = {}
    if not breaking.empty:
        for k, cnt in breaking["lines_broken_count"].value_counts().items():
            lines_broken_dist[int(k)] = int(cnt)  # type: ignore[call-overload]

    def _group_stats(col: str) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for key in sorted(labeled[col].dropna().unique()):
            mask = labeled[col] == key
            result[str(key)] = {
                "breaking": int(labeled[mask]["is_line_breaking"].eq(True).sum()),
                "total": int(mask.sum()),
            }
        return result

    return {
        "total": total,
        "n_labeled": n_labeled,
        "n_skipped": n_skipped,
        "n_breaking": n_breaking,
        "n_not_breaking": n_not_breaking,
        "lines_broken_dist": lines_broken_dist,
        "by_team": _group_stats("team"),
        "by_period": {int(k): v for k, v in _group_stats("period").items()},
        "by_subtype": _group_stats("subtype"),
    }
