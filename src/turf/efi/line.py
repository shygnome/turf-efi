from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage  # type: ignore[import-untyped]


def find_goalkeeper(df: pd.DataFrame, team: str) -> str:
    """Return player ID of the goalkeeper by majority vote across all frames.

    For each frame the player with the highest abs(x) is nominated; the player
    nominated most often across the clip is returned.  This is robust to
    occasional tracklet swaps that would fool a single-frame heuristic.
    """
    x_cols = [c for c in df.columns if c.startswith(team + "_") and c.endswith("_x")]
    if not x_cols:
        raise ValueError(f"No {team} player positions found")

    votes: dict[str, int] = {}
    for _, row in df.iterrows():
        valid = row[x_cols].dropna()
        if valid.empty:
            continue
        gk_col = str(valid.abs().idxmax())
        player_num = gk_col.rsplit("_x", 1)[0].rsplit("_", 1)[1]
        votes[player_num] = votes.get(player_num, 0) + 1

    if not votes:
        raise ValueError(f"No {team} player positions in any frame")

    return max(votes, key=lambda p: votes[p])


def _calinski_harabasz(
    x: npt.NDArray[np.float64], labels: npt.NDArray[np.intp]
) -> float:
    """CH index for 1-D data; 0.0 when undefined (k<2 or k>=n)."""
    n = len(x)
    clusters = np.unique(labels)
    k = len(clusters)
    if k < 2 or k >= n:
        return 0.0
    overall_mean = x.mean()
    ssb = sum(
        float((labels == c).sum()) * float((x[labels == c].mean() - overall_mean) ** 2)
        for c in clusters
    )
    ssw = sum(
        float(((x[labels == c] - x[labels == c].mean()) ** 2).sum()) for c in clusters
    )
    if ssw < 1e-12:
        return float("inf")
    return (ssb / (k - 1)) / (ssw / (n - k))


def _cluster(x: npt.NDArray[np.float64], k: int) -> npt.NDArray[np.intp]:
    """Ward agglomerative clustering; returns 0-indexed labels."""
    if k == 1:
        return np.zeros(len(x), dtype=np.intp)
    Z: Any = linkage(x.reshape(-1, 1), method="ward")
    result: npt.NDArray[np.intp] = (
        fcluster(Z, t=k, criterion="maxclust").astype(np.intp) - 1
    )
    return result


def _merge_singletons(
    x: npt.NDArray[np.float64], labels: npt.NDArray[np.intp], min_size: int
) -> npt.NDArray[np.intp]:
    """Merge clusters smaller than min_size into the nearest cluster by mean x."""
    labels = labels.copy()
    changed = True
    while changed:
        changed = False
        unique: npt.NDArray[np.intp]
        counts: npt.NDArray[np.intp]
        unique, counts = np.unique(labels, return_counts=True)
        for c, cnt in zip(unique, counts, strict=False):
            if cnt < min_size:
                c_mean = float(x[labels == c].mean())
                others = [o for o in unique if o != c]
                if not others:
                    break

                def _key(o: np.intp, cm: float = c_mean) -> float:
                    return abs(float(x[labels == o].mean()) - cm)

                nearest = min(others, key=_key)
                labels[labels == c] = nearest
                changed = True
                break  # restart after each merge
    return labels


def detect_lines_frame(
    x_positions: dict[str, float],
    min_per_line: int = 2,
    min_lines: int = 2,
    max_lines: int = 4,
    min_line_gap: float = 0.0,
) -> dict[str, int]:
    """
    Detect defensive unit lines for outfield players in a single frame.

    x_positions: {player_number -> x coordinate} — GK already excluded
    Returns: {player_number -> line_number} where 1 = deepest (lowest x)
    """
    if not x_positions:
        return {}

    players = list(x_positions.keys())
    x: npt.NDArray[np.float64] = np.array(
        [x_positions[p] for p in players], dtype=np.float64
    )
    n = len(players)

    if n == 1:
        return {players[0]: 1}

    lo = max(1, min(min_lines, n // min_per_line))
    hi = min(max_lines, n // min_per_line)

    best_labels: npt.NDArray[np.intp] | None = None
    best_score = -1.0

    for k in range(lo, hi + 1):
        labels = _cluster(x, k)
        unique: npt.NDArray[np.intp]
        counts: npt.NDArray[np.intp]
        unique, counts = np.unique(labels, return_counts=True)
        valid = (
            len(unique) == k
            and len(unique) >= min_lines
            and bool(np.all(counts >= min_per_line))
        )
        if valid:
            score = _calinski_harabasz(x, labels)
            if score > best_score:
                best_score = score
                best_labels = labels.copy()

    if best_labels is None:
        # Fallback: no clustering met all constraints; use smallest allowed k.
        # Only accept singleton merges that don't drop below min_lines.
        k = max(1, min(min_lines, max(1, n // min_per_line)))
        best_labels = _cluster(x, k)
        merged = _merge_singletons(x, best_labels, min_per_line)
        if len(np.unique(merged)) >= min_lines:
            best_labels = merged

    # Merge adjacent lines that are too close together
    assert best_labels is not None
    if min_line_gap > 0.0:
        changed = True
        while changed:
            changed = False
            uc = np.unique(best_labels)
            if len(uc) <= min_lines:
                break
            means = {int(c): float(x[best_labels == c].mean()) for c in uc}
            sorted_c = sorted(uc, key=lambda c: means[int(c)])
            for i in range(len(sorted_c) - 1):
                c1, c2 = int(sorted_c[i]), int(sorted_c[i + 1])
                if means[c2] - means[c1] < min_line_gap:
                    s1 = int((best_labels == c1).sum())
                    s2 = int((best_labels == c2).sum())
                    keep, drop = (c1, c2) if s1 >= s2 else (c2, c1)
                    best_labels = best_labels.copy()
                    best_labels[best_labels == drop] = keep
                    changed = True
                    break

    # Remap to 1-indexed labels ordered by mean x (1 = deepest)
    unique_clusters = np.unique(best_labels)
    ordered = sorted(unique_clusters, key=lambda c: float(x[best_labels == c].mean()))
    remap = {int(old): new + 1 for new, old in enumerate(ordered)}

    return {p: remap[int(best_labels[i])] for i, p in enumerate(players)}


def smooth_line_assignments(
    df: pd.DataFrame, team: str, window: int = 5
) -> pd.DataFrame:
    """Apply rolling majority-vote to line assignments to reduce flicker."""
    result = df.copy()
    line_cols = [
        c for c in df.columns if c.startswith(team + "_") and c.endswith("_line")
    ]

    def _mode(arr: npt.NDArray[np.float64]) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return float("nan")
        counts = np.bincount(valid.astype(np.intp))
        return float(np.argmax(counts))

    for col in line_cols:
        series: pd.Series = df[col]
        if series.isna().all():
            continue
        smoothed = series.rolling(window, center=True, min_periods=1).apply(
            _mode, raw=True
        )
        # Preserve original NaN positions — smoothing must not invent values
        result[col] = smoothed.where(series.notna())
    return result


def fix_player_assignments(lines_df: pd.DataFrame, team: str) -> pd.DataFrame:
    """Lock each outfield player to their majority line assignment across all frames.

    After force_n_lines stabilises K, individual players may still oscillate between
    adjacent lines as they move across cluster boundaries.  This replaces each
    player's per-frame assignment with the line they occupy most often, making
    the stored assignments fully stable across the clip.

    GK assignments (line == 0) and NaN positions are preserved unchanged.
    """
    result = lines_df.copy()
    line_cols = [
        c for c in lines_df.columns if c.startswith(team + "_") and c.endswith("_line")
    ]
    for col in line_cols:
        series = lines_df[col]
        outfield_mask = series.notna() & (series > 0)
        outfield = series[outfield_mask]
        if outfield.empty:
            continue
        majority = float(int(outfield.mode()[0]))
        result.loc[outfield_mask, col] = majority
    return result


def vote_line_count(lines_df: pd.DataFrame) -> int:
    """Return the most common non-zero line count from an analyze_lines result."""
    counts = lines_df["line_count"]
    non_zero = counts[counts > 0]
    if non_zero.empty:
        raise ValueError("No non-zero line counts found in lines_df")
    return int(non_zero.mode()[0])


def analyze_lines(
    df: pd.DataFrame,
    team: str,
    min_line_gap: float = 0.0,
    force_n_lines: int | None = None,
) -> pd.DataFrame:
    """
    Add unit line assignments to every frame in df.

    Adds {team}_{player}_line columns: 0 for GK, 1-4 for outfield, NaN if absent.
    Also adds line_count column with the number of lines detected per frame.
    """
    result: pd.DataFrame = df.copy()
    gk_num = find_goalkeeper(df, team)
    gk_x_col = f"{team}_{gk_num}_x"

    x_cols = [
        c
        for c in df.columns
        if c.startswith(team + "_") and c.endswith("_x") and c != gk_x_col
    ]
    player_nums = [c.rsplit("_x", 1)[0].rsplit("_", 1)[1] for c in x_cols]

    result[f"{team}_{gk_num}_line"] = np.nan
    for num in player_nums:
        result[f"{team}_{num}_line"] = np.nan
    result["line_count"] = 0

    for idx, row in df.iterrows():
        if pd.notna(row[gk_x_col]):
            result.at[idx, f"{team}_{gk_num}_line"] = 0

        x_pos: dict[str, float] = {}
        for col, num in zip(x_cols, player_nums, strict=False):
            val = row[col]
            if pd.notna(val):
                x_pos[num] = float(val)

        if force_n_lines is not None:
            lines = detect_lines_frame(
                x_pos,
                min_line_gap=min_line_gap,
                min_lines=force_n_lines,
                max_lines=force_n_lines,
            )
        else:
            lines = detect_lines_frame(x_pos, min_line_gap=min_line_gap)
        for num, line_num in lines.items():
            result.at[idx, f"{team}_{num}_line"] = float(line_num)
        result.at[idx, "line_count"] = len(set(lines.values())) if lines else 0

    return result
