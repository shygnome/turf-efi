"""Pitch visualization helpers for turf.efi outputs.

Developing module — see CLAUDE.md stability tiers.

Requires the viz dependency group:
    uv sync --group viz

Themes
------
Light theme (default) matches the jfa-wc26-analysis reference style.
Pass dark=True to get the dark grass style.

Data conventions
----------------
Tracking frames are wide-format DataFrames with columns:
    period, time_s, {team}_{n}_x, {team}_{n}_y, ball_x, ball_y
    (team label is lowercase: 'home' or 'away')

After analyze_lines, each player also has a {team}_{n}_line column:
    0 = goalkeeper, 1–4 = outfield defensive lines (front to back)

Coordinates use the SkillCorner/centered system:
    x: −52.5 (own goal end) → +52.5 (opponent goal end)
    y: −34 (left touchline) → +34 (right touchline)

Player index n in {team}_{n}_x is used as the jersey number for display.
"""

from __future__ import annotations

import collections
import contextlib
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.animation import FuncAnimation  # type: ignore[import-not-found]
    from matplotlib.axes import Axes  # type: ignore[attr-defined]
    from mplsoccer import Pitch as PitchType  # type: ignore[import-untyped]

# ── pitch / coordinate constants ───────────────────────────────────────────────

_PITCH_TYPE   = "skillcorner"
_PITCH_LENGTH = 105
_PITCH_WIDTH  = 68
_HALF_WIDTH   = _PITCH_WIDTH / 2    # 34.0
_YLIM_BOTTOM  = -42.0
_LABEL_Y      = -38.0

# ── themes (matches jfa-wc26-analysis animate_reception_event.py) ──────────────

_LIGHT_THEME: dict[str, str] = {
    "pitch_color": "#f5f5f0",
    "line_color":  "#999999",
    "bg_color":    "#ffffff",
    "text_color":  "#111111",
    "edge_color":  "#333333",
    "num_color":   "white",    # jersey number text drawn on team-colored dots
    "bar_bg":      "#e8e8e8",  # depth panel + legend background
}

_DARK_THEME: dict[str, str] = {
    "pitch_color": "#1a7a1a",
    "line_color":  "white",
    "bg_color":    "#0d0d0d",
    "text_color":  "white",
    "edge_color":  "white",
    "num_color":   "black",
    "bar_bg":      "#222244",
}

# ── marker / color constants ───────────────────────────────────────────────────

_TEAM_COLORS: dict[str, str] = {
    "Home": "#4e79a7",
    "Away": "#e15759",
}

_DEFENSIVE_LINE_COLORS: dict[int, str] = {
    1: "#4e79a7",
    2: "#59a14f",
    3: "#f28e2b",
    4: "#e15759",
}

_BALL_COLOR = "white"
_BALL_EDGE  = "#555555"

_TRAIL_LEN       = 15
_TRAIL_ALPHA_MIN = 0.06
_TRAIL_ALPHA_MAX = 0.50
_MAX_LINES       = 4

# Line depth panel (axes-fraction units, bottom-left corner)
_DP_X0     = 0.04    # left edge of bar track
_DP_Y0     = 0.06    # bottom bar centre-y
_DP_BAR_H  = 0.038   # bar height
_DP_STRIDE = 0.065   # vertical gap between bar centres
_DP_W_MAX  = 0.20    # max bar width (proportion of axes width)
_DP_MAX_M  = 52.5    # x-axis max for depth normalisation


# ── public pitch factory ───────────────────────────────────────────────────────


def make_pitch(
    half: bool = False,
    dark: bool = False,
) -> tuple[PitchType, dict[str, str]]:
    """Return a configured (Pitch, theme) pair.

    Light theme (default) matches the jfa-wc26-analysis reference style.
    Pass dark=True for the dark grass / dark background style.

    Usage::

        pitch, theme = make_pitch()
        fig, ax = pitch.draw(figsize=(12, 8))
        fig.patch.set_facecolor(theme["bg_color"])
    """
    from mplsoccer import Pitch

    theme = _DARK_THEME if dark else _LIGHT_THEME
    pitch = Pitch(
        pitch_type=_PITCH_TYPE,
        pitch_length=_PITCH_LENGTH,
        pitch_width=_PITCH_WIDTH,
        pitch_color=theme["pitch_color"],
        line_color=theme["line_color"],
        linewidth=0.8,
        half=half,
    )
    return pitch, theme


# ── static frame plots ─────────────────────────────────────────────────────────


def plot_frame(
    ax: Axes,
    frame: pd.Series,
    home_team: str = "Home",
    away_team: str = "Away",
    theme: dict[str, str] | None = None,
) -> None:
    """Plot players and ball for a single wide-format tracking row.

    home_team players are drawn solid; away_team at reduced alpha.
    GK: square (s) marker.  Outfielders: circle (o) marker.
    Jersey number (player index n) shown inside each marker.
    """
    if theme is None:
        theme = _LIGHT_THEME

    for team, alpha in [(home_team, 1.0), (away_team, 0.55)]:
        color = _TEAM_COLORS.get(team, "#aaaaaa")
        gk_id = _find_gk_id(frame, team)
        n = 1
        while True:
            xc, yc = f"{team}_{n}_x", f"{team}_{n}_y"
            if xc not in frame.index:
                break
            try:
                xf, yf = float(frame[xc]), float(frame[yc])
                if not (math.isnan(xf) or math.isnan(yf)):
                    _draw_player(
                        ax, xf, yf, color,
                        is_gk=(str(n) == gk_id),
                        label=str(n),
                        alpha=alpha,
                        edge_color=theme["edge_color"],
                        num_color=theme["num_color"],
                    )
            except (TypeError, ValueError):
                pass
            n += 1

    # Ball
    try:
        bx, by = float(frame["ball_x"]), float(frame["ball_y"])
        if not (math.isnan(bx) or math.isnan(by)):
            ax.scatter(
                [bx], [by],
                c=_BALL_COLOR, s=100, zorder=7,
                edgecolors=_BALL_EDGE, linewidths=1.5,
            )
    except (KeyError, TypeError, ValueError):
        pass


def plot_lines_frame(
    ax: Axes,
    frame: pd.Series,
    defending_team: str,
    attacking_team: str,
    theme: dict[str, str] | None = None,
) -> None:
    """Plot a tracking frame with defensive unit line overlays.

    frame must have {defending_team}_{n}_line columns (from analyze_lines output).
    Each non-GK line is drawn as a zig-zag with a dashed representative-x vertical,
    labelled L1…L4 (L1 = most advanced line).
    """
    if theme is None:
        theme = _LIGHT_THEME

    # Identify defending GK from the _line column (line==0) so marker shape
    # matches the algorithm's GK assignment, not a position heuristic.
    def_gk_id = _find_gk_id_from_lines(frame, defending_team)
    _draw_team(ax, frame, defending_team, alpha=1.0, theme=theme, gk_id=def_gk_id)
    _draw_team(ax, frame, attacking_team, alpha=0.55, theme=theme)

    # Ball
    try:
        bx, by = float(frame["ball_x"]), float(frame["ball_y"])
        if not (math.isnan(bx) or math.isnan(by)):
            ax.scatter(
                [bx], [by],
                c=_BALL_COLOR, s=100, zorder=7,
                edgecolors=_BALL_EDGE, linewidths=1.5,
            )
    except (KeyError, TypeError, ValueError):
        pass

    gk_x = _gk_x(frame, defending_team)
    lines_players: dict[int, list[tuple[float, float]]] = {}
    n = 1
    while True:
        xc = f"{defending_team}_{n}_x"
        if xc not in frame.index:
            break
        yc       = f"{defending_team}_{n}_y"
        line_col = f"{defending_team}_{n}_line"
        try:
            xf     = float(frame[xc])
            yf     = float(frame[yc])
            ln_raw = (
                float(frame[line_col])
                if line_col in frame.index else float("nan")
            )
            if not (math.isnan(xf) or math.isnan(yf) or math.isnan(ln_raw)):
                lines_players.setdefault(int(ln_raw), []).append((xf, yf))
        except (TypeError, ValueError):
            pass
        n += 1

    active = {
        ln: pts
        for ln, pts in lines_players.items()
        if ln != 0 and pts
    }
    sorted_lns = sorted(active, key=lambda ln: abs(_repr_x(active[ln], gk_x)))
    for user_idx, ln in enumerate(sorted_lns, start=1):
        pts   = active[ln]
        color = _DEFENSIVE_LINE_COLORS.get(user_idx, "white")
        sp    = sorted(pts, key=lambda p: p[1])
        if len(sp) >= 2:
            ax.plot(
                [p[0] for p in sp], [p[1] for p in sp],
                color=color, lw=2.0, alpha=0.85, zorder=5,
            )
        rx = _repr_x(pts, gk_x)
        ax.axvline(rx, color=color, linestyle="--", lw=1.2, alpha=0.5, zorder=3)
        ax.text(
            rx, _LABEL_Y, f"L{user_idx}",
            color=color, fontsize=9, ha="center", va="top",
            fontweight="bold", zorder=6,
        )


# ── animation ─────────────────────────────────────────────────────────────────


def animate_lines(
    defending_frames: pd.DataFrame,
    attacking_frames: pd.DataFrame,
    defending_team: str,
    attacking_team: str,
    metadata: dict[str, Any],
    fps: float = 25.0,
    smooth_lines: bool = True,
    dark: bool = False,
    show_depth_panel: bool = True,
    pass_labels: dict[str, Any] | None = None,
    freeze_duration: float = 5.0,
) -> FuncAnimation:
    """Return a FuncAnimation stepping through a tracking clip with line overlays.

    defending_frames / attacking_frames must have {team}_{n}_line columns
    (output of analyze_lines + fix_player_assignments).

    pass_labels (optional) triggers a reveal phase after the tracking ends:
      {
        "is_line_breaking": bool,
        "lines_broken_count": int,
        "direction_per_line": list[str],
        "location_after_break": str | None,
        "hull_vertices": list[list[tuple[float, float]]],
      }
    """
    from matplotlib.animation import FuncAnimation
    from matplotlib.lines import Line2D  # type: ignore[import-not-found]

    from turf.efi.line import smooth_line_assignments

    if smooth_lines:
        defending_frames = smooth_line_assignments(defending_frames, defending_team)

    pitch, theme = make_pitch(dark=dark)
    fig, ax      = pitch.draw(figsize=(13, 8.5))
    ax.set_ylim(_YLIM_BOTTOM, _HALF_WIDTH)
    fig.patch.set_facecolor(theme["bg_color"])

    n_frames = min(len(defending_frames), len(attacking_frames))
    n_reveal = int(freeze_duration * fps) if pass_labels is not None else 0
    period   = metadata.get("period", 1)

    # Stable L-label mapping: L1 = most advanced line
    gk_x0   = _gk_x(defending_frames.iloc[0], defending_team)
    last_lp  = _players_by_line(defending_frames.iloc[-1], defending_team)
    last_rx: dict[int, float] = {
        ln: _repr_x(last_lp[ln], gk_x0)
        for ln in range(1, _MAX_LINES + 1)
        if last_lp.get(ln)
    }
    ln_to_user: dict[int, int] = {
        ln: i + 1
        for i, ln in enumerate(sorted(last_rx, key=lambda k: abs(last_rx[k])))
    }

    start_ball = _ball_xy(defending_frames.iloc[0])
    end_ball   = _ball_xy(defending_frames.iloc[-1])

    # Ball trail deque
    ball_trail: collections.deque[tuple[float, float] | None] = collections.deque(
        maxlen=_TRAIL_LEN
    )

    # Legend (fig-level so ax.cla() doesn't remove it)
    def_color = _TEAM_COLORS.get(defending_team, "#aaaaaa")
    atk_color = _TEAM_COLORS.get(attacking_team, "#aaaaaa")
    fig.legend(
        handles=[
            Line2D(
                [], [], color=def_color, marker="o", ms=7, lw=0,
                label=f"{defending_team} (def.)",
            ),
            Line2D(
                [], [], color=def_color, marker="s", ms=7, lw=0,
                label=f"{defending_team} GK",
            ),
            Line2D(
                [], [], color=atk_color, marker="o", ms=7, lw=0, alpha=0.5,
                label=f"{attacking_team} (atk.)",
            ),
            Line2D(
                [], [], color=atk_color, marker="s", ms=7, lw=0, alpha=0.5,
                label=f"{attacking_team} GK",
            ),
            Line2D(
                [], [], color=_BALL_COLOR, marker="o", ms=7, lw=0,
                markeredgecolor=_BALL_EDGE, label="Ball",
            ),
        ],
        loc="lower center",
        ncol=5,
        facecolor=theme["bar_bg"],
        labelcolor=theme["text_color"],
        fontsize=7.5,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )

    def _update(frame_i: int) -> None:
        eff    = min(frame_i, n_frames - 1)
        reveal = frame_i - n_frames

        ax.cla()
        pitch.draw(ax=ax)
        ax.set_ylim(_YLIM_BOTTOM, _HALF_WIDTH)

        row_def = defending_frames.iloc[eff]
        row_atk = attacking_frames.iloc[eff]

        # ── ball trail ────────────────────────────────────────────────────────
        if reveal < 0:
            ball_trail.append(_ball_xy(row_def))

        n_tr = len(ball_trail)
        for tidx, bp in enumerate(ball_trail):
            if bp is None:
                continue
            alpha  = _TRAIL_ALPHA_MIN + (_TRAIL_ALPHA_MAX - _TRAIL_ALPHA_MIN) * (
                tidx / max(n_tr - 1, 1)
            )
            sz_fac = 0.3 + 0.5 * (tidx / max(n_tr - 1, 1))
            ax.scatter(
                bp[0], bp[1],
                c=_BALL_COLOR, s=int(60 * sz_fac),
                alpha=alpha, zorder=3, edgecolors="none",
            )

        # ── players ───────────────────────────────────────────────────────────
        def_gk = _find_gk_id_from_lines(row_def, defending_team)
        _draw_team(ax, row_def, defending_team, alpha=1.0, theme=theme, gk_id=def_gk)
        _draw_team(ax, row_atk, attacking_team, alpha=0.5, theme=theme)

        # ── unit lines + depth panel ──────────────────────────────────────────
        line_players = _players_by_line(row_def, defending_team)
        active_lns: dict[int, list[tuple[float, float]]] = {
            ln: pts
            for ln, pts in line_players.items()
            if ln != 0 and pts
        }
        for ln, pts in active_lns.items():
            user_ln = ln_to_user.get(ln, ln)
            color   = _DEFENSIVE_LINE_COLORS.get(user_ln, "white")
            sp      = sorted(pts, key=lambda p: p[1])
            if len(sp) >= 2:
                ax.plot(
                    [p[0] for p in sp], [p[1] for p in sp],
                    color=color, lw=2.0, alpha=0.85, zorder=5,
                )
            rx = _repr_x(pts, gk_x0)
            ax.axvline(rx, color=color, linestyle="--", lw=1.2, alpha=0.4, zorder=3)
            ax.text(
                rx, _LABEL_Y, f"L{user_ln}",
                color=color, fontsize=8, ha="center", va="top",
                fontweight="bold", zorder=6,
            )

        if show_depth_panel and active_lns:
            _draw_depth_panel(ax, active_lns, ln_to_user, gk_x0, theme)

        # ── ball ──────────────────────────────────────────────────────────────
        bp = _ball_xy(row_def)
        if bp:
            ax.scatter(
                [bp[0]], [bp[1]],
                c=_BALL_COLOR, s=130, zorder=7,
                edgecolors=_BALL_EDGE, linewidths=1.5,
            )

        # ── pass reveal phase ─────────────────────────────────────────────────
        if pass_labels is not None and reveal >= 0:
            import matplotlib.patches as mpatches

            hv_list = pass_labels.get("hull_vertices") or []
            if reveal >= int(fps * 1):
                for hv in hv_list:
                    if hv and len(hv) >= 3:
                        ax.add_patch(mpatches.Polygon(
                            hv, closed=True, fill=False, hatch="////",
                            edgecolor="#FFB300", lw=1.0, zorder=2,
                        ))

            if reveal >= int(fps * 2) and start_ball and end_ball:
                ax.add_patch(mpatches.FancyArrowPatch(
                    posA=start_ball, posB=end_ball,
                    arrowstyle="->,head_width=4,head_length=3",
                    color=theme["text_color"], lw=2.5, zorder=8,
                ))

            if reveal >= int(fps * 3):
                ax.text(
                    -48.0, 22.0, _build_info_text(pass_labels),
                    color=theme["text_color"], fontsize=9, va="top",
                    bbox={
                        "facecolor": theme["bar_bg"],
                        "alpha": 0.85,
                        "boxstyle": "round",
                    },
                    zorder=10,
                )

        # ── title ─────────────────────────────────────────────────────────────
        ax.set_title(
            f"Unit Lines  |  {_timestamp_str(row_def, period)}",
            color=theme["text_color"], fontsize=10, pad=8,
        )
        fig.patch.set_facecolor(theme["bg_color"])

    return FuncAnimation(
        fig, _update,
        frames=n_frames + n_reveal,
        interval=1000.0 / fps,
        blit=False,
    )


# ── private draw helpers ───────────────────────────────────────────────────────


def _draw_player(
    ax: Axes,
    xf: float,
    yf: float,
    color: str,
    is_gk: bool,
    label: str,
    alpha: float,
    edge_color: str,
    num_color: str,
) -> None:
    marker = "s" if is_gk else "o"
    size   = 200 if is_gk else 160
    edgew  = 1.5 if is_gk else 0.8
    ax.scatter(
        [xf], [yf],
        c=color, s=size, marker=marker,
        zorder=4 + int(is_gk),
        alpha=alpha,
        edgecolors=edge_color,
        linewidths=edgew,
    )
    ax.text(
        xf, yf, label,
        fontsize=5.5, ha="center", va="center",
        color=num_color, fontweight="bold",
        zorder=6 + int(is_gk),
        alpha=alpha,
    )


def _draw_team(
    ax: Axes,
    row: pd.Series,
    team: str,
    alpha: float,
    theme: dict[str, str],
    gk_id: str | None = None,
) -> None:
    color = _TEAM_COLORS.get(team, "#aaaaaa")
    if gk_id is None:
        gk_id = _find_gk_id(row, team)
    n = 1
    while True:
        xc, yc = f"{team}_{n}_x", f"{team}_{n}_y"
        if xc not in row.index:
            break
        try:
            xf, yf = float(row[xc]), float(row[yc])
            if not (math.isnan(xf) or math.isnan(yf)):
                _draw_player(
                    ax, xf, yf, color,
                    is_gk=(str(n) == gk_id),
                    label=str(n),
                    alpha=alpha,
                    edge_color=theme["edge_color"],
                    num_color=theme["num_color"],
                )
        except (TypeError, ValueError):
            pass
        n += 1


def _draw_depth_panel(
    ax: Axes,
    active_lns: dict[int, list[tuple[float, float]]],
    ln_to_user: dict[int, int],
    gk_x: float,
    theme: dict[str, str],
) -> None:
    """Draw the line depth panel in the bottom-left corner (axes-fraction coords).

    Each detected line gets a bar proportional to its depth from the GK,
    labelled L1–L4 with the matching line color.
    """
    import matplotlib.patches as mpatches

    n_lines  = len(active_lns)
    tc       = theme["text_color"]
    bg       = theme["bar_bg"]
    is_light = bg.startswith("#e")
    track_bg = "#c8c8c8" if is_light else "#444466"

    panel_h = _DP_BAR_H + (n_lines - 1) * _DP_STRIDE + 0.02
    ax.add_patch(mpatches.FancyBboxPatch(
        (_DP_X0 - 0.035, _DP_Y0 - 0.015),
        _DP_W_MAX + 0.065, panel_h,
        boxstyle="round,pad=0.005",
        transform=ax.transAxes,
        fc=bg, ec="none", alpha=0.82, zorder=20,
    ))

    sorted_lns = sorted(
        active_lns, key=lambda ln: abs(_repr_x(active_lns[ln], gk_x))
    )
    for i, ln in enumerate(sorted_lns):
        user_ln  = ln_to_user.get(ln, ln)
        color    = _DEFENSIVE_LINE_COLORS.get(user_ln, "white")
        rx       = _repr_x(active_lns[ln], gk_x)
        depth    = max(0.0, abs(rx) - abs(gk_x))
        bar_frac = min(depth / _DP_MAX_M, 1.0)
        bar_w    = bar_frac * _DP_W_MAX
        yc       = _DP_Y0 + i * _DP_STRIDE

        # Track background
        ax.add_patch(mpatches.Rectangle(
            (_DP_X0, yc - _DP_BAR_H / 2), _DP_W_MAX, _DP_BAR_H,
            transform=ax.transAxes,
            fc=track_bg, ec="none", alpha=0.4, zorder=21,
        ))
        # Filled bar
        if bar_w > 0:
            ax.add_patch(mpatches.Rectangle(
                (_DP_X0, yc - _DP_BAR_H / 2), bar_w, _DP_BAR_H,
                transform=ax.transAxes,
                fc=color, ec="none", alpha=0.88, zorder=22,
            ))
        # Line label
        ax.text(
            _DP_X0 - 0.012, yc, f"L{user_ln}",
            transform=ax.transAxes, ha="right", va="center",
            fontsize=7, color=color, fontweight="bold", zorder=23,
        )
        # Depth value
        ax.text(
            _DP_X0 + _DP_W_MAX + 0.010, yc, f"{depth:.1f} m",
            transform=ax.transAxes, ha="left", va="center",
            fontsize=6.5, color=tc, fontweight="bold", zorder=23,
        )


# ── private data helpers ───────────────────────────────────────────────────────


def _find_gk_id(row: pd.Series, team: str) -> str | None:
    """Fallback GK detection used when _line columns are absent.

    Infers defending direction from the team's median x (negative → defending
    left, positive → defending right), then picks the signed extreme player.
    This avoids misidentifying an outfielder who pushed into the opponent's half.
    """
    x_vals: dict[str, float] = {}
    n = 1
    while True:
        xc = f"{team}_{n}_x"
        if xc not in row.index:
            break
        try:
            xf = float(row[xc])
            if not math.isnan(xf):
                x_vals[str(n)] = xf
        except (TypeError, ValueError):
            pass
        n += 1
    if not x_vals:
        return None
    xs = list(x_vals.values())
    median_x = sorted(xs)[len(xs) // 2]
    if median_x < 0:
        return min(x_vals, key=lambda k: x_vals[k])
    return max(x_vals, key=lambda k: x_vals[k])


def _find_gk_id_from_lines(row: pd.Series, team: str) -> str | None:
    """GK detection from analyze_lines output: player where _line == 0.
    Falls back to position-based _find_gk_id if no _line columns found."""
    n = 1
    found_any_line_col = False
    while True:
        xc = f"{team}_{n}_x"
        if xc not in row.index:
            break
        lc = f"{team}_{n}_line"
        if lc in row.index:
            found_any_line_col = True
            try:
                if int(float(row[lc])) == 0:
                    return str(n)
            except (TypeError, ValueError):
                pass
        n += 1
    # No _line column found — fall back to position heuristic
    if not found_any_line_col:
        return _find_gk_id(row, team)
    return None


def _gk_x(row: pd.Series, team: str) -> float:
    gk_id = _find_gk_id_from_lines(row, team)
    if gk_id is None:
        return 0.0
    with contextlib.suppress(KeyError, TypeError, ValueError):
        return float(row[f"{team}_{gk_id}_x"])
    return 0.0


def _repr_x(players: list[tuple[float, float]], gk_x: float) -> float:
    sign = math.copysign(1.0, gk_x) if gk_x != 0.0 else 1.0
    return max((p[0] for p in players), key=lambda x: x * sign)


def _players_by_line(
    row: pd.Series, team: str
) -> dict[int, list[tuple[float, float]]]:
    result: dict[int, list[tuple[float, float]]] = {}
    n = 1
    while True:
        xc = f"{team}_{n}_x"
        if xc not in row.index:
            break
        yc = f"{team}_{n}_y"
        lc = f"{team}_{n}_line"
        try:
            xf, yf = float(row[xc]), float(row[yc])
            ln = float(row[lc]) if lc in row.index else float("nan")
            if not (math.isnan(xf) or math.isnan(yf) or math.isnan(ln)):
                result.setdefault(int(ln), []).append((xf, yf))
        except (TypeError, ValueError):
            pass
        n += 1
    return result


def _player_xy_with_ids(
    row: pd.Series, team: str
) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    n = 1
    while True:
        xc, yc = f"{team}_{n}_x", f"{team}_{n}_y"
        if xc not in row.index:
            break
        try:
            xf, yf = float(row[xc]), float(row[yc])
            if not (math.isnan(xf) or math.isnan(yf)):
                out.append((str(n), xf, yf))
        except (TypeError, ValueError):
            pass
        n += 1
    return out


def _ball_xy(row: pd.Series) -> tuple[float, float] | None:
    try:
        bx, by = float(row["ball_x"]), float(row["ball_y"])
        return None if (math.isnan(bx) or math.isnan(by)) else (bx, by)
    except (KeyError, TypeError, ValueError):
        return None


def _closest_player(
    row: pd.Series, team: str, ref_x: float, ref_y: float
) -> tuple[float, float] | None:
    best: tuple[float, float] | None = None
    best_d = float("inf")
    n = 1
    while True:
        xc, yc = f"{team}_{n}_x", f"{team}_{n}_y"
        if xc not in row.index:
            break
        try:
            xf, yf = float(row[xc]), float(row[yc])
            if not (math.isnan(xf) or math.isnan(yf)):
                d = (xf - ref_x) ** 2 + (yf - ref_y) ** 2
                if d < best_d:
                    best_d, best = d, (xf, yf)
        except (TypeError, ValueError):
            pass
        n += 1
    return best


def _timestamp_str(row: pd.Series, period: object) -> str:
    try:
        secs = int(float(row.get("time_s", row.get("Time [s]", 0))))
        m, s = divmod(secs, 60)
        half = "H1" if int(str(period)) == 1 else "H2"
        return f"{half} {m:02d}:{s:02d}"
    except (KeyError, TypeError, ValueError):
        return ""


def _build_info_text(pass_labels: dict[str, Any]) -> str:
    is_lb     = pass_labels.get("is_line_breaking")
    lbc       = pass_labels.get("lines_broken_count", 0)
    direction = pass_labels.get("direction_per_line", [])
    location  = pass_labels.get("location_after_break")
    lines = [
        f"Line-breaking: {'Yes' if is_lb else 'No'}",
        f"Lines broken: {lbc}",
    ]
    if direction and isinstance(direction, list):
        lines.append(f"Direction: {', '.join(str(d) for d in direction)}")
    if location:
        lines.append(f"Location: {location}")
    return "\n".join(lines)
