"""Pitch visualization helpers for turf.efi outputs.

Developing module — see CLAUDE.md stability tiers.

Requires the viz dependency group:
    uv sync --group viz

Data conventions
----------------
Tracking frames are wide-format DataFrames with columns:
    Period, Time [s], {Team}_{n}_x, {Team}_{n}_y, ball_x, ball_y

After analyze_lines, each player also has a {Team}_{n}_line column:
    0 = goalkeeper, 1–4 = outfield defensive lines (front to back)

Coordinates use the SkillCorner/centered system:
    x: −52.5 (own goal end) → +52.5 (opponent goal end)
    y: −34 (left touchline) → +34 (right touchline)
"""

from __future__ import annotations

import contextlib
import math
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.animation import FuncAnimation
    from matplotlib.axes import Axes
    from mplsoccer import Pitch as PitchType  # type: ignore[import-untyped]

# ── style constants ────────────────────────────────────────────────────────────

_PITCH_TYPE = "skillcorner"
_PITCH_LENGTH = 105
_PITCH_WIDTH = 68
_HALF_WIDTH = _PITCH_WIDTH / 2  # 34.0
_YLIM_BOTTOM = -42.0
_LABEL_Y = -38.0

_PITCH_COLOR = "#1a7a1a"
_LINE_COLOR = "white"
_BG_COLOR = "#0d0d0d"

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
_GK_EDGE_COLOR = "lime"
_BALL_COLOR = "#FFA500"
_TRAIL_RGB = (1.0, 0.647, 0.0)
_TRAIL_ALPHA_MIN = 0.08
_TRAIL_ALPHA_MAX = 0.65
_MAX_LINES = 4

# ── public pitch factory ───────────────────────────────────────────────────────


def make_pitch(half: bool = False) -> PitchType:
    """Return a configured mplsoccer Pitch (dark grass style)."""
    from mplsoccer import Pitch

    return Pitch(
        pitch_type=_PITCH_TYPE,
        pitch_length=_PITCH_LENGTH,
        pitch_width=_PITCH_WIDTH,
        pitch_color=_PITCH_COLOR,
        line_color=_LINE_COLOR,
        half=half,
    )


# ── static frame plots ─────────────────────────────────────────────────────────


def plot_frame(
    ax: Axes,
    frame: pd.Series,
    home_team: str = "Home",
    away_team: str = "Away",
) -> None:
    """Plot players and ball for a single wide-format tracking row.

    home_team players are drawn solid; away_team at reduced alpha.
    Goalkeepers are rendered as diamond markers.
    """
    for team, alpha in [(home_team, 1.0), (away_team, 0.55)]:
        color = _TEAM_COLORS.get(team, "#aaaaaa")
        gk_id = _find_gk_id(frame, team)
        out_xs, out_ys, gk_xs, gk_ys = [], [], [], []
        n = 1
        while True:
            xc, yc = f"{team}_{n}_x", f"{team}_{n}_y"
            if xc not in frame.index:
                break
            try:
                xf, yf = float(frame[xc]), float(frame[yc])
                if not (math.isnan(xf) or math.isnan(yf)):
                    if str(n) == gk_id:
                        gk_xs.append(xf)
                        gk_ys.append(yf)
                    else:
                        out_xs.append(xf)
                        out_ys.append(yf)
            except (TypeError, ValueError):
                pass
            n += 1
        if out_xs:
            ax.scatter(
                out_xs, out_ys,
                c=color, s=140, zorder=4, alpha=alpha,
                edgecolors="white", linewidths=0.8,
            )
        if gk_xs:
            ax.scatter(
                gk_xs, gk_ys,
                c=color, s=170, marker="D", zorder=5, alpha=alpha,
                edgecolors=_GK_EDGE_COLOR, linewidths=1.5,
            )

    # Ball
    try:
        bx, by = float(frame["ball_x"]), float(frame["ball_y"])
        if not (math.isnan(bx) or math.isnan(by)):
            ax.scatter(
                [bx], [by],
                c=_BALL_COLOR, s=100, zorder=6,
                edgecolors="black", linewidths=1.0,
            )
    except (KeyError, TypeError, ValueError):
        pass


def plot_lines_frame(
    ax: Axes,
    frame: pd.Series,
    defending_team: str,
    attacking_team: str,
) -> None:
    """Plot a tracking frame with defensive unit line overlays.

    frame must have {defending_team}_{n}_line columns (from analyze_lines output).
    Each non-GK line is drawn as a zig-zag connecting players and a dashed
    representative-x vertical line, labelled L1…L4.
    """
    plot_frame(ax, frame, home_team=defending_team, away_team=attacking_team)

    gk_x = _gk_x(frame, defending_team)
    lines_players: dict[int, list[tuple[float, float]]] = {}
    n = 1
    while True:
        xc = f"{defending_team}_{n}_x"
        if xc not in frame.index:
            break
        yc = f"{defending_team}_{n}_y"
        line_col = f"{defending_team}_{n}_line"
        try:
            xf = float(frame[xc])
            yf = float(frame[yc])
            ln_raw = (
                float(frame[line_col]) if line_col in frame.index else float("nan")
            )
            if not (math.isnan(xf) or math.isnan(yf) or math.isnan(ln_raw)):
                lines_players.setdefault(int(ln_raw), []).append((xf, yf))
        except (TypeError, ValueError):
            pass
        n += 1

    # Sort lines by distance from GK (ascending abs x) → L1 = most advanced
    active = {
        ln: players
        for ln, players in lines_players.items()
        if ln != 0 and players
    }
    sorted_lns = sorted(
        active, key=lambda ln: abs(_repr_x(active[ln], gk_x))
    )
    for user_idx, ln in enumerate(sorted_lns, start=1):
        players = active[ln]
        color = _DEFENSIVE_LINE_COLORS.get(user_idx, "white")
        sorted_p = sorted(players, key=lambda p: p[1])
        if len(sorted_p) >= 2:
            ax.plot(
                [p[0] for p in sorted_p], [p[1] for p in sorted_p],
                color=color, lw=2.0, alpha=0.85, zorder=5,
            )
        rx = _repr_x(players, gk_x)
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
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    from turf.efi.line import smooth_line_assignments

    if smooth_lines:
        defending_frames = smooth_line_assignments(defending_frames, defending_team)

    first_def_row = defending_frames.iloc[0]
    gk_id = _find_gk_id(first_def_row, defending_team)
    def_gk_x: float = 0.0
    if gk_id is not None:
        gk_xcol = f"{defending_team}_{gk_id}_x"
        with contextlib.suppress(KeyError, TypeError, ValueError):
            def_gk_x = float(first_def_row[gk_xcol])

    pitch = make_pitch()
    fig, ax = pitch.draw()
    ax.set_ylim(_YLIM_BOTTOM, _HALF_WIDTH)
    fig.patch.set_facecolor(_BG_COLOR)

    n_frames = min(len(defending_frames), len(attacking_frames))

    # Stable L-label mapping: L1 = most advanced (lowest abs repr_x)
    last_lp = _players_by_line(defending_frames.iloc[-1], defending_team)
    last_repr_xs: dict[int, float] = {
        ln: _repr_x(last_lp[ln], def_gk_x)
        for ln in range(1, _MAX_LINES + 1)
        if last_lp.get(ln)
    }
    sorted_by_advance = sorted(last_repr_xs, key=lambda ln: abs(last_repr_xs[ln]))
    ln_to_user_stable: dict[int, int] = {
        ln: i + 1 for i, ln in enumerate(sorted_by_advance)
    }

    atk_gk_id = _find_gk_id(attacking_frames.iloc[0], attacking_team)

    ball_trail: list[tuple[float, float] | None] = [
        _ball_xy(defending_frames.iloc[i]) for i in range(n_frames)
    ]
    start_ball = ball_trail[0] if ball_trail else None
    end_ball = ball_trail[-1] if ball_trail else None

    # ── artists ───────────────────────────────────────────────────────────────
    atk_color = _TEAM_COLORS.get(attacking_team, "gray")
    def_color = _TEAM_COLORS.get(defending_team, "gray")

    atk_out_scat = ax.scatter([], [], c=atk_color, s=120, alpha=0.5, zorder=3)
    atk_gk_scat = ax.scatter(
        [], [], c=atk_color, marker="D", s=160,
        edgecolors=_GK_EDGE_COLOR, linewidths=2, alpha=0.5, zorder=5,
    )
    def_out_scat = ax.scatter([], [], c=def_color, s=140, zorder=4)
    def_gk_scat = ax.scatter(
        [], [], c=def_color, marker="D", s=160,
        edgecolors=_GK_EDGE_COLOR, linewidths=2, zorder=5,
    )
    ball_scat = ax.scatter(
        [], [], c=_BALL_COLOR, edgecolors="black", s=80, zorder=6,
    )
    trail_scat = ax.scatter([], [], s=18, edgecolors="none", zorder=5)

    zigzag_lines: dict[int, Any] = {}
    repr_lines: dict[int, Any] = {}
    line_labels: dict[int, Text] = {}
    for ln in range(1, _MAX_LINES + 1):
        (zz,) = ax.plot(
            [], [], color=_DEFENSIVE_LINE_COLORS[ln], lw=1.5, alpha=0.7, zorder=3
        )
        zigzag_lines[ln] = zz
        (rl,) = ax.plot(
            [], [], color=_DEFENSIVE_LINE_COLORS[ln],
            linestyle="--", lw=1.5, alpha=0.5, zorder=2,
        )
        repr_lines[ln] = rl
        line_labels[ln] = ax.text(
            0.0, _LABEL_Y, "",
            color=_DEFENSIVE_LINE_COLORS[ln], fontsize=8, rotation=45,
            ha="center", va="top", visible=False,
        )

    # legend
    legend_handles = [
        Line2D(
            [], [], color=def_color, marker="o", ms=7, lw=0,
            label=f"{defending_team} (def.)",
        ),
        Line2D(
            [], [], color=def_color, marker="D", ms=7, lw=0,
            markeredgecolor=_GK_EDGE_COLOR, markeredgewidth=1.5,
            label=f"{defending_team} GK",
        ),
        Line2D(
            [], [], color=atk_color, marker="o", ms=7, lw=0, alpha=0.5,
            label=f"{attacking_team} (atk.)",
        ),
        Line2D(
            [], [], color=atk_color, marker="D", ms=7, lw=0,
            markeredgecolor=_GK_EDGE_COLOR, markeredgewidth=1.5, alpha=0.5,
            label=f"{attacking_team} GK",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=7, framealpha=0.7)
    title = ax.set_title("")
    period = metadata.get("period", 1)

    # pass_labels reveal phase
    n_reveal = int(freeze_duration * fps) if pass_labels is not None else 0
    hull_patches: list[Any] = []
    pass_arrow = None
    passer_ring = None
    receiver_ring = None
    info_text = None

    if pass_labels is not None:
        import matplotlib.patches as mpatches

        hv_list = pass_labels.get("hull_vertices")
        if hv_list and isinstance(hv_list, list):
            for hv in hv_list:
                if hv and len(hv) >= 3:
                    patch = mpatches.Polygon(
                        hv, closed=True, fill=False, hatch="////",
                        edgecolor="#FFB300", lw=1.0, visible=False, zorder=2,
                    )
                    ax.add_patch(patch)
                    hull_patches.append(patch)

        if start_ball is not None and end_ball is not None:
            pass_arrow = mpatches.FancyArrowPatch(
                posA=start_ball, posB=end_ball,
                arrowstyle="->,head_width=4,head_length=3",
                color="white", lw=2.5, visible=False, zorder=8,
            )
            ax.add_patch(pass_arrow)

        passer_ring = ax.scatter(
            [], [], s=350, facecolors="none", edgecolors="yellow", lw=2.5, zorder=9,
        )
        receiver_ring = ax.scatter(
            [], [], s=350, facecolors="none", edgecolors="lime", lw=2.5, zorder=9,
        )
        info_text = ax.text(
            -48.0, 22.0, "", color="white", fontsize=9, va="top",
            bbox={"facecolor": "#111111", "alpha": 0.8, "boxstyle": "round"},
            visible=False, zorder=10,
        )

    # ── static start-ball marker ──────────────────────────────────────────────
    if start_ball is not None:
        sbx, sby = start_ball
        ax.scatter(
            [sbx], [sby], color="yellow", edgecolors="darkorange",
            marker="*", s=140, lw=1.5, zorder=7,
        )

    def _update(frame_i: int) -> tuple[object, ...]:
        eff = min(frame_i, n_frames - 1)
        reveal = frame_i - n_frames

        row_def = defending_frames.iloc[eff]
        row_atk = attacking_frames.iloc[eff]

        # attacking team
        atk_all = _player_xy_with_ids(row_atk, attacking_team)
        atk_out = [(x, y) for pid, x, y in atk_all if pid != atk_gk_id]
        atk_gk = [(x, y) for pid, x, y in atk_all if pid == atk_gk_id]
        atk_out_scat.set_offsets(atk_out if atk_out else np.empty((0, 2)))
        atk_gk_scat.set_offsets(atk_gk if atk_gk else np.empty((0, 2)))

        # ball
        bp = _ball_xy(row_def)
        ball_scat.set_offsets([[bp[0], bp[1]]] if bp else np.empty((0, 2)))

        # ball trail
        valid_trail = [p for p in ball_trail[: eff + 1] if p is not None]
        if valid_trail:
            n_t = len(valid_trail)
            alphas = (
                np.array([_TRAIL_ALPHA_MAX])
                if n_t == 1
                else np.linspace(_TRAIL_ALPHA_MIN, _TRAIL_ALPHA_MAX, n_t)
            )
            rgba = np.empty((n_t, 4))
            rgba[:, 0], rgba[:, 1], rgba[:, 2] = _TRAIL_RGB
            rgba[:, 3] = alphas
            trail_scat.set_offsets(valid_trail)
            trail_scat.set_facecolor(rgba)
        else:
            trail_scat.set_offsets(np.empty((0, 2)))

        # defending team
        line_players = _players_by_line(row_def, defending_team)
        gk_pl = line_players.get(0, [])
        def_gk_scat.set_offsets(gk_pl if gk_pl else np.empty((0, 2)))
        all_out = [
            p for ln in range(1, _MAX_LINES + 1) for p in line_players.get(ln, [])
        ]
        def_out_scat.set_offsets(all_out if all_out else np.empty((0, 2)))

        repr_xs_active: dict[int, float] = {
            ln: _repr_x(line_players[ln], def_gk_x)
            for ln in range(1, _MAX_LINES + 1)
            if line_players.get(ln)
        }

        for ln in range(1, _MAX_LINES + 1):
            players = line_players.get(ln, [])
            if players:
                user_ln = ln_to_user_stable.get(ln, ln)
                color = _DEFENSIVE_LINE_COLORS[user_ln]
                sorted_p = sorted(players, key=lambda p: p[1])
                if len(sorted_p) >= 2:
                    zigzag_lines[ln].set_data(
                        [p[0] for p in sorted_p], [p[1] for p in sorted_p]
                    )
                else:
                    zigzag_lines[ln].set_data([], [])
                zigzag_lines[ln].set_color(color)
                rx = repr_xs_active[ln]
                repr_lines[ln].set_data([rx, rx], [-_HALF_WIDTH, _HALF_WIDTH])
                repr_lines[ln].set_color(color)
                line_labels[ln].set_position((rx, _LABEL_Y))
                line_labels[ln].set_text(f"L{user_ln}")
                line_labels[ln].set_color(color)
                line_labels[ln].set_visible(pass_labels is None or reveal < 0)
            else:
                zigzag_lines[ln].set_data([], [])
                repr_lines[ln].set_data([], [])
                line_labels[ln].set_visible(False)

        ts = _timestamp_str(row_def, period)
        title.set_text(f"Unit Lines | {ts}")

        reveal_artists: tuple[object, ...] = ()
        if pass_labels is not None:
            p1 = int(fps * 1)
            p2 = int(fps * 2)
            p3 = int(fps * 3)
            for hp in hull_patches:
                hp.set_visible(reveal >= p1)
            if pass_arrow is not None:
                pass_arrow.set_visible(reveal >= p2)
            if passer_ring is not None and receiver_ring is not None:
                if reveal >= p2:
                    if start_ball is not None:
                        pp = _closest_player(
                            attacking_frames.iloc[0], attacking_team, *start_ball
                        )
                        passer_ring.set_offsets([pp] if pp else np.empty((0, 2)))
                    if end_ball is not None:
                        rp = _closest_player(
                            attacking_frames.iloc[-1], attacking_team, *end_ball
                        )
                        receiver_ring.set_offsets([rp] if rp else np.empty((0, 2)))
                else:
                    passer_ring.set_offsets(np.empty((0, 2)))
                    receiver_ring.set_offsets(np.empty((0, 2)))
            if info_text is not None:
                info_text.set_visible(reveal >= p3)
                if reveal >= p3 and not info_text.get_text():
                    info_text.set_text(_build_info_text(pass_labels))

            ra: list[object] = [*hull_patches]
            if pass_arrow is not None:
                ra.append(pass_arrow)
            if passer_ring is not None:
                ra.append(passer_ring)
            if receiver_ring is not None:
                ra.append(receiver_ring)
            if info_text is not None:
                ra.append(info_text)
            reveal_artists = tuple(ra)

        return (
            atk_out_scat, atk_gk_scat, trail_scat, ball_scat,
            def_out_scat, def_gk_scat,
            *zigzag_lines.values(), *repr_lines.values(),
            *line_labels.values(),
            title,
            *reveal_artists,
        )


    return FuncAnimation(
        fig, _update,  # type: ignore[arg-type]
        frames=n_frames + n_reveal,
        interval=1000.0 / fps,
        blit=True,
    )


# ── private helpers ────────────────────────────────────────────────────────────


def _find_gk_id(row: pd.Series, team: str) -> str | None:
    best_id: str | None = None
    best_abs = -1.0
    n = 1
    while True:
        xc = f"{team}_{n}_x"
        if xc not in row.index:
            break
        try:
            xf = float(row[xc])
            a = abs(xf)
            if not math.isnan(a) and a > best_abs:
                best_abs = a
                best_id = str(n)
        except (TypeError, ValueError):
            pass
        n += 1
    return best_id


def _gk_x(row: pd.Series, team: str) -> float:
    gk_id = _find_gk_id(row, team)
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
        secs = int(float(row["Time [s]"]))
        m, s = divmod(secs, 60)
        half = "H1" if int(str(period)) == 1 else "H2"
        return f"{half} {m:02d}:{s:02d}"
    except (KeyError, TypeError, ValueError):
        return ""


def _build_info_text(pass_labels: dict[str, Any]) -> str:
    is_lb = pass_labels.get("is_line_breaking")
    lbc = pass_labels.get("lines_broken_count", 0)
    direction = pass_labels.get("direction_per_line", [])
    location = pass_labels.get("location_after_break")
    lines = [
        f"Line-breaking: {'Yes' if is_lb else 'No'}",
        f"Lines broken: {lbc}",
    ]
    if direction and isinstance(direction, list):
        lines.append(f"Direction: {', '.join(str(d) for d in direction)}")
    if location:
        lines.append(f"Location: {location}")
    return "\n".join(lines)
