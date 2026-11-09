"""Read and write PFF FC raw data (JSONL tracking + JSON events + JSON metadata).

Public entry points:
    load_match(data_dir, match_id)              -> MatchData   (from raw)
    save_match(match, out_dir, match_id)        -> None        (to parquet)
    read_match(out_dir, match_id)               -> MatchData   (from parquet)

Column convention (tracking DataFrame):
    period, time_s, ball_x, ball_y,
    home_<jersey>_x, home_<jersey>_y,
    away_<jersey>_x, away_<jersey>_y

Team labels throughout: 'home' and 'away' (lowercase).
"""
from __future__ import annotations

import bz2
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class AttackingDirections:
    """Attacking direction (+1 or -1) per team per period.

    +1  → team attacks toward positive x in that period
    -1  → team attacks toward negative x in that period

    Source: metadata homeTeamStartLeft (ground truth from PFF).
    Teams swap sides in period 2.
    """

    home_p1: int
    away_p1: int
    home_p2: int
    away_p2: int

    def for_team(self, team: str, period: int) -> int:
        """Return attacking direction for team ('home'|'away') and period (1|2).

        Raises ValueError for periods other than 1 or 2 (extra time not yet
        supported — homeTeamStartLeftExtraTime is in metadata but ignored here).
        """
        if period not in (1, 2):
            raise ValueError(f"Period {period} not supported (only 1 or 2)")
        if team == "home":
            return self.home_p1 if period == 1 else self.home_p2
        if team == "away":
            return self.away_p1 if period == 1 else self.away_p2
        raise ValueError(f"Unknown team '{team}' — expected 'home' or 'away'")


@dataclass
class MatchData:
    """Preprocessed data for one match.

    tracking   : wide DataFrame — period, time_s, ball_x, ball_y,
                 home_<jersey>_x/y, away_<jersey>_x/y per player
    events     : tidy DataFrame — one row per possession event
    directions : attacking direction per team per period
    meta       : raw metadata dict (fps, team names, period boundaries, …)
    """

    tracking: pd.DataFrame
    events: pd.DataFrame
    directions: AttackingDirections
    meta: dict[str, Any]


# ── internal helpers ──────────────────────────────────────────────────────────


def _parse_meta(meta_path: Path) -> dict[str, Any]:
    with open(meta_path) as f:
        raw: Any = json.load(f)
    result: dict[str, Any] = raw[0] if isinstance(raw, list) else raw
    return result


def _attacking_directions(meta: dict[str, Any]) -> AttackingDirections:
    """Derive attacking directions from homeTeamStartLeft.

    homeTeamStartLeft=True  → home attacks right (+1) in period 1.
    Teams swap in period 2.
    """
    home_starts_left: bool = bool(meta["homeTeamStartLeft"])
    home_p1 = 1 if home_starts_left else -1
    return AttackingDirections(
        home_p1=home_p1,
        away_p1=-home_p1,
        home_p2=-home_p1,
        away_p2=home_p1,
    )


def _parse_tracking(tracking_path: Path) -> pd.DataFrame:
    """Parse JSONL.bz2 tracking file → wide DataFrame.

    Output columns (all lowercase, no spaces):
        period, time_s, ball_x, ball_y,
        home_<jersey>_x, home_<jersey>_y,
        away_<jersey>_x, away_<jersey>_y
    """
    rows: list[dict[str, Any]] = []
    with bz2.open(tracking_path, "rt") as f:
        for line in f:
            frame: dict[str, Any] = json.loads(line)
            row: dict[str, Any] = {
                "period": int(frame["period"]),
                "time_s": float(frame["periodElapsedTime"]),
            }

            # ballsSmoothed is a single dict; balls is a list — handle both
            ball_raw = frame.get("ballsSmoothed") or (
                frame["balls"][0] if frame.get("balls") else None
            )
            has_pos = (
                bool(ball_raw)
                and isinstance(ball_raw, dict)
                and ball_raw.get("x") is not None
            )
            if has_pos:
                assert isinstance(ball_raw, dict)
                row["ball_x"] = float(ball_raw["x"])
                row["ball_y"] = float(ball_raw["y"])
            else:
                row["ball_x"] = float("nan")
                row["ball_y"] = float("nan")

            home = frame.get("homePlayersSmoothed") or frame.get("homePlayers") or []
            for p in home:
                n = str(p["jerseyNum"])
                row[f"home_{n}_x"] = float(p["x"])
                row[f"home_{n}_y"] = float(p["y"])

            away = frame.get("awayPlayersSmoothed") or frame.get("awayPlayers") or []
            for p in away:
                n = str(p["jerseyNum"])
                row[f"away_{n}_x"] = float(p["x"])
                row[f"away_{n}_y"] = float(p["y"])

            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values(["period", "time_s"]).reset_index(drop=True)
    return df


def _parse_events(events_path: Path) -> pd.DataFrame:
    """Parse event JSON → tidy possession-event DataFrame.

    One row per event. Key columns:
        game_event_id, possession_event_id,
        period, start_time, end_time,
        team_id, team_name, is_home,
        player_id, player_name,
        event_type,       ← gameEventType (e.g. OTB, FIRSTKICKOFF)
        possession_type,  ← possessionEventType (PA=pass, CR=carry, …)
        lines_broken,     ← linesBrokenType (None / 'NONE' / 'MD' / 'ATT')
        ball_x, ball_y,   ← ball position at event start
        passer_id, passer_name, receiver_id, receiver_name,
        pass_outcome,     ← passOutcomeType (success / incomplete / …)
        subtype,          ← gameEvents.subType
    """
    with open(events_path) as f:
        raw: list[dict[str, Any]] = json.load(f)

    rows: list[dict[str, Any]] = []
    for ev in raw:
        ge: dict[str, Any] = ev.get("gameEvents") or {}
        pe: dict[str, Any] = ev.get("possessionEvents") or {}
        balls: list[dict[str, Any]] = ev.get("ball") or []
        bx = float(balls[0]["x"]) if balls else float("nan")
        by = float(balls[0]["y"]) if balls else float("nan")

        rows.append(
            {
                "game_event_id": ev.get("gameEventId"),
                "possession_event_id": ev.get("possessionEventId"),
                "period": ge.get("period"),
                "start_time": ev.get("startTime"),
                "end_time": ev.get("endTime"),
                "team_id": ge.get("teamId"),
                "team_name": ge.get("teamName"),
                "is_home": ge.get("homeTeam"),
                "player_id": ge.get("playerId"),
                "player_name": ge.get("playerName"),
                "event_type": ge.get("gameEventType"),
                "possession_type": pe.get("possessionEventType"),
                "lines_broken": pe.get("linesBrokenType"),
                "ball_x": bx,
                "ball_y": by,
                "passer_id": pe.get("passerPlayerId"),
                "passer_name": pe.get("passerPlayerName"),
                "receiver_id": pe.get("receiverPlayerId"),
                "receiver_name": pe.get("receiverPlayerName"),
                "pass_outcome": pe.get("passOutcomeType"),
                "subtype": ge.get("subType"),
            }
        )

    return pd.DataFrame(rows)


# ── public API ────────────────────────────────────────────────────────────────


def load_match(data_dir: str | Path, match_id: int | str) -> MatchData:
    """Load one PFF FC match from raw data into a MatchData container.

    data_dir should be the root of the raw PFF dataset, e.g.:
        data/raw/FIFA_WC_2022/

    Reads:
        {data_dir}/Tracking Data/{match_id}.jsonl.bz2
        {data_dir}/Event Data/{match_id}.json
        {data_dir}/Metadata/{match_id}.json
    """
    root = Path(data_dir)
    mid = str(match_id)

    meta = _parse_meta(root / "Metadata" / f"{mid}.json")
    directions = _attacking_directions(meta)
    tracking = _parse_tracking(root / "Tracking Data" / f"{mid}.jsonl.bz2")
    events = _parse_events(root / "Event Data" / f"{mid}.json")

    return MatchData(
        tracking=tracking,
        events=events,
        directions=directions,
        meta=meta,
    )


def save_match(match: MatchData, out_dir: str | Path, match_id: int | str) -> Path:
    """Write a MatchData to parquet + JSON under {out_dir}/{match_id}/.

    Creates the directory if needed. Returns the match directory path.

    Requires pyarrow: uv sync --group io

    Writes:
        {out_dir}/{match_id}/tracking.parquet
        {out_dir}/{match_id}/events.parquet
        {out_dir}/{match_id}/directions.json
        {out_dir}/{match_id}/meta.json
    """
    match_dir = Path(out_dir) / str(match_id)
    match_dir.mkdir(parents=True, exist_ok=True)

    match.tracking.to_parquet(match_dir / "tracking.parquet", index=False)
    match.events.to_parquet(match_dir / "events.parquet", index=False)

    with open(match_dir / "directions.json", "w") as f:
        json.dump(asdict(match.directions), f)

    with open(match_dir / "meta.json", "w") as f:
        json.dump(match.meta, f)

    return match_dir


def read_match(out_dir: str | Path, match_id: int | str) -> MatchData:
    """Load a MatchData previously saved with save_match.

    Requires pyarrow: uv sync --group io
    """
    match_dir = Path(out_dir) / str(match_id)

    tracking = pd.read_parquet(match_dir / "tracking.parquet")
    events = pd.read_parquet(match_dir / "events.parquet")

    with open(match_dir / "directions.json") as f:
        dirs: dict[str, int] = json.load(f)
    directions = AttackingDirections(**dirs)

    with open(match_dir / "meta.json") as f:
        meta: dict[str, Any] = json.load(f)

    return MatchData(tracking=tracking, events=events, directions=directions, meta=meta)
