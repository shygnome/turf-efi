# Data Contract — turf.efi modules

Authoritative column-level spec for inputs and outputs of the three stable
`turf.efi` modules. All coordinate values use the **SkillCorner centered
system** unless otherwise noted.

**Coordinate system:**
- `x`: −52.5 (own goal line) → +52.5 (opponent goal line), metres
- `y`: −34 (right touchline) → +34 (left touchline), metres
- Attacking direction is **not encoded** in the coordinate values; callers
  must track which team is attacking which direction.

---

## turf.efi.line

### Input — tracking DataFrame

One row per tracking frame. Both teams combined in a single wide-format
DataFrame (produced by merging home and away per-team CSVs on `Period` +
`Time [s]`).

| Column | Dtype | Required | Description |
|--------|-------|----------|-------------|
| `Period` | int64 | yes | Match half: 1 or 2 |
| `Time [s]` | float64 | yes | Seconds elapsed within the period |
| `ball_x` | float64 | yes | Ball x-position; NaN if ball out of play |
| `ball_y` | float64 | yes | Ball y-position |
| `{Team}_{n}_x` | float64 | yes | Player *n* x-position for team *Team*; NaN if off-pitch |
| `{Team}_{n}_y` | float64 | yes | Player *n* y-position for team *Team* |

`{Team}` is any string (e.g. `"Home"`, `"Away"`, club abbreviation).  
`{n}` is a 1-based integer jersey/squad identifier.  
Player count per team is flexible; the module discovers columns by pattern.

### Output — tracking DataFrame (augmented)

Same DataFrame as input, plus one column per player of the **defending team**:

| Column | Dtype | Description |
|--------|-------|-------------|
| `{DefendingTeam}_{n}_line` | float64 | 0 = GK; 1 = most advanced line; 2–4 = deeper lines; NaN = not detected |

---

## turf.efi.label

### Input — lines DataFrame + team string

Accepts the output of `turf.efi.line` directly (see above). No filesystem
handoff required.

```python
label_event(lines_df: pd.DataFrame, defending_team: str) -> dict[str, Any]
```

`lines_df` must contain the `{DefendingTeam}_{n}_line` columns produced by
`analyze_lines` / `fix_player_assignments`, plus `ball_x`, `ball_y`.

### Output — event dict

| Key | Type | Description |
|-----|------|-------------|
| `is_line_breaking` | bool | True if the ball crossed at least one defensive line |
| `lines_broken_count` | int | Number of distinct lines the ball trajectory crossed |
| `direction_per_line` | list[str] | Per broken line: `"forward"` (toward opp. goal) or `"backward"` |

---

## turf.efi.risk

### Input — event-level DataFrame

One row per line-breaking pass event. Derived features; callers are
responsible for computing them from raw event + annotation data.

| Column | Dtype | Required | Description |
|--------|-------|----------|-------------|
| `zone_thirds` | object (str) | yes | Pitch third of pass start: `"t_def"` / `"t_mid"` / `"t_att"` |
| `score_diff` | int64 | yes | Score differential at time of pass (positive = leading) |
| `actual_xt_gain` | float64 | yes | Normalised forward progression ∈ [0, 1]: Δx.clip(0) / 52.5 |
| `pressure` | bool | no | Whether the passer was under pressure (adds bin dimension) |

`game_state` is derived internally from `score_diff`:

| score_diff | game_state |
|-----------|------------|
| ≤ −2 | `"losing_2+"` |
| −1 | `"losing_1"` |
| 0 | `"level"` |
| +1 | `"winning_1"` |
| ≥ +2 | `"winning_2+"` |

### Output of `build_expected_xt`

```python
dict[tuple[str, str], float]
# key: (zone_thirds, game_state)
# value: mean actual_xt_gain across all passes in that bin
```

### Output of `add_residuals`

Same DataFrame as input, plus:

| Column | Dtype | Description |
|--------|-------|-------------|
| `xt_expected` | float64 | Lookup value for this row's bin; NaN if bin unseen in training data |
| `xt_residual` | float64 | `actual_xt_gain − xt_expected`; NaN if `xt_expected` is NaN |

---

## Coordinate system note for turf-io

Providers other than SkillCorner use different systems and must be remapped
before passing data to any `turf.efi` function:

| Provider | x range | y range | Unit |
|----------|---------|---------|------|
| SkillCorner | −52.5 → +52.5 | −34 → +34 | metres (centered) |
| StatsBomb | 0 → 120 | 0 → 80 | yards (corner origin) |
| Tracab / STATS | provider-specific | provider-specific | cm or mm |

Remapping to the SkillCorner convention is `turf-io`'s responsibility.
