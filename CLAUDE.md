# turf-efi — Claude Code Guide

Open research library that proposes and validates algorithms approximating
concepts from FIFA's Enhanced Football Intelligence (EFI) system.

**What EFI is:** FIFA's proprietary, closed-source analytics framework that
defines metrics such as unit lines, line-breaking passes, and space control.
FIFA does not publish the algorithms.

**What this repo is:** A research-grade open implementation that independently
proposes methods to produce EFI-style metrics from raw tracking data. The unit
line detection algorithm is the author's proposed approach, grounded in Ward
agglomerative clustering adapted from published academic work, designed to
approximate how FIFA's EFI likely defines defensive unit lines. Results may
differ from FIFA's internal system — that divergence is itself a research
finding.

Part of the **turf** ecosystem. Installable as `pip install turf-efi`;
importable as `turf.efi.*`.

---

## Package layout

```
src/
  turf/               ← no __init__.py  (implicit namespace — see below)
    efi/
      __init__.py     ← public API re-exports
      line.py         ← unit line detection from tracking frames
      label.py        ← line-break labeling for pass/dribble events
      risk.py         ← xT-based risk model for line-breaking passes
      viz.py          ← pitch visualization (developing; requires viz extras)
tests/
  fixtures/           ← synthetic tracking frames only (no real data)
  test_line.py
  test_label.py
  test_risk.py
experiments/
  README.md           ← index of experiments and key findings
  {topic}/
    notebook.ipynb    ← committed; all results inside are gitignored
    results/          ← gitignored
data/                 ← gitignored (real tracking data — never commit)
```

### Namespace package

`src/turf/` has **no `__init__.py`**. This makes `turf` an implicit namespace
so any `turf-*` package can contribute sub-modules to the same `turf.*` import
tree without coordination.

```python
# works after only pip install turf-efi — no turf install needed
from turf.efi.line import detect_lines_frame, analyze_lines
from turf.efi.label import label_event, detect_line_break
from turf.efi.risk import build_expected_xt
```

Never add `src/turf/__init__.py`. If the main `turf` package is also installed,
both coexist cleanly in the namespace.

---

## Running commands

Always use `uv run` — do not assume the venv is activated.

```bash
uv run pytest
uv run pytest -x              # stop on first failure
uv run pytest -k "name"       # filter by name
uv run ruff check .
uv run ruff check . --fix
uv run ruff format .
uv run mypy src
```

---

## Stability tiers

Modules are classified into two tiers. The tier controls how strictly
development discipline is applied.

### Stable modules

Full TDD required. All functions type-annotated. `mypy` must pass clean.
No `# type: ignore` without an explanation comment.

| Module              | Description                              |
|---------------------|------------------------------------------|
| `turf.efi.line`     | Unit line detection from tracking frames |
| `turf.efi.label`    | Line-break event labeling                |
| `turf.efi.risk`     | xT-based risk model                      |

**If a module is in this table, TDD is non-negotiable.**

### Developing modules

New modules under active research. TDD is encouraged but not enforced.
`mypy` errors may be temporarily silenced with `# type: ignore[<code>]`.
When a developing module stabilises, add it to the stable table above
and retroactively close any missing test coverage before merging.

To add a new developing module, create it under `turf/efi/` and add a
row to the table below (no approval process, just track it):

| Module              | Status      | Notes                                |
|---------------------|-------------|--------------------------------------|
| `turf.efi.viz`      | developing  | mplsoccer pitch plots; needs viz extras (`uv sync --group viz`) |

---

## Development workflow

### Stable modules — red/green TDD

1. Write a failing test first.
2. Write minimum code to make it pass.
3. Refactor with tests green.
4. Never write implementation code without a driving test.

### Developing modules — research mode

Write the algorithm first to explore the idea. Tests follow after the
behaviour is roughly understood. Keep the test coverage honest: if
a function is exported, it should have at least a smoke test.

### Paper-first discipline

Before implementing any algorithm, locate the source paper and write
a `definition/` document that captures:
- The exact formula or procedure as stated in the paper
- Any parameters and their stated values
- Edge cases the paper explicitly addresses

Only then write code. Reference implementations found online may deviate
from the paper — do not use them as ground truth.

---

## Experiments

Experiments live in `experiments/{topic}/`. Each experiment is a committed
Jupyter notebook that documents the question, method, and conclusion.

```
experiments/
  README.md                    ← one-line entry per experiment, date + outcome
  line_detection_ablation/
    notebook.ipynb             ← committed
    results/                   ← gitignored (CSV outputs, PNGs)
  footvision_line_comparison/
    notebook.ipynb
    results/
```

### Conventions

- Freeze the random seed and environment in the notebook header cell.
- Keep notebooks self-contained: they should run top-to-bottom in a clean kernel.
- Output cells are committed (so findings are readable on GitHub without re-running).
  Exception: cells with large base64 images — clear those before committing.
- Results directories are gitignored. If a result needs to be shared, export it
  to a `reports/` folder and commit it there explicitly.
- No production code in experiments. If a new algorithm is worth keeping,
  implement it in `turf/efi/` as a developing module — do not import from `experiments/`.

### Ablation studies

For systematic comparisons (parameter sensitivity, algorithm variants),
create one notebook per study. Log the result table directly in the notebook
and add a summary row to `experiments/README.md`.

### Ground truth policy

Tests and experiments may use:
- Synthetic fixtures (`tests/fixtures/` or inline in notebooks) — safe to commit
- Public open datasets — load via URL / download script, do not commit the data;
  document the exact download command in the notebook
- Real / proprietary data — **never committed**; loaded from `data/` (gitignored);
  experiments that depend on real data note this in their first cell

---

## Tests

### Fixtures

`tests/fixtures/` contains small synthetic tracking DataFrames constructed
by hand. No real tracking data, no reduced real data. If a test requires
properties that are hard to synthesise, use `pytest.mark.skipif` to skip
it when the real `data/` path is absent.

### Marks

```python
@pytest.mark.slow       # runs > 1 s — skipped in fast CI pass
@pytest.mark.real_data  # requires data/ to be populated
```

Run only fast tests: `uv run pytest -m "not slow and not real_data"`

---

## Code style

Linting and formatting via `ruff`. Type checking via `mypy`. Both must pass
before committing stable module changes.

```bash
uv run ruff check .
uv run ruff check . --fix
uv run ruff format .
uv run mypy src
```

---

## Commit and branch conventions

Same as the main `turf` repository.

**Commit format (Conventional Commits, title only — no body):**

```
feat(line): add force_n_lines parameter to analyze_lines
fix(label): handle empty frame in detect_line_break
test(line): add red/green for single-player edge case
experiment: add footvision line comparison notebook
chore: update numpy dependency to >=2.0
```

Use `experiment:` type for experiment notebooks and `definition:` for
`definition/` documents.

**Branch naming:**

```
feature/no-ref/add-force-n-lines
bugfix/no-ref/fix-empty-frame-label
experiment/no-ref/footvision-line-comparison
docs/no-ref/efi-line-definition
```

---

## Future / TODO

- [ ] Evaluate whether `turf.efi` warrants a separate `_experimental` submodule
      once the user base grows beyond the author. For now, the stability table
      above is sufficient.
- [ ] Publish to PyPI once the three stable modules have adequate test coverage
      and the API has been validated against at least one real dataset.
- [ ] Consider a `turf` meta-package that lists `turf-efi` (and future `turf-*`)
      as dependencies, so `pip install turf` pulls in the full ecosystem.
