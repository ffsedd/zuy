# Spectrum processing extraction

Preserve numerical results, public imports, Jdx mutation and return conventions,
metadata, processing defaults, and automatic tidy behavior. Retain the JCAMP
`dataprocessing` header field. Keep data-model unification as a separate follow-up.
Preserve unrelated user changes. Repository-wide quality cleanup was approved
by the user before the extraction; minimal lint/type fixes in the Raman converter
are included in that cleanup, without changing its conversion behavior.

## Preparation — Done

Cleanup commit: `716984a`. Checks: Ruff format and lint, mypy, deptry,
pytest (31 passed), final Ruff checks; all passed.

- Approved baseline snapshot: commit `add6b3b`, containing only the existing
  `src/zuy/spectrum/jdx.py` and `tests/test_jdx.py`.
- Approved dependency addition: deptry installed with `uv add --dev deptry`;
  pyproject.toml and uv.lock changes remain uncommitted. The pre-existing
  sortsamples entry must not be included in the dependency commit.
- Sample is now at `src/zuy/data/RMP00084.jdx` (moved since the initial check).
- Baseline quality commands executed from the repository root, in order:
  - `uv run ruff format .`: passed, 9 files reformatted.
  - `uv run ruff check --fix --unsafe-fixes .`: passed, 3 fixes.
  - Inspected changes and restored all Python files from a byte-for-byte snapshot
    to preserve unrelated/pre-existing work before subsequent checks.
  - `uv run mypy .`: failed, 44 existing errors in 7 files.
  - `uv run deptry .`: failed, 59 existing dependency issues.
  - `uv run pytest`: passed, 24 tests.
  - `uv run ruff format --check .`: failed, 9 existing unformatted files.
  - `uv run ruff check . --no-fix`: failed, 3 existing unused imports. The flag
    prevents the project's `fix = true` setting from changing protected files.
- Commands used `UV_CACHE_DIR=/tmp/zuy-uv-cache` because the default cache is
  outside the writable sandbox. Full logs: `/tmp/zuy-refactor-checks/`.
- User approved fixing repository-wide checks first. Fix dependency declarations,
  install missing type stubs, use bundled Matplotlib types, fix incorrect source
  annotations and lint/format issues. Preserve behavior and all pre-existing work.
  Run all required checks and make a separate cleanup commit before Step 1.
- User authorized replacing obsolete zumpy with internal zuy readers, referencing
  `/mnt/wd/ZUMI-XXL/zzz_zumpy/ir/read/scn.py`. Implement and test the Buck SCN
  format internally; use the existing internal SPA parser. No zumpy dependency.

## Step 1 — Characterize numerical behavior

Add regression coverage for baseline correction, calibration, region replacement,
descending axes, nonfinite inputs, mutation, and the RMP00084 sample. Capture
expected results before extracting algorithms. Run required checks and commit.

Validation: all required quality commands passed; mypy checked 42 files, deptry
reported no issues, pytest passed 42 tests. Final Ruff checks passed. One existing
SciPy sparse efficiency warning remains; numerical outputs are unchanged.

## Step 2 — Extract numerical processing

Move reusable array calculations into processing.py. Keep separate algorithm
names for moving average, Savitzky–Golay, legacy ASLS, pybaselines ASLS, and arPLS.
Keep Jdx methods and existing utility imports as compatibility wrappers/re-exports.
Functions must not mutate their inputs. Preserve results and validate with Step 1
regressions plus direct function checks. Run required checks and commit.

## Step 3 — Separate plotting and processing workflows

Extract baseline diagnostic plotting and make processing order explicit in reusable
workflow functions, preserving legacy tidy/subtract/auto_tidy behavior. Test plot
content and workflow equivalence. Run required checks and commit.

## Follow-up outside this extraction

Unify Jdx and Spectrum only after separately agreeing axis ordering, metadata,
validation, and mutation conventions. Do not change scientific algorithms or their
defaults as part of extraction.

### Cleanup validation

All required checks passed: Ruff formatting, Ruff unsafe autofixes (reviewed),
mypy (41 files), deptry (no issues), pytest (31 passed), final Ruff format check,
and final Ruff lint check. Legacy external examples were read without modifying
sources: SCN arrays match all 4,250 original points; SPA loads 1,764 finite points.
RMP00084 arrays were captured before extraction; input SHA256 is
`0439c6e9f2ad5e4e89f815d9eb924fc142a4704433fcda060c605a33794469c8`.
Untracked Raman and filename scripts were formatted/linted in the working tree;
their pre-existing contents and the sortsamples entry remain outside our commits.
