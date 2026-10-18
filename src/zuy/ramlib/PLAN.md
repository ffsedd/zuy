# Raman library search: rewrite plan (ramanlib2)

Goal: given a measured Raman spectrum, return a ranked, explainable list of library matches (single compounds and mixtures) that holds up against real-world variation in noise, baseline, x calibration, resolution and spectral range.

## 0. Decision and status

- The old `ramanlib` (ramfind.py, make_library.py) has no users, so nothing needs to stay compatible. It is **kept untouched as the baseline**, and a new package **`ramanlib2`** is written beside it. Both are run through the same benchmark and compared side by side. Rename the package freely.
- The new library data is extracted from the old pickle for now. It can be rebuilt from the source `.jdx` files later (Phase 6) if the benchmark shows that preprocessing control matters.

| Phase | What | Status |
|-------|------|--------|
| 1 | Library format, pickle converter, converted library, tests | **done** (section 3) |
| 2 | Benchmark + adapter for the old code + plain-Pearson baseline | next |
| 3 | Preprocessing module | |
| 4 | Scoring + vectorised search + grouping | |
| 5 | Mixtures | |
| 6 | CLI, plotting, optional rebuild from source `.jdx` | |
| 7 | Clean-up, docs | |

## 1. What the old code does and where it is weak

`LibrarySearch.find()` preprocesses the query (baseline, `smooth(21)`, `resample(1)`, rescale), intersects it by index with every library spectrum, and ranks by Pearson r (called `r2`). `search_recursive()` takes the top hit, sets the x-regions where its smoothed (71) version exceeds 0.2 to NaN, and searches again, up to 6 components.

### Bugs
| # | Where | Problem | Effect |
|---|-------|---------|--------|
| B1 | `search_recursive` + `find` | The masked query has NaN, `intersected_spectra` does not drop NaN, `np.corrcoef` returns NaN for every entry. | The 2nd to 6th component search returns arbitrary entries. |
| B2 | `search_recursive(..., results=[])` | Mutable default argument. | Results leak between calls (`multisearch`); works only by accident. |
| B3 | `find` | `if a:` / `if b:` treat 0 as no limit. | A ROI starting at 0 is ignored. |
| B4 | `find` | Relies on `process()` mutating the input in place; ROI is cut from `spectrum`, not `self.spectrum`. | Fragile; changes the caller's object. |
| B5 | `make_library.process` | After an exception `s` may be unbound or half processed and is still stored. | Broken spectra can enter the library. |
| B6 | `r2` | It is Pearson r, not r². | Misleading name; anti-correlation ranks as bad. |
| B7 | `smooth` | `s.smooth(21)` is used as in place, `fit = fit.smooth(71)` as returning a copy. | One of the two is wrong. Check in the `Spectrum` class. |

### Algorithmic weaknesses
| # | Problem | Why it hurts |
|---|---------|--------------|
| A1 | Pearson over the whole overlap. | Dominated by broad features and leftover baseline, not by the narrow bands that identify a compound. |
| A2 | No x-shift tolerance. | A 2 to 5 cm-1 calibration error is normal. |
| A3 | Coverage is not penalised; minimum overlap is an absolute `< 100` points. | A library entry covering 100 to 800 cm-1 can win on a partial match. Strong peaks outside the compared range are ignored. |
| A4 | Smoothing mismatch (library `smooth=71` in `make_library.main`, query `smooth(21)`). | A 71 cm-1 window at a 1 cm-1 step erases most Raman structure; the two sides have different resolution. |
| A5 | No noise model, no cosmic-ray removal. | One spike moves r. |
| A6 | Mixtures: masking instead of subtracting a fitted component. | Overlapping peaks of the second component are deleted along with the first. |
| A7 | One scalar score. | No confidence, no explanation of which peaks matched. |
| A8 | Duplicates are not grouped. | Top 10 can be ten entries of one pigment. |

### Engineering issues
- One class does I/O, preprocessing, scoring, plotting and printing; hard imports of `IPython.display` and `matplotlib`; `print()` in `process`.
- A 47 MB pickle is reloaded per `ramfind()` call, and per library in `multisearch`.
- A Python loop over all entries with `np.intersect1d` + `isin` per entry.
- The pickle stores `qq.spectrum.spectrum.BaseSpectrum` objects, so it breaks on a package rename (the code imports `zumpy.qq...`, the repo is now `zuy`), and pickle is unsafe for untrusted files.
- Personal paths hard-coded in `main()` defaults.

## 2. Facts about the library (measured on `RamanLibIrugSop.pkl`)

- **1,322 spectra**: 1,003 IRUG, 319 SOPRANO (KIK-IRPA, all 785 nm, `baseline corr.: Y` by the provider).
- All on an integer 1 cm-1 grid, y in [0, 1], no NaN, no gaps.
- **Coverage varies a lot**: median about 100 to 2000 cm-1; 546 entries end below 1800, 1007 end below 3000, 575 start above 100. Original SOPRANO files cover 100 to 1800, 2000 or 3200.
- **Median peak FWHM about 13 cm-1.** The smoothing used for this pickle is not recorded; it may be lighter or heavier than intended.
- **Laser** (parsed from headers): 785 nm 1,045 entries (including all SOPRANO), 532 nm 112, 633 nm 66 (+13 at 632), 514 nm 51, 1064 nm 13, unknown 20.
- **Groups**: 488 groups when grouped by pigment code (e.g. `PR105`) or, if the entry has none, by the normalised name (97 entries). **250 groups occur in both IRUG and SOPRANO**, which makes a realistic cross-source benchmark possible (section 6). The largest groups have 13 to 19 entries (`PY34`, `PR106`, `PG39`, `PB29`, ...).
- Group heuristics are not perfect, e.g. `PB15` (13) and `PB151` (13) look like the same family written two ways. Fix such cases by hand in the `group` column of the csv.
- The two sample queries differ strongly: 2269 is 532 nm, 0.42 cm-1 step, 60 to 3200 cm-1; 2271 is 488 nm, 3.6 cm-1 step, 79 to 7225 cm-1, y offset 50k to 250k (fluorescence).

## 3. Library format (implemented)

```
data/RamanLibIrugSop.npz        axis (5000,) int32, Y (1322, 5000) float32 (NaN outside own coverage), ids, config JSON   (9.9 MB)
data/RamanLibIrugSop.meta.csv   one row per entry, key column `id`, hand-editable
```

- `id` is the old pickle key (e.g. `IRUG_RMP00001 Red Lead, Pigment Red 105 (PR105), CI77578`), so old and new results can be mapped one to one.
- Meta columns: `source, entry_id, name, pigment_code, group, ci, cas, color, laser_nm, title, x_min, x_max, n_points, orig_x_min, orig_x_max, orig_n_points, orig_step, origin_url, file_stem`.
- `ramanlib2/library.py`: `Library(axis, Y, meta, config)` with `.save`, `.load`, `.spectrum(id)`, `.crop(lo, hi)`, `.coverage`. On load, the csv is aligned to the matrix by `id`; missing ids raise.
- `ramanlib2/convert_pkl.py`: converts the old pickle without needing `zumpy`/`zuy` (stub classes, allowlisted unpickler; run it only on your own pickles).
- `tests/test_library.py`: round trip, csv alignment, error cases, invariants of the real library (6 tests, all passing).

```
python -m ramanlib2.convert_pkl RamanLibIrugSop.pkl data/RamanLibIrugSop
lib = Library.load("data/RamanLibIrugSop.npz")
```

Caveat: the stored y values are what the old pipeline produced (baseline corrected, smoothed with an unrecorded window, rescaled). SOPRANO spectra were baseline corrected by the provider and then again by the old pipeline.

## 4. Algorithm design

### 4.1 Preprocessing (`preprocess.py`, pure functions, one `PreprocessConfig`)
Order matters; all widths are in cm-1, not points.
1. **Despike on the native grid, before any resampling.** A one-point spike on a coarse grid (3.6 cm-1) would otherwise become a band after interpolation. Hampel / median-difference filter, threshold in MAD units.
2. **Resample** onto the library axis by linear interpolation; no extrapolation (NaN outside the query range).
3. **Baseline**: arPLS or SNIP (width about 100 cm-1), robust to bands, with parameters set in cm-1. Test first on the 488 nm sample (huge offset, fluorescence).
4. **Light smoothing** only: Savitzky-Golay, window about 7 to 11 cm-1, order 2 or 3. No `smooth=71`.
5. **Noise** `sigma = 1.4826 * MAD(second difference) / sqrt(6)`, used for peak thresholds.
6. **Normalise** by a robust maximum (99.5th percentile) over the compared region; clip negatives.
7. Default ROI 100 to 3000 cm-1 (the old CLI default), configurable.

The config is stored in the library (`config`) and checked at search time; a library built with a different config than the query is rejected.

### 4.2 Scores (`scoring.py`)
Let `q` be the processed query on the library axis and `Y_i` entry i, `W_i` the mask where both have data inside the ROI.

1. **Broadened correlation `r_b`**: Gaussian broadening (sigma about 4 cm-1) of query and library, then masked Pearson over `W_i`. For all entries at once this is a few matrix products (masked sums `n, Σq, Σy, Σqy, Σq², Σy²` as `W @ q`, `(W*Y) @ q`, ...), milliseconds for 1,322 entries. Broadening is the shift tolerance; a lag search of +/-3 cm-1 is a refinement for the top candidates only.
2. **Peak match F-score `F`**:
   - peaks: `scipy.signal.find_peaks` with prominence >= max(3 sigma, 0.02), position refined parabolically; library peaks are computed once and stored with the library;
   - one-to-one assignment within a tolerance (start at 6 cm-1), greedy by intensity;
   - recall = matched library peak weight / library peak weight inside the query range; precision = matched query weight / query peak weight inside the library range; weights `sqrt(height)` so that band ratios (which differ between lasers and instruments) matter less than band positions;
   - `F = 2PR/(P+R)`.
3. **Coverage `cov`**: library peak weight inside the query range / library peak weight inside the ROI. Prevents partial-range matches from winning.
4. **Combined score** `S = max(r_b, 0)^a * F^b * cov^c`, start with a = b = c = 1; tune on the benchmark, not on the two sample spectra. Rank fusion (reciprocal rank) is the fallback if the product is unstable.

### 4.3 Search (`search.py`)
- Stage 1: `r_b` for all entries (vectorised). Take the union of the top 100 by `r_b` and the top 100 by peak recall.
- Stage 2: full scoring of the candidates (lag refinement, `F`, `cov`).
- Group results by `group` (best entry per group, number of supporting entries); report `margin` = score of group 1 minus group 2.
- Optional filters: laser, source.

```python
@dataclass
class SearchResult:
    table: pd.DataFrame        # group, id, name, score, r_b, f1, cov, n_matched, laser_nm, n_entries
    query: np.ndarray          # processed query on the library axis
    matches: dict[str, list]   # id -> matched peak pairs (query pos, library pos, weights)
    margin: float
```

### 4.4 Mixtures
Replaces the NaN-mask recursion (B1, A6).
1. Candidates: the best entry of each of the top 30 groups by `S`.
2. Greedy forward selection with non-negative least squares (`scipy.optimize.nnls`) on the broadened spectra over the ROI; after each addition refit all selected components and compute the residual.
3. Stop when the gain in explained variance is below 0.02, when no residual peak exceeds 3 sigma, or at 5 components.
4. Report coefficients (relative contributions), the residual and the explained fraction. Restricting to one entry per group avoids collinearity between near-duplicate entries.

### 4.5 Later, only if the benchmark says so
Library augmentation (shifts, broadening); a small learned re-ranker over (`r_b`, `F`, `cov`, margin); a flag for fluorescence-dominated queries.

## 5. Package layout

```
ramanlib2/
  library.py       done   Library container, npz + csv
  convert_pkl.py   done   old pickle -> Library
  preprocess.py           PreprocessConfig + pure functions
  peaks.py                peak detection (query and library)
  scoring.py              broadened correlation, peak match, coverage, combined
  search.py               LibrarySearch.search() -> SearchResult; mixtures
  plotting.py             all matplotlib code
  cli.py                  argparse, tables, logging
  build_library.py        source .jdx -> Library (Phase 6)
benchmarks/
  run.py                  runs any Searcher over the benchmark sets
  adapters.py             OldSearcher (wraps ramanlib), NewSearcher
tests/
data/
```

Rules: numpy arrays at every boundary (`Spectrum` is used only to read `.jdx` into `(x, y)`); no in-place mutation; `logging`, never `print`, in library code; no IPython or matplotlib import in the core.

## 6. Benchmark (Phase 2, build this before any new algorithm)

Common interface, so old and new are run identically:

```python
class Searcher(Protocol):
    name: str
    def search(self, x, y, roi=(100, 3000)) -> list[tuple[str, float]]:  # (id, score), best first
```

- `OldSearcher` writes the `(x, y)` to a temporary `.jdx` (same layout as the sample files) and calls the old `LibrarySearch.find`, because the old code only takes a `Spectrum` made from a path. Library ids are identical to the old stems.
- A plain Pearson-on-library baseline is implemented in the new package too, so the effect of each added piece (preprocessing, broadening, peaks, coverage) can be measured separately.
- Results are scored at the **group** level (the 10 chrome yellows count as one answer).

Test sets:
1. **Cross-source (realistic, no synthetic data):** the 250 groups present in both sources. Query = each SOPRANO entry with the library restricted to IRUG, and the reverse. Different lab, different instrument, different preparation: this is the honest test and answers the objection that leave-one-out on one family is too easy.
2. **Perturbed leave-one-out:** query = a library entry (itself excluded, and excluded together with all same-group entries of the same source) with synthetic perturbations applied one at a time and in combination: additive noise at several SNR levels; x shift +/-1 to 6 cm-1 and a slight stretch; fluorescence baseline (polynomial + exponential); cosmic-ray spikes; cropped range (e.g. 200 to 1200 only); resampling to a 3.6 cm-1 step; band intensity ratio distortion.
3. **Mixtures:** two and three component mixtures with known weights (including overlapping bands).
4. **Real samples:** the two `.jdx` files (2269, 2271); add the known answers when available, plus any further measurements of your own with known identity.

Metrics: top-1, top-5, MRR at group level; mixture component recall and weight error; runtime per query; for the cross-source set also a breakdown by laser. Results are written to `benchmarks/results/<name>.md` so that changes can be compared over time.

## 7. Phases and acceptance criteria

**Phase 1: library (done).** Converter, container, 6 tests, library files.

**Phase 2: benchmark and baseline.**
- `benchmarks/run.py`, `OldSearcher`, plain-Pearson `NewSearcher`, the four test sets.
- Accept: baseline table for old code and plain Pearson committed. Expect the old code to do reasonably on clean single-compound queries and to fail on mixtures (B1).

**Phase 3: preprocessing.**
- `preprocess.py`, config stored in the library, unit tests: baseline removal on synthetic fluorescence + peaks; spike removal before resampling; same result for a 0.42 and a 3.6 cm-1 sampled version of the same synthetic spectrum within tolerance.
- Accept: plain Pearson with the new preprocessing is at least as good as before on the benchmark; the 488 nm sample gives a sensible baseline.

**Phase 4: scoring and search.**
- Broadened correlation, peak matching, coverage, combined score, grouping, margin.
- Accept: each added component improves (or does not hurt) the benchmark; the combined search beats the old code and plain Pearson on the cross-source set at top-1 and top-5. If it does not, stop and find out why before going on.

**Phase 5: mixtures.**
- NNLS with greedy selection and residual stopping.
- Accept: on synthetic two-component mixtures, both components appear in the result clearly more often than with the old recursion; weights are in the right order.

**Phase 6: CLI, plotting, optional rebuild.**
- `cli.py` with the old flags (`-p`, `-v`, `-a`, `-b`), library path relative to the package, table and JSON output; plotting moved out of the core.
- `build_library.py` from the source `.jdx` files with the new preprocessing; rebuild and rerun the benchmark. Keep the rebuilt library only if it is better than the converted one (the pickle's smoothing and the double baseline on SOPRANO are the suspects).

**Phase 7: clean-up.** README with the config parameters and how to rebuild; decide when to retire the old package.

## 8. Risks and open questions

- Which smoothing was used for `RamanLibIrugSop.pkl` (the comment in `ramfind.py` mentions `RAMAN_IRUG_SOPR_sm71.pkl`; the median peak width suggests something lighter)? Rebuilding from `.jdx` answers it.
- Does `Spectrum.smooth()` return a copy or modify in place (B7)? Needed for the `OldSearcher` and for anything that wraps old code.
- SOPRANO files are baseline corrected twice in the old library. Check whether skipping the second correction changes the benchmark.
- Group heuristics (`PB15` vs `PB151`, natural pigments without a code) need a manual pass over the csv; edits are picked up on load.
- Laser values `632` and `633` are the same He-Ne line; normalise if laser filtering is used.
- The score weights (a, b, c), tolerance (6 cm-1) and broadening (4 cm-1) are starting values and must be tuned on the benchmark, on sets 1 and 2, and verified on sets 3 and 4 without retuning.
- Leave-one-out inside one lab's measurements flatters any method; judge by the cross-source set first.

## 9. Related file

`untitled.py` is a one-off scraper that extracts JCAMP-DX download URLs from SOPRANO pages. It is not part of the search code; if kept, move it to `tools/` beside `build_library.py` and replace the hard-coded HTML by a fetched page.
