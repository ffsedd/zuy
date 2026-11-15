# Arrow labels for imannotator

Status: Implemented. User approved proceeding with the documented pre-existing mypy/deptry failures.

## Goal

Add numbered and free-text callouts to `semangui`, persist them in `annotations.yaml`, and render them with `semandraw`. Preserve existing circles, crosses, rectangles, file paths, autosave, and undo/redo behavior.

## Pre-implementation baseline (historical)

- `models.py`: frozen `Shape` dataclass; three shape types, image-space coordinates, optional size and second endpoint. Serialization omits `None`; loading ignores unknown fields but does not validate shape-specific requirements.
- `storage.py`: version-1 YAML, image-relative keys, atomic writes, and timestamps preserved for unchanged shape lists. Storage already serializes through `Shape`.
- `gui.py`: Tkinter canvas, fit-to-window coordinate conversion, click placement for circles/crosses, drag placement for rectangles, list snapshots for undo/redo, autosave on navigation/close. Right-click deletes the nearest shape center without a distance limit. Keyboard shortcuts currently ignore only Spinbox input.
- `draw.py`: independent Pillow drawing functions and a dispatch table. GUI and export already differ in stroke styling; new callouts should use one rendering implementation.
- No annotator-specific tests currently exist. Project development dependencies include all five required quality tools.

## Proposed behavior

1. Add **Numbered label (B)** and **Text label (T)** tools. Keep existing shortcuts, including N for Next.
2. Press on the image feature to establish the arrow tip; drag to the label location and release. Show a live preview. The arrow points from the label toward the feature.
3. Numbered labels display a positive integer inside a high-contrast badge. Default to one greater than the highest existing numbered label on the current image, or 1. A Next number field permits choosing a different unused positive integer. Reject duplicates within an image. Advance only after successful placement; derive the default again after undo, redo, deletion, or image loading. Never renumber existing labels automatically.
4. Text labels use a toolbar entry containing single-line text. Require non-whitespace content, retain Unicode, and reuse the entered text for repeated placements. Disable application shortcuts while an editable input has focus.
5. A separate Font size control uses original-image pixels; do not reuse the existing shape radius control. Scale text, badge padding, line width, and arrowhead with the image. Use a readable white background, dark text, and a contrasting arrow with a light halo.
6. Escape cancels a pending drag. Tool switches and image navigation also clear pending placement. A short drag is ignored; use Euclidean screen distance so horizontal and vertical arrows are valid. Reject starts outside the displayed image and constrain label placement to the image, keeping its box visible where possible.
7. Each callout is one undoable shape. Existing Delete-last and Clear apply to it. For right-click deletion, measure a callout against its label box and arrow segment, rather than treating the arrow tip as its center; preserve existing shape behavior.
8. Version one supports placement and deletion/recreation. Moving endpoints, editing existing labels, multiline layout, standalone arrows, automatic label placement, and global numbering are outside this proposal.

## Data and rendering design

Extend `ShapeType` with `callout` and append optional fields to the frozen dataclass, preserving existing positional constructors:

- `x`, `y`: feature location / arrow tip.
- `x2`, `y2`: label-box center. Preserve direction; never normalize these like rectangle corners.
- `text: str | None`: free-text content.
- `number: int | None`: numeric badge content.
- `font_size: float | None`: persisted image-space font size.

A callout requires both endpoints, a positive font size, and exactly one of non-empty text or a positive number. Keep callout validation focused; do not change legacy shape semantics or silently drop malformed callouts. Report the image and invalid field. Unknown future shape types retain the current unsupported-type handling.

Use YAML version 2 when callouts are present. Continue reading version 1 unchanged, preserve an existing version-2 document when saving, and reject unsupported document versions clearly. Version 2 signals the new capability; older applications cannot be expected to render it. Add a short example fixture with legacy shapes and both callout variants.

Add `callouts.py` for pure layout, validation helpers, and Pillow RGBA rendering. Compute the text/badge bounds, arrow attachment at the box boundary, and triangle arrowhead in image coordinates; draw arrow first and label last. Handle coincident/overlapping endpoints without division by zero. Keep font resolution and metrics common to layout and rendering, with a deterministic available font and explicit fallback. Test representative Unicode; document that glyph coverage depends on the selected font. Do not add a dependency or bundle a font without approval.

Use the Pillow callout renderer in `draw.py` and a scaled RGBA callout overlay in the Tkinter canvas. Cache the committed overlay until shapes change, retain `PhotoImage` references, and draw the pending callout separately. Resize the overlay with the same viewport scale as the image. This shares text metrics and arrow geometry while leaving existing shape renderers intact.

## Ordered implementation steps

Each step begins Pending. On approval, implement one step at a time, run its checks, and create one focused local commit before starting the next. Record its actual hash and results below afterward; do not amend a commit to insert its own hash. Do not push.

### 1. Model and YAML compatibility — Done

Implemented model fields/validation, version handling, image-specific errors, duplicate-number rejection, and mixed legacy/callout fixtures and tests. No commit created because required repository checks fail in unchanged code.

Checks performed:
- `uv run ruff format .`: passed (two new/changed files formatted; no unrelated tracked changes).
- `uv run ruff check --fix --unsafe-fixes .`: passed; no unsafe fixes introduced.
- `uv run mypy .`: failed with 7 errors in 6 unchanged files: stale `zuy.spectrum.models.spectrum` imports in Raman scripts, missing `zumpy`, `ramanlib2.library`, and `library`, a duplicate `Library` definition, and an incompatible return type in `ramlib/library.py`.
- `uv run deptry .`: failed with 6 existing dependency issues: unused `jinja2`, missing `zumpy`, transitive `IPython`, missing `library`, development `pytest` imported under src, and missing `ramanlib2`.
- `uv run pytest`: blocked during collection by missing `ramanlib2` in `src/zuy/ramlib/test_library.py` (75 tests collected, one collection error).
- `uv run pytest tests/test_imannotator_storage.py`: passed, 18 tests.
- `uv run ruff format --check .`: passed, 57 files.
- `uv run ruff check .`: passed.

User approved continuing with these documented pre-existing failures. Step 1 is already committed as `4b48642` (present when work resumed); no duplicate commit was created.


Extend the model, add callout validation, and implement version handling in the store. Add focused tests for both callout variants, old-file loading, mixed-shape round trips, unchanged timestamps, relative paths, and invalid/unsupported input.

Acceptance: existing shapes round-trip unchanged; new fields survive save/reload; malformed callouts produce useful errors; version-1 fixtures continue to load.

### 2. Shared layout and export — Done

Implemented shared Pillow layout/rendering, constrained placement, hit distance, cached fonts, and export dispatch. Added 13 tests covering eight arrow directions, degenerate geometry, edge constraints, Unicode, and shared-renderer export parity. Inspected diff; no unrelated formatter/autofix changes.

Checks: `uv run ruff format .`, `uv run ruff check --fix --unsafe-fixes .`, `uv run ruff format --check .`, `uv run ruff check .`, and `git diff --check` passed. `uv run mypy .` reported 6 existing errors in 5 unrelated Raman files; `uv run deptry .` reported 4 existing dependency issues. `uv run pytest` passed (88 tests, 5 existing SciPy warnings). Focused storage/render tests passed (31). Proceeding under the user's approval of the baseline failures. Commit: `dfe1c4d`.


Implement callout layout/rendering and integrate export dispatch. Add geometry and image-output tests for all arrow directions, horizontal/vertical and coincident endpoints, label-boundary attachment, edge placement, varying font sizes, Unicode, and mixed legacy/new shapes. Verify unchanged rendering of existing shapes.

Acceptance: exported arrows point to the saved feature coordinate; labels remain readable; layout is deterministic under a known font; degenerate geometry does not crash.

### 3. Editor tools and interaction — Done

Added two tools and a second toolbar, validated text/number/font controls, directional drag preview, Escape cancellation, bounds constraints, stable numbering, overlay caching and resizing, input shortcut protection, and callout-aware deletion. Export now draws callouts above legacy markers to match the editor overlay. Added seven headless tests exercising actual event handlers with Tcl variables, including history, deletion, invalid input, navigation, reload, and resize/cache invalidation.

Checks: `uv run ruff format .`, `uv run ruff check --fix --unsafe-fixes .`, `uv run ruff format --check .`, `uv run ruff check .`, and `git diff --check` passed. `uv run mypy .` still reports the same 6 existing errors in 5 unrelated files; `uv run deptry .` still reports the same 4 baseline issues. `uv run pytest` passed (95 tests, 5 existing SciPy warnings). Initial new lint findings were corrected before the final sequence. Tk 9 is installed; no working display or Xvfb is available, so window-level manual interaction remains unavailable and will be recorded in final verification. Commit: `3fbb51f`.


Add tools, text/number/font controls, drag preview, cancellation, image bounds handling, numbering rules, input focus protection, and callout hit testing. Connect overlay invalidation to add/delete/clear, undo/redo, load, and viewport changes. Use small pure helpers for numbering and geometry where practical.

Acceptance: create both variants, cancel without modifying history, save/reopen, navigate, undo/redo, delete, and resize successfully. Typing shortcut characters in text fields never navigates or activates tools. Preview and export use the same callout layout.

### 4. Documentation and end-to-end verification — Done

Added the annotator guide and root README link. Added an end-to-end editor-handler save/reload/export test, including mixed-shape overlay order and exact scaled-overlay comparisons at 0.5×, 1×, and 1.5×.

Checks (in required order): `uv run ruff format .` passed; `uv run ruff check --fix --unsafe-fixes .` passed; `uv run mypy .` reported the same 6 baseline errors in 5 unrelated files; `uv run deptry .` reported the same 4 baseline dependency issues; `uv run pytest` passed all 96 tests (5 existing SciPy warnings). Final `uv run ruff format --check .`, `uv run ruff check .`, and `git diff --check` passed. No dependencies or quality settings changed.

The display became available during final verification. Ran a real Tk window with generated canvas mouse events for both label types, undo/redo, save and reload. Inspected a rendered mixed-annotation sample and window captures at 1100×760 and 900×650 after allowing the existing resize debounce to settle; labels, arrow directions, and scaling were correct and controls remained visible. Live viewport scales were approximately 1.222 and 1.0. Earlier lack of display did not remain a blocker. Inspection artifacts were generated under `/tmp/imannotator-*.png` and are not repository files. Commit: `aac879a`.


Document placement direction, shortcuts, numbering rules, font sizing, version compatibility, and export usage in the repository README or a linked annotator guide. Add an end-to-end temporary-image/YAML/export test and manually inspect a representative output image and GUI at multiple viewport sizes.

Acceptance: mixed annotations survive a GUI-save/reload/export workflow; exported labels match the editor in position, arrow direction, and proportions. Record manual checks explicitly, including any unavailable GUI environment.

## Checks required before every implementation-step commit

From the repository root, inspect the working tree first and protect any pre-existing changes before project-wide formatting/autofixes. Run in this order:

```bash
uv run ruff format .
uv run ruff check --fix --unsafe-fixes .
uv run mypy .
uv run deptry .
uv run pytest
```

Inspect every autofix; revert only unrelated changes created by this work. Reformat again if autofixes require it. Confirm the final state with `uv run ruff format --check .` and `uv run ruff check .`. Run focused tests during development. Record exact commands/results and the commit hash for each completed step. If a required check is unavailable or fails for a pre-existing reason, mark the step Blocked and report the next action; do not commit it as complete without approval. Scope/order/acceptance changes require approval.

## Planning validation

Inspected all four annotator modules, repository status, project configuration, test inventory, and ancestor/repository instruction locations. Working tree was clean. At planning time only documentation had changed; no implementation checks had run yet. Completed implementation checks and GUI verification are recorded above.

## GUI review fixes — Done (uncommitted)

Fixed the three issues identified in the follow-up review:

- Drag previews render directly at viewport resolution using shared image-space layout; committed overlays and export retain their existing rendering. A regression test verifies that an 8192×6144 image allocates only a 1024×768 preview. The measured preview renderer averaged 1.12 ms/frame over 30 iterations, compared with the review's 1.18 seconds/frame for full-resolution allocation/resizing; these measurements exclude Tk image transfer and other event-handler work.
- New callout font input is limited to the toolbar's 1–300 image-pixel range; non-finite values are rejected and font-loading errors produce normal validation messages. Tests cover extreme, empty, non-finite, and out-of-range input without history mutation.
- Automatic next-number refresh tracks numbered callouts only, preserving a custom number when adding/removing legacy markers or text labels. Image loading still resets the default.

Relabeled the original implementation description as a historical baseline and clarified the planning-time validation note. Updated the user guide with font limits.

Validation: all required commands ran in order. `uv run ruff format .` and `uv run ruff check --fix --unsafe-fixes .` passed; `uv run mypy .` reports only the same 6 pre-existing errors in 5 unrelated Raman files; `uv run deptry .` reports the same 4 existing dependency issues; `uv run pytest` passes 106 tests with 5 existing SciPy warnings. Final `uv run ruff format --check .`, `uv run ruff check .`, and `git diff --check` passed. The newly introduced tuple typing issue found during development was fixed before the final sequence. No dependency/configuration changes. These are ordinary bug fixes and remain uncommitted under the user's global working preferences.

## Arrow-label color picker — Done (uncommitted)

Added a native Color picker button with a swatch in the label toolbar. The selected color applies to the arrow, label text, and border of new callouts; the white background/halo remain. Existing callouts retain their own colors. Optional `color: '#RRGGBB'` is persisted per callout; missing color defaults to the original black styling. Picker cancellation does not modify selection, annotations, or history. Prior uncommitted GUI fixes were preserved.

Validation: focused annotator tests passed (57). In the required full sequence, `uv run ruff format .` and `uv run ruff check --fix --unsafe-fixes .` passed; `uv run mypy .` reported the same 6 pre-existing errors in unrelated files; `uv run deptry .` reported the same 4 existing issues; `uv run pytest` passed 114 tests with 5 existing SciPy warnings. Final `uv run ruff format --check .`, `uv run ruff check .`, and `git diff --check` passed. A live Tk window verified button invocation and swatch update with the dialog result supplied programmatically. No dependencies or quality settings changed. No commit created for this ordinary feature change.
