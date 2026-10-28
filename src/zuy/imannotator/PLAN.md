# Arrow labels for imannotator

Status: Approved for implementation by the user.

## Goal

Add numbered and free-text callouts to `semangui`, persist them in `annotations.yaml`, and render them with `semandraw`. Preserve existing circles, crosses, rectangles, file paths, autosave, and undo/redo behavior.

## Current implementation

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

### 1. Model and YAML compatibility — Blocked

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

Next action: obtain approval to proceed with these documented pre-existing check failures, or approval to expand scope to repair them. Steps 2–4 have not started. The user’s repository instructions prohibit committing this step as complete without that approval.


Extend the model, add callout validation, and implement version handling in the store. Add focused tests for both callout variants, old-file loading, mixed-shape round trips, unchanged timestamps, relative paths, and invalid/unsupported input.

Acceptance: existing shapes round-trip unchanged; new fields survive save/reload; malformed callouts produce useful errors; version-1 fixtures continue to load.

### 2. Shared layout and export — Pending

Implement callout layout/rendering and integrate export dispatch. Add geometry and image-output tests for all arrow directions, horizontal/vertical and coincident endpoints, label-boundary attachment, edge placement, varying font sizes, Unicode, and mixed legacy/new shapes. Verify unchanged rendering of existing shapes.

Acceptance: exported arrows point to the saved feature coordinate; labels remain readable; layout is deterministic under a known font; degenerate geometry does not crash.

### 3. Editor tools and interaction — Pending

Add tools, text/number/font controls, drag preview, cancellation, image bounds handling, numbering rules, input focus protection, and callout hit testing. Connect overlay invalidation to add/delete/clear, undo/redo, load, and viewport changes. Use small pure helpers for numbering and geometry where practical.

Acceptance: create both variants, cancel without modifying history, save/reopen, navigate, undo/redo, delete, and resize successfully. Typing shortcut characters in text fields never navigates or activates tools. Preview and export use the same callout layout.

### 4. Documentation and end-to-end verification — Pending

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

Inspected all four annotator modules, repository status, project configuration, test inventory, and ancestor/repository instruction locations. Working tree was clean. This proposal changes documentation only; no implementation, Python quality checks, or GUI tests have run.
