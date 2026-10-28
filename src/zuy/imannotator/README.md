# Image annotator

Start the editor on a folder of images:

```bash
uv run semangui --dir /path/to/images
```

Choose **Numbered label (B)** or **Text label (T)**. Set the text or next number and
font size in the second toolbar. Press on the image feature, drag to the label
position, and release. The arrow points back to the feature. Horizontal and
vertical arrows work. A drag shorter than three screen pixels is ignored.

Text labels accept non-empty, single-line Unicode text; the text stays in the
field for repeated placements. Numbered badges start at 1 for each image. The
default next number is one greater than the highest existing numbered badge.
You can enter a different unused positive integer. Deleting a badge does not
renumber the others. Undo, redo, deletion, and image loading recalculate the next
number from the current annotations. Failed or cancelled placement consumes no
number.

Font size is measured in original-image pixels, independently of the circle or
cross Size control. Labels and arrows scale with the image when resizing the
window. Label boxes stay inside the image where possible; oversized text is
centered and clipped at the image boundary. Use a smaller font for long labels.
DejaVu Sans is preferred, with Pillow's default font as a fallback. Unicode glyph
coverage and text metrics depend on the available font. Use the same font
installation when editing and exporting for consistent layout.

Escape cancels a pending drag; changing tools or images also cancels placement.
Undo/redo, Delete-last, and Clear treat each callout as one shape. Right-click
deletes the nearest annotation, measuring callouts against their box and arrow.
To change an existing label, delete it and place it again. Application shortcuts
are ignored while typing in text or numeric inputs.

Existing tools remain Circle (C), Cross (X), and Rectangle (R). Use Next (N or
Right), Prev (P or Left), Save (S), Ctrl+Z for undo, and Ctrl+Y or Ctrl+Shift+Z for
redo. Images save automatically when navigating or closing the editor.

Annotations are saved to `annotations.yaml` beside the images, using relative
image paths. Existing version-1 files continue to load; adding a callout upgrades
the document to version 2, which is retained even if the callouts are deleted.
Older application versions do not support the new labels. Keep the YAML file
with its images when moving a folder.

Export annotated copies:

```bash
uv run semandraw --dir /path/to/images --out-dir /path/to/output
# Or specify the annotation file directly:
uv run semandraw /path/to/images/annotations.yaml --suffix _annotated
```

Exports use the original resolution. Callouts use the same Pillow layout as the
editor and appear above legacy markers. Existing circles, crosses, and rectangles
retain their previous rendering styles. Output names default to the `_x` suffix.
