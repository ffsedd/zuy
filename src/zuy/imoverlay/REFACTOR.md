auto init points do not work well with SEM-EDS and microscope image, features are very different. Can I find transformation based on sample boundary and boundaries between layers?


Yes — that's a much better idea than trying to fix ORB for this use case. SEM/SEM-EDS and optical microscope images differ so much in imaging physics (topography/composition contrast vs. reflected light/color) that local feature descriptors like ORB rarely find true correspondences across modalities — they're matching on local gradient patterns that mean physically different things in each domain. Boundary-based registration works better here because a physical edge (the sample outline, or the interface between layers) is a *structural* fact that exists in both images regardless of how each modality renders texture inside the regions it bounds.

Two viable strategies, in order of how much I'd trust them for this:

**1. Landmark points from boundary geometry (most robust — what I'd start with)**

Segment the sample outline and layer boundaries independently in each modality (thresholds/edge detectors will differ between SEM and optical, that's expected and fine), then extract a *small number of well-defined geometric landmarks* — corners, boundary intersections — rather than trying to match arbitrary points along a curve. Corners and T-junctions (where a layer boundary meets the sample edge) are strong landmarks because their existence and rough ordering is preserved across modalities even when local appearance is unrecognizable.

```python
def segment_boundary(img: np.ndarray, *, blur_ksize: int = 5) -> np.ndarray:
    """Binary mask of the sample: largest contour after Otsu threshold."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [largest], -1, 255, thickness=cv2.FILLED)
    return out


def boundary_corners(mask: np.ndarray, *, epsilon_frac: float = 0.01) -> List[Point]:
    """Polygon-approximate the outer contour and return its corner points, in order."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest = max(contours, key=cv2.contourArea)
    epsilon = epsilon_frac * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    return [(int(p[0][0]), int(p[0][1])) for p in approx]
```

This alone gets you the outer sample boundary corners in both images as a **small, ordered set of points** — which is exactly what `homography_disp` needs, and it's a far smaller, more reliable correspondence problem than ORB's hundreds of spurious feature matches. The main thing you still have to solve is *correspondence order* (point 1 in OPT = which point in SEM?) — since the sample is roughly a known shape (often near-rectangular for a cross-section or die), sorting corners by angle around the centroid and picking a consistent starting corner (e.g. topmost-then-leftmost) usually gets you a consistent cyclic order in both images, up to a rotation/reflection ambiguity you'd resolve once per session (or with one manual click to anchor "this is corner 1").

**2. Layer boundaries, if the sample is a cross-section**

Same idea, one level deeper: for each internal layer boundary, take the mask, run `cv2.findContours` with `RETR_LIST` instead of `RETR_EXTERNAL`, and either fit a line/curve per boundary (`cv2.fitLine` if roughly straight, useful for cross-sections) or take its intersections with the outer boundary as landmark points. Line-boundary intersections are usually the sharpest, most identifiable points in both modalities.

**3. Fallback: segmentation-assisted manual clicking**

Given how ambiguous cross-modal correspondence can get with noisy real SEM/EDS data, a very practical middle ground — and probably worth building regardless of how far you take (1)/(2) — is to **display the segmented boundary overlay** (mask contour drawn on the image) as an additional clickable view alongside the raw image, so a human is clicking on a clean boundary line rather than hunting for a landmark in noisy texture. That turns your existing manual-click workflow into a much faster, more accurate one without needing full automation to work perfectly.

**On ICP, if you want closer-to-automatic:** once you have a rough initial homography from a handful of boundary corners (from 1), you could refine it by extracting dense point sets along both boundaries (`cv2.findContours` with `CHAIN_APPROX_NONE`) and running iterative closest point (not in OpenCV directly — `open3d` has a 2D-friendly ICP, or it's a ~30-line loop: transform points, find nearest neighbors, re-fit homography, repeat). I'd only reach for this after (1) gives a decent starting alignment — ICP without a good initialization will converge to nonsense given how different the layer patterns can look.

Want me to sketch this as a replacement `auto_init_points_boundary()` that slots into the existing `semvis()` flow (same return shape as `auto_init_points`, so it's a drop-in swap), and add it as a new step in REFACTOR.md?
