"""Mask geometry helpers for prediction cleanup and display.

Human labels are never changed by training loaders. The PMC annotation UI also
uses these helpers on explicit save, with class reassignment disabled, according
to the user's single-target annotation rule.

The single-component rule is a task prior, not a universal anatomical fact.
Joint A/V cleanup repairs constrained class mixing, closes small gaps and fills
enclosed holes. No opening is used, preserving thin lines. Empty input stays empty.
Area/radius are configurable input-grid pixels; a sufficiently large false positive
still cannot be identified by connected components alone. clean_binary retains the
legacy deletion-only rule for reproducible comparisons.
"""
import cv2
import numpy as np


def clean_binary(binary, min_area=1, max_dist=40.0, keep_largest=True):
    """Keep at most one 8-connected region, including a one-pixel-wide line.

An explicit keep_largest=False retains the historical distance-gate option for
experiments. min_area is in input-grid pixels; increasing it can remove real
compressed veins. No component is created, enlarged, or joined by this function.
    """
    b = (np.asarray(binary) > 0).astype(np.uint8)
    if b.ndim != 2 or min_area < 0 or max_dist < 0:
        raise ValueError("Expected a 2D mask and nonnegative area/distance")
    n, labels, stats, _ = cv2.connectedComponentsWithStats(b, connectivity=8)
    survivors = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not survivors:
        return np.zeros_like(b, dtype=bool)
    anchor = max(survivors, key=lambda i: stats[i, cv2.CC_STAT_AREA])
    if keep_largest or max_dist == 0:
        return labels == anchor
    distance = cv2.distanceTransform((labels != anchor).astype(np.uint8), cv2.DIST_L2, 5)
    keep = [i for i in survivors if i == anchor or distance[labels == i].min() <= max_dist]
    return np.isin(labels, keep)


def complete_region(binary, min_area=4, closing_radius=2, fill_holes=True):
    """One region, local closing and enclosed-hole filling; never opening/erosion.

    Radius and area are pixels on the supplied grid. Pad with background so closing
    cannot flood the image when a vessel touches an image edge. Empty stays empty.
    """
    if not isinstance(closing_radius, (int, np.integer)) or closing_radius < 0:
        raise ValueError("closing_radius must be a nonnegative integer")
    b = clean_binary(binary, min_area=min_area)
    if not b.any():
        return b
    if closing_radius:
        pad = 2 * closing_radius + 1
        padded = np.pad(b.astype(np.uint8), pad)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                          (2 * closing_radius + 1,) * 2)
        closed = cv2.morphologyEx(padded, cv2.MORPH_CLOSE, kernel)
        b |= closed[pad:-pad, pad:-pad].astype(bool)
    if fill_holes:
        padded = np.pad(b.astype(np.uint8), 1)
        exterior = padded.copy()
        # Use 4-connected background as the dual of 8-connected foreground.
        cv2.floodFill(exterior, None, (0, 0), 2, flags=4)
        b |= exterior[1:-1, 1:-1] == 0
    return clean_binary(b, min_area=min_area)


def clean_labels(labels, min_area=4, *, closing_radius=2, fill_holes=True,
                 reassign=True, dominance=0.6):
    """Repair local class mixing, then keep at most one region per A/V class.

    Each class's largest original component is an anchor. A raw vessel-union
    component can change minority-class pixels only if it contains exactly ONE
    class anchor and that class owns at least `dominance` of its pixels. Components
    containing both anchors are ambiguous (possibly touching real A/V): never
    majority-vote them into one class. Detached discarded components are not moved.
    Morphology cannot overwrite the other class's retained pixels; new collisions
    go to the nearest pre-completion mask. This is a heuristic, not anatomy truth.
    """
    labels = np.asarray(labels)
    if labels.ndim != 2 or not np.isin(labels, [0, 1, 2]).all():
        raise ValueError("Expected model ids 0=background, 1=vein, 2=artery")
    if not 0.5 < dominance <= 1:
        raise ValueError("dominance must be in (0.5, 1]")
    anchors = {c: clean_binary(labels == c, min_area=min_area) for c in (1, 2)}
    repaired = labels.copy()
    if reassign:
        n, components = cv2.connectedComponents((labels > 0).astype(np.uint8), connectivity=8)
        for i in np.unique(components[anchors[1] | anchors[2]]):
            region = components == i
            owners = [c for c in (1, 2) if np.any(anchors[c] & region)]
            if len(owners) == 1:
                owner = owners[0]
                if np.count_nonzero((labels == owner) & region) >= dominance * region.sum():
                    repaired[region] = owner
    bases = {c: clean_binary(repaired == c, min_area=min_area) for c in (1, 2)}
    masks = {c: complete_region(bases[c], min_area, closing_radius, fill_holes)
             & ~bases[3-c] for c in (1, 2)}
    overlap = masks[1] & masks[2]
    if overlap.any():
        distance = {c: cv2.distanceTransform((~bases[c]).astype(np.uint8), cv2.DIST_L2, 5)
                    for c in (1, 2)}
        vein_wins = distance[1] <= distance[2]
        masks[1][overlap & ~vein_wins] = False
        masks[2][overlap & vein_wins] = False
    out = np.zeros(labels.shape, dtype=np.uint8)
    for c in (1, 2):
        out[clean_binary(masks[c], min_area=min_area)] = c
    return out


def av_outputs(vein, artery, min_area=4, **completion):
    """Independent masks -> cleaned A/V, union, intersection and RGB display.

display_labels: 0 background, 1 blue vein, 2 red artery, 3 green overlap.
Id 3 here is DISPLAY ONLY, never a fourth model class or a training target.
vessel_rgb renders the ENTIRE union green; display_rgb renders only overlap green.
Softmax argmax A/V masks are disjoint, so overlap is normally empty today.
    """
    vein, artery = np.asarray(vein), np.asarray(artery)
    if vein.shape != artery.shape:
        raise ValueError("Vein and artery masks must have the same shape")
    vein, artery = vein.astype(bool), artery.astype(bool)
    if np.any(vein & artery):
        # Independent overlapping masks have no mutually exclusive class evidence.
        # Preserve their overlap as a display layer instead of inventing ownership.
        v = complete_region(vein, min_area, **completion)
        a = complete_region(artery, min_area, **completion)
    else:
        labels = vein.astype(np.uint8) + 2 * artery.astype(np.uint8)
        labels = clean_labels(labels, min_area, **completion)
        v, a = labels == 1, labels == 2
    union, overlap = v | a, v & a
    display = v.astype(np.uint8) + 2 * a.astype(np.uint8)
    palette = np.array([[0, 0, 0], [77, 159, 255], [255, 96, 104], [68, 221, 136]], dtype=np.uint8)
    return dict(vein=v, artery=a, vessel=union, overlap=overlap,
                display_labels=display, display_rgb=palette[display],
                vessel_rgb=np.where(union[..., None], palette[3], palette[0]).astype(np.uint8))
