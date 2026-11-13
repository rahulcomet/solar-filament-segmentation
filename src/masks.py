"""Convert instance polygons and binary masks using COCO mask conventions."""

import numpy as np
from pycocotools import mask as mask_utils


def polygon_to_mask(polygons, height, width):
    """Rasterize COCO polygons belonging to one instance into a uint8 mask.

    Multiple polygons are merged within this instance only. An empty polygon
    list returns an empty mask. Coordinates use the original image geometry.
    """
    if not isinstance(height, (int, np.integer)) or not isinstance(width, (int, np.integer)):
        raise ValueError("Mask height and width must be positive integers.")
    if height <= 0 or width <= 0:
        raise ValueError("Mask height and width must be positive integers.")
    validated = []
    for polygon in polygons:
        coordinates = np.asarray(polygon, dtype=np.float64)
        if coordinates.ndim != 1 or coordinates.size < 6 or coordinates.size % 2:
            raise ValueError("Each polygon must contain at least three x/y pairs.")
        if not np.isfinite(coordinates).all():
            raise ValueError("Polygon coordinates must be finite.")
        validated.append(coordinates.tolist())
    if not validated:
        return np.zeros((height, width), dtype=np.uint8)
    rles = mask_utils.frPyObjects(validated, height, width)
    return mask_utils.decode(mask_utils.merge(rles)).astype(np.uint8)


def encode_mask(binary_mask):
    """Encode a 2D mask as JSON-compatible compressed COCO RLE.

    Positive values are foreground. COCO requires column-major storage.
    """
    binary_mask = np.asarray(binary_mask)
    if binary_mask.ndim != 2:
        raise ValueError("Expected a 2D binary mask.")
    if 0 in binary_mask.shape:
        raise ValueError("Mask dimensions must be nonzero.")
    binary_mask = np.asfortranarray((binary_mask > 0).astype(np.uint8))
    rle = mask_utils.encode(binary_mask)
    return {"size": list(rle["size"]), "counts": rle["counts"].decode("ascii")}


def decode_mask(rle):
    """Decode one compressed COCO RLE with string or bytes counts."""
    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("ascii")
    decoded = mask_utils.decode({"size": rle["size"], "counts": counts})
    return decoded.astype(np.uint8)


def mask_iou(mask_a, mask_b):
    """Return pixel IoU for two equally shaped 2D masks; empty union is 0."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("Mask shapes must match.")
    if a.ndim != 2:
        raise ValueError("Expected 2D binary masks.")
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / union) if union else 0.0
