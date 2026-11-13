"""Admit corroborated, disjoint filament candidates after boundary voting."""
import cv2
import numpy as np
from pycocotools import mask as rle

from boundary_vote import eligible_indices
from validation_pq import rle_iou


FEATURES = ('confidence', 'coarse_iou', 'other_view_iou', 'second_view_iou',
            'other_view_supported_confidence', 'log_area', 'local_contrast',
            'original_view', 'flipped_view')


def contrast(mask, image):
    x, y, w, h = rle.toBbox(mask).astype(int)
    ys = slice(max(0, y - 4), y + h + 4)
    xs = slice(max(0, x - 4), x + w + 4)
    foreground = rle.decode(mask)[ys, xs]
    ring = (cv2.dilate(foreground, np.ones((9, 9), np.uint8)) > 0) & (foreground == 0)
    patch = image[ys, xs]
    ring &= patch > 0
    return float(np.median(patch[ring]) - np.median(patch[foreground > 0])) if ring.any() else 0.


def candidate_rows(anchor, views, image):
    masks, coarse, confidence, source = [], [], [], []
    for view_id, row in enumerate(views):
        for i in eligible_indices(row):
            masks.append(row['masks'][i])
            coarse.append(row['coarse_masks'][i])
            confidence.append(row['confidence'][i])
            source.append(view_id)
    confidence = np.asarray(confidence)
    source = np.asarray(source)
    cross = rle_iou(masks, masks)
    coarse_agreement = rle_iou(masks, coarse).diagonal()
    isolated = ~rle_iou(masks, anchor).any(axis=1)
    rows = []
    for i in np.flatnonzero(isolated):
        support = sorted((cross[i, source == other].max(initial=0)
                          for other in range(3) if other != source[i]), reverse=True)
        corroboration = confidence[(source != source[i]) & (cross[i] >= .55)].max(initial=0)
        features = [confidence[i], coarse_agreement[i], *support, corroboration,
                    np.log1p(rle.area(masks[i])), contrast(masks[i], image),
                    float(source[i] == 0), float(source[i] == 1)]
        rows.append(dict(mask=masks[i], view=int(source[i]), features=[float(value) for value in features]))
    return rows


def select(rows, priorities):
    selected = []
    masks = [row['mask'] for row in rows]
    overlap = rle_iou(masks, masks) > 0
    for i in np.argsort(-np.asarray(priorities, dtype=float), kind='stable'):
        if not np.isfinite(priorities[i]):
            break
        if not overlap[i, selected].any():
            selected.append(int(i))
    return selected


def contrast_priority(features):
    confidence, coarse, support, _, _, _, darkness, *_ = features
    return confidence * support if support >= .55 and coarse >= .50 and darkness >= 12.5 else -np.inf


def admit(anchor, views, image):
    rows = candidate_rows(anchor, views, image)
    priorities = [contrast_priority(row['features']) for row in rows]
    return anchor + [rows[i]['mask'] for i in select(rows, priorities)]
