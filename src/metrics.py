"""Competition PQ: all IoU > 0.5 pairs, pooled over annotator-image records.

Verified against Azim Ahmadzadeh's Self_Evaluation_Notebook, version 6:
https://www.kaggle.com/code/azimahmadzadeh/self-evaluation-notebook
This deliberately does not perform one-to-one assignment or category matching.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from pycocotools import mask as mask_utils

from src.masks import encode_mask


@dataclass(frozen=True)
class PQStats:
    """Additive statistics; TP counts qualifying pairs, not unique instances."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0

    @property
    def pq(self):
        denominator = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / denominator if denominator else 0.0

    def __add__(self, other):
        if not isinstance(other, PQStats):
            return NotImplemented
        return PQStats(
            self.tp + other.tp,
            self.fp + other.fp,
            self.fn + other.fn,
            self.iou_sum + other.iou_sum,
        )


def _mask_sequence(masks):
    """Validate a mask list or (N, H, W) array, including empty stacks."""
    shape = None
    if isinstance(masks, np.ndarray):
        if masks.ndim != 3:
            raise ValueError("Mask stacks must have shape (N, H, W).")
        shape = masks.shape[1:]
    result = []
    for mask in masks:
        mask = np.asarray(mask)
        if mask.ndim != 2:
            raise ValueError("Each instance mask must be 2D.")
        if shape is None:
            shape = mask.shape
        if mask.shape != shape:
            raise ValueError("Mask shapes must match within each record.")
        if not np.isfinite(mask).all():
            raise ValueError("Mask values must be finite.")
        result.append(mask.astype(bool))
    if shape is not None and 0 in shape:
        raise ValueError("Mask dimensions must be nonzero.")
    return result, shape


def pairwise_mask_iou(gt_masks, pred_masks):
    """Return (N_gt, N_pred) IoUs using COCO RLE without a pixel-pair tensor.

    Inputs are lists of 2D masks or (N, H, W) arrays. Nonzero pixels are
    foreground. Empty masks have IoU zero, including empty/empty pairs.
    """
    gt, gt_shape = _mask_sequence(gt_masks)
    pred, pred_shape = _mask_sequence(pred_masks)
    if gt_shape is not None and pred_shape is not None and gt_shape != pred_shape:
        raise ValueError("Ground-truth and prediction mask shapes must match.")
    if not gt or not pred:
        return np.zeros((len(gt), len(pred)), dtype=np.float64)
    # COCO computes the same intersection/union from compressed masks, with
    # much less temporary memory than expanding all full-resolution pairs.
    return mask_utils.iou(
        [encode_mask(mask) for mask in gt],
        [encode_mask(mask) for mask in pred],
        [0] * len(pred),
    )


def pq_stats_from_iou(iou_matrix):
    """Count all strict > 0.5 hits; FP/FN are unmatched columns/rows."""
    ious = np.asarray(iou_matrix, dtype=np.float64)
    if ious.ndim != 2:
        raise ValueError("IoU matrix must be 2D (N_gt, N_pred).")
    if not np.isfinite(ious).all() or np.any((ious < 0) | (ious > 1)):
        raise ValueError("IoU values must be finite and within [0, 1].")
    hits = ious > 0.5
    return PQStats(
        tp=int(hits.sum()),
        fp=int((~hits.any(axis=0)).sum()),
        fn=int((~hits.any(axis=1)).sum()),
        iou_sum=float(ious[hits].sum()),
    )


def evaluate_record(gt_masks, pred_masks):
    """Evaluate one annotator's masks against the physical image predictions."""
    return pq_stats_from_iou(pairwise_mask_iou(gt_masks, pred_masks))


def aggregate_pq(record_stats):
    """Pool record counts and IoUs, then calculate PQ via the result's .pq."""
    total = PQStats()
    for stats in record_stats:
        total = total + stats
    return total


def evaluate_dataset(dataset, predictions_by_filename: Mapping):
    """Score every FilamentDataset record independently and pool statistics.

    Predictions map exact filenames (including extensions) to mask lists or
    (N, H, W) arrays. A repeated filename reuses the same predictions for each
    annotator record. Supply [] explicitly for images with no predictions.
    Missing or extra filenames are errors so the evaluation scope is explicit.
    """
    expected_names = {info["file_name"] for info in dataset.images}
    supplied_names = set(predictions_by_filename)
    if supplied_names != expected_names:
        raise ValueError(
            "Prediction filenames must match the dataset: "
            f"missing={sorted(expected_names - supplied_names)}, "
            f"extra={sorted(supplied_names - expected_names)}"
        )
    return aggregate_pq(
        evaluate_record(sample["masks"], predictions_by_filename[sample["file_name"]])
        for sample in (dataset[index] for index in range(len(dataset)))
    )
