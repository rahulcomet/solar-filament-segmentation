"""Annotation-free confidence/geometry selection, isolated from production export."""
from dataclasses import dataclass
import cv2
import numpy as np
from pycocotools import mask as mu
from refiner import decode_mask, encode_mask
from validation_pq import rle_iou


@dataclass(frozen=True)
class Policy:
    name: str
    cutoff: float = .20
    area_power: float = 0.
    coherence_power: float = 0.
    mask_nms: bool = True
    weak_only: bool = False
    min_remaining: float = .10


POLICIES = [
    Policy('original_025', cutoff=.25, mask_nms=False, min_remaining=0.),
    Policy('original_020', mask_nms=False, min_remaining=0.),
    Policy('previous_weak_cleanup_020', weak_only=True),
    Policy('global_cleanup_025', cutoff=.25),
    Policy('global_cleanup_020'),
    Policy('confidence_area_quarter_020', area_power=.25),
    Policy('confidence_area_half_020', area_power=.5),
    Policy('confidence_area_half_coherence_020', area_power=.5, coherence_power=.5),
]


def features(row):
    decoded = [decode_mask(r).astype(bool) for r in row['masks']]
    areas = np.array([int(mu.area(r)) for r in row['masks']], dtype=float)
    coherence = []
    for rle, mask, area in zip(row['masks'], decoded, areas):
        x, y, w, h = [int(v) for v in mu.toBbox(rle)]
        count, _, stats, _ = cv2.connectedComponentsWithStats(mask[y:y+h, x:x+w].astype(np.uint8), connectivity=8)
        coherence.append(float(stats[1:, cv2.CC_STAT_AREA].max())/area if count > 1 else 0.)
    return dict(decoded=decoded, areas=areas, coherence=np.array(coherence), pair_iou=rle_iou(row['masks'], row['masks']))


def apply_policy(row, policy, cached=None, shape=(2048,2048)):
    f = features(row) if cached is None else cached
    confidence = np.asarray(row['confidence'])
    ids = [i for i, c in enumerate(confidence) if c >= policy.cutoff]
    scores = confidence * f['areas']**policy.area_power * f['coherence']**policy.coherence_power
    order = sorted(ids, key=lambda i:(-scores[i], -confidence[i], i))
    kept, suppressed = [], []
    for i in order:
        eligible = policy.mask_nms and (not policy.weak_only or confidence[i] < .25)
        if eligible and kept and (f['pair_iou'][i, kept] >= .5).any():
            suppressed.append(i)
        else:
            kept.append(i)
    occupied = np.zeros(shape, dtype=bool)
    encoded, fractions = {}, {}
    dropped = []
    for i in kept:
        mask = f['decoded'][i]
        assert mask.shape == shape and mask.any()
        remain = mask & ~occupied
        occupied |= mask
        fraction = int(remain.sum())/f['areas'][i]
        fractions[i] = fraction
        eligible = not policy.weak_only or confidence[i] < .25
        if not remain.any() or (eligible and fraction < policy.min_remaining):
            dropped.append(i)
        else:
            encoded[i] = encode_mask(remain)
    final = sorted(encoded)
    source_ids = row['source_proposal_ids']
    return dict(file_name=row['file_name'], masks=[encoded[i] for i in final],
                confidence=[row['confidence'][i] for i in final], source_proposal_ids=[source_ids[i] for i in final]), dict(
        input_candidates=len(ids), suppressed=[source_ids[i] for i in suppressed],
        removed_after_clipping=[source_ids[i] for i in dropped],
        output_candidates=len(final), ranking_scores={str(source_ids[i]):float(scores[i]) for i in ids})
