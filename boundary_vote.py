"""Combine three aligned instance masks while preserving disjoint outputs."""
from pycocotools import mask as mask_utils

from validation_pq import rle_iou


MIN_SCORE = .20
MIN_ANCHOR_IOU = .65


def eligible_indices(row):
    return [index for index, (mask, score) in enumerate(zip(row['masks'], row['confidence']))
            if score >= MIN_SCORE and mask_utils.area(mask) > 0]


def mutual_matches(matrix, cutoff):
    if not matrix.size:
        return {}
    reverse = matrix.argmax(axis=0)
    return {i: int(j) for i, j in enumerate(matrix.argmax(axis=1))
            if reverse[j] == i and matrix[i, j] >= cutoff}


def majority_boundaries(anchor, mirror, scale):
    mirror_matches = mutual_matches(rle_iou(anchor, mirror), MIN_ANCHOR_IOU)
    scale_matches = mutual_matches(rle_iou(anchor, scale), MIN_ANCHOR_IOU)
    output = list(anchor)
    for i in sorted(mirror_matches.keys() & scale_matches.keys()):
        members = [anchor[i], mirror[mirror_matches[i]], scale[scale_matches[i]]]
        if rle_iou(members[1:2], members[2:])[0, 0] < MIN_ANCHOR_IOU:
            continue
        votes = [mask_utils.merge([members[a], members[b]], intersect=True)
                 for a, b in ((0, 1), (0, 2), (1, 2))]
        voted = mask_utils.merge(votes)
        # Keep the original ownership and respect earlier accepted boundary changes.
        if not rle_iou([voted], anchor[:i] + anchor[i + 1:] + output[:i]).any():
            output[i] = dict(size=voted['size'], counts=voted['counts'].decode('ascii'))
    return output


def vote_from_views(anchor, mirror, scale):
    return majority_boundaries(anchor, [mirror['masks'][i] for i in eligible_indices(mirror)],
                               [scale['masks'][i] for i in eligible_indices(scale)])
