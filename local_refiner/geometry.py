"""Proposal-derived crop geometry, independent of training targets."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class CropTransform:
    shape: tuple
    box: tuple
    resized: tuple
    offset: tuple
    size: int = 512

    @classmethod
    def from_mask(cls, mask, size=512, padding=32):
        yy, xx = np.nonzero(mask)
        if not len(xx):
            raise ValueError('Cannot construct a crop from an empty proposal')
        h, w = mask.shape
        x0, y0 = max(0, int(xx.min())-padding), max(0, int(yy.min())-padding)
        x1, y1 = min(w, int(xx.max())+padding+1), min(h, int(yy.max())+padding+1)
        ch, cw = y1-y0, x1-x0
        scale = min(1., size/max(ch, cw))
        rh, rw = max(1, round(ch*scale)), max(1, round(cw*scale))
        return cls((h, w), (x0, y0, x1, y1), (rh, rw), ((size-rh)//2, (size-rw)//2), size)

    def extract(self, array, mask=False):
        x0, y0, x1, y1 = self.box
        rh, rw = self.resized
        oy, ox = self.offset
        crop = array[y0:y1, x0:x1]
        if crop.shape != (rh, rw):
            crop = cv2.resize(crop, (rw, rh), interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR)
        canvas = np.zeros((self.size, self.size), dtype=array.dtype)
        canvas[oy:oy+rh, ox:ox+rw] = crop
        return canvas

    def restore(self, canvas, mask=False):
        x0, y0, x1, y1 = self.box
        rh, rw = self.resized
        oy, ox = self.offset
        crop = canvas[oy:oy+rh, ox:ox+rw]
        if crop.shape != (y1-y0, x1-x0):
            crop = cv2.resize(crop, (x1-x0, y1-y0), interpolation=cv2.INTER_NEAREST if mask else cv2.INTER_LINEAR)
        result = np.zeros(self.shape, dtype=canvas.dtype)
        result[y0:y1, x0:x1] = crop
        return result

    def valid(self):
        rh, rw = self.resized
        oy, ox = self.offset
        canvas = np.zeros((self.size, self.size), dtype=np.uint8)
        canvas[oy:oy+rh, ox:ox+rw] = 1
        return canvas


def clear_associations(ious, coverage, minimum=.1):
    """Conservative per-record training matches; official PQ remains unchanged."""
    result = []
    for proposal in range(ious.shape[1]):
        if not ious.shape[0]:
            continue
        order = np.argsort(ious[:, proposal])[::-1]
        target = int(order[0])
        best = float(ious[target, proposal])
        second = float(ious[order[1], proposal]) if len(order) > 1 else 0.
        if (best >= minimum and second < best*.5
                and int(ious[target].argmax()) == proposal
                and np.count_nonzero(coverage[:, proposal] >= .25) <= 1):
            result.append((proposal, target, best))
    return result


def restore_prediction(probability, geometry, original, threshold=.5):
    if not np.isfinite(probability).all():
        raise ValueError('Nonfinite refiner probabilities')
    prediction = (geometry.restore(probability)>=threshold).astype(np.uint8)
    empty = not prediction.any()
    return (original.copy() if empty else prediction), empty
