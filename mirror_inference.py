import cv2
import numpy as np
import torch
from PIL import Image
from refiner import CropTransform, decode_mask, encode_mask, make_input, restore_prediction
from validation_pq import rle_iou


def mirror_coarse(detector, path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot read {path}')
    result = detector.predict(cv2.flip(image, 1), imgsz=2048, conf=.15, iou=.7,
                              max_det=100, retina_masks=True, device='0', verbose=False)[0]
    if result.masks is None:
        return dict(file_name=path.name, masks=[], confidence=[])
    assert len(result.masks) == len(result.boxes) < 100
    native = result.masks.data.cpu().numpy().astype('uint8')
    return dict(file_name=path.name, masks=[encode_mask(mask[:, ::-1]) for mask in native],
                confidence=result.boxes.conf.cpu().tolist())


def high_resolution_coarse(detector, path):
    result = detector.predict(str(path), imgsz=2560, conf=.18, iou=.7, max_det=100,
                              retina_masks=True, device='0', verbose=False)[0]
    if result.masks is None:
        return dict(file_name=path.name, masks=[], confidence=[])
    masks = [encode_mask(mask) for mask in result.masks.data.cpu().numpy().astype('uint8')]
    confidence = result.boxes.conf.cpu().tolist()
    assert len(masks) == len(confidence)
    return dict(file_name=path.name, masks=masks, confidence=confidence)


def supported(raw, mirror):
    agreement = rle_iou(raw['coarse_masks'], mirror['coarse_masks'])
    if not agreement.size:
        return []
    reverse = agreement.argmax(axis=0)
    matched = []
    for i, j in enumerate(agreement.argmax(axis=1)):
        if reverse[j] != i or agreement[i, j] < .45:
            continue
        if rle_iou([raw['masks'][i]], [mirror['masks'][j]])[0, 0] < .50:
            continue
        confidence = (raw['confidence'][i] + mirror['confidence'][j]) / 2
        # Prefer the original-view shape. The mirrored view supplies corroboration.
        matched.append((raw['masks'][i], confidence))
    return matched


def append_isolated(anchor, candidates, cutoff):
    output = list(anchor)
    for mask, confidence in sorted(candidates, key=lambda pair: -pair[1]):
        if confidence >= cutoff and not rle_iou([mask], output).any():
            output.append(mask)
    return output


def refine(net,source,path):
    with Image.open(path) as im:
        assert im.size==(2048,2048);image=np.asarray(im.convert('L')).copy()
    result=dict(file_name=source['file_name'],confidence=source['confidence'],coarse_masks=source['masks'],masks=[])
    for rle in source['masks']:
        original=decode_mask(rle);geometry=CropTransform.from_mask(original,size=512,padding=32)
        x=torch.from_numpy(make_input(geometry.extract(image),geometry.extract(original,mask=True),geometry.valid(),'image'))[None].cuda()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            probability=net(x).float().sigmoid()[0,0].cpu().numpy()
        result['masks'].append(encode_mask(restore_prediction(probability,geometry,original,.55)[0]))
    return result
