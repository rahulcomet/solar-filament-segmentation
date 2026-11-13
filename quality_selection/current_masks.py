"""Shared exact preparation for the selected confidence0.18 research pipeline."""
import cv2,numpy as np
from pycocotools import mask as mu
from refiner import DEFAULT_PROJECT,encode_mask

def prepare(raw,minimum_added_fraction=0.,*,image_path=None):
    ids=[i for i,c in enumerate(raw['confidence']) if c>=.18]
    row=dict(file_name=raw['file_name'],
        **{k:[raw[k][i] for i in ids] for k in ['masks','coarse_masks','confidence']},
        source_proposal_ids=list(range(len(ids))))
    if image_path is not None:row["image_path"]=str(image_path)
    image=None
    for i,(m,c) in enumerate(zip(row['masks'],row['coarse_masks'])):
        a=mu.decode(m).astype(bool);co=mu.decode(c).astype(bool)
        if (a&co).sum()/a.sum()<.95 or not .5<=a.sum()/co.sum()<1:continue
        if image is None:image=cv2.imread(str(row.get('image_path',DEFAULT_PROJECT/'data/train/images'/row['file_name'])),0)
        ring=cv2.dilate(a.astype('uint8'),np.ones((17,17),'uint8')).astype(bool)&~a&(image>0)
        if not ring.any():continue
        add=co&~a&(image.astype(float)<=float(image[ring].mean())-12.5)
        if add.any() and int(add.sum())>=minimum_added_fraction*int(a.sum()):
            row['masks'][i]=encode_mask((a|add).astype('uint8'))
    return row
