"""Three fixed post-pipeline ablations; never reassign overlapping pixels."""
import cv2
import numpy as np
from refiner import encode_mask,decode_mask
from reaudit_fp_progress import parents,parent_union


def prune_tiny_satellites(mask):
    count,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),connectivity=8)
    areas=stats[1:,cv2.CC_STAT_AREA]
    if not len(areas):return mask.copy()
    largest=int(areas.argmax())+1;total=int(mask.sum())
    remove=[j+1 for j,a in enumerate(areas) if j+1!=largest and a<64 and a<.10*total]
    return mask&~np.isin(labels,remove)


def apply_option(row,coarse,option):
    assert option in ['confidence025','tiny_satellites','yolo_consensus']
    kept=[];masks=[];modified=[];removed=[]
    for j,rle in enumerate(row['masks']):
        source=row['source_proposal_ids'][j]
        if option=='confidence025':
            if row['confidence'][j]<.25:removed.append(source);continue
            new=rle
        else:
            old=decode_mask(rle).astype(bool)
            out=prune_tiny_satellites(old) if option=='tiny_satellites' else old&decode_mask(parent_union(coarse,parents(row,j))).astype(bool)
            assert not (out&~old).any()
            if not out.any():removed.append(source);continue
            new=encode_mask(out)
            if new!=rle:modified.append(source)
        kept.append(j);masks.append(new)
    return dict(file_name=row['file_name'],masks=masks,confidence=[row['confidence'][j] for j in kept],
        source_proposal_ids=[row['source_proposal_ids'][j] for j in kept],source_groups=[parents(row,j) for j in kept]),dict(modified=modified,removed=removed)
