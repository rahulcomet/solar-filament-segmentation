"""Conservative post-cleanup rejection from original-image and mask evidence."""
from copy import deepcopy
import cv2
import numpy as np
from common import ROOT
from refiner import DEFAULT_PROJECT
from pycocotools import mask as mu
from validation_pq import rle_iou
from remnant_candidates import filter_history


def fragmented_disagreement(e):
    return (e['confidence']<.50 and e['main_fraction']<.65 and
            e['coarse_agreement']<.50 and e['contrast'] is not None and e['contrast']<12.5)


def reject(prepared,history,fragmented=False,remnant=False,unsupported=False):
    result=filter_history(prepared,history,confidence_limit=.55) if remnant else deepcopy(history)
    removed=[];image=None
    for j,m in enumerate(result['masks'] if fragmented or unsupported else []):
        sources=[o['source'] for o in result['outcomes'] if j in o['final_indices']]
        confidence=max(prepared['confidence'][s] for s in sources)
        if confidence>=.5:continue
        agreement=float(rle_iou([m],[mu.merge([prepared['coarse_masks'][s] for s in sources])])[0,0])
        if unsupported and confidence<.30 and agreement<.50:
            others=[i for i in range(len(prepared['masks'])) if i not in sources]
            support=float(rle_iou([m],[prepared['masks'][i] for i in others]).max(initial=0))
            if support<.50:
                removed.append(dict(prediction=j,sources=sources,confidence=confidence,coarse_agreement=agreement,
                    other_proposal_iou=support,reason='Weak changed mask lacks coarse and competing-proposal support'))
                continue
        if not fragmented:continue
        x,y,w,h=map(int,mu.toBbox(m));mask=mu.decode(m)
        _,_,stats,_=cv2.connectedComponentsWithStats(mask[y:y+h,x:x+w],8)
        main=float(stats[1:,cv2.CC_STAT_AREA].max()/mu.area(m))
        if main>=.65:continue
        if agreement>=.50:continue
        if image is None:image=cv2.imread(str(prepared.get('image_path',DEFAULT_PROJECT/'data/train/images'/prepared['file_name'])),0)
        x0,y0=max(0,x-8),max(0,y-8);x1,y1=min(mask.shape[1],x+w+8),min(mask.shape[0],y+h+8)
        patch=mask[y0:y1,x0:x1];im=image[y0:y1,x0:x1]
        ring=cv2.dilate(patch,np.ones((17,17),'uint8')).astype(bool)&~patch.astype(bool)&(im>0)
        contrast=float(im[ring].mean()-im[patch.astype(bool)].mean()) if ring.any() else None
        evidence=dict(confidence=confidence,main_fraction=main,coarse_agreement=agreement,contrast=contrast)
        if fragmented_disagreement(evidence):removed.append(dict(prediction=j,sources=sources,**evidence))
    rejected={r['prediction'] for r in removed}
    keep=[j for j in range(len(result['masks'])) if j not in rejected];mapping={j:i for i,j in enumerate(keep)}
    result['masks']=[result['masks'][j] for j in keep]
    for o in result['outcomes']:
        before=o['final_indices'];o['final_indices']=[mapping[j] for j in before if j in mapping]
        if before and not o['final_indices']:o['status']='Removed by final evidence filter'
    if 'parent_masks' in result:result['parent_masks']=[result['parent_masks'][j] for j in keep]
    result['fragmented_rejections']=removed
    return result


def predict(prepared):
    from near_candidate_repair import predict as base
    history=base(prepared,protect_strong=True,strong_confidence_limit=.55)
    return reject(prepared,history,unsupported=True)['masks']
