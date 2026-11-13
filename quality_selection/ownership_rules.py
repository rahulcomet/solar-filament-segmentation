"""Disjoint assignment with a bounded preference for mask-interior pixels."""
import cv2,numpy as np
from refiner import decode_mask,encode_mask
from pycocotools import mask as mu

def assign_interiors(row,weight):
    owner=np.full((2048,2048),-1,dtype=np.int16)
    best=np.zeros((2048,2048),dtype=np.float32)
    order=sorted(range(len(row['masks'])),key=lambda i:(-row['confidence'][i],i))
    for i in order:
        m=decode_mask(row['masks'][i]).astype('uint8')
        x,y,w,h=map(int,mu.toBbox(row['masks'][i]))
        cut=m[y:y+h,x:x+w]
        padded=np.pad(cut,1)
        distance=cv2.distanceTransform(padded,cv2.DIST_L2,5)[1:-1,1:-1]
        scale=max(float(np.quantile(distance[cut.astype(bool)],.95)),1)
        priority=row['confidence'][i]*(1+weight*np.clip(distance/scale,0,1))
        target=best[y:y+h,x:x+w]
        take=cut.astype(bool)&(priority>target)
        target[take]=priority[take]
        owner[y:y+h,x:x+w][take]=i
    keep=[];masks=[]
    for i,m in enumerate(row['masks']):
        assigned=owner==i
        if assigned.sum()/float(mu.area(m))<.10:continue
        keep.append(i);masks.append(encode_mask(assigned.astype('uint8')))
    return dict(file_name=row['file_name'],masks=masks,
        confidence=[row['confidence'][i] for i in keep],
        source_proposal_ids=[row['source_proposal_ids'][i] for i in keep])
