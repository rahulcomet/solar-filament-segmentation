"""Shared paths and prediction-only ranking for the isolated quality pilot."""
import sys
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT/'local_refiner'))
from refiner import read_json,write_json,sha,encode_mask,decode_mask
from validation_pq import rle_iou
from merge_recovery import edges,run_policy

OUT=HERE/'outputs/mask_quality_20261103'
BASE=ROOT/'local_refiner/outputs/expanded_scratch_continued_to10_20261101'
WEIGHTS=BASE/'checkpoints/epoch_06.pt'
TRAIN_CACHE=ROOT/'box_guided_refiner/outputs/box_data_20261031/prediction_cache'
VAL_CACHE=ROOT/'local_refiner/outputs/low_confidence_candidates_20261019/predictions'
ALPHAS=[0.,.25,.50]


class QualityHead(nn.Module):
    def __init__(self,features):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(features,64),nn.ReLU(),nn.Dropout(.1),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))

    def forward(self,x):return self.layers(x)


def quality_targets(gt_records,predictions):
    """Expected max IoU and qualifying frequency across separate annotators."""
    best=[]
    for gts in gt_records:
        m=rle_iou(gts,predictions)
        best.append(m.max(axis=0) if len(gts) else np.zeros(len(predictions)))
    best=np.stack(best)
    return np.column_stack([best.mean(axis=0),(best>.5).mean(axis=0)]).astype('float32')


def ordered_cleanup(raw,scores,nms_only=False):
    """Alter only NMS/ownership priority; retain original confidence downstream."""
    scores=np.asarray(scores);conf=np.asarray(raw['confidence']);assert scores.shape==conf.shape and np.isfinite(scores).all()
    order=sorted((i for i,c in enumerate(conf) if c>=.20),key=lambda i:(-scores[i],-conf[i],i))
    pair=rle_iou(raw['masks'],raw['masks']);kept=[]
    for i in order:
        if not kept or not (pair[i,kept]>=.5).any():kept.append(i)
    if nms_only:kept.sort(key=lambda i:(-conf[i],i))
    shape=tuple(raw['masks'][0]['size']) if raw['masks'] else (2048,2048)
    occupied=np.zeros(shape,bool);masks={}
    for i in kept:
        mask=decode_mask(raw['masks'][i]).astype(bool);remain=mask&~occupied;occupied|=mask
        if remain.any() and remain.sum()/mask.sum()>=.10:masks[i]=encode_mask(remain)
    ids=sorted(masks);clean=dict(file_name=raw['file_name'],masks=[masks[i] for i in ids],
        confidence=[raw['confidence'][i] for i in ids],source_proposal_ids=[raw['source_proposal_ids'][i] for i in ids])
    final,_=run_policy(clean,raw,edges(raw,clean),'majority_veto')
    return final


def predict_quality(state,x):
    net=QualityHead(len(state['mean']));net.load_state_dict(state['model']);net.eval()
    norm=np.clip((x-state['mean'])/state['std'],-5,5)
    with torch.inference_mode():p=net(torch.from_numpy(norm.astype('float32'))).sigmoid().numpy()
    return p[:,0]*p[:,1]
