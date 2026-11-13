"""Bounded fuller-candidate reconsideration before NMS and ownership."""
import cv2
import numpy as np
from pycocotools import mask as mu
from common import ROOT
from refiner import DEFAULT_PROJECT
from contained_extension_selection import propose, priorities
from continuation_features import measure_continuation
from hybrid_lineage import trace
from remnant_candidates import filter_history


def actions(prepared,image_supported=False,protect_strong=False,strong_confidence_limit=.50):
    existing=propose(prepared)
    occupied={i for p in existing for i in [p['big'],p['small']]}
    extras=[p for p in propose(prepared,confidence_gap_limit=.30)
            if p['confidence_gap']>.10 and not occupied.intersection([p['big'],p['small']])]
    if protect_strong:
        extras=[p for p in extras if prepared['confidence'][p['small']]<strong_confidence_limit]
    if not image_supported or not extras:
        return existing,extras
    image=cv2.imread(str(prepared.get('image_path',DEFAULT_PROJECT/'data/train/images'/prepared['file_name'])),0)
    accepted=[]
    for p in extras:
        small,big=[prepared['masks'][p[k]] for k in ['small','big']]
        x,y,w,h=map(int,mu.toBbox(mu.merge([small,big])))
        # Three Gaussian sigmas of actual surrounding image are needed for
        # a local darkness estimate; a tight mask box can contain no background.
        x0,y0=max(0,x-36),max(0,y-36)
        x1,y1=min(image.shape[1],x+w+36),min(image.shape[0],y+h+36)
        anchor=mu.decode(small)[y0:y1,x0:x1].astype(bool)
        added=mu.decode(big)[y0:y1,x0:x1].astype(bool)&~anchor
        evidence=measure_continuation(anchor,added,image[y0:y1,x0:x1])
        fraction=evidence['added_image']['dark_fraction']
        if fraction is not None and fraction>=.50:
            accepted.append(dict(p,image_evidence=evidence))
    return existing,accepted


def predict(prepared,image_supported=False,protect_strong=False,strong_confidence_limit=.50):
    old,new=actions(prepared,image_supported,protect_strong,strong_confidence_limit)
    rank,chosen=priorities(prepared,old+new)
    history=trace(prepared,rank,curved_extension=True)
    result=filter_history(prepared,history)
    result['fuller_selection']=chosen
    return result


def selected_predict(prepared):
    """Conservative arm: preserve strong blockers and all previous preferences."""
    return predict(prepared,protect_strong=True)['masks']


def moderate_blocker_predict(prepared):
    """Bounded0.55guard variant; all existing shape/cleanup rules retained."""
    return predict(prepared,protect_strong=True,strong_confidence_limit=.55)['masks']
