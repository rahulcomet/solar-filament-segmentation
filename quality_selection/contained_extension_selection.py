"""A guarded alternative to confidence-only duplicate selection."""
import numpy as np
from pycocotools import mask as mu
from continuation_features import measure_continuation
from structural_cleanup import largest
from validation_pq import rle_iou


def propose(row,confidence_gap_limit=.10):
    masks=row['masks'];confidence=row['confidence']
    if not masks:return []
    areas=mu.area(masks)
    pair=rle_iou(masks,masks);actions=[]
    kept=[];suppressed=[]
    for i in sorted(range(len(masks)),key=lambda i:(-confidence[i],i)):
        (suppressed if kept and (pair[i,kept]>=.50).any() else kept).append(i)
    for big in suppressed:
        for small in kept:
            if not confidence[small]>confidence[big] or confidence[small]-confidence[big]>confidence_gap_limit:continue
            if not 1.2<=areas[big]/areas[small]<=3 or pair[big,small]<.50:continue
            shared=float(pair[big,small]*(areas[big]+areas[small])/(1+pair[big,small]))
            if shared/areas[small]<.95:continue
            x,y,w,h=map(int,mu.toBbox(mu.merge([masks[big],masks[small]])))
            a=mu.decode(masks[small])[y:y+h,x:x+w].astype(bool)
            b=mu.decode(masks[big])[y:y+h,x:x+w].astype(bool)
            if min(largest(z).sum()/z.sum() for z in [a,b])<.95:continue
            geometry=measure_continuation(a,b&~a,np.zeros(a.shape,'uint8'))
            angle=geometry['tangent_angle_degrees']
            if angle is None or angle>25 or geometry['width_ratio']>1.5:continue
            if any(v is None or v<3 for v in [geometry['anchor_axis_ratio'],geometry['added_axis_ratio']]):continue
            actions.append(dict(big=big,small=small,confidence_gap=confidence[small]-confidence[big],
                containment=shared/areas[small],geometry=geometry))
    return actions


def priorities(row,actions):
    result=list(row['confidence']);used=set();selected=[]
    for action in sorted(actions,key=lambda a:(a['confidence_gap'],-a['containment'],a['big'],a['small'])):
        big,small=action['big'],action['small']
        if big in used or small in used:continue
        result[big]=max(result[big],result[small])+1e-8
        used.update([big,small]);selected.append(action)
    return result,selected


def predict(prepared):
    """Guarded duplicate selection followed by the selected hybrid cleanup."""
    from early_guarded_merger import predict as hybrid_predict
    rank,chosen=priorities(prepared,propose(prepared))
    return hybrid_predict(prepared,selection_priorities=rank if chosen else None)
