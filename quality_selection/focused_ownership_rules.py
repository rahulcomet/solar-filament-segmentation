"""Prediction-only contained-tip and secondary-component ownership corrections."""
import cv2,numpy as np
from pycocotools import mask as mu
from refiner import encode_mask
from continuation_features import measure_continuation
from validation_pq import rle_iou

def axis(mask):
    yy,xx=np.nonzero(mask)
    if len(xx)<4:return None,0
    values,vectors=np.linalg.eigh(np.cov(np.stack([xx,yy])))
    return vectors[:,-1],float(values[-1]/max(values[0],1))

def find_actions(row,features,survivors,include_tip=True):
    tip=[];component=[]
    areas=features['areas'];decoded=features['decoded']
    coherent=[i for i in survivors if features['coherence'][i]>=.95]
    for big in coherent if include_tip else []:
        for small in coherent:
            if big==small or areas[big]<4*areas[small]:continue
            iou=float(features['pair_iou'][big,small])
            if not .10<=iou<.20:continue
            inter=iou*(areas[big]+areas[small])/(1+iou)
            if inter/areas[small]<.98:continue
            x,y,w,h=map(int,mu.toBbox(mu.merge([row['masks'][big],row['masks'][small]])))
            anchor=decoded[small][y:y+h,x:x+w];added=decoded[big][y:y+h,x:x+w]&~anchor
            g=measure_continuation(anchor,added,np.zeros(anchor.shape,'uint8'))
            if g['tangent_angle_degrees'] is None or g['tangent_angle_degrees']>25 or g['width_ratio']>1.5:continue
            if any(v is None or v<3 for v in [g['anchor_axis_ratio'],g['added_axis_ratio']]):continue
            tip.append(dict(big=big,small=small,containment=float(inter/areas[small])))
    for owner in survivors:
        if not .8<=features['coherence'][owner]<.99:continue
        x,y,w,h=map(int,mu.toBbox(row['masks'][owner]))
        part=decoded[owner][y:y+h,x:x+w].astype('uint8')
        count,labels,stats,_=cv2.connectedComponentsWithStats(part,connectivity=8)
        main=int(1+np.argmax(stats[1:,cv2.CC_STAT_AREA]))
        for k in range(1,count):
            fraction=stats[k,cv2.CC_STAT_AREA]/areas[owner]
            if k==main or not .01<=fraction<=.20:continue
            cm=labels==k;direction,ratio=axis(cm)
            if ratio<3:continue
            possible=[]
            for target in coherent:
                if target==owner or areas[target]<2*cm.sum():continue
                cut=decoded[target][y:y+h,x:x+w]
                containment=float((cut&cm).sum()/cm.sum())
                if containment<.95:continue
                if float((cut&(labels==main)).sum()/stats[main,cv2.CC_STAT_AREA])>=.05:continue
                xx,yy,ww,hh=map(int,mu.toBbox(row['masks'][target]))
                target_axis,target_ratio=axis(decoded[target][yy:yy+hh,xx:xx+ww])
                if target_ratio<3:continue
                angle=float(np.degrees(np.arccos(np.clip(abs(np.dot(direction,target_axis)),0,1))))
                if angle>20:continue
                possible.append((row['confidence'][target],target,containment))
            if possible:
                _,target,contained=max(possible)
                exclude=np.zeros(decoded[owner].shape,dtype='uint8')
                exclude[y:y+h,x:x+w]=(cm&decoded[target][y:y+h,x:x+w]).astype('uint8')
                complete=np.zeros_like(exclude)
                complete[y:y+h,x:x+w]=cm.astype('uint8')
                component.append(dict(owner=owner,target=target,exclude=encode_mask(exclude),
                    component_mask=encode_mask(complete),containment=contained))
    return dict(tip=tip,component=component)

def transform(ownership,row,actions,mode):
    result={k:list(v) if isinstance(v,list) else v for k,v in ownership.items()}
    ids=result['source_proposal_ids']
    if mode in ['tip','both']:
        dropped=set()
        for action in sorted(actions['tip'],key=lambda a:-a['containment']):
            big,small=action['big'],action['small']
            if big not in ids or small not in ids or big in dropped or small in dropped:continue
            index=ids.index(big)
            overlaps=rle_iou([result['masks'][index]],result['masks'])[0]
            result['confidence'][index]=max(c for c,o in zip(result['confidence'],overlaps) if o>.05)+1e-8
            dropped.add(small)
        keep=[j for j,i in enumerate(ids) if i not in dropped]
        for k in ['masks','confidence','source_proposal_ids']:result[k]=[result[k][j] for j in keep]
        ids=result['source_proposal_ids']
    if mode in ['component','both']:
        for action in actions['component']:
            if action['owner'] not in ids or action['target'] not in ids:continue
            j=ids.index(action['owner']);mask=mu.decode(result['masks'][j]).astype(bool)
            mask&=~mu.decode(action['exclude']).astype(bool)
            assert mask.any()
            result['masks'][j]=encode_mask(mask.astype('uint8'))
    return result
