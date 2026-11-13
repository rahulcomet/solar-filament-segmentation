"""Annotation-free, union-preserving pairing and component separation."""
import cv2
import numpy as np
from pycocotools import mask as mu
from refiner import decode_mask,encode_mask
from validation_pq import rle_iou

SETTINGS=dict(min_raw_iou=.20,max_raw_iou=.50,min_smaller_coverage=.50,
    min_shared_in_main_components=.90,min_component_area=64,min_component_fraction=.10,
    merge_groups='disjoint greedy pairs; no transitive chaining',small_components='retain with largest component')


def components(mask):
    yy,xx=np.nonzero(mask);x0,x1=int(xx.min()),int(xx.max())+1;y0,y1=int(yy.min()),int(yy.max())+1
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask[y0:y1,x0:x1].astype(np.uint8),connectivity=8)
    return (x0,y0,x1,y1),labels,stats[1:,cv2.CC_STAT_AREA]


def largest(mask):
    box,labels,areas=components(mask);x0,y0,x1,y1=box;out=np.zeros_like(mask)
    out[y0:y1,x0:x1]=labels==(int(areas.argmax())+1)
    return out


def structural_policy(raw,final,merge=False,split=False,confidence_guard_gap=None):
    source_to_raw={s:i for i,s in enumerate(raw['source_proposal_ids'])}
    ids=final['source_proposal_ids'];n=len(ids);shapes=[decode_mask(r).astype(bool) for r in final['masks']]
    if not n:return dict(final,source_groups=[]),dict(merges=[],splits=[])
    raw_rles=[raw['masks'][source_to_raw[s]] for s in ids]
    raw_areas=mu.area(raw_rles).astype(float);pair=rle_iou(raw_rles,raw_rles)
    intersections=pair*(raw_areas[:,None]+raw_areas)/(1+pair)
    main={};decoded={};edges=[];blocked=[]
    if merge:
        for a in range(n):
            for b in range(a+1,n):
                overlap=pair[a,b]
                if not .20<=overlap<.50:continue
                if intersections[a,b]/min(raw_areas[a],raw_areas[b])<.50:continue
                for i in [a,b]:
                    if i not in decoded:decoded[i]=decode_mask(raw_rles[i]).astype(bool);main[i]=largest(decoded[i])
                common=decoded[a]&decoded[b]
                main_fraction=float((common&main[a]&main[b]).sum()/common.sum())
                if main_fraction<.90:continue
                if confidence_guard_gap is not None:
                    small,big=(a,b) if raw_areas[a]<raw_areas[b] else (b,a)
                    containment=float(intersections[a,b]/raw_areas[small])
                    gap=float(final['confidence'][small]-final['confidence'][big])
                    if raw_areas[big]>raw_areas[small] and containment>=.90 and gap>=confidence_guard_gap:
                        blocked.append(dict(smaller_source=ids[small],larger_source=ids[big],containment=containment,
                                            confidence_gap=gap,raw_iou=float(overlap)))
                        continue
                edges.append((float(overlap),a,b,main_fraction))
    used=set();groups=[];audit=dict(merges=[],splits=[])
    if confidence_guard_gap is not None:audit['blocked_eligible_pairs']=blocked
    for iou,a,b,fraction in sorted(edges,key=lambda e:(-e[0],e[1],e[2])):
        if a in used or b in used:continue
        used.update([a,b]);groups.append([a,b]);audit['merges'].append(dict(sources=[ids[a],ids[b]],raw_iou=iou,shared_in_main=fraction))
    groups.extend([i] for i in range(n) if i not in used);groups.sort(key=lambda g:min(g))
    result_masks=[];confidence=[];out_ids=[];lineage=[]
    for group in groups:
        mask=np.logical_or.reduce([shapes[i] for i in group]);sources=[ids[i] for i in group];label='+'.join(map(str,sources))
        pieces=[mask]
        if split:
            box,labels,areas=components(mask)
            qualifying=[j+1 for j,area in enumerate(areas) if area>=64 and area/int(mask.sum())>=.10]
            if len(qualifying)>=2:
                biggest=int(areas.argmax())+1;x0,y0,x1,y1=box;pieces=[]
                for component in qualifying:
                    crop=labels==component
                    if component==biggest:crop|=(labels>0)&~np.isin(labels,qualifying)
                    piece=np.zeros_like(mask);piece[y0:y1,x0:x1]=crop;pieces.append(piece)
                audit['splits'].append(dict(sources=sources,areas=[int(p.sum()) for p in pieces]))
        for j,piece in enumerate(pieces):
            result_masks.append(encode_mask(piece));confidence.append(max(final['confidence'][i] for i in group))
            out_ids.append(label if len(pieces)==1 else f'{label}.{j+1}');lineage.append(sources)
    assert len(out_ids)==len(set(out_ids))
    # No foreground is removed or introduced by pairing/separation.
    before=mu.merge(final['masks']);after=mu.merge(result_masks)
    assert before==after
    return dict(file_name=final['file_name'],masks=result_masks,confidence=confidence,source_proposal_ids=out_ids,source_groups=lineage),audit
