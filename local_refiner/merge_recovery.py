"""Bounded prediction-only merge experiments; no annotations enter decisions."""
import numpy as np
from pycocotools import mask as mu
from refiner import decode_mask, encode_mask
from structural_cleanup import largest
from continuation_features import measure_continuation
from validation_pq import rle_iou
from fp_improvement_options import apply_option
from reaudit_fp_progress import parent_union

RULES = {
    'shape_veto': 'Keep guard; veto pairs with final main-component width ratio>2 or local direction angle>45 degrees.',
    'continuation_override': 'Keep existing merges; override guard only for angle<=25 degrees, width ratio<=1.5, local axis ratios>=3 and dominant components>=80% each.',
    'majority_veto_with_continuation': 'Keep the majority check and allow the guarded continuation exception.',
    'combined': 'Apply both the shape veto and the guarded continuation override.',
}


def edges(raw, clean, max_iou=.50, min_iou=.20):
    ids = clean['source_proposal_ids']
    lookup = {s:i for i,s in enumerate(raw['source_proposal_ids'])}
    rles = [raw['masks'][lookup[s]] for s in ids]
    areas = mu.area(rles).astype(float) if rles else np.array([])
    pair = rle_iou(rles, rles)
    decoded = {}; main = {}; final = {}; found = []
    for a in range(len(ids)):
        for b in range(a+1, len(ids)):
            iou = float(pair[a,b])
            if not min_iou <= iou < max_iou: continue
            inter = iou*(areas[a]+areas[b])/(1+iou)
            if inter/min(areas[a], areas[b]) < .50: continue
            for j in [a,b]:
                if j not in decoded:
                    decoded[j] = decode_mask(rles[j]).astype(bool)
                    main[j] = largest(decoded[j])
                    final[j] = decode_mask(clean['masks'][j]).astype(bool)
            common = decoded[a]&decoded[b]
            if (common&main[a]&main[b]).sum()/common.sum() < .90: continue
            small,big = (a,b) if areas[a]<areas[b] else (b,a)
            gap = clean['confidence'][small]-clean['confidence'][big]
            guarded = bool(areas[big]>areas[small] and inter/areas[small]>=.90 and gap>=.10)
            u = final[a]|final[b]; yy,xx=np.nonzero(u)
            x0,x1=int(xx.min()),int(xx.max())+1; y0,y1=int(yy.min()),int(yy.max())+1
            fa=final[a][y0:y1,x0:x1]; fb=final[b][y0:y1,x0:x1]
            f=measure_continuation(fa,fb,np.zeros(fa.shape,np.uint8))
            coherence=[float(largest(z).sum()/z.sum()) for z in [fa,fb]]
            found.append(dict(a=a,b=b,sources=[ids[a],ids[b]],raw_iou=iou,guarded=guarded,
                confidence_gap=float(gap),containment=float(inter/areas[small]),
                angle=f['tangent_angle_degrees'],width_ratio=f['width_ratio'],
                axis_ratios=[f['anchor_axis_ratio'],f['added_axis_ratio']],coherence=coherence))
    return sorted(found,key=lambda e:(-e['raw_iou'],e['a'],e['b']))


def accept(edge, policy):
    if policy=='none': return False
    if policy=='reference': return not edge['guarded']
    angle=edge['angle']
    if policy in ['shape_veto','combined','majority_veto','majority_veto_with_continuation']:
        if angle is None or angle>45 or edge['width_ratio']>2: return False
    if policy in ['majority_veto','majority_veto_with_continuation'] and min(edge['coherence'])<.5: return False
    if not edge['guarded']: return True
    return bool(policy in ['continuation_override','combined','majority_veto_with_continuation'] and angle is not None and
        angle<=25 and edge['width_ratio']<=1.5 and all(v is not None and v>=3 for v in edge['axis_ratios']) and
        min(edge['coherence'])>=.8)


def group(clean, chosen):
    used=set(); groups=[]
    for e in chosen:
        a,b=e['a'],e['b']
        if a in used or b in used: continue
        used.update([a,b]);groups.append([a,b])
    groups.extend([j] for j in range(len(clean['masks'])) if j not in used);groups.sort(key=min)
    ids=clean['source_proposal_ids']
    return dict(file_name=clean['file_name'],masks=[mu.merge([clean['masks'][j] for j in g]) for g in groups],
        confidence=[max(clean['confidence'][j] for j in g) for g in groups],
        source_proposal_ids=['+'.join(str(ids[j]) for j in g) for g in groups],source_groups=[[ids[j] for j in g] for g in groups])


def finish(row,raw):
    pruned,_=apply_option(row,None,'tiny_satellites')
    keep=[j for j,m in enumerate(pruned['masks']) if not(pruned['confidence'][j]<.5 and
        float(mu.area(m))/float(mu.area(parent_union(raw,pruned['source_groups'][j])))<.5)]
    return dict(file_name=row['file_name'],**{k:[pruned[k][j] for j in keep] for k in ['masks','confidence','source_proposal_ids','source_groups']})


def selected_edges(candidates,policy):
    chosen=[];used=set()
    for e in candidates:
        if accept(e,policy) and e['a'] not in used and e['b'] not in used:
            chosen.append(e);used.update([e['a'],e['b']])
    return chosen


def run_policy(clean,raw,candidates,policy):
    chosen=selected_edges(candidates,policy)
    return finish(group(clean,chosen),raw),chosen
