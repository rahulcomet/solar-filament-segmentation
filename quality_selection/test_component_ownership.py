"""Recognize a duplicate of one component before assigning shared pixels."""
from copy import deepcopy
import cv2,numpy as np
from pycocotools import mask as mu
from common import ROOT
from refiner import HERE,read_json,write_json
from build_visual_review import cache
from selection_policies import Policy,features,apply_policy
from merge_recovery import edges,selected_edges,group
from ablate_pass6_cleanup import final_filters
from sweep_confidence import assert_disjoint
from validation_pq import rle_iou
from evaluate_cohort_guarded_replay import score
from test_ten_fp_repairs import contrast_filter,pipeline

OUT=HERE/'outputs/component_ownership_20261105'

def repair(row,geometry_guard=False,connected_extension=False,return_trace=False,direction_limit=25,whole_group_priority=False,connected_direction_limit=None,interior_weight=0,fragment_protection=True,agreement_weight=True,merge_policy="majority_veto_with_continuation",ownership_exclude=(),return_metadata=False,ownership_transform=None,selection_priorities=None):
    f=features(row)
    ranked=dict(row,confidence=selection_priorities) if selection_priorities is not None else row
    _,audit=apply_policy(ranked,Policy('nms',cutoff=.18),cached=f)
    survivors=[i for i in range(len(row['masks'])) if i not in audit['suppressed']]
    stable=([float(rle_iou([m],[c])[0,0]) for m,c in zip(row['masks'],row['coarse_masks'])]
            if agreement_weight else [1.0]*len(row['masks']))
    priorities=row['confidence'] if selection_priorities is None else selection_priorities
    scores=[priorities[i]*(max(stable[i],.01)**.5 if agreement_weight else 1) for i in range(len(stable))]
    proposed=[]
    for big in (survivors if fragment_protection else []):
        x,y,w,h=map(int,mu.toBbox(row['masks'][big]))
        mask=f['decoded'][big][y:y+h,x:x+w].astype('uint8')
        count,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        significant=[k for k in range(1,count) if stats[k,cv2.CC_STAT_AREA]/f['areas'][big]>=.20]
        if len(significant)!=2:continue
        for small in survivors:
            if small==big or f['areas'][small]>=f['areas'][big]:continue
            patch=f['decoded'][small][y:y+h,x:x+w]
            for component in significant:
                cm=labels==component
                inter=int((cm&patch).sum());ca=int(cm.sum());sa=f['areas'][small]
                similarity=inter/(ca+sa-inter)
                if similarity>=.70 and inter/sa>=.90:
                    # Comparable fragments, with a second substantial component.
                    other=[k for k in significant if k!=component][0]
                    ratio=max(ca,stats[other,cv2.CC_STAT_AREA])/min(ca,stats[other,cv2.CC_STAT_AREA])
                    if geometry_guard:
                        from continuation_features import measure_continuation
                        a=(labels==component)
                        b=(labels==other)
                        geometry=measure_continuation(a,b,np.zeros(a.shape,dtype='uint8'))
                        angle=geometry['tangent_angle_degrees']
                        if angle is None or angle>direction_limit or geometry['width_ratio']>2:
                            continue
                    if ratio<=3:
                        proposed.append(dict(big=big,small=small,component_iou=float(similarity)))
    if connected_extension:
        from continuation_features import measure_continuation
        for big in (survivors if fragment_protection else []):
            if f['coherence'][big]<.80:continue
            for small in survivors:
                if small==big or f['areas'][big]<1.5*f['areas'][small] or f['coherence'][small]<.80:continue
                pair=float(f['pair_iou'][big,small])
                if not .10<=pair<.50:continue
                inter=pair*(f['areas'][big]+f['areas'][small])/(1+pair)
                if inter/f['areas'][small]<.90:continue
                union=mu.merge([row['masks'][big],row['masks'][small]])
                x,y,w,h=map(int,mu.toBbox(union))
                anchor=f['decoded'][small][y:y+h,x:x+w]
                added=f['decoded'][big][y:y+h,x:x+w]&~anchor
                geometry=measure_continuation(anchor,added,np.zeros(anchor.shape,dtype='uint8'))
                angle=geometry['tangent_angle_degrees']
                limit=direction_limit if connected_direction_limit is None else connected_direction_limit
                if angle is None or angle>limit or geometry['width_ratio']>1.5:continue
                if any(v is None or v<3 for v in [geometry['anchor_axis_ratio'],geometry['added_axis_ratio']]):continue
                proposed.append(dict(big=big,small=small,component_iou=pair,kind='connected_extension'))
    used=set();chosen=[]
    for action in sorted(proposed,key=lambda r:-r['component_iou']):
        big,small=action['big'],action['small']
        if big in used or small in used:continue
        chosen.append(action);used.update([big,small])
        priority=max(scores[big],scores[small])
        if whole_group_priority:
            priority=max([priority]+[scores[i] for i in survivors if f['pair_iou'][big,i]>.05])
        scores[big]=priority+1e-8
    dropped={c['small'] for c in chosen}
    permitted=[i for i in survivors if i not in dropped and i not in ownership_exclude]
    altered=dict(file_name=row['file_name'],masks=[row['masks'][i] for i in permitted],
        confidence=[scores[i] for i in permitted],source_proposal_ids=permitted)
    if ownership_transform is not None:
        altered=ownership_transform(altered)
    if interior_weight:
        from ownership_rules import assign_interiors
        clean=assign_interiors(altered,interior_weight)
    else:
        clean,_=apply_policy(altered,Policy('component',cutoff=0,mask_nms=False))
    clean['confidence']=[row['confidence'][i] for i in clean['source_proposal_ids']]
    merge_choices=[] if merge_policy=='none' else selected_edges(edges(row,clean),merge_policy)
    merged=group(clean,merge_choices)
    final=contrast_filter(final_filters(merged,row),row)
    assert_disjoint(final['masks'])
    if return_metadata:
        retained=sorted({i for sources in final['source_groups'] for i in sources})
        return final['masks'],chosen,dict(permitted=permitted,retained=retained,
            nms_suppressed=audit['suppressed'],ownership=clean,final=final)
    if return_trace:
        ownership,_=apply_policy(altered,Policy('ownership_trace',cutoff=0,mask_nms=False,min_remaining=0))
        from fp_improvement_options import apply_option
        pruned,_=apply_option(merged,None,'tiny_satellites')
        remnant=final_filters(merged,row)
        stages=dict(refined_input=row['masks'],
            duplicate_suppression=[row['masks'][i] for i in survivors],
            fragment_choice=[row['masks'][i] for i in permitted],
            pixel_ownership=ownership['masks'],minimum_area=clean['masks'],
            merging=merged['masks'],pruning=pruned['masks'],
            remnant_filter=remnant['masks'],contrast=final['masks'])
        return final['masks'],chosen,stages
    return final['masks'],chosen

def main():
    OUT.mkdir(exist_ok=True)
    raw=cache(HERE/'outputs/weak_admission_20261105/raw')
    records=read_json(HERE/'outputs/fp_progress_three_options_20261028/validation_records.json')
    outputs={k:{} for k in ['retained_ownership','component_grouping']};actions=[]
    for number,(name,r) in enumerate(raw.items(),1):
        ids=[i for i,c in enumerate(r['confidence']) if c>=.18]
        row=dict(file_name=name,**{k:[r[k][i] for i in ids] for k in ['masks','coarse_masks','confidence']},
            source_proposal_ids=list(range(len(ids))))
        outputs['retained_ownership'][name]=pipeline(row,'stable_ownership')
        result,chosen=repair(row)
        outputs['component_grouping'][name]=result
        if chosen:actions.append(dict(file_name=name,actions=chosen))
        if number%25==0:print(number,flush=True)
    baseline,old=score(records,outputs['retained_ownership'])
    assert abs(baseline['pq']-.4491122498274961)<1e-10
    metrics,new=score(records,outputs['component_grouping'])
    metrics.update(gained=sum(int(((old[k]<=.5)&(new[k]>.5)).sum()) for k in old),
        lost=sum(int(((old[k]>.5)&(new[k]<=.5)).sum()) for k in old))
    traces=read_json(HERE/'outputs/ten_fp_repair_20261105/trace.json')
    details=[dict(id=c['id'],old_iou=float(old[c['record']][c['target']]),
        new_iou=float(new[c['record']][c['target']])) for c in traces]
    write_json(OUT/'result.json',dict(control=baseline,trial=metrics,actions=actions,ten_cases=details))
    write_json(OUT/'predictions.json',outputs['component_grouping'])
    print(metrics,actions,details,flush=True)
if __name__=='__main__':main()
