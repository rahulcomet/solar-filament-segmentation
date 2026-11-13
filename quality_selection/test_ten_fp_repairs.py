"""Bounded full-validation tests for qualifying-parent FP losses."""
from copy import deepcopy
from collections import defaultdict
import numpy as np
from pycocotools import mask as mu
from common import ROOT
from refiner import HERE,read_json,write_json,encode_mask
from build_visual_review import cache
from selection_policies import Policy,features,apply_policy
from merge_recovery import edges,selected_edges,group
from ablate_pass6_cleanup import final_filters
from sweep_confidence import assert_disjoint
from validation_pq import rle_iou
from evaluate_cohort_guarded_replay import score
OUT=HERE/'outputs/ten_fp_repair_20261105'

def contrast_filter(row,prepared):
    # Match current contrast12.5 definition by original image pixels.
    import cv2
    from refiner import DEFAULT_PROJECT
    image=None;keep=[]
    for j,m in enumerate(row['masks']):
        weak=all(prepared['confidence'][i]<.20 for i in row['source_groups'][j])
        if weak:
            if image is None:image=cv2.imread(str(prepared.get('image_path',DEFAULT_PROJECT/'data/train/images'/row['file_name'])),0)
            fg=mu.decode(m).astype('uint8')
            ring=cv2.dilate(fg,np.ones((17,17),'uint8')).astype(bool)&~fg.astype(bool)&(image>0)
            if ring.any() and float(image[ring].mean()-image[fg.astype(bool)].mean())<12.5:continue
        keep.append(j)
    return {k:([row[k][i] for i in keep] if isinstance(row[k],list) else row[k]) for k in row}

def pipeline(row,mode):
    f=features(row)
    clean,audit=apply_policy(row,Policy('control',cutoff=.18),cached=f)
    if mode=='stable_ownership':
        # NMS decisions remain those from original confidence. Only pixel order changes.
        suppressed=set(audit['suppressed'])
        stable=[float(rle_iou([m],[c])[0,0]) for m,c in zip(row['masks'],row['coarse_masks'])]
        permitted=[i for i in range(len(row['masks'])) if i not in suppressed]
        # Reuse apply_policy on the permitted candidates only, preserving original proposal IDs.
        subset={k:([row[k][i] for i in permitted] if k in ['masks','coarse_masks','confidence','source_proposal_ids'] else row[k]) for k in row}
        subset['confidence']=[row['confidence'][i]*max(stable[i],.01)**.5 for i in permitted]
        clean,_=apply_policy(subset,Policy('stable',cutoff=0,mask_nms=False))
        clean['confidence']=[row['confidence'][i] for i in clean['source_proposal_ids']]
    merged=group(clean,selected_edges(edges(row,clean),'majority_veto_with_continuation'))
    final=final_filters(merged,row)
    final=contrast_filter(final,row)
    assert_disjoint(final['masks'])
    return final['masks']

def main():
    OUT.mkdir(exist_ok=True)
    raw=cache(HERE/'outputs/weak_admission_20261105/raw')
    records=read_json(HERE/'outputs/fp_progress_three_options_20261028/validation_records.json')
    cases=read_json(OUT/'trace.json')
    outputs={k:{} for k in ['control','stable_ownership','small_shrink_union','small_shrink_coarse']}
    for number,(name,r) in enumerate(raw.items(),1):
        ids=[i for i,c in enumerate(r['confidence']) if c>=.18]
        row=dict(file_name=name,**{k:[r[k][i] for i in ids] for k in ['masks','coarse_masks','confidence']},
                 source_proposal_ids=list(range(len(ids))))
        for mode in outputs:
            candidate=deepcopy(row)
            if mode in ['small_shrink_union','small_shrink_coarse']:
                for i,(m,c) in enumerate(zip(row['masks'],row['coarse_masks'])):
                    ratio=float(mu.area(m))/float(mu.area(c))
                    similarity=float(rle_iou([m],[c])[0,0])
                    if .85<=ratio<1 and similarity>=.8:
                        replacement=mu.merge([m,c]) if mode=='small_shrink_union' else c
                        if isinstance(replacement['counts'],bytes):replacement=dict(replacement,counts=replacement['counts'].decode('ascii'))
                        candidate['masks'][i]=replacement
            outputs[mode][name]=pipeline(candidate,mode)
        if number%25==0:print(number,flush=True)
    base,old=score(records,outputs['control'])
    assert abs(base['pq']-.44892017525556277)<1e-10
    results={}
    for mode,masks in outputs.items():
        metrics,new=score(records,masks)
        metrics.update(gained=sum(int(((old[k]<=.5)&(new[k]>.5)).sum()) for k in old),
            lost=sum(int(((old[k]>.5)&(new[k]<=.5)).sum()) for k in old),
            ten_cases=[dict(id=c['id'],old_iou=float(old[c['record']][c['target']]),
                new_iou=float(new[c['record']][c['target']])) for c in cases])
        results[mode]=metrics
    write_json(OUT/'bounded_trials.json',results)
    print(results,flush=True)
if __name__=='__main__':main()
