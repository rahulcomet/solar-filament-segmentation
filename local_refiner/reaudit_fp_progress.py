"""Revisit the original FP taxonomy on the frozen combined candidate."""
from collections import Counter
import json
import numpy as np
import torch
from pycocotools import mask as mu
from refiner import HERE,read_json,write_json,sha,WORKSPACE
from build_visual_review import cache
from review_current_errors import LOW
from validation_pq import ValidationPQ,rle_iou
from audit_current_false_positives import primary,fractions

OUT=HERE/'outputs/fp_progress_three_options_20261028'
CURRENT=HERE/'outputs/combined_pixel055_guard010_20261028/predictions'
RAW=HERE/'outputs/pixel_thresholds_basic_20261028/pixel_0.55/raw_predictions'
OLD=HERE/'outputs/visual_review_20261019/fp_audit_20261028'


def parents(row,j):
    return row['source_groups'][j] if 'source_groups' in row else [int(s) for s in str(row['source_proposal_ids'][j]).split('+')]


def parent_union(row,ids):
    index={s:i for i,s in enumerate(row.get('source_proposal_ids',range(len(row['masks']))))}
    return mu.merge([row['masks'][index[s]] for s in ids])


def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    v=ValidationPQ();current=cache(CURRENT);raw=cache(RAW);coarse=cache(LOW)
    old=read_json(OLD/'cases.json');old_by_record={}
    for c in old:old_by_record.setdefault(c['record'],[]).append(c)
    cases=[];transitions=[];records=[]
    for number,(fn,recs) in enumerate(v.records.items()):
        row=current[fn];groups=[parents(row,j) for j in range(len(row['masks']))]
        rawunion=[parent_union(raw[fn],g) for g in groups];coarseunion=[parent_union(coarse[fn],g) for g in groups]
        area=mu.area(row['masks']).astype(float);matrices={r['image_id']:rle_iou(r['gt'],row['masks']) for r in recs}
        for rec in recs:
            rid=rec['image_id'];matrix=matrices[rid];hits=(matrix>.5).any(axis=1);predhits=(matrix>.5).any(axis=0)
            records.append(dict(record=rid,gt=rec['gt'],file_name=fn))
            for oldcase in old_by_record.get(rid,[]):
                source=oldcase['source_id'];dest=next((j for j,g in enumerate(groups) if source in g),None)
                fate='Removed' if dest is None else 'Now matched' if predhits[dest] else 'Still FP'
                transitions.append(dict(old_id=oldcase['id'],old_category=oldcase['category'],record=rid,file_name=fn,
                    source_id=source,current_id=None if dest is None else row['source_proposal_ids'][dest],fate=fate,
                    best_iou=0. if dest is None else float(matrix[:,dest].max(initial=0))))
            precision,recall=fractions(matrix,mu.area(rec['gt']).astype(float),area)
            rm=rle_iou(rec['gt'],rawunion);cm=rle_iou(rec['gt'],coarseunion)
            for j in np.flatnonzero(~predhits):
                j=int(j);scores=matrix[:,j];best=float(scores.max(initial=0));t=int(scores.argmax()) if best else None
                other=max((float(m[:,j].max(initial=0)) for k,m in matrices.items() if k!=rid),default=0.)
                p=float(precision[t,j]) if t is not None else 0.;r=float(recall[t,j]) if t is not None else 0.
                c=dict(record=rid,file_name=fn,source_id=row['source_proposal_ids'][j],parents=groups[j],final_index=j,
                    confidence=row['confidence'][j],best_iou=best,best_target_index=t,pixel_precision=p,pixel_recall=r,
                    raw_best_iou=float(rm[:,j].max(initial=0)),coarse_best_iou=float(cm[:,j].max(initial=0)),
                    other_annotator_best_iou=other,substantial_reference_count=int(((precision[:,j]>=.1)&(recall[:,j]>=.1)).sum()),
                    overlaps_matched_reference=bool(((scores>0)&hits).any()),foreground_pixels=int(area[j]))
                c['category']=primary(c);cases.append(c)
        if (number+1)%30==0:print(f'FP re-audit {number+1}/146',flush=True)
    assert len(cases)==481 and len(transitions)==520
    categories=Counter(c['category'] for c in cases)
    bycategory={cat:dict(Counter(c['fate'] for c in transitions if c['old_category']==cat)) for cat in sorted({c['category'] for c in old})}
    summary=dict(original=read_json(OLD/'summary.json')['primary_categories'],current=dict(categories),original_fp=520,current_fp=481,
        original_fp_fates=dict(Counter(c['fate'] for c in transitions)),fates_by_original_category=bycategory,
        near_threshold_flags=sum(c['best_iou']>=.4 for c in cases),zero_overlap_flags=sum(c['best_iou']==0 for c in cases),
        low_confidence_fp=sum(c['confidence']<.25 for c in cases),other_annotator_supported=sum(c['other_annotator_best_iou']>.5 for c in cases),
        limitation='Primary category precedence unchanged. For merged predictions, raw/coarse comparisons use union of parent candidates. Old FP fates are lineage entries, not unique current predictions; several may map to one mask. Removed does not imply background or a recovered annotation.')
    write_json(OUT/'reaudit_summary.json',summary);write_json(OUT/'current_fp_cases.json',cases);write_json(OUT/'old_fp_transitions.json',transitions)
    write_json(OUT/'validation_records.json',records)
    hashes={str(p.relative_to(WORKSPACE)):sha(p) for f in [CURRENT,RAW,LOW] for p in f.glob('*.json')}
    write_json(OUT/'audit_plan.json',dict(source_hashes=hashes,script_sha256=sha(__file__),reference_sha256=sha(HERE/'working_reference.json')))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
