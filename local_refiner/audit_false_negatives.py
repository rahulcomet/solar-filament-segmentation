"""Annotation-record FN taxonomy; diagnostics never enter prediction policies."""
from collections import Counter,defaultdict
import itertools
import numpy as np
from pycocotools import mask as mu
from refiner import HERE,WORKSPACE,read_json,write_json,sha,PQStats,pq_stats_from_iou
from build_visual_review import cache
from reaudit_fp_progress import RAW,parents
from review_current_errors import LOW
from sweep_confidence import select_candidates
from validation_pq import rle_iou

OUT=HERE/'outputs/fn_audit_20261028'
LOW_REFINED=HERE/'outputs/confidence_sweep_basic_unet_20261026/refined_candidates'

def best(matrix,t):
    return float(matrix[t].max(initial=0))

def coverage(gt,pred):
    if not pred:return 0.
    u=mu.merge(pred);iou=float(rle_iou([gt],[u])[0,0]);ga=float(mu.area(gt));ua=float(mu.area(u))
    return iou*(ga+ua)/(1+iou)/ga

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'summary.json').exists()
    manifest=HERE/'current_best_pipeline.json';ref=read_json(manifest);basepath=WORKSPACE/ref['validation_predictions']
    base=cache(basepath);raws=cache(RAW);lowref=cache(LOW_REFINED)
    records=defaultdict(list)
    for r in read_json(HERE/'outputs/fp_progress_three_options_20261028/validation_records.json'):records[r['file_name']].append(r)
    hashes={str(p.relative_to(WORKSPACE)):sha(p) for f in [basepath,RAW,LOW,LOW_REFINED] for p in f.glob('*.json')}
    write_json(OUT/'plan.json',dict(reference_sha256=sha(manifest),source_hashes=hashes,runner_sha256=sha(__file__),
        scope='Primary diagnostic categories have precedence and sum to567FN. Overlapping flags reported separately. Low-confidence refined cache uses pixel0.50, unlike current0.55; explicitly a mixed-setting availability diagnostic, not confidence-only attribution.',
        grouping='Try union of up to three existing final predictions with individually >50% pixel precision against this target; annotation oracle for diagnosis only.'))
    cases=[];totals=PQStats()
    for number,p in enumerate(sorted(LOW.glob('*.json'))):
        low=read_json(p);fn=low['file_name'];pred=base[fn];raw=raws[fn];coarse=select_candidates(low,.20);lr=lowref[fn]
        lowids=[j for j,c in enumerate(lr['confidence']) if .05<=c<.20]
        lmasks=[lr['masks'][j] for j in lowids]
        final_areas=mu.area(pred['masks']).astype(float) if pred['masks'] else np.array([])
        remaining_sources={s for j in range(len(pred['masks'])) for s in parents(pred,j)}
        for rec in records[fn]:
            gt=rec['gt'];m=rle_iou(gt,pred['masks']);rm=rle_iou(gt,raw['masks']);cm=rle_iou(gt,coarse['masks']);lrm=rle_iou(gt,lmasks);allcm=rle_iou(gt,low['masks'])
            totals+=pq_stats_from_iou(m)
            for t in np.flatnonzero(~(m>.5).any(axis=1)):
                t=int(t);ga=float(mu.area(gt[t]));bv=best(m,t);j=int(m[t].argmax()) if bv else None
                inter=0 if j is None else bv*(ga+final_areas[j])/(1+bv)
                precision=0 if j is None else inter/final_areas[j];recall=inter/ga
                ri=int(rm[t].argmax()) if best(rm,t) else None
                lowbest=best(lrm,t);li=lowids[int(lrm[t].argmax())] if lowbest else None
                union_recall=coverage(gt[t],raw['masks']);best_union=bv;union_sources=[]
                if len(final_areas):
                    intersections=m[t]*(ga+final_areas)/(1+m[t]);qualified=np.flatnonzero(intersections/final_areas>.5)
                    qualified=sorted(qualified,key=lambda k:-m[t,k])[:6]
                    for size in [2,3]:
                        for group in itertools.combinations(qualified,size):
                            score=float(rle_iou([gt[t]],[mu.merge([pred['masks'][k] for k in group])])[0,0])
                            if score>best_union:best_union=score;union_sources=[pred['source_proposal_ids'][k] for k in group]
                if best(rm,t)>.5:
                    source=raw['source_proposal_ids'][ri]
                    category='Qualifying refined candidate removed' if source not in remaining_sources else 'Qualifying refined candidate damaged or merged'
                elif best(cm,t)>.5:category='Qualifying YOLO mask lost during refinement'
                elif lowbest>.5:category='Qualifying lower-confidence candidate available'
                elif best_union>.5:category='Recoverable by grouping existing final pieces (oracle)'
                elif bv>=.40:category='Near matching threshold (IoU0.40-0.50)'
                elif union_recall<=.50:category='Insufficient foreground in current candidate union'
                elif recall>.50 and precision<=.50:category='Excess foreground or merged instance'
                elif recall<=.50 and bv>0:category='Incomplete coverage or scattered fragments'
                elif bv==0:category='No final-mask overlap despite available foreground'
                else:category='Other boundary or instance mismatch'
                cases.append(dict(id=f'fn{len(cases)+1}',record=rec['record'],file_name=fn,target_index=t,category=category,
                    best_iou=bv,best_final_source=None if j is None else pred['source_proposal_ids'][j],pixel_precision=precision,pixel_recall=recall,
                    raw_best_iou=best(rm,t),raw_best_source=None if ri is None else raw['source_proposal_ids'][ri],coarse_best_iou=best(cm,t),
                    lower_confidence_best_iou=lowbest,lower_confidence_source=None if li is None else lr['source_proposal_ids'][li],lower_confidence_conf=None if li is None else lr['confidence'][li],
                    all_coarse001_best_iou=best(allcm,t),raw_union_recall=union_recall,best_union_iou=best_union,best_union_sources=union_sources,
                    area=int(ga),near_threshold=bv>=.4,raw_qualifies=best(rm,t)>.5,coarse_qualifies=best(cm,t)>.5,low_qualifies=lowbest>.5,grouping_qualifies=best_union>.5))
        if (number+1)%30==0:print(f'FN taxonomy {number+1}/146',flush=True)
    assert totals.tp==955 and totals.fn==567 and abs(totals.pq-ref['validation']['pq'])<1e-12 and len(cases)==567
    assert sha(manifest)==read_json(OUT/'plan.json')['reference_sha256'] and all(sha(WORKSPACE/p)==h for p,h in hashes.items())
    summary=dict(fn=len(cases),primary_categories=dict(Counter(c['category'] for c in cases)),
        overlapping_flags={k:sum(c[k] for c in cases) for k in ['near_threshold','raw_qualifies','coarse_qualifies','low_qualifies','grouping_qualifies']},
        raw_union_insufficient=sum(c['raw_union_recall']<=.5 for c in cases),images=146,records=234,baseline_pq=totals.pq,
        limitation='Categories are deterministic diagnostics, not proven causal ground truth. Lower-confidence refined masks use historical pixel0.50. Grouping oracle can conflict across annotators; sufficient foreground is not proof of a valid instance mask.')
    write_json(OUT/'cases.json',cases);write_json(OUT/'summary.json',summary);print(summary,flush=True)

if __name__=='__main__':main()
