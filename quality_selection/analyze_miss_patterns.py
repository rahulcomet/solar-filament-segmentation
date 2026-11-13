"""Fresh structural audit of the best saved CNN pipeline; no prediction changes."""
from collections import Counter,defaultdict
from dataclasses import asdict
import itertools
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from pycocotools import mask as mu
from common import ROOT,OUT,read_json,write_json,sha,decode_mask,encode_mask
from ranker import RANK,predict
from group_cnn import DEST as CNN
from group_actions import signature
from decision_selector import finish as fast_finish
from refiner import PQStats,pq_stats_from_iou,DEFAULT_PROJECT,FilamentDataset,CropTransform
from build_visual_review import cache
from sweep_confidence import assert_disjoint,select_candidates
from validation_pq import rle_iou
from merge_recovery import edges,selected_edges,group,finish
from fp_improvement_options import apply_option
from reaudit_fp_progress import parents
from audit_pass6_candidate import fn_category
from audit_false_negatives import coverage
from audit_current_false_positives import fractions,primary
from official_notebook_diagnostics import official_functions
from report_group_actions import overlay

DEST=ROOT/'quality_selection/outputs/miss_patterns_20261104'
LOW=ROOT/'local_refiner/outputs/low_confidence_candidates_20261019/predictions'
OLD=ROOT/'local_refiner/outputs/pass6_error_audit_20261101'
STAGES=['raw','cnn_rejection','mask_nms','overlap_clipping','minimum_remaining','guarded_merge','tiny_satellites','final']


def subset(raw,saved):
    ids=sorted(saved)
    return dict(file_name=raw['file_name'],masks=[saved[i] for i in ids],
                confidence=[raw['confidence'][i] for i in ids],source_proposal_ids=[raw['source_proposal_ids'][i] for i in ids])


def trace(raw,scores,rejected,ownership_priority=()):
    """Mirror MLP NMS plus the actual saved CNN rejection and all later stages."""
    assert all(a['action']['kind']=='reject' for a in rejected)
    removed={i for a in rejected for i in a['action']['ids']}
    order=sorted((i for i in range(len(scores)) if i not in removed),key=lambda i:(-scores[i],-raw['confidence'][i],i))
    pair=rle_iou(raw['masks'],raw['masks']);kept=[];logs={}
    for i in order:
        blockers=[j for j in kept if pair[i,j]>=.5]
        logs[i]=dict(nms_blockers=blockers)
        if not blockers:kept.append(i)
    shape=tuple(raw['masks'][0]['size']) if raw['masks'] else (2048,2048)
    occupied=np.zeros(shape,bool);allocated=[];clipped={};clean={}
    for i in sorted(kept,key=lambda i:(i not in ownership_priority,-raw['confidence'][i],i)):
        mask=decode_mask(raw['masks'][i]).astype(bool);remain=mask&~occupied;occupied|=mask
        owners=[dict(index=j,pixels=int(mu.area(mu.merge([raw['masks'][i],rle],intersect=True)))) for j,rle in allocated]
        owners=[o for o in owners if o['pixels']];owners.sort(key=lambda o:-o['pixels'])
        fraction=float(remain.sum()/mask.sum())
        logs[i].update(owners=owners,remaining_fraction=fraction)
        if remain.any():
            clipped[i]=encode_mask(remain);allocated.append((i,clipped[i]))
            if fraction>=.1:clean[i]=clipped[i]
    c=subset(raw,clean);chosen=selected_edges(edges(raw,c),'majority_veto');merged=group(c,chosen)
    pruned,_=apply_option(merged,None,'tiny_satellites');final=finish(merged,raw)
    assert signature(fast_finish(c,raw))==signature(final)
    stages=dict(raw=raw,cnn_rejection=subset(raw,{i:m for i,m in enumerate(raw['masks']) if i not in removed}),mask_nms=subset(raw,{i:raw['masks'][i] for i in kept}),
                overlap_clipping=subset(raw,clipped),minimum_remaining=c,guarded_merge=merged,tiny_satellites=pruned,final=final)
    return stages,logs,pair


def component_stats(rle):
    x,y,w,h=map(int,mu.toBbox(rle));cut=decode_mask(rle)[y:y+h,x:x+w].astype('uint8')
    _,_,stats,_=cv2.connectedComponentsWithStats(cut,connectivity=8)
    area=stats[1:,cv2.CC_STAT_AREA]
    return dict(components=len(area),largest_fraction=float(area.max()/area.sum()) if len(area) else 0.)


def competitor_pattern(raw,source,blocker,pair,records,target,final):
    area=mu.area(raw['masks']).astype(float);iou=float(pair[source,blocker])
    intersection=iou*(area[source]+area[blocker])/(1+iou)
    inside=intersection/area[source];reverse=intersection/area[blocker]
    sid=raw['source_proposal_ids'][blocker]
    # Prediction-only evidence that a broad mask covers two smaller, mutually
    # distinct proposals. This is a flag, never a claim about physical identity.
    smaller=[]
    for j in range(len(area)):
        if j==blocker or area[j]>=.85*area[blocker]:continue
        x=pair[j,blocker]*(area[j]+area[blocker])/(1+pair[j,blocker])
        if x/area[j]>=.7:smaller.append(j)
    support=[]
    for a,b in itertools.combinations(smaller,2):
        if pair[a,b]>=.1:continue
        u=mu.merge([raw['masks'][a],raw['masks'][b]])
        covered=float(mu.area(mu.merge([u,raw['masks'][blocker]],intersect=True)))/area[blocker]
        if covered>=.6:support.append(dict(sources=[raw['source_proposal_ids'][a],raw['source_proposal_ids'][b]],coverage=covered))
    match_records=[];other_targets=[]
    for rec in records:
        values=rle_iou(rec['gt'],[raw['masks'][blocker]])[:,0]
        for t in np.flatnonzero(values>.5):
            match_records.append(rec['record'])
            if rec['record']==target['record'] and int(t)!=target['target_index']:other_targets.append(int(t))
    survives=any(sid in parents(final,j) for j in range(len(final['masks'])))
    return dict(source=sid,confidence=raw['confidence'][blocker],pair_iou=iou,
        candidate_inside_competitor=float(inside),competitor_inside_candidate=float(reverse),
        competitor_to_candidate_area=float(area[blocker]/area[source]),
        competitor_components=component_stats(raw['masks'][blocker]),two_distinct_smaller_support=support,
        competitor_matches_records=sorted(set(match_records)),competitor_matches_other_targets=other_targets,
        competitor_survives_final=survives)


def main():
    torch.set_num_threads(2);cv2.setNumThreads(1);DEST.mkdir(exist_ok=True)
    raw=cache(OUT/'val');final=cache(CNN/'epoch6_mlp/predictions');low=cache(LOW)
    records=defaultdict(list)
    for rec in read_json(ROOT/'local_refiner/outputs/fp_progress_three_options_20261028/validation_records.json'):records[rec['file_name']].append(rec)
    old=read_json(OLD/'cases.json');ids={(c['record'],c['target_index']):c['id'] for c in old if c['type']=='FN'}
    cohort={(c['record'],c['target_index']) for c in old if c['type']=='FN' and c['category']=='Qualifying refined candidate removed'}
    assert len(cohort)==50
    decisions=defaultdict(list)
    for d in read_json(CNN/'epoch6_mlp/decisions.json'):decisions[d['file_name']].append(d)
    state=torch.load(RANK/'best.pt',map_location='cpu',weights_only=False)
    sourcepaths={read_json(p)['file_name']:p for p in (OUT/'val').glob('*.json')}
    protected=[ROOT/'quality_selection/current_candidate.json',ROOT/'local_refiner/current_best_pipeline.json',CNN/'research_candidate.json']
    protected_hash={str(p.relative_to(ROOT)):sha(p) for p in protected}
    hashes={str(p.relative_to(ROOT)):sha(p) for folder in [OUT/'val',CNN/'epoch6_mlp/predictions',LOW] for p in folder.glob('*.json')}
    write_json(DEST/'plan.json',dict(reference_pq=.44938154635013317,inputs=hashes,protected=protected_hash,
        scope='Fresh saved-mask audit on current highest measured epoch6+CNN. Annotation-dependent flags are diagnosis only. No training, detector/refiner inference or new cleanup policy.',
        prediction_flags='Containment>=90%andarea ratio>=1.5; broad candidate supported by two smaller masks with mutualIoU<0.1,each70%contained,union covers60%ofbroadmask. Descriptive, not tuned selection rules.',
        runner_sha256=sha(__file__)))
    cases=[];fps=[];tables=[];total=PQStats();traces={}
    for number,(fn,recs) in enumerate(sorted(records.items())):
        r=raw[fn];f=final[fn];assert_disjoint(f['masks']);coarse=select_candidates(low[fn],.2)
        with np.load(sourcepaths[fn].with_suffix('.npz')) as data:scores=predict(state,data['X'],r['confidence'],.5) if len(r['masks']) else np.zeros(0)
        stages,log,pair=trace(r,scores,decisions[fn]);assert stages['final']==f,fn
        traces[fn]=dict(first_rejections=decisions[fn],logs={str(r['source_proposal_ids'][j]):v for j,v in log.items()})
        retained={s for j in range(len(f['masks'])) for s in parents(f,j)}
        weak=[i for i,c in enumerate(low[fn]['confidence']) if .01<=c<.2]
        matrices={rec['record']:rle_iou(rec['gt'],f['masks']) for rec in recs}
        boxes=[CropTransform.from_mask(decode_mask(m)).box for m in coarse['masks']]
        raw_union=mu.merge(r['masks']) if r['masks'] else None
        for rec in recs:
            gt=rec['gt'];m=matrices[rec['record']];rm=rle_iou(gt,r['masks']);cm=rle_iou(gt,coarse['masks'])
            wm=rle_iou(gt,[low[fn]['masks'][i] for i in weak]);prec,recall=fractions(m,mu.area(gt).astype(float),mu.area(f['masks']).astype(float) if f['masks'] else np.zeros(0))
            s=pq_stats_from_iou(m);total+=s;tables.append(dict(iou_matrix=torch.from_numpy(m),n_gt=len(gt),n_pred=len(f['masks'])))
            for t0 in np.flatnonzero(~(m>.5).any(axis=1)):
                t=int(t0);key=(rec['record'],t);rb=float(rm[t].max(initial=0));bv=float(m[t].max(initial=0))
                ri=int(rm[t].argmax()) if rb else None;j=int(m[t].argmax()) if bv else None
                sid=None if ri is None else r['source_proposal_ids'][ri]
                pp=float(prec[t,j]) if j is not None else 0.;rr=float(recall[t,j]) if j is not None else 0.
                ui=bv;unions=[];poss=sorted(np.flatnonzero(prec[t]>.5),key=lambda k:-m[t,k])[:6]
                for size in [2,3]:
                    for selected in itertools.combinations(poss,size):
                        v=float(rle_iou([gt[t]],[mu.merge([f['masks'][k] for k in selected])])[0,0])
                        if v>ui:ui=v;unions=[str(f['source_proposal_ids'][k]) for k in selected]
                raw_cov=coverage(gt[t],r['masks'])
                target_pixels=decode_mask(gt[t]);target_area=float(mu.area(gt[t]))
                visible=max((int(target_pixels[y0:y1,x0:x1].sum())/target_area for x0,y0,x1,y1 in boxes),default=0.)
                weakbest=float(wm[t].max(initial=0));cb=float(cm[t].max(initial=0))
                category=fn_category(rb,sid in retained,cb,weakbest,ui,bv,raw_cov,pp,rr)
                c=dict(id=ids.get(key,'NEW_'+rec['record']+'_G'+str(t)),record=rec['record'],file_name=fn,target_index=t,
                    category=category,original50=key in cohort,final_iou=bv,raw_iou=rb,raw_source=sid,
                    raw_qualifier_count=int((rm[t]>.5).sum()),coarse_iou=cb,weak_coarse_iou=weakbest,
                    pixel_precision=pp,pixel_recall=rr,raw_union_recall=raw_cov,grouping_union_iou=ui,grouping_sources=unions,
                    crop_visibility=visible,near_threshold=bv>=.4,final_source=None if j is None else str(f['source_proposal_ids'][j]))
                if rb>.5:
                    timeline=[]
                    for stage in STAGES:
                        p=stages[stage];children=[k for k in range(len(p['masks'])) if sid in parents(p,k)]
                        values=rle_iou([gt[t]],[p['masks'][k] for k in children])[0]
                        timeline.append(dict(stage=stage,present=bool(children),iou=float(values.max(initial=0))))
                    first=next(x['stage'] for x in timeline if x['iou']<=.5)
                    entry=log.get(ri,{})
                    blockerids=entry.get('nms_blockers',[]) if first=='mask_nms' else [o['index'] for o in entry.get('owners',[])]
                    if first=='guarded_merge':
                        blockerids=[r['source_proposal_ids'].index(other) for k in range(len(stages['guarded_merge']['masks']))
                                    if sid in parents(stages['guarded_merge'],k) for other in parents(stages['guarded_merge'],k) if other!=sid]
                    competitors=[competitor_pattern(r,ri,b,pair,recs,c,f) for b in blockerids]
                    c.update(first_loss=first,timeline=timeline,remaining_fraction=entry.get('remaining_fraction'),competitors=competitors,
                        contained_by_larger=any(p['candidate_inside_competitor']>=.9 and p['competitor_to_candidate_area']>=1.5 for p in competitors),
                        broader_than_blocker=any(p['competitor_inside_candidate']>=.9 and p['competitor_to_candidate_area']<=2/3 for p in competitors),
                        composite_blocker_support=any(p['two_distinct_smaller_support'] for p in competitors),
                        blocker_all_gone=bool(competitors) and all(not p['competitor_survives_final'] for p in competitors),
                        blocker_matches_other_same_record=any(p['competitor_matches_other_targets'] for p in competitors),
                        blocker_supported_another_annotator=any(any(rid!=rec['record'] for rid in p['competitor_matches_records']) for p in competitors))
                cases.append(c)
            for j0 in np.flatnonzero(~(m>.5).any(axis=0)):
                j=int(j0);bv=float(m[:,j].max(initial=0));t=int(m[:,j].argmax()) if bv else None
                lineage=parents(f,j);rawparents=mu.merge([r['masks'][r['source_proposal_ids'].index(s)] for s in lineage])
                c=dict(record=rec['record'],file_name=fn,source_id=str(f['source_proposal_ids'][j]),best_iou=bv,
                       other_annotator_best_iou=max((float(matrix[:,j].max(initial=0)) for rid,matrix in matrices.items() if rid!=rec['record']),default=0.),
                       raw_best_iou=float(rle_iou(gt,[rawparents]).max(initial=0)),coarse_best_iou=0.,pixel_precision=float(prec[t,j]) if t is not None else 0.,
                       pixel_recall=float(recall[t,j]) if t is not None else 0.,substantial_reference_count=int(((prec[:,j]>=.1)&(recall[:,j]>=.1)).sum()),
                       overlaps_matched_reference=bool(((m[:,j]>0)&(m>.5).any(axis=1)).any()))
                c['category']=primary(c);fps.append(c)
        if (number+1)%25==0:print(f'Fresh best-pipeline pattern audit {number+1}/146',flush=True)
    assert (total.tp,total.fp,total.fn)==(963,454,559)
    assert abs(float(official_functions()['get_pq_score'](pd.DataFrame(tables)))-total.pq)<1e-10
    qualifying=[c for c in cases if c['raw_iou']>.5];remaining50=[c for c in cases if c['original50']]
    flag_names=['contained_by_larger','broader_than_blocker','composite_blocker_support','blocker_all_gone','blocker_matches_other_same_record','blocker_supported_another_annotator']
    slices={}
    for name,selected in [('remaining_original50',remaining50),('all_existing_qualifier_losses',qualifying)]:
        slices[name]=dict(entries=len(selected),physical_images=len({c['file_name'] for c in selected}),
            unique_raw_candidates=len({(c['file_name'],c['raw_source']) for c in selected}),
            first_loss=dict(Counter(c['first_loss'] for c in selected)),flags={k:sum(c[k] for c in selected) for k in flag_names},
            flag_ids={k:[c['id'] for c in selected if c[k]] for k in flag_names})
    summary=dict(**asdict(total),pq=total.pq,categories=dict(Counter(c['category'] for c in cases)),fp_categories=dict(Counter(c['category'] for c in fps)),
        qualifier_slices=slices,near_threshold_all=sum(c['near_threshold'] for c in cases),
        near_threshold_has_qualifier=sum(c['near_threshold'] and c['raw_iou']>.5 for c in cases),
        visible_undercoverage=sum(c['raw_iou']<=.5 and c['crop_visibility']>=.9 and c['pixel_precision']>.8 and c['pixel_recall']<.5 for c in cases),
        no_single_crop90=sum(c['crop_visibility']<.9 for c in cases),
        no_training_or_new_inference=True,exact_pipeline_replay=True,organizer_parity=True,disjoint=True)
    for path,digest in hashes.items():assert sha(ROOT/path)==digest
    for path,digest in protected_hash.items():assert sha(ROOT/path)==digest
    write_json(DEST/'cases.json',cases);write_json(DEST/'fp_cases.json',fps);write_json(DEST/'summary.json',summary);write_json(DEST/'traces.json',traces)
    write_json(DEST/'completed.json',dict(complete=True));print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
