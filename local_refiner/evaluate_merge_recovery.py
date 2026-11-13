"""Full-validation experiments and isolated-pair diagnostic, frozen current reference."""
from dataclasses import asdict
from collections import defaultdict
import json
import numpy as np
import pandas as pd
import torch
from refiner import HERE,WORKSPACE,read_json,write_json,sha,PQStats,pq_stats_from_iou
from build_visual_review import cache
from reaudit_fp_progress import RAW
from selection_policies import apply_policy,Policy
from merge_recovery import RULES,edges,selected_edges,finish,group
from sweep_confidence import assert_disjoint
from validation_pq import rle_iou
from official_notebook_diagnostics import official_functions

OUT=HERE/'outputs/merge_recovery_20261028'

def compare(records,before,after):
    changes=[]
    for rec in records:
        a=rle_iou(rec['gt'],before['masks']);b=rle_iou(rec['gt'],after['masks'])
        ah=(a>.5).any(axis=1);bh=(b>.5).any(axis=1)
        for t in np.flatnonzero(ah!=bh):
            t=int(t);av=float(a[t].max(initial=0));bv=float(b[t].max(initial=0))
            changes.append(dict(record=rec['record'],file_name=rec['file_name'],target_index=t,
                group='Gained' if bh[t] else 'Lost',before_iou=av,after_iou=bv,
                before_source_id=before['source_proposal_ids'][int(a[t].argmax())] if av else None,
                after_source_id=after['source_proposal_ids'][int(b[t].argmax())] if bv else None))
    return changes

def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    manifest=HERE/'current_best_pipeline.json';current=read_json(manifest)
    basepath=WORKSPACE/current['validation_predictions'];base=cache(basepath)
    records=defaultdict(list)
    for rec in read_json(HERE/'outputs/fp_progress_three_options_20261028/validation_records.json'):records[rec['file_name']].append(rec)
    hashes={str(p.relative_to(WORKSPACE)):sha(p) for f in [RAW,basepath] for p in f.glob('*.json')}
    plan=dict(reference_sha256=sha(manifest),input_hashes=hashes,rules=RULES,
        code_sha256=sha(HERE/'merge_recovery.py'),runner_sha256=sha(__file__),
        controls=['reference','none'],scope='Three fixed geometry policies; no fitted scorer or threshold sweep. Per-edge GT effects are diagnostic only, never used in acceptance. No automatic promotion.',
        acceptance='No lost reference matches, improved PQ, plus report outstanding matches lost relative to no-merging control. Validation only; no guarantee on unseen images.')
    assert not (OUT/'summary.json').exists(),'Use a new output folder for a new experiment.'
    write_json(OUT/'plan.json',plan)
    names=['reference','none']+list(RULES);scores={n:PQStats() for n in names};tables={n:[] for n in names}
    changes=[];history=[];operations=[];pairaudit=[];allrecords=[]
    for n in names:(OUT/n/'predictions').mkdir(parents=True,exist_ok=True)
    for number,path in enumerate(sorted(RAW.glob('*.json'))):
        raw=read_json(path);fn=raw['file_name'];clean,_=apply_policy(raw,Policy('global_cleanup_020'))
        es=edges(raw,clean);variants={};chosen={};unique={}
        for n in names:
            chosen[n]=selected_edges(es,n);key=tuple((e['a'],e['b']) for e in chosen[n])
            if key not in unique:unique[key]=finish(group(clean,chosen[n]),raw)
            variants[n]=unique[key]
        assert variants['reference']==base[fn],f'Reconstruction mismatch {fn}'
        operations.append(dict(file_name=fn,candidates=es,selected={n:[e['sources'] for e in chosen[n]] for n in names}))
        historical=compare(records[fn],variants['none'],variants['reference']);history.extend(historical)
        for n in names:
            pred=variants[n];assert_disjoint(pred['masks']);write_json(OUT/n/'predictions'/path.name,pred)
            for c in compare(records[fn],base[fn],pred):changes.append(dict(policy=n,**c))
            for rec in records[fn]:
                m=rle_iou(rec['gt'],pred['masks']);s=pq_stats_from_iou(m);scores[n]+=s
                tables[n].append(dict(iou_matrix=torch.from_numpy(m),n_gt=len(rec['gt']),n_pred=m.shape[1]))
                allrecords.append(dict(policy=n,record=rec['record'],**asdict(s),matched=np.flatnonzero((m>.5).any(axis=1)).tolist()))
        if es:
            separate=variants['none']
            for e in es:
                isolated=finish(group(clean,[e]),raw)
                effect=compare(records[fn],separate,isolated)
                pairaudit.append(dict(file_name=fn,**e,selected_in_reference=e in chosen['reference'],
                    gains=sum(c['group']=='Gained' for c in effect),losses=sum(c['group']=='Lost' for c in effect),changes=effect))
        if (number+1)%20==0:print(f'Merge audit {number+1}/146',flush=True)
    official=official_functions();results=[]
    lostkeys={(c['record'],c['target_index']) for c in history if c['group']=='Lost'}
    for n in names:
        score=scores[n];assert abs(float(official['get_pq_score'](pd.DataFrame(tables[n])))-score.pq)<1e-10
        saved=cache(OUT/n/'predictions');check=PQStats()
        for fn,recs in records.items():
            assert_disjoint(saved[fn]['masks'])
            for rec in recs:check+=pq_stats_from_iou(rle_iou(rec['gt'],saved[fn]['masks']))
        assert asdict(check)==asdict(score)
        gain=[c for c in changes if c['policy']==n and c['group']=='Gained'];loss=[c for c in changes if c['policy']==n and c['group']=='Lost']
        recovered=[c for c in gain if (c['record'],c['target_index']) in lostkeys]
        results.append(dict(policy=n,**asdict(score),pq=score.pq,delta_pq=score.pq-scores['reference'].pq,
            gained=len(gain),lost=len(loss),historical_merge_losses_recovered=len(recovered),historical_merge_losses_remaining=len(lostkeys)-len(recovered),
            passes_zero_loss_gate=bool(n not in ['reference','none'] and not loss and score.pq>scores['reference'].pq)))
    assert abs(scores['reference'].pq-current['validation']['pq'])<1e-12
    assert sha(manifest)==plan['reference_sha256'] and all(sha(WORKSPACE/p)==h for p,h in hashes.items())
    write_json(OUT/'summary.json',dict(results=results,images=len(records),records=sum(map(len,records.values())),
        historical_merge_gains=sum(c['group']=='Gained' for c in history),historical_merge_losses=len(lostkeys),
        organizer_parity=True,reload_exact=True,disjoint=True,baseline_reconstruction=True,production_unchanged=True))
    write_json(OUT/'changes.json',changes);write_json(OUT/'historical_merge_changes.json',history)
    write_json(OUT/'pair_audit.json',pairaudit);write_json(OUT/'operations.json',operations)
    write_json(OUT/'records.json',allrecords)
    print(json.dumps(read_json(OUT/'summary.json'),indent=2),flush=True)

if __name__=='__main__':main()
