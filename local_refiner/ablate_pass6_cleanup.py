"""Single-step cleanup ablations on the same frozen pass6 raw masks."""
from collections import defaultdict,Counter
from dataclasses import asdict
import json
from pycocotools import mask as mu
import pandas as pd
import torch
from refiner import HERE,WORKSPACE,read_json,write_json,sha,PQStats,pq_stats_from_iou
from build_visual_review import cache
from selection_policies import features,apply_policy,Policy
from merge_recovery import edges,selected_edges,group,finish
from fp_improvement_options import apply_option
from reaudit_fp_progress import parent_union
from sweep_confidence import assert_disjoint
from validation_pq import rle_iou
from official_notebook_diagnostics import official_functions
from evaluate_merge_recovery import compare

OUT=HERE/'outputs/pass6_cleanup_ablations_20261103'
VAL=HERE/'outputs/expanded_scratch_continued_to10_20261101/epoch_06_validation'
NAMES=['control','no_nms','no_minimum_remaining','no_merge','no_tiny_pruning','no_remnant_filter']


def final_filters(merged,raw,tiny=True,remnant=True):
    row=apply_option(merged,None,'tiny_satellites')[0] if tiny else merged
    keep=[j for j,m in enumerate(row['masks']) if not(remnant and row['confidence'][j]<.5 and
        float(mu.area(m))/float(mu.area(parent_union(raw,row['source_groups'][j])))<.5)]
    out=dict(file_name=row['file_name'],**{k:[row[k][j] for j in keep] for k in ['masks','confidence','source_proposal_ids','source_groups']})
    # pycocotools merge returns bytes; pruning normally converts them, but the
    # no-pruning arm still needs the same JSON-safe, lossless representation.
    out['masks']=[dict(m,counts=m['counts'].decode('ascii')) if isinstance(m['counts'],bytes) else m for m in out['masks']]
    return out


def variants(raw):
    f=features(raw);clean,_=apply_policy(raw,Policy('control'),cached=f)
    es=edges(raw,clean);chosen=selected_edges(es,'majority_veto');merged=group(clean,chosen)
    control=final_filters(merged,raw);assert control==finish(merged,raw)
    rows=dict(control=control,no_merge=final_filters(group(clean,[]),raw),
        no_tiny_pruning=final_filters(merged,raw,tiny=False),no_remnant_filter=final_filters(merged,raw,remnant=False))
    for name,policy in [('no_nms',Policy('no_nms',mask_nms=False)),('no_minimum_remaining',Policy('no_minimum_remaining',min_remaining=0.))]:
        altered,_=apply_policy(raw,policy,cached=f);chosen=selected_edges(edges(raw,altered),'majority_veto')
        rows[name]=final_filters(group(altered,chosen),raw)
    return rows


def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if (OUT/'completed.json').exists():print(json.dumps(read_json(OUT/'summary.json'),indent=2));return
    raw=cache(VAL/'raw_predictions');base=cache(VAL/'predictions');records=defaultdict(list)
    for rec in read_json(HERE/'outputs/fp_progress_three_options_20261028/validation_records.json'):records[rec['file_name']].append(rec)
    removed=read_json(HERE/'outputs/removed50_diagnosis_20261103/cases.json');audit={(c['record'],c['target_index']):c for c in removed}
    manifest=HERE/'current_best_pipeline.json';hashes={str(p.relative_to(WORKSPACE)):sha(p) for folder in [VAL/'raw_predictions',VAL/'predictions'] for p in folder.glob('*.json')}
    plan=dict(production_sha256=sha(manifest),input_hashes=hashes,code_sha256=sha(__file__),reference_pq=.44849475084018126,
        variants=NAMES,protocol='Five single-step removals, no combinations or tuning. Frozen pass6 raw predictions, admission0.20,pixel0.55. Disjoint confidence-first ownership remains required in every arm.',
        distinctions='Report exact removed50 subset, first removal steps and competitor support for other same-record targets. That support is a diagnostic flag, not complete same-filament/different-filament labeling. Unreviewed identity is unknown.',
        scope='Saved-mask CPU evaluation only; no training, inference, source editing, promotion or Kaggle submission; reused validation.')
    write_json(OUT/'plan.json',plan)
    scores={n:PQStats() for n in NAMES};tables={n:[] for n in NAMES};changes=[];operations=[]
    for n in NAMES:(OUT/n/'predictions').mkdir(parents=True,exist_ok=True)
    for k,fn in enumerate(sorted(raw)):
        rows=variants(raw[fn]);assert rows['control']==base[fn],f'Control mismatch: {fn}'
        operations.append(dict(file_name=fn,physical_counts={n:len(row['masks']) for n,row in rows.items()}))
        for name,row in rows.items():
            assert_disjoint(row['masks']);write_json(OUT/name/'predictions'/f'{k:04d}.json',row)
            for c in compare(records[fn],base[fn],row):
                info=audit.get((c['record'],c['target_index']))
                changes.append(dict(policy=name,audit_id=None if info is None else info['id'],removed50_first_removal=None if info is None else info['first_removal'],
                    competitor_matches_other_target=None if info is None else info['any_competitor_matches_other_target_final'],instance_identity='Unreviewed; do not infer solely from overlap',**c))
            for rec in records[fn]:
                m=rle_iou(rec['gt'],row['masks']);scores[name]+=pq_stats_from_iou(m)
                tables[name].append(dict(iou_matrix=torch.from_numpy(m),n_gt=len(rec['gt']),n_pred=m.shape[1]))
        write_json(OUT/'progress.json',dict(images=k+1,total=len(raw),stage='evaluation'))
        if (k+1)%20==0:print(f'Cleanup ablations {k+1}/{len(raw)}',flush=True)
    official=official_functions();results=[]
    for name in NAMES:
        assert abs(float(official['get_pq_score'](pd.DataFrame(tables[name])))-scores[name].pq)<1e-10
        check=PQStats();saved=cache(OUT/name/'predictions')
        for fn in sorted(saved):
            assert_disjoint(saved[fn]['masks'])
            for rec in records[fn]:check+=pq_stats_from_iou(rle_iou(rec['gt'],saved[fn]['masks']))
        assert asdict(check)==asdict(scores[name])
        cs=[c for c in changes if c['policy']==name];recovered=[c for c in cs if c['group']=='Gained' and c['audit_id']]
        results.append(dict(policy=name,**asdict(check),pq=check.pq,delta_pq=check.pq-scores['control'].pq,
            gained=sum(c['group']=='Gained' for c in cs),lost=sum(c['group']=='Lost' for c in cs),removed50_recovered=len(recovered),
            removed50_recovered_by_step=dict(Counter(c['removed50_first_removal'] for c in recovered)),
            recovered_with_competitor_matching_other_target=sum(c['competitor_matches_other_target'] for c in recovered)))
    assert abs(scores['control'].pq-plan['reference_pq'])<1e-12
    assert sha(manifest)==plan['production_sha256'] and all(sha(WORKSPACE/p)==h for p,h in hashes.items())
    for name,value in [('summary',dict(results=results,organizer_parity=True,disjoint=True,reload_exact=True,control_exact=True,production_unchanged=True)),('changes',changes),('operations',operations)]:write_json(OUT/f'{name}.json',value)
    write_json(OUT/'completed.json',dict(complete=True));print(json.dumps(read_json(OUT/'summary.json'),indent=2),flush=True)


if __name__=='__main__':main()
