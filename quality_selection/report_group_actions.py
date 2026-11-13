"""Summarize available training-only gains and render actual action outcomes."""
from collections import Counter
from dataclasses import asdict
import html
import json

import numpy as np
from PIL import Image, ImageDraw

from group_actions import DEST, REFERENCE_PQ, choose, utility, outcome, signature
from common import ROOT, OUT, read_json, write_json, sha, decode_mask
from decision_selector import DEST as OLD, training_records
from refiner import DEFAULT_PROJECT, FilamentDataset, PQStats
from validation_pq import rle_iou
from sweep_confidence import assert_disjoint


def overlay(image, masks, box, labels):
    x0,y0,x1,y1=box
    base=np.array(image.crop(box).convert('RGB')).copy()
    colors=[(255,170,40),(30,220,160),(220,70,240),(70,150,255),(240,220,50)]
    centers=[]
    for i,rle in enumerate(masks):
        cut=decode_mask(rle)[y0:y1,x0:x1].astype(bool)
        if not cut.any(): continue
        base[cut]=(base[cut]*.55+np.array(colors[i%len(colors)])*.45).astype('uint8')
        yy,xx=np.nonzero(cut); centers.append((float(xx.mean()),float(yy.mean()),str(labels[i])))
    scale=min(480/(x1-x0),480/(y1-y0))
    width,height=max(1,round((x1-x0)*scale)),max(1,round((y1-y0)*scale))
    left,top=(480-width)//2,(480-height)//2
    panel=Image.new('RGB',(480,480),(22,26,35))
    panel.paste(Image.fromarray(base).resize((width,height)),(left,top))
    draw=ImageDraw.Draw(panel)
    for x,y,label in centers:
        draw.text((left+int(x*scale),top+int(y*scale)),label,fill='white',stroke_width=1,stroke_fill='black')
    return panel


def main():
    summary=read_json(DEST/'summary.json'); assert summary['complete_training_pool']
    rows=[read_json(p) for p in sorted((DEST/'images').glob('*.json'))]
    records=training_records(); previous={}; comparisons={}; signed=Counter(); recovery=[]; removed_raw={}
    for part in ['fit','tune']:
        selected=[r for r in rows if r['partition']==part]; previous_rows=[]
        qualifying_removed=0; available_rescues=set(); chosen_rescues=set()
        chosen_by_name={c['file_name']:c['alternative'] for c in summary['partitions'][part]['choices']}
        for row in selected:
            prior=read_json(OLD/'training'/ (row['raw_path'].split('/')[-1].split('\\')[-1]))
            base=row['alternatives'][0]['outcome']
            assert all(abs(base[k]-prior['baseline'][k])<1e-9 for k in ['tp','fp','fn','iou_sum'])
            previous_rows.append(dict(alternatives=[dict(outcome=prior['baseline'])]+
                                     [dict(outcome=e['stats']) for e in prior['examples']]))
            hits0={r['record']:set(r['hits']) for r in base['records']}
            raw=read_json(ROOT/row['raw_path']); missed={}
            for rec in records[row['file_name']]:
                matrix=rle_iou(rec['gt'],raw['masks'])
                missed[rec['record']]=set(np.flatnonzero((matrix>.5).any(axis=1)))-hits0[rec['record']]
                qualifying_removed+=len(missed[rec['record']])
            for j,alt in enumerate(row['alternatives'][1:],1):
                s=alt['outcome']; delta=utility(s,REFERENCE_PQ)-utility(base,REFERENCE_PQ)
                signed['positive' if delta>1e-9 else 'negative' if delta < -1e-9 else 'neutral']+=1
                gained=sum(len(set(r['hits'])-hits0[r['record']]) for r in s['records'])
                lost=sum(len(hits0[r['record']]-set(r['hits'])) for r in s['records'])
                rescued={(rec['record'],t) for rec in s['records'] for t in set(rec['hits']) & missed[rec['record']]}
                available_rescues.update(rescued)
                if j==chosen_by_name[row['file_name']]: chosen_rescues.update(rescued)
                if gained: recovery.append(dict(file_name=row['file_name'],alternative=j,kind=alt['action']['kind'],
                                                gained=gained,lost=lost,delta=delta,partition=part,
                                                qualifying_removed_recovered=len(rescued)))
        _,old=choose(previous_rows); previous[part]=old
        comparisons[part]=dict(old_swap_oracle=old,new_group_oracle=summary['partitions'][part]['oracle'],
                               action_sets_not_strict_supersets=True)
        removed_raw[part]=dict(qualifying_raw_targets_lost_by_control=qualifying_removed,
                              recoverable_by_at_least_one_action=len(available_rescues),
                              recovered_by_pooled_oracle=len(chosen_rescues))
    write_json(DEST/'diagnostics.json',dict(comparison=comparisons,action_labels=dict(signed),
        alternatives_recovering_targets=len(recovery),recovery_actions=recovery,training_cleanup_cohort=removed_raw))

    # Reload the selected legal predictions and independently verify labels.
    for part,s in summary['partitions'].items():
        by_name={r['file_name']:r for r in rows if r['partition']==part}
        total=PQStats()
        for choice in s['choices']:
            row=by_name[choice['file_name']]; alt=row['alternatives'][choice['alternative']]
            pred=alt['prediction']; assert_disjoint(pred['masks'])
            rescored=outcome(records[row['file_name']],pred)
            assert rescored==alt['outcome']
            total+=PQStats(**{k:rescored[k] for k in ['tp','fp','fn','iou_sum']})
        assert abs(total.pq-s['oracle']['pq'])<1e-12

    review=DEST/'review'; review.mkdir(exist_ok=True)
    reader=FilamentDataset(DEFAULT_PROJECT,'train')
    by_name={r['file_name']:r for r in rows}
    # Different actions, genuine recoveries, harmful changes, and disagreement.
    picks=[]; used=set()
    for kind in ['prefer','allocate','merge','reject']:
        options=sorted([r for r in recovery if r['kind']==kind],key=lambda r:(-r['qualifying_removed_recovered'],-r['delta']))
        if options:
            pick=options[0]; picks.append(pick); used.add((pick['file_name'],pick['alternative']))
    for row in rows:
        b=row['alternatives'][0]['outcome']
        for j,a in enumerate(row['alternatives'][1:],1):
            d=[utility(s,REFERENCE_PQ)-utility(t,REFERENCE_PQ) for s,t in zip(a['outcome']['records'],b['records'])]
            if min(d,default=0)<-1e-9 and max(d,default=0)>1e-9 and (row['file_name'],j) not in used:
                picks.append(dict(file_name=row['file_name'],alternative=j,kind='annotator_disagreement'))
                used.add((row['file_name'],j)); break
        if len(picks)>=7: break
    index=[]
    for number,pick in enumerate(picks):
        row=by_name[pick['file_name']]; alt=row['alternatives'][pick['alternative']]
        action=alt['action']; raw=read_json(ROOT/row['raw_path'])
        from pycocotools import mask as mu
        x,y,w,h=map(int,mu.toBbox(mu.merge([raw['masks'][j] for j in action['members']])))
        box=(max(0,x-40),max(0,y-40),min(2048,x+w+40),min(2048,y+h+40))
        with Image.open(reader.image_dir/row['file_name']) as im:
            image=im.convert('L')
            panels=[('Original',overlay(image,[],box,[])),
                    ('Raw proposals',overlay(image,[raw['masks'][j] for j in action['members']],box,
                                             [f'#{raw["source_proposal_ids"][j]}' for j in action['members']])),
                    ('Current final',overlay(image,row['alternatives'][0]['prediction']['masks'],box,
                                             row['alternatives'][0]['prediction']['source_proposal_ids'])),
                    ('Alternative final',overlay(image,alt['prediction']['masks'],box,alt['prediction']['source_proposal_ids']))]
            # Separate panels for annotators, with separate instance IDs.
            for rec in records[row['file_name']]:
                panels.append((rec['record'],overlay(image,rec['gt'],box,[f'G{i}' for i in range(len(rec['gt']))])))
        sheet=Image.new('RGB',(960,65+520*((len(panels)+1)//2)),(22,26,35)); draw=ImageDraw.Draw(sheet)
        draw.text((12,10),f'{row["file_name"]} | {pick["kind"]} | {action["kind"]} {action["ids"]}',fill='white')
        for i,(title,panel) in enumerate(panels):
            x=(i%2)*480; y=65+(i//2)*520
            draw.text((x+8,y),title,fill='white'); sheet.paste(panel,(x,y+25))
        name=f'case_{number+1:02d}.jpg'; sheet.save(review/name,quality=90)
        index.append(dict(id=f'TRAIN{number+1:02d}',**pick,action=action,sheet=name,
                          baseline=row['alternatives'][0]['outcome'],alternative_outcome=alt['outcome']))
    write_json(review/'cases.json',index)
    (review/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Training action review</title>'
        '<style>body{font:18px system-ui;background:#151922;color:#eee;margin:24px}img{max-width:100%}</style>'
        '<h1>Training-only action review</h1><p>Diagnostic alternatives, not CNN predictions. Separate panels preserve independent annotations.</p>'+
        ''.join(f'<h2>{p["id"]}: {html.escape(p["file_name"])} — {p["kind"]}</h2><img src="{p["sheet"]}">' for p in index),encoding='utf-8')
    lines=['# Conflict-group/action dataset — 3 November 2026','',
           '**Training-only diagnostic completed; no CNN trained and no production changes.**','',
           'Frozen YOLO11s + epoch6 U-Net + bounded NMS ranker; admission 0.20, pixel threshold 0.55. '
           '561 physical training images, 920 independent annotation records. Labels retain separate records. '
           'Current upstream predictions are in-sample, not out of fold.','',
           '| Training partition | Baseline PQ | Previous swap oracle | Group/action oracle | Oracle TP / FP / FN | Gained / lost |',
           '|---|---:|---:|---:|---|---|']
    for part,s in summary['partitions'].items():
        o=s['oracle']; lines.append(f'| {part} ({s["images"]} images) | {s["baseline"]["pq"]:.6f} | {previous[part]["pq"]:.6f} | {o["pq"]:.6f} | {o["tp"]} / {o["fp"]} / {o["fn"]} | {s["gained"]} / {s["lost"]} |')
    lines+=['',f'{summary["groups"]} overlapping groups; {summary["alternatives"]} distinct changed full-image outputs. '
            f'{summary["images_without_conflicts"]} images have no overlapping conflict group. Label signs at fixed reference utility: {dict(signed)}.',
            '', 'Training analog of the original cleanup-loss cohort (independent annotation targets): '+json.dumps(removed_raw)+'.',
            '', '## What the numbers mean','',
            'The oracle uses training annotations to choose at most one changed group per image, or unchanged. '
            'It optimizes pooled matched-IoU and TP/FP/FN contributions, not a mean of group scores. '
            'It is an action-space ceiling on these training images, not a CNN result or validation score. '
            'Training-partition PQ must not be compared numerically with validation PQ 0.448878 or 0.449307.',
            '', 'Every member may be preferred/rejected regardless of confidence gap; six most-overlapping pairs per group receive two allocation choices and an optional merge. '
            'Prefer changes both NMS precedence and ownership. Reject reruns NMS and may rescue a previously suppressed candidate. '
            'Allocate keeps both pair members despite duplicate suppression and assigns shared pixels to the first member. '
            'Merge uses their surviving disjoint union before unchanged downstream pruning. '
            'The previous swap action space is different, not a strict subset. These options do not invent foreground or join nonoverlapping fragments.',
            '', '## Verification and limitations','',
            'All generated outputs disjoint; organizer notebook PQ parity for both partition ceilings. '
            'Selected saved outputs reloaded and rescored; all 561 baseline contributions match the previous verified selector dataset. '
            'Protected model manifests unchanged. Action construction receives no annotations; annotations enter only full-output labeling. '
            'Several alternatives have opposite effects across annotators; labels preserve these, rather than forcing a winner.',
            '', 'No new raw masks were generated for the update2500 checkpoint: its 0.449307 validation configuration remains a separate required deployment comparator. '
            'A CNN trained from this epoch6 dataset cannot be assumed to transfer to it without matched checks. '
            'No claim is made that the original 50 validation misses have been recovered.',
            '', '## Next decision','',
            'The new ceiling exceeds the previous action ceiling by 0.008994 on fit and 0.008213 on training tune, '
            'with useful recoveries represented in the cleanup-loss cohort. This supports one bounded group/action CNN pilot. '
            'It does not establish that the CNN can learn to realize those gains. Include unchanged and protected-success examples, '
            'keep original training-day fit/tune separation, choose a conservative action threshold on tuning days, '
            'then evaluate the frozen selector once against both validation references. '
            'Improvement in this diagnostic alone does not establish learnability or generalization.',
            '', 'Visual inspection: [VISUAL_FINDINGS.md](VISUAL_FINDINGS.md). Actual rendered examples: [review/index.html](review/index.html). Machine-readable outcomes: images/*.json; '
            'summary.json; diagnostics.json; plan.json. No test-set access, gradients, training or submission.']
    (DEST/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plan=read_json(DEST/'plan.json')
    for path,digest in plan['protected'].items(): assert sha(ROOT/path)==digest
    for config in ['quality_selection/current_candidate.json',
                   'local_refiner/outputs/realistic_matched_2500_20261103/selected_confidence_diagnostic.json']:
        bundle=read_json(ROOT/config)
        for key in ['detector','refiner','mask_ranking']:
            assert sha(ROOT/bundle[key]['path'])==bundle[key]['sha256']
    inspected=read_json(DEST/'visual_review.json')['inspected_sheets'] if (DEST/'visual_review.json').exists() else []
    write_json(DEST/'report_verification.json',dict(baseline_images_matched=561,selected_reload_exact=True,
               disjoint=True,protected_unchanged=True,rendered_sheets=len(index),visually_inspected_sheets=inspected,
               protected_model_hashes_match_manifests=True,
               report_source_sha256=sha(__file__),no_cnn_training=True,no_validation_labels=True))
    print(json.dumps(dict(comparison=comparisons,action_labels=dict(signed),training_cleanup_cohort=removed_raw,rendered_sheets=len(index)),indent=2))


if __name__=='__main__': main()
