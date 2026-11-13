"""Replay frozen production cleanup, explain losses, and optionally ablate one operation."""
import argparse
import copy
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mu
import torch
from refiner import HERE, WORKSPACE, DEFAULT_PROJECT, FilamentDataset, read_json, write_json, decode_mask, encode_mask, sha, PQStats, pq_stats_from_iou
from build_visual_review import cache, bbox, view_box
from validation_pq import rle_iou
from selection_policies import Policy, features, apply_policy
from merge_recovery import edges, selected_edges, group, run_policy, finish
from fp_improvement_options import apply_option
from reaudit_fp_progress import parents
from sweep_confidence import assert_disjoint
from official_notebook_diagnostics import official_functions

OUT = HERE / 'outputs/production_cleanup_trace_20261101'
STAGES = ['raw', 'mask_nms', 'overlap_clipping', 'minimum_remaining', 'guarded_merge', 'tiny_satellites', 'final']
LABELS = dict(raw='Raw U-Net candidate', mask_nms='Duplicate suppression', overlap_clipping='Confidence-first pixel ownership',
    minimum_remaining='10% remaining-area filter', guarded_merge='Guarded instance merging', tiny_satellites='Tiny-fragment pruning', final='Clipped-remnant filter')


def subset(raw, masks):
    ids = sorted(masks)
    return dict(file_name=raw['file_name'], masks=[masks[i] for i in ids],
        confidence=[raw['confidence'][i] for i in ids], source_proposal_ids=[raw['source_proposal_ids'][i] for i in ids])


def replay(raw, mask_nms=True, shape=(2048, 2048)):
    """Mirror production decisions and retain evidence; annotations never enter this function."""
    f = features(raw); confidence = raw['confidence']; sources = raw['source_proposal_ids']
    order = sorted((i for i,c in enumerate(confidence) if c >= .20), key=lambda i:(-confidence[i],-confidence[i],i))
    kept=[]; log={}
    for i in order:
        blockers = [j for j in kept if f['pair_iou'][i,j] >= .5] if mask_nms else []
        log[i]=dict(source_id=sources[i],confidence=confidence[i],raw_area=int(f['areas'][i]),
            nms_blockers=[dict(source_id=sources[j],confidence=confidence[j],raw_pair_iou=float(f['pair_iou'][i,j])) for j in blockers])
        if not blockers: kept.append(i)
    occupied=np.zeros(shape,bool); allocations=[]; clipped={}; clean_masks={}
    for i in kept:
        mask=f['decoded'][i]; remain=mask&~occupied; occupied|=mask
        owners=[]
        for owner,claimed in allocations:
            count=int(mu.area(mu.merge([raw['masks'][i],claimed],intersect=True)))
            if count: owners.append(dict(source_id=sources[owner],confidence=confidence[owner],claimed_pixels=count))
        owners.sort(key=lambda x:-x['claimed_pixels'])
        fraction=int(remain.sum())/f['areas'][i]
        log[i].update(remaining_pixels=int(remain.sum()),remaining_fraction=float(fraction),pixel_owners=owners,
            dropped_by_remaining_filter=bool(not remain.any() or fraction<.10))
        if remain.any():
            encoded=encode_mask(remain); clipped[i]=encoded; allocations.append((i,encoded))
            if fraction>=.10: clean_masks[i]=encoded
    clean=subset(raw,clean_masks)
    expected,_=apply_policy(raw,Policy('trace_reference',mask_nms=mask_nms),cached=f,shape=shape)
    assert clean==expected, 'Trace replay diverged from actual production confidence cleanup'
    stages=dict(raw=raw,mask_nms=subset(raw,{i:raw['masks'][i] for i in kept}),overlap_clipping=subset(raw,clipped),minimum_remaining=clean)
    candidates=edges(raw,clean);chosen=selected_edges(candidates,'majority_veto');merged=group(clean,chosen)
    pruned,prune_log=apply_option(merged,None,'tiny_satellites');final=finish(merged,raw)
    actual,actual_chosen=run_policy(clean,raw,candidates,'majority_veto');assert final==actual and chosen==actual_chosen
    stages.update(guarded_merge=merged,tiny_satellites=pruned,final=final)
    return stages,dict(candidates=list(log.values()),merge_edges=candidates,chosen_merges=chosen,pruning=prune_log)


def same_predictions(actual,saved):
    return all(actual[k]==saved[k] for k in ['file_name','masks','confidence','source_proposal_ids','source_groups'])


def candidate_stage(gt,row,source):
    ids=[j for j in range(len(row['masks'])) if source in parents(row,j)]
    matrix=rle_iou([gt],row['masks'])[0]
    best=max(ids,key=lambda j:matrix[j]) if ids else None
    return dict(present=bool(ids),candidate_iou=0. if best is None else float(matrix[best]),
        target_best_iou=float(matrix.max(initial=0)),source_id=None if best is None else str(row['source_proposal_ids'][best]),
        candidate_area=0 if best is None else int(mu.area(row['masks'][best])))


def render(case,raw,final,gt,image,path):
    selected=decode_mask(gt).astype(bool); candidate=case['candidate']; source=candidate['source_id']
    j=raw['source_proposal_ids'].index(source); initial=decode_mask(raw['masks'][j]).astype(bool)
    evidence=candidate['decision']; competitors=evidence.get('merge_partners',[]) if candidate['first_loss_stage']=='guarded_merge' else evidence.get('nms_blockers') or evidence.get('pixel_owners',[])
    owner=competitors[0]['source_id'] if competitors else None
    competitor=None if owner is None else decode_mask(raw['masks'][raw['source_proposal_ids'].index(owner)]).astype(bool)
    descendants=[decode_mask(m).astype(bool) for k,m in enumerate(final['masks']) if source in parents(final,k)]
    best=case['final_best_index']; final_best=None if best is None else decode_mask(final['masks'][best]).astype(bool)
    scope=selected|initial
    if competitor is not None: scope|=competitor
    crop=tuple(int(v) for v in view_box(bbox(scope),64)); x0,y0,x1,y1=crop; side=380
    base=np.array(image.crop(crop).resize((side,side)))
    panels=[[],[(selected,(50,255,95),f'TARGET G{case["target_index"]}')],[(initial,(0,220,240),f'#{source} raw')],
        [] if competitor is None else [(competitor,(255,160,40),f'#{owner} competitor')],
        [(m,(0,220,240),f'#{source} descendant') for m in descendants],
        [] if final_best is None else [(final_best,(0,220,240),f'#{final["source_proposal_ids"][best]} final')]]
    titles=['Original image','Selected annotation ONLY','Qualifying raw U-Net candidate','Competing raw candidate','Same candidate AFTER pipeline','Best surviving final prediction']
    sheet=Image.new('RGB',(side*3,side*2+130),'white');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',14)
    draw.text((8,4),f"{case['id']} | {case['record']} | target G{case['target_index']} | raw #{source} confidence {candidate['confidence']:.4f}",fill='black',font=font)
    draw.text((8,27),f"Raw IoU {candidate['raw_iou']:.4f} -> final best {case['final_best_iou']:.4f} | First loss: {LABELS[candidate['first_loss_stage']]}",fill='black',font=font)
    draw.text((8,50),'Green: ONE selected annotation. Cyan: predictions. Orange: competitor. Raw masks may overlap; final masks are disjoint.',fill='black',font=font)
    for k,items in enumerate(panels):
        pixels=base.astype(float).copy();tags=[]
        for mask,color,label in items:
            small=np.array(Image.fromarray(mask[y0:y1,x0:x1]).resize((side,side),Image.Resampling.NEAREST)).astype(bool)
            pixels[small]=.45*pixels[small]+.55*np.array(color)
            if small.any():
                yy,xx=np.nonzero(small);tags.append((int(xx.min()),int(yy.min()),label,color))
        tile=Image.fromarray(pixels.astype(np.uint8));td=ImageDraw.Draw(tile)
        for x,y,label,color in tags:td.text((x,y),label,font=font,fill=color,stroke_width=2,stroke_fill='black')
        x=(k%3)*side;y=110+(k//3)*(side+20);sheet.paste(tile,(x,y));draw.text((x+5,y-21),titles[k],fill='black',font=font)
    sheet.save(path,quality=90)


def main(ablation=None):
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True);(OUT/'sheets').mkdir(exist_ok=True)
    refpath=HERE/'current_best_pipeline.json';digest=sha(refpath);ref=read_json(refpath)
    rawdir=WORKSPACE/ref['raw_validation_predictions'];raw=cache(rawdir);final=cache(WORKSPACE/ref['validation_predictions'])
    records=defaultdict(list)
    for r in read_json(HERE/'outputs/fp_progress_three_options_20261028/validation_records.json'):records[r['file_name']].append(r)
    production_cases=read_json(HERE/'outputs/production_best_audit_20261101/cases.json')
    selected=[c for c in production_cases if c['type']=='FN' and c['category'] in ['Qualifying refined candidate removed','Qualifying refined candidate damaged or merged']]
    assert len(selected)==60
    byimage=defaultdict(list)
    for c in selected:byimage[c['file_name']].append(c)
    cases=[];total=PQStats();tables=[];changes=[]
    reader=FilamentDataset(DEFAULT_PROJECT,'val')
    for number,(fn,recs) in enumerate(records.items(),1):
        if ablation:
            clean,_=apply_policy(raw[fn],Policy('single_operation_ablation',mask_nms=False))
            pred,_=run_policy(clean,raw[fn],edges(raw[fn],clean),'majority_veto')
        else:
            stages,log=replay(raw[fn]);pred=stages['final']
        assert_disjoint(pred['masks'])
        if ablation is None:
            assert same_predictions(pred,final[fn]),f'Production replay mismatch: {fn}'
        for r in recs:
            matrix=rle_iou(r['gt'],pred['masks']);total+=pq_stats_from_iou(matrix)
            tables.append(dict(iou_matrix=torch.from_numpy(matrix),n_gt=len(r['gt']),n_pred=matrix.shape[1]))
            if ablation:
                before=rle_iou(r['gt'],final[fn]['masks']);h0=(before>.5).any(axis=1);h1=(matrix>.5).any(axis=1)
                for t in np.flatnonzero(h0!=h1):changes.append(dict(record=r['record'],file_name=fn,target_index=int(t),group='Gained' if h1[t] else 'Lost',before_iou=float(before[t].max(initial=0)),after_iou=float(matrix[t].max(initial=0))))
        if ablation:
            folder=OUT/ablation/'predictions';folder.mkdir(parents=True,exist_ok=True)
            write_json(folder/f'{number-1:04d}.json',pred)
        elif fn in byimage:
            with Image.open(reader.image_dir/fn) as im:image=im.convert('RGB')
            for c in byimage[fn]:
                rec=next(r for r in recs if r['record']==c['record']);gt=rec['gt'][c['target_index']]
                rm=rle_iou([gt],raw[fn]['masks'])[0];qualified=[]
                for i in np.flatnonzero(rm>.5):
                    i=int(i);source=raw[fn]['source_proposal_ids'][i]
                    timeline=[dict(stage=stage,label=LABELS[stage],**candidate_stage(gt,stages[stage],source)) for stage in STAGES]
                    first=next(s['stage'] for s in timeline if s['candidate_iou']<=.5)
                    decision=copy.deepcopy(next(x for x in log['candidates'] if x['source_id']==source))
                    decision['merge_partners']=[]
                    for merge in log['chosen_merges']:
                        if source in merge['sources']:
                            for partner in merge['sources']:
                                if partner!=source:
                                    idx=raw[fn]['source_proposal_ids'].index(partner)
                                    decision['merge_partners'].append(dict(source_id=partner,confidence=raw[fn]['confidence'][idx],raw_iou_with_same_target=float(rm[idx]),final=candidate_stage(gt,pred,partner)))
                    competitors=decision.get('nms_blockers') or decision.get('pixel_owners',[])
                    for competitor in competitors:
                        sid=competitor['source_id'];idx=raw[fn]['source_proposal_ids'].index(sid)
                        competitor['raw_iou_with_same_target']=float(rm[idx])
                        competitor['final']=candidate_stage(gt,pred,sid)
                    qualified.append(dict(source_id=source,confidence=raw[fn]['confidence'][i],raw_iou=float(rm[i]),first_loss_stage=first,timeline=timeline,decision=decision))
                qualified.sort(key=lambda x:-x['raw_iou']);assert qualified
                m=rle_iou([gt],pred['masks'])[0];best=float(m.max(initial=0));assert abs(best-c['best_iou'])<1e-12
                case=dict(id=c['id'],record=c['record'],file_name=fn,target_index=c['target_index'],category=c['category'],
                    final_best_iou=best,final_best_index=int(m.argmax()) if best else None,candidate=qualified[0],all_qualifying_candidates=qualified,
                    merges=log['chosen_merges'],sheet=f'sheets/{c["id"]}.jpg')
                render(case,raw[fn],pred,gt,image,OUT/case['sheet']);cases.append(case)
        if number%25==0:print(f'{ablation or "Trace"}: {number}/146 images',flush=True)
    official=float(official_functions()['get_pq_score'](pd.DataFrame(tables)));assert abs(official-total.pq)<1e-10
    assert sha(refpath)==digest
    summary=dict(**asdict(total),pq=total.pq,organizer_parity=True,all_final_masks_disjoint=True,production_unchanged=True,reference_sha256=digest)
    if ablation:
        summary.update(delta_pq=total.pq-ref['validation']['pq'],gained=sum(c['group']=='Gained' for c in changes),lost=sum(c['group']=='Lost' for c in changes),
            intervention='Disable mask NMS only; keep confidence ownership, remaining-area filter, merging, satellite pruning and clipped-remnant filter. No GT used for decisions.')
        write_json(OUT/ablation/'summary.json',summary);write_json(OUT/ablation/'changes.json',changes)
    else:
        assert (total.tp,total.fp,total.fn)==(956,460,566) and abs(total.pq-ref['validation']['pq'])<1e-12
        cases.sort(key=lambda c:int(c['id'][2:]));summary.update(cases=60,first_loss_of_best_raw_candidate=dict(Counter(c['candidate']['first_loss_stage'] for c in cases)),
            unique_raw_candidates=len({(c['file_name'],q['source_id']) for c in cases for q in c['all_qualifying_candidates']}))
        write_json(OUT/'cases.json',cases);write_json(OUT/'summary.json',summary)
        data=json.dumps(cases).replace('</','<\\/')
        page=r'''<!doctype html><meta charset="utf-8"><title>Best pipeline: cleanup history</title><style>body{font:16px system-ui;background:#f4f6f8;margin:24px}select,button{font:inherit;padding:8px}img{max-width:100%}table{border-collapse:collapse;background:white}td,th{border:1px solid #ccd;padding:8px}#details{white-space:pre-line;background:white;padding:12px}</style><h1>Best pipeline: candidate history</h1><p><strong>Production reference PQ 0.445799 | TP956 / FP460 / FN566.</strong> Frozen models; exact saved final predictions reproduced on all146images. This is a diagnostic history of the 60 cleanup-related FN entries, not a new pipeline.</p><p>Green = selected target only; cyan = prediction; orange = competing candidate. Raw candidates may overlap; final masks are disjoint. IoU above0.50 qualifies. Each annotation record is scored separately.</p><button id="prev">Previous</button><select id="pick"></select><button id="next">Next</button><p id="details"></p><table><thead><tr><th>Step</th><th>Selected candidate / descendant</th><th>IoU with target</th><th>Pixels</th><th>Best IoU from any candidate</th></tr></thead><tbody id="timeline"></tbody></table><img id="sheet"><details><summary>All qualifying candidates and exact decision evidence</summary><pre id="json"></pre></details><script>const cases=DATA;const $=id=>document.getElementById(id);
for(const [i,c]of cases.entries()){const o=document.createElement('option');o.value=i;o.textContent=c.id+' · '+c.record+' · raw #'+c.candidate.source_id;$('pick').append(o)}
function show(){const c=cases[Number($('pick').value)],q=c.candidate,d=q.decision;const competitors=q.first_loss_stage==='guarded_merge'?d.merge_partners||[]:d.nms_blockers.length?d.nms_blockers:d.pixel_owners||[];let why=d.nms_blockers.length?'Suppressed as a duplicate of '+d.nms_blockers.map(x=>'#'+x.source_id+' (confidence '+x.confidence.toFixed(4)+', mask IoU '+x.raw_pair_iou.toFixed(3)+')').join(', '):'Kept after duplicate suppression. Retained '+(100*(d.remaining_fraction||0)).toFixed(1)+'% after pixel ownership.';$('details').textContent=c.id+' · target G'+c.target_index+' · raw candidate #'+q.source_id+' confidence '+q.confidence.toFixed(4)+'\n'+why+'\nFirst loss: '+q.timeline.find(s=>s.stage===q.first_loss_stage).label+'. Raw target IoU '+q.raw_iou.toFixed(4)+' → best final IoU '+c.final_best_iou.toFixed(4)+'.\n'+(competitors.length?'Competitors: '+competitors.map(x=>'#'+x.source_id+' target IoU '+x.raw_iou_with_same_target.toFixed(4)).join('; '):'');$('timeline').replaceChildren(...q.timeline.map(s=>{const tr=document.createElement('tr');for(const x of [s.label,s.present?'#'+s.source_id:'Removed',s.candidate_iou.toFixed(4),s.candidate_area,s.target_best_iou.toFixed(4)]){const td=document.createElement('td');td.textContent=x;tr.append(td)}if(s.stage===q.first_loss_stage)tr.style.background='#ffe1d7';return tr}));$('sheet').src=c.sheet;$('json').textContent=JSON.stringify(c.all_qualifying_candidates,null,2);history.replaceState(null,'','#'+c.id)}
$('pick').onchange=show;for(const[id,step]of[['prev',-1],['next',1]])$(id).onclick=()=>{$('pick').value=Math.max(0,Math.min(cases.length-1,Number($('pick').value)+step));show()};const desired=cases.findIndex(c=>c.id===location.hash.slice(1));$('pick').value=desired<0?0:desired;show();</script>'''.replace('DATA',data)
        (OUT/'index.html').write_text(page,encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--ablation',choices=['no_mask_nms']);args=parser.parse_args();main(args.ablation)
