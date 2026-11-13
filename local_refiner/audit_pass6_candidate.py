"""Fresh pass6 FP/FN audit and optional exact cleanup histories; saved predictions only."""
import argparse
import copy
import itertools
from collections import Counter,defaultdict
from dataclasses import asdict
import json
import numpy as np
import pandas as pd
import torch
from PIL import Image,ImageDraw,ImageFont
from pycocotools import mask as mu
from refiner import HERE,WORKSPACE,DEFAULT_PROJECT,FilamentDataset,CropTransform,read_json,write_json,decode_mask,sha,PQStats,pq_stats_from_iou
from build_visual_review import cache,bbox,view_box
from sweep_confidence import select_candidates,assert_disjoint
from review_current_errors import LOW
from validation_pq import rle_iou
from audit_current_false_positives import primary,fractions
from audit_false_negatives import coverage
from reaudit_fp_progress import parents,parent_union
from official_notebook_diagnostics import official_functions
from trace_production_cleanup import replay,candidate_stage,STAGES,LABELS,render as render_history

RUN=HERE/'outputs/expanded_scratch_continued_to10_20261101'
VALIDATION=RUN/'epoch_06_validation'
OUT=HERE/'outputs/pass6_error_audit_20261101'


def fn_category(raw_iou,raw_source_retained,coarse_iou,weak_iou,union_iou,final_iou,raw_coverage,precision,recall):
    if raw_iou>.5:return 'Qualifying refined candidate damaged or merged' if raw_source_retained else 'Qualifying refined candidate removed'
    if coarse_iou>.5:return 'Qualifying YOLO mask worsened by refinement'
    if weak_iou>.5:return 'Qualifying lower-confidence YOLO proposal available'
    if union_iou>.5:return 'Grouping existing final pieces can qualify (oracle)'
    if final_iou>=.4:return 'Near matching threshold (IoU0.40-0.50)'
    if raw_coverage<=.5:return 'Insufficient foreground in current raw union'
    if recall>.5 and precision<=.5:return 'Excess foreground or merged instance'
    if recall<=.5 and final_iou>0:return 'Incomplete coverage or scattered fragments'
    if final_iou==0:return 'No final overlap despite available raw foreground'
    return 'Other boundary or instance mismatch'


def best(values):
    return float(values.max(initial=0))


def render_case(c,image,gts,others,rawmask,final,path):
    t=c['target_index'];j=c['final_index'];focus=None if j is None else decode_mask(final['masks'][j]).astype(bool)
    scope=np.zeros((2048,2048),bool)
    if t is not None:scope|=gts[t]
    if focus is not None:scope|=focus
    if rawmask is not None:scope|=rawmask
    crop=tuple(int(v) for v in view_box(bbox(scope),64));x0,y0,x1,y1=crop;side=380
    base=np.array(image.crop(crop).resize((side,side)));font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',14)
    sheet=Image.new('RGB',(side*3,side*2+130),'white');draw=ImageDraw.Draw(sheet)
    draw.text((8,3),f"PASS6 | {c['id']} | {c['record']} | target G{t if t is not None else 'none'} | final #{c['source_id']} | IoU {c['best_iou']:.4f}",font=font,fill='black')
    draw.text((8,25),c['category'],font=font,fill='black')
    draw.text((8,47),'GREEN = ONE selected reference. CYAN = prediction. Faded blue G labels = separate neighbouring references. #IDs identify outputs.',font=font,fill='black')
    panels=[[],[(m,f'G{k}',k==t,'annotation') for k,m in enumerate(gts)],
        [] if rawmask is None else [(rawmask,c['raw_source_label'],False,'prediction')],
        [] if focus is None else [(focus,'#'+str(c['source_id']),False,'prediction')],
        [(decode_mask(m).astype(bool),'#'+str(final['source_proposal_ids'][k]),False,'nearby') for k,m in enumerate(final['masks'])],
        [(m,label,False,'other') for m,label in others]]
    titles=['1. Original image','2. Selected target GREEN','3. '+c['raw_panel_title'],'4. Best/focus FINAL prediction','5. All nearby FINAL masks','6. Other annotators (context only)']
    for panel,items in enumerate(panels):
        pixels=base.astype(float).copy();tags=[]
        for mask,label,selected,kind in items:
            cut=mask[y0:y1,x0:x1]
            if not cut.any():continue
            small=np.array(Image.fromarray(cut).resize((side,side),Image.Resampling.NEAREST)).astype(bool)
            if not small.any():continue
            color=(50,255,95) if selected else (140,160,200) if kind=='annotation' else (185,145,215) if kind=='other' else (0,220,240) if kind=='prediction' or label=='#'+str(c['source_id']) else (80,140,255)
            alpha=.65 if selected else .22 if kind in ['annotation','other'] else .5
            pixels[small]=(1-alpha)*pixels[small]+alpha*np.array(color)
            yy,xx=np.nonzero(small);tags.append((int(xx.min()),int(yy.min()),'TARGET '+label if selected else label,color))
        tile=Image.fromarray(pixels.astype(np.uint8));td=ImageDraw.Draw(tile)
        for x,y,label,color in tags:td.text((x,y),label,font=font,fill=color,stroke_width=2,stroke_fill='black')
        x=(panel%3)*side;y=110+(panel//3)*(side+20)
        sheet.paste(tile,(x,y));draw.text((x+4,y-21),titles[panel],font=font,fill='black')
    sheet.save(path,quality=89)


def write_page(cases):
    data=json.dumps(cases).replace('</','<\\/')
    page=r'''<!doctype html><meta charset="utf-8"><title>Pass6 candidate: FP/FN audit</title><style>body{font:16px system-ui;max-width:1400px;margin:24px auto;background:#f4f6f8;color:#152b3b}button,select{font:inherit;padding:8px;margin:4px}img{width:100%}#detail{white-space:pre-line;background:white;padding:14px}textarea{width:98%;height:90px;font:inherit}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #ccd}</style>
<h1>Pass6 candidate — new local best: error audit</h1><p><strong>PQ 0.448495 | TP962 | FP458 | FN560</strong> — expanded-data basic U-Net, pass6, batch2, YOLOconf0.20/pixel0.55 and existing cleanup. Production selection remains unchanged. This viewer shows pass6, not the older model or probability-ownership experiment.</p>
<p>Green = ONE selected annotation. Faded blue G labels = separate neighbouring annotations. Cyan = prediction. Other annotators are context, not consensus. IDs are zero-based. A match requires IoU &gt;0.50.</p>
<p>FN/FP categories are diagnostic and have precedence. Low-confidence availability refers to YOLO coarse masks, not proven refined outputs. Recovered targets are successful comparison cases, not additional errors. Stable IDs refer to the same annotation/proposal lineage as the original audit where available.</p>
<label>View <select id="kind"><option>FN</option><option>FP</option><option>Recovered</option></select></label><label>Category <select id="category"></select></label><label><input id="changed" type="checkbox">Changes vs original only</label><br><button id="prev">Previous</button><select id="pick"></select><button id="next">Next</button><button id="history" disabled>Before/after cleanup history</button><p id="detail"></p><div id="timeline"></div><img id="sheet"><h3>Your observations</h3><textarea id="notes" placeholder="Describe what you see: contrast, gaps, grouping, boundaries…"></textarea><button id="export">Download observations</button><span id="save"></span>
<script>const cases=DATA;const $=id=>document.getElementById(id);let items=[],viewHistory=false,notes={};try{notes=JSON.parse(localStorage.getItem('pass6-audit-notes')||'{}')}catch(e){}
function show(){const c=items[Number($('pick').value)];if(!c){$('detail').textContent='No cases in this filter.';$('sheet').removeAttribute('src');return;}const h=c.history;const target=c.target_index===null?'No annotation overlaps the focus prediction.':'Selected target: G'+c.target_index+' (GREEN); other G labels are separate instances.';let extra=c.type==='Recovered'?'This target now matches; it was missed by the original pipeline.':c.type==='FN'?'No final prediction matches this annotation.':'The focus prediction does not match this annotation record.';$('detail').textContent=c.id+' · '+c.record+' · '+c.category+'\n'+target+'\nFinal IoU '+c.best_iou.toFixed(4)+'. '+extra+'\n'+c.note+(c.other_annotator_best_iou!==undefined?'\nOther-annotator best IoU '+c.other_annotator_best_iou.toFixed(4):'');$('history').disabled=!h;$('history').textContent=viewHistory&&h?'Show final-mask review':'Before/after cleanup history';$('sheet').src=viewHistory&&h?h.sheet:c.sheet;$('timeline').replaceChildren();if(viewHistory&&h){const table=document.createElement('table');for(const s of h.candidate.timeline){const tr=document.createElement('tr');for(const value of [s.label,s.present?'#'+s.source_id:'Removed','target IoU '+s.candidate_iou.toFixed(4),s.candidate_area+' px']){const td=document.createElement('td');td.textContent=value;tr.append(td)}if(s.stage===h.candidate.first_loss_stage)tr.style.background='#ffe1d7';table.append(tr)}$('timeline').append(table)}$('notes').value=notes[c.id]||'';history.replaceState(null,'','#'+c.id)}
function filter(){viewHistory=false;items=cases.filter(c=>c.type===$('kind').value&&c.category===$('category').value&&(!$('changed').checked||c.changed_vs_original));$('pick').replaceChildren(...items.map((c,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+'/'+items.length+' · '+c.id+' · '+c.record;return o}));show()}
function categories(){const rows=cases.filter(c=>c.type===$('kind').value);$('category').replaceChildren(...[...new Set(rows.map(c=>c.category))].map(cat=>{const o=document.createElement('option');o.value=cat;o.textContent=cat+' ('+rows.filter(c=>c.category===cat).length+')';return o}));filter()}
$('kind').onchange=categories;$('category').onchange=filter;$('changed').onchange=filter;$('pick').onchange=()=>{viewHistory=false;show()};$('history').onclick=()=>{viewHistory=!viewHistory;show()};for(const[id,step]of[['prev',-1],['next',1]])$(id).onclick=()=>{$('pick').value=Math.max(0,Math.min(items.length-1,Number($('pick').value)+step));viewHistory=false;show()};
$('notes').oninput=()=>{const c=items[Number($('pick').value)];if(!c)return;notes[c.id]=$('notes').value;try{localStorage.setItem('pass6-audit-notes',JSON.stringify(notes));$('save').textContent='Saved in this browser.'}catch(e){$('save').textContent='Use Download observations to preserve notes.'}};
$('export').onclick=()=>{const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(notes,null,2)],{type:'application/json'}));a.download='pass6-audit-observations.json';a.click();URL.revokeObjectURL(a.href)};
const desired=cases.find(c=>c.id===location.hash.slice(1));categories();if(desired){$('kind').value=desired.type;categories();$('category').value=desired.category;filter();$('pick').value=items.findIndex(c=>c.id===desired.id);show()}
</script>'''.replace('DATA',data)
    (OUT/'index.html').write_text(page,encoding='utf-8')


def analyze():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True);(OUT/'sheets').mkdir(exist_ok=True)
    if (OUT/'audit_complete.json').exists():print(json.dumps(read_json(OUT/'summary.json'),indent=2));return
    reference_path=HERE/'current_best_pipeline.json';refhash=sha(reference_path);ref=read_json(reference_path)
    result=read_json(RUN/'epoch_06_result.json');checkpoint=RUN/result['checkpoint']
    assert abs(result['stages']['full']['pq']-.44849475084018126)<1e-12
    raw=cache(VALIDATION/'raw_predictions');final=cache(VALIDATION/'predictions');original=cache(WORKSPACE/ref['validation_predictions'])
    pass3result=read_json(HERE/'outputs/expanded_scratch_4589_3passes_20261101/epoch_03_result.json');pass3=cache(__import__('pathlib').Path(pass3result['prediction_folder'])/'predictions')
    oldcases=read_json(HERE/'outputs/production_best_audit_20261101/cases.json');oldsummary=read_json(HERE/'outputs/production_best_audit_20261101/summary.json')
    oldfn={(c['record'],c['target_index']):c['id'] for c in oldcases if c['type']=='FN'}
    oldfp={(c['record'],tuple(int(s) for s in c['source_id'].split('+'))):c['id'] for c in oldcases if c['type']=='FP'}
    nextfn=max(int(v[2:]) for v in oldfn.values())+1;nextfp=max(int(v[2:]) for v in oldfp.values())+1
    records=defaultdict(list)
    for r in read_json(HERE/'outputs/fp_progress_three_options_20261028/validation_records.json'):records[r['file_name']].append(r)
    hashes={str(p.relative_to(WORKSPACE)):sha(p) for folder in [VALIDATION/'raw_predictions',VALIDATION/'predictions',WORKSPACE/ref['validation_predictions'],LOW] for p in folder.glob('*.json')}
    write_json(OUT/'plan.json',dict(candidate_epoch=6,candidate_checkpoint=str(checkpoint.resolve()),candidate_checkpoint_sha256=sha(checkpoint),
        production_reference_sha256=refhash,input_hashes=hashes,scope='Saved pass6outputs only; fresh taxonomy; same primary precedence as original audit; no training/policy changes.',
        primary_fp='Use current raw-parent union and final masks; coarse-parent quality is a separate flag for comparability.',
        low_confidence='Actual cached coarse YOLO masks0.01<=confidence<0.20 only; no old-refiner weak-mask cache.',
        crop_geometry='Native crops constructed from admitted YOLO coarse masks with production512canvas/padding32; not box-guided geometry.',
        source_sha256=sha(__file__)))
    reader=FilamentDataset(DEFAULT_PROJECT,'val');cases=[];total=PQStats();tables=[];changes=[];paired=Counter()
    for number,path in enumerate(sorted(LOW.glob('*.json')),1):
        low=read_json(path);fn=low['file_name'];coarse=select_candidates(low,.20);r=raw[fn];pred=final[fn]
        assert r['source_proposal_ids']==coarse['source_proposal_ids'];assert_disjoint(pred['masks'])
        groups=[parents(pred,j) for j in range(len(pred['masks']))];retained={s for group in groups for s in group}
        rawparents=[parent_union(r,g) for g in groups];coarseparents=[parent_union(coarse,g) for g in groups]
        weakids=[i for i,c in enumerate(low['confidence']) if .01<=c<.20];weak=[low['masks'][i] for i in weakids]
        areas=mu.area(pred['masks']).astype(float) if pred['masks'] else np.array([])
        rawunion=[mu.merge(r['masks'])] if r['masks'] else []
        boxes=[CropTransform.from_mask(decode_mask(m),size=512,padding=32).box for m in coarse['masks']]
        matrices={rec['record']:rle_iou(rec['gt'],pred['masks']) for rec in records[fn]}
        decoded_gt={rec['record']:[decode_mask(g).astype(bool) for g in rec['gt']] for rec in records[fn]}
        with Image.open(reader.image_dir/fn) as im:image=im.convert('RGB')
        for rec in records[fn]:
            rid=rec['record'];gt=rec['gt'];m=matrices[rid];rm=rle_iou(gt,r['masks']);cm=rle_iou(gt,coarse['masks']);wm=rle_iou(gt,weak)
            om=rle_iou(gt,original[fn]['masks']);p3m=rle_iou(gt,pass3[fn]['masks'])
            h=(m>.5).any(axis=1);oh=(om>.5).any(axis=1);p3h=(p3m>.5).any(axis=1)
            total+=pq_stats_from_iou(m);tables.append(dict(iou_matrix=torch.from_numpy(m),n_gt=len(gt),n_pred=m.shape[1]))
            paired.update(both=int((h&oh).sum()),original_only=int((~h&oh).sum()),candidate_only=int((h&~oh).sum()),neither=int((~h&~oh).sum()),
                pass3_only=int((~h&p3h).sum()),candidate_only_vs_pass3=int((h&~p3h).sum()))
            for t in np.flatnonzero(h!=oh):changes.append(dict(record=rid,file_name=fn,target_index=int(t),group='Gained' if h[t] else 'Lost',before_iou=best(om[t]),after_iou=best(m[t])))
            precision,recall=fractions(m,mu.area(gt).astype(float),areas)
            parentm=rle_iou(gt,rawparents);coarsem=rle_iou(gt,coarseparents);local=[]
            for t0 in np.flatnonzero(~h):
                t=int(t0);bv=best(m[t]);j=int(m[t].argmax()) if bv else None;rb=best(rm[t]);ri=int(rm[t].argmax()) if rb else None
                source=None if ri is None else r['source_proposal_ids'][ri]
                pp=float(precision[t,j]) if j is not None else 0.;rr=float(recall[t,j]) if j is not None else 0.
                rawcoverage=coverage(gt[t],rawunion);union_iou=bv;union_sources=[]
                candidates=sorted(np.flatnonzero(precision[t]>.5),key=lambda k:-m[t,k])[:6]
                for size in [2,3]:
                    for selected in itertools.combinations(candidates,size):
                        value=float(rle_iou([gt[t]],[mu.merge([pred['masks'][k] for k in selected])])[0,0])
                        if value>union_iou:union_iou=value;union_sources=[str(pred['source_proposal_ids'][k]) for k in selected]
                weakbest=best(wm[t]);wi=weakids[int(wm[t].argmax())] if weakbest else None
                category=fn_category(rb,source in retained,best(cm[t]),weakbest,union_iou,bv,rawcoverage,pp,rr)
                identity=(rid,t)
                if identity in oldfn:cid=oldfn[identity]
                else:cid=f'FN{nextfn:03d}';nextfn+=1
                ga=int(mu.area(gt[t]));visible=max((int(decoded_gt[rid][t][y0:y1,x0:x1].sum())/ga for x0,y0,x1,y1 in boxes),default=0.)
                note='A coarse lower-confidence proposal qualifies; this does not establish its refined quality.' if 'lower-confidence' in category else 'Raw/union measurements are diagnostic; annotations never select predictions.'
                local.append(dict(id=cid,type='FN',category=category,record=rid,file_name=fn,target_index=t,final_index=j,
                    source_id=None if j is None else str(pred['source_proposal_ids'][j]),best_iou=bv,pixel_precision=pp,pixel_recall=rr,
                    raw_best_iou=rb,raw_best_source=source,raw_confidence=None if ri is None else r['confidence'][ri],
                    coarse_best_iou=best(cm[t]),lower_coarse_best_iou=weakbest,lower_coarse_source=None if wi is None else wi,
                    lower_coarse_confidence=None if wi is None else low['confidence'][wi],raw_union_recall=rawcoverage,
                    union_iou=union_iou,union_sources=union_sources,best_admitted_mask_crop_visibility=visible,
                    changed_vs_original=bool(oh[t]),matched_by_original=bool(oh[t]),near_threshold=bv>=.4,area=ga,
                    raw_source_label='raw #'+str(source),raw_panel_title='Best RAW target candidate',note=note,
                    _raw_rle=None if ri is None else r['masks'][ri]))
            for j0 in np.flatnonzero(~(m>.5).any(axis=0)):
                j=int(j0);bv=best(m[:,j]);t=int(m[:,j].argmax()) if bv else None
                other=max((best(v[:,j]) for k,v in matrices.items() if k!=rid),default=0.)
                identity=(rid,tuple(groups[j]))
                if identity in oldfp:cid=oldfp[identity]
                else:cid=f'FP{nextfp:03d}';nextfp+=1
                c=dict(id=cid,type='FP',record=rid,file_name=fn,target_index=t,final_index=j,parents=groups[j],
                    source_id=str(pred['source_proposal_ids'][j]),best_iou=bv,confidence=pred['confidence'][j],
                    other_annotator_best_iou=other,raw_best_iou=best(parentm[:,j]),coarse_best_iou=0.,
                    coarse_parent_iou=best(coarsem[:,j]),pixel_precision=0. if t is None else float(precision[t,j]),
                    pixel_recall=0. if t is None else float(recall[t,j]),
                    substantial_reference_count=int(((precision[:,j]>=.1)&(recall[:,j]>=.1)).sum()),
                    overlaps_matched_reference=bool(((m[:,j]>0)&h).any()),changed_vs_original=identity not in oldfp,
                    no_overlap_any_record=bool(bv==0 and other==0),near_threshold=bv>=.4,
                    raw_source_label='raw parents #'+str(pred['source_proposal_ids'][j]),raw_panel_title='RAW parent mask(s)',
                    note='Scored independently against this annotation record. No annotation overlap is not proof of physical background.',
                    _raw_rle=rawparents[j])
                c['category']=primary(c);local.append(c)
            for t0 in np.flatnonzero(h&~oh):
                t=int(t0);j=int(m[t].argmax());ri=int(rm[t].argmax())
                local.append(dict(id=f'GAIN{sum(c["type"]=="Recovered" for c in cases)+sum(c["type"]=="Recovered" for c in local)+1:03d}',
                    type='Recovered',category='Recovered vs original reference',record=rid,file_name=fn,target_index=t,final_index=j,
                    source_id=str(pred['source_proposal_ids'][j]),best_iou=best(m[t]),changed_vs_original=True,
                    raw_source_label='raw #'+str(r['source_proposal_ids'][ri]),raw_panel_title='Best RAW target candidate',
                    note='Successful new match, not an additional FP/FN error.',_raw_rle=r['masks'][ri]))
            others=[(mask,f'A{a+1}:G{k}') for a,otherrec in enumerate([x for x in records[fn] if x['record']!=rid]) for k,mask in enumerate(decoded_gt[otherrec['record']])]
            for c in local:
                raw_rle=c.pop('_raw_rle');rawmask=None if raw_rle is None else decode_mask(raw_rle).astype(bool)
                c['sheet']=f'sheets/{c["id"]}.jpg';render_case(c,image,decoded_gt[rid],others,rawmask,pred,OUT/c['sheet']);cases.append(c)
        write_json(OUT/'progress.json',dict(stage='classify_and_render',images=number,total=146,cases=len(cases)))
        if number%25==0:print(f'Pass6audit {number}/146 images; {len(cases)} cases',flush=True)
    assert (total.tp,total.fp,total.fn)==(962,458,560) and abs(total.pq-result['stages']['full']['pq'])<1e-12
    assert abs(float(official_functions()['get_pq_score'](pd.DataFrame(tables)))-total.pq)<1e-10
    assert Counter(c['type'] for c in cases)==dict(FN=560,FP=458,Recovered=12)
    assert len({c['id'] for c in cases})==len(cases)
    assert paired['candidate_only']==12 and paired['original_only']==6
    assert all(sha(WORKSPACE/p)==h for p,h in hashes.items()) and sha(reference_path)==refhash
    summary=dict(**asdict(total),pq=total.pq,errors=1018,comparison_recovered=12,rendered_cases=len(cases),paired=dict(paired),
        categories={kind:dict(Counter(c['category'] for c in cases if c['type']==kind)) for kind in ['FN','FP']},
        original_categories=oldsummary['categories'],
        fn_flags=dict(near_threshold=sum(c['type']=='FN' and c['near_threshold'] for c in cases),
            no_single_admitted_crop_covers90pct=sum(c['type']=='FN' and c['best_admitted_mask_crop_visibility']<.9 for c in cases)),
        fp_flags=dict(other_annotator_supported=sum(c['type']=='FP' and c['other_annotator_best_iou']>.5 for c in cases),
            no_overlap_any_record=sum(c['type']=='FP' and c['no_overlap_any_record'] for c in cases),
            near_threshold=sum(c['type']=='FP' and c['near_threshold'] for c in cases)),
        organizer_parity=True,disjoint=True,production_unchanged=True,no_training_or_new_inference=True,
        caveat='Primary categories have precedence; overlapping flags must not be added to them. Grouping unions and stage matches are annotation-guided diagnostics only.')
    write_json(OUT/'cases.json',cases);write_json(OUT/'summary.json',summary);write_json(OUT/'changes_vs_original.json',changes);write_page(cases)
    write_json(OUT/'audit_complete.json',dict(stage='final_review_ready',case_count=len(cases)))
    print(json.dumps(summary,indent=2))


def trace():
    torch.set_num_threads(2);cases=read_json(OUT/'cases.json');(OUT/'history').mkdir(exist_ok=True)
    selected=[c for c in cases if c['type']=='FN' and c['raw_best_iou']>.5 and 'history' not in c]
    byimage=defaultdict(list)
    for c in selected:byimage[c['file_name']].append(c)
    raw=cache(VALIDATION/'raw_predictions');final=cache(VALIDATION/'predictions')
    records={r['record']:r for r in read_json(HERE/'outputs/fp_progress_three_options_20261028/validation_records.json')}
    reader=FilamentDataset(DEFAULT_PROJECT,'val')
    for number,(fn,rows) in enumerate(byimage.items(),1):
        stages,log=replay(raw[fn])
        assert all(stages['final'][k]==final[fn][k] for k in ['masks','confidence','source_proposal_ids','source_groups'])
        with Image.open(reader.image_dir/fn) as im:image=im.convert('RGB')
        for c in rows:
            gt=records[c['record']]['gt'][c['target_index']];values=rle_iou([gt],raw[fn]['masks'])[0];qualified=[]
            for i0 in np.flatnonzero(values>.5):
                i=int(i0);source=raw[fn]['source_proposal_ids'][i];decision=copy.deepcopy(next(d for d in log['candidates'] if d['source_id']==source))
                timeline=[dict(stage=s,label=LABELS[s],**candidate_stage(gt,stages[s],source)) for s in STAGES]
                decision['merge_partners']=[]
                for merge in log['chosen_merges']:
                    if source in merge['sources']:
                        for partner in merge['sources']:
                            if partner!=source:decision['merge_partners'].append(dict(source_id=partner,confidence=raw[fn]['confidence'][raw[fn]['source_proposal_ids'].index(partner)]))
                qualified.append(dict(source_id=source,confidence=raw[fn]['confidence'][i],raw_iou=float(values[i]),decision=decision,
                    timeline=timeline,first_loss_stage=next(s['stage'] for s in timeline if s['candidate_iou']<=.5)))
            qualified.sort(key=lambda x:-x['raw_iou'])
            history=dict(id=c['id'],record=c['record'],file_name=fn,target_index=c['target_index'],final_best_iou=c['best_iou'],
                final_best_index=c['final_index'],candidate=qualified[0],all_qualifying_candidates=qualified,merges=log['chosen_merges'],sheet=f'history/{c["id"]}.jpg')
            render_history(history,raw[fn],final[fn],gt,image,OUT/history['sheet']);c['history']=history
        write_json(OUT/'cases.json',cases)
        write_json(OUT/'progress.json',dict(stage='trace_cleanup',images=number,total=len(byimage),cases_completed=sum('history' in c for c in cases)))
        if number%10==0:print(f'Pass6cleanup history {number}/{len(byimage)} images',flush=True)
    write_page(cases)
    histories=[c for c in cases if 'history' in c]
    summary=dict(cases=len(histories),first_loss=dict(Counter(c['history']['candidate']['first_loss_stage'] for c in histories)),
        unique_best_raw_candidates=len({(c['file_name'],c['history']['candidate']['source_id']) for c in histories}),exact_final_replay=True)
    write_json(OUT/'history_summary.json',summary);write_json(OUT/'progress.json',dict(stage='complete',errors=1018,history_cases=len(histories)))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--trace',action='store_true');args=parser.parse_args()
    trace() if args.trace else analyze()
