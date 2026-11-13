"""Audit all 520 FP annotation/prediction pairs in the official-0.35 pipeline."""
from collections import Counter, defaultdict
import csv
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools import mask as mu
import torch
from refiner import HERE, WORKSPACE, read_json, write_json, decode_mask, sha
from validation_pq import ValidationPQ, rle_iou
from build_visual_review import cache, bbox, view_box
from review_current_errors import RAW, FINAL, LOW

OUT=HERE/'outputs/visual_review_20261019/fp_audit_20261028'


def primary(c):
    if c['other_annotator_best_iou']>.5:return 'Matches another annotator'
    if c['raw_best_iou']>.5:return 'Qualifying raw mask damaged by cleanup'
    if c['coarse_best_iou']>.5:return 'Qualifying YOLO mask lost during refinement'
    if c['substantial_reference_count']>=2:return 'Possible merge: substantial overlap with multiple references'
    if c['overlaps_matched_reference']:return 'Extra piece around an already-matched reference'
    if c['best_iou']>=.4:return 'Near matching threshold (IoU 0.40–0.50)'
    if c['best_iou']==0:return 'No overlap with this annotation'
    if c['pixel_precision']>=.7 and c['pixel_recall']<.5:return 'Mostly inside reference, incomplete coverage'
    if c['pixel_recall']>=.7 and c['pixel_precision']<.5:return 'Covers reference with substantial excess'
    return 'Other partial or inaccurate overlap'


def fractions(matrix,gt_areas,pred_areas):
    intersections=matrix*(gt_areas[:,None]+pred_areas[None,:])/(1+matrix)
    return intersections/pred_areas[None,:],intersections/gt_areas[:,None]


def render(c,image,gt,other_gt,coarse,raw,final,path):
    t=c['best_target_index'];focus=decode_mask(final['masks'][c['final_index']]).astype(bool)
    scope=focus.copy()
    if t is not None:scope|=gt[t]
    crop=view_box(bbox(scope),margin=96);side=360;top=125;gap=32
    sheet=Image.new('RGB',(side*3,top+2*(side+gap)),'#f4f6f8');draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',14);bold=ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf',14)
    headers=[f"{c['record']} | FP candidate #{c['source_id']} | conf {c['confidence']:.3f}",c['category'],
        f"Best IoU: YOLO {c['coarse_best_iou']:.3f} -> raw U-Net {c['raw_best_iou']:.3f} -> final {c['best_iou']:.3f}; retained {100*c['retained_fraction']:.1f}%",
        'Yellow = chosen reference (reference panels only). Cyan = focus prediction. Blue = other final predictions.']
    for j,line in enumerate(headers):draw.text((8,4+23*j),line,font=font,fill='black')
    base=image.crop(crop).convert('RGBA')
    titles=['Original image','This record: annotation instances','Original YOLO candidate',
            'U-Net before cleanup','Final masks: focus + nearby outputs','Other annotator: '+(c['other_annotator_record'] or 'not available')]
    visible_ids=[]
    for i,title in enumerate(titles):
        layer=np.zeros((crop[3]-crop[1],crop[2]-crop[0],4),np.uint8);tags=[]
        if i in [1,5]:
            selected=gt if i==1 else other_gt;target=t if i==1 else c['other_annotator_target']
            order=sorted(range(len(selected)),key=lambda j:j==target)
            for j in order:
                m=selected[j][crop[1]:crop[3],crop[0]:crop[2]]
                if not m.any():continue
                color=(255,220,20) if j==target else (100,155,255)
                layer[m]=[*color,145];yy,xx=np.nonzero(m);tags.append((str(j+1),int(xx.min()),int(yy.min()),color))
        elif i in [2,3]:
            mask=coarse if i==2 else raw;m=mask[crop[1]:crop[3],crop[0]:crop[2]]
            layer[m]=[0,220,240,145]
        elif i==4:
            order=sorted(range(len(final['masks'])),key=lambda j:j==c['final_index'])
            for j in order:
                m=decode_mask(final['masks'][j]).astype(bool)[crop[1]:crop[3],crop[0]:crop[2]]
                if not m.any():continue
                source=final['source_proposal_ids'][j];color=(0,220,240) if j==c['final_index'] else (100,155,255)
                layer[m]=[*color,145];yy,xx=np.nonzero(m);tags.append(('#'+str(source),int(xx.min()),int(yy.min()),color));visible_ids.append(source)
        panel=base.copy();panel.alpha_composite(Image.fromarray(layer));panel=panel.convert('RGB').resize((side,side),Image.Resampling.LANCZOS);pd=ImageDraw.Draw(panel)
        for text,x,y,color in tags:
            w=int(pd.textlength(text,font=bold))+8;x=max(1,min(side-w-1,round(x*side/(crop[2]-crop[0]))));y=max(1,min(side-23,round(y*side/(crop[3]-crop[1]))-23))
            pd.rectangle((x,y,x+w,y+22),fill='#15212d');pd.text((x+4,y),text,font=bold,fill=color)
        x=(i%3)*side;y=top+(i//3)*(side+gap);draw.text((x+3,y-22),title,font=font,fill='black');sheet.paste(panel,(x,y))
    sheet.save(path,quality=90)
    c['crop']=crop;c['visible_final_source_ids']=visible_ids


def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True);(OUT/'sheets').mkdir(exist_ok=True)
    v=ValidationPQ();raw=cache(RAW);final=cache(FINAL);coarse=cache(LOW)
    sources={str(p.relative_to(WORKSPACE)):sha(p) for folder in [RAW,FINAL,LOW] for p in folder.glob('*.json')}
    reference_sha=sha(HERE/'working_reference.json');cases=[];physical=defaultdict(list);total_tp=total_fn=0
    for number,(filename,records) in enumerate(v.records.items()):
        f=final[filename];r=raw[filename];co=coarse[filename]
        ids={s:i for i,s in enumerate(r['source_proposal_ids'])}
        final_areas=mu.area(f['masks']).astype(float) if f['masks'] else np.array([])
        matrices={rec['image_id']:rle_iou(rec['gt'],f['masks']) for rec in records}
        decoded_gt={rec['image_id']:[decode_mask(g).astype(bool) for g in rec['gt']] for rec in records}
        with Image.open(v.reader.image_dir/filename) as im:image=im.convert('RGB')
        for rec in records:
            gt=rec['gt'];matrix=matrices[rec['image_id']];hits=(matrix>.5).any(axis=1)
            total_tp+=int(hits.sum());total_fn+=int((~hits).sum())
            gt_areas=mu.area(gt).astype(float) if gt else np.array([])
            precision,recall=fractions(matrix,gt_areas,final_areas)
            rm=rle_iou(gt,r['masks']);cm=rle_iou(gt,co['masks'])
            for j in np.flatnonzero(~(matrix>.5).any(axis=0)):
                j=int(j);source=f['source_proposal_ids'][j];ri=ids[source];scores=matrix[:,j]
                assert f['confidence'][j]==r['confidence'][ri]==co['confidence'][source]
                best=float(scores.max(initial=0));target=int(scores.argmax()) if best>0 else None
                other_options=[(float(m[:,j].max(initial=0)),rid,int(m[:,j].argmax()) if m.shape[0] else None)
                    for rid,m in matrices.items() if rid!=rec['image_id']]
                other_iou,other_record,other_target=max(other_options,key=lambda a:a[0]) if other_options else (0.,None,None)
                if other_iou==0:other_target=None
                retained=float(final_areas[j]/mu.area(r['masks'][ri]))
                mask=decode_mask(f['masks'][j]).astype(bool);x,y,w,h=[int(a) for a in mu.toBbox(f['masks'][j])]
                count,_,stats,_=cv2.connectedComponentsWithStats(mask[y:y+h,x:x+w].astype(np.uint8),connectivity=8)
                substantial=(precision[:,j]>=.1)&(recall[:,j]>=.1)
                p=float(precision[target,j]) if target is not None else 0.;rr=float(recall[target,j]) if target is not None else 0.
                ident=f'{rec["image_id"]}_pred{source}'
                c=dict(id=ident,physical_id=f'{filename}#{source}',record=rec['image_id'],file_name=filename,source_id=source,final_index=j,
                    confidence=f['confidence'][j],best_iou=best,best_target_index=target,
                    pixel_precision=p,pixel_recall=rr,coarse_best_iou=float(cm[:,source].max(initial=0)),raw_best_iou=float(rm[:,ri].max(initial=0)),
                    retained_fraction=retained,foreground_pixels=int(final_areas[j]),overlapping_reference_count=int((scores>0).sum()),
                    substantial_reference_count=int(substantial.sum()),substantial_reference_indices=np.flatnonzero(substantial).tolist(),
                    overlaps_matched_reference=bool(((scores>0)&hits).any()),best_reference_already_matched=bool(hits[target]) if target is not None else False,
                    other_annotator_best_iou=other_iou,other_annotator_record=other_record,other_annotator_target=other_target,
                    connected_components=count-1,largest_component_fraction=float(stats[1:,cv2.CC_STAT_AREA].max()/final_areas[j]),
                    sheet=f'sheets/fp_{len(cases):03d}.jpg',visual_review_status='unreviewed')
                c['category']=primary(c)
                c['flags']=[name for name,flag in {
                    'Matches another annotator':other_iou>.5,'Cleanup damaged qualifying mask':c['raw_best_iou']>.5,
                    'Refinement lost qualifying YOLO mask':c['coarse_best_iou']>.5 and c['raw_best_iou']<=.5,
                    'Overlaps multiple references substantially':c['substantial_reference_count']>=2,
                    'Touches already-matched reference':c['overlaps_matched_reference'],
                    'Near threshold':best>=.4,'No annotation overlap':best==0,
                    'High precision / low coverage':p>=.7 and rr<.5,
                    'High coverage / excess foreground':rr>=.7 and p<.5,
                    'Lost at least half of raw area':retained<.5,
                    'Disconnected final mask':count>2}.items() if flag]
                render(c,image,decoded_gt[rec['image_id']],decoded_gt.get(other_record,[]),
                    decode_mask(co['masks'][source]).astype(bool),decode_mask(r['masks'][ri]).astype(bool),f,OUT/c['sheet'])
                with Image.open(OUT/c['sheet']) as im:im.verify()
                cases.append(c);physical[c['physical_id']].append(ident)
        if (number+1)%15==0:print(f'FP audit {number+1}/146 images; {len(cases)} FP records rendered',flush=True)
    assert len(cases)==520 and total_tp==945 and total_fn==577
    assert len({c['id'] for c in cases})==520
    for c in cases:c['same_physical_prediction_fp_records']=physical[c['physical_id']]
    counts=Counter(c['category'] for c in cases);flags=Counter(flag for c in cases for flag in c['flags'])
    assert sum(counts.values())==520
    summary=dict(pipeline='basic U-Net + confidence-first global_cleanup_020; official displayed score 0.35',
        tp=945,fp=520,fn=577,physical_images=146,independent_records=234,
        unique_physical_predictions_with_at_least_one_fp=len(physical),primary_categories=counts,overlapping_flags=flags,
        confidence_bins=Counter('0.20–0.25' if c['confidence']<.25 else '0.25–0.50' if c['confidence']<.5 else '0.50–0.75' if c['confidence']<.75 else '0.75–1.00' for c in cases),
        all_520_sheets_verified=True,no_training=True,no_new_inference=True,models_and_reference_unchanged=True,
        limitation='Primary categories use a precedence rule to sum to 520; flags overlap. Geometric labels are hypotheses, not completed visual classifications. No overlap does not prove background; multiple overlaps do not prove a merge. Counts are annotation-record pairs, not unique anatomical errors.')
    assert all(sha(WORKSPACE/p)==h for p,h in sources.items()) and sha(HERE/'working_reference.json')==reference_sha
    write_json(OUT/'plan.json',dict(script_sha256=sha(__file__),source_hashes=sources,reference_sha256=reference_sha,
        primary_precedence=['other annotator','cleanup damage','refinement damage','possible merge','already matched reference','near threshold','zero overlap','incomplete coverage','excess foreground','other partial'],
        definitions=dict(substantial_overlap='>=10% of predicted mask AND >=10% of reference',incomplete='>=70% prediction precision and <50% reference recall',excess='>=70% reference recall and <50% prediction precision')))
    write_json(OUT/'cases.json',cases);write_json(OUT/'summary.json',summary)
    with (OUT/'cases.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fields=['id','physical_id','category','confidence','best_iou','pixel_precision','pixel_recall','coarse_best_iou','raw_best_iou','retained_fraction','foreground_pixels','other_annotator_best_iou','flags']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows({k:json.dumps(c[k]) if isinstance(c[k],list) else c[k] for k in fields} for c in cases)
    (OUT/'data.js').write_text('const DATA='+json.dumps(dict(cases=cases,summary=summary))+';',encoding='utf-8')
    (OUT/'index.html').write_text(HTML,encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


HTML='''<!doctype html><meta charset="utf-8"><title>All 520 false positives · best basic U-Net pipeline</title>
<style>body{font:16px system-ui;background:#f3f5f8;color:#173044;margin:22px}main{max-width:1200px;margin:auto}select,button,textarea{font:inherit;padding:8px;margin:5px}img{width:100%;height:auto}#detail,.box{background:white;padding:14px;line-height:1.6}textarea{width:96%;min-height:75px}#tags label{display:inline-block;margin:8px}a{color:#006580}table{border-collapse:collapse}td,th{padding:7px;border:1px solid #ccc}</style><main>
<h1>Audit all 520 false positives</h1><p>Frozen best basic U-Net + confidence-first cleanup, cutoff 0.20. Local PQ 0.43124; TP 945 / FP 520 / FN 577. Official displayed score 0.35. Attention U-Net excluded.</p>
<p>Each entry is one unmatched prediction under one independent annotator. The same physical prediction may appear several times. Measured groups are starting hypotheses; your visual tags are stored separately.</p><details><summary>Category counts and definitions</summary><div id="counts"></div><p>Substantial multi-reference overlap requires at least 10% of prediction AND reference pixels. Near threshold: IoU 0.40–0.50. High precision / low coverage: at least 70% of predicted pixels inside reference but under 50% of reference covered. Primary categories use precedence to sum to 520; flags may overlap.</p></details>
<label>Primary group <select id="group"><option value="All">All 520</option></select></label><label>Flag <select id="flag"><option value="All">Any flag</option></select></label>
<button id="prev">← Previous</button><select id="picker" aria-label="FP case"></select><button id="next">Next →</button><div id="detail"></div><img id="sheet" alt="Original, current references, YOLO proposal, raw U-Net, final masks, other annotator">
<h2>Your visual interpretation</h2><div id="tags"></div><textarea id="note" aria-label="Your observation"></textarea><button id="export">Export review notes</button>
<p class="box">No annotation overlap is not proof of background noise. A prediction overlapping several annotations is not automatically a merge. Another annotator's support is evidence of disagreement, not a reason to rewrite annotations. The sixth panel identifies the alternative record explicitly. No masks or models have been changed.</p><p><a href="cases.csv">All 520 measured entries (CSV)</a> · <a href="summary.json">Audit summary</a> · <a href="../current_errors_20261027/index.html">Current FN/FP review</a></p></main>
<script src="data.js"></script><script>const $=id=>document.getElementById(id),key='solar-best-basic-fp-audit-20261028',labels=['Background detection','Real filament fragment','Boundary mismatch','Merged instances','Duplicate / remnant','Annotator disagreement','Uncertain'];let notes=JSON.parse(localStorage.getItem(key)||'{}'),items=[],current;
for(const [g,n] of Object.entries(DATA.summary.primary_categories)){let o=document.createElement('option');o.value=g;o.textContent=g+' ('+n+')';$('group').append(o)}for(const [g,n] of Object.entries(DATA.summary.overlapping_flags)){let o=document.createElement('option');o.value=g;o.textContent=g+' ('+n+')';$('flag').append(o)}$('counts').textContent=Object.entries(DATA.summary.primary_categories).map(([g,n])=>g+': '+n).join(' · ')+' | Unique physical predictions flagged: '+DATA.summary.unique_physical_predictions_with_at_least_one_fp;
for(const text of labels){let l=document.createElement('label'),b=document.createElement('input');b.type='checkbox';b.value=text;b.onchange=save;l.append(b,document.createTextNode(text));$('tags').append(l)}
function save(){if(!current)return;notes[current.id]={tags:[...$('tags').querySelectorAll('input:checked')].map(x=>x.value),text:$('note').value};localStorage.setItem(key,JSON.stringify(notes))}
function show(){current=items[Number($('picker').value)];if(!current){$('detail').textContent='No cases match both filters.';$('sheet').removeAttribute('src');return}const c=current,n=notes[c.id]||{};$('sheet').src=c.sheet;$('note').value=n.text||'';for(const b of $('tags').querySelectorAll('input'))b.checked=(n.tags||[]).includes(b.value);$('detail').textContent=c.record+' | candidate #'+c.source_id+' | confidence '+c.confidence.toFixed(3)+' | final IoU '+c.best_iou.toFixed(3)+' | pixel precision '+c.pixel_precision.toFixed(3)+' / recall '+c.pixel_recall.toFixed(3)+' | raw IoU '+c.raw_best_iou.toFixed(3)+' | retained '+(100*c.retained_fraction).toFixed(1)+'% | '+c.foreground_pixels+' pixels | other annotator '+(c.other_annotator_record||'none')+' IoU '+c.other_annotator_best_iou.toFixed(3)+' | flags: '+c.flags.join('; ')+' | same physical prediction is FP in '+c.same_physical_prediction_fp_records.length+' records';$('prev').disabled=Number($('picker').value)===0;$('next').disabled=Number($('picker').value)===items.length-1;}
function filter(){items=DATA.cases.filter(c=>($('group').value==='All'||c.category===$('group').value)&&($('flag').value==='All'||c.flags.includes($('flag').value)));$('picker').replaceChildren(...items.map((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=(i+1)+'/'+items.length+' · '+c.record+' · #'+c.source_id;return o}));show()}
$('group').onchange=filter;$('flag').onchange=filter;$('picker').onchange=show;for(const [id,step] of [['prev',-1],['next',1]])$(id).onclick=()=>{$('picker').value=Number($('picker').value)+step;show()};$('note').oninput=save;$('export').onclick=()=>{let a=document.createElement('a'),u=URL.createObjectURL(new Blob([JSON.stringify({schema:'best-basic-fp-review-v1',pipeline:'global_cleanup_020',notes},null,2)],{type:'application/json'}));a.href=u;a.download='best_basic_fp_review.json';a.click();URL.revokeObjectURL(u)};filter();</script>'''


if __name__=='__main__':main()
