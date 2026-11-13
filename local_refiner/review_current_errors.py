"""Stratified visual review of the official 0.35 confidence-first pipeline."""
from collections import Counter
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from refiner import HERE, read_json, write_json, decode_mask, sha
from validation_pq import ValidationPQ, rle_iou
from build_visual_review import bbox, view_box, cache

OUT = HERE / 'outputs/visual_review_20261019/current_errors_20261027'
RAW = HERE / 'outputs/confidence_sweep_basic_unet_20261026/refined_candidates'
FINAL = HERE / 'outputs/selection_policies_20261026/global_cleanup_020/predictions'
LOW = HERE / 'outputs/low_confidence_candidates_20261019/predictions'


def subset(row, low, high=1.01):
    ids = [i for i,c in enumerate(row['confidence']) if low <= c < high]
    sources = row.get('source_proposal_ids', row.get('proposal_ids', list(range(len(row['masks'])))))
    return dict(masks=[row['masks'][i] for i in ids], confidence=[row['confidence'][i] for i in ids],
                source_proposal_ids=[sources[i] for i in ids])


def render(case, image, gt, coarse, raw, final, weak, path):
    focus = gt[case['target']] if case['kind']=='FN' else final['masks'][case['prediction']]
    crop = view_box(bbox(decode_mask(focus)), margin=110)
    side=400; top=105; sheet=Image.new('RGB',(3*side,top+2*(side+30)),'#f4f6f8')
    draw=ImageDraw.Draw(sheet); font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',14)
    label=ImageFont.truetype('C:/Windows/Fonts/segoeuib.ttf',15)
    draw.text((8,6),f"{case['group']} | {case['record']} | {case['kind']} | best final IoU {case['iou']:.3f}",font=font,fill='black')
    draw.text((8,30),'Yellow: selected reference (GT panel only). Cyan: focus prediction. Blue: other predictions.',font=font,fill='black')
    draw.text((8,54),'All panels use the same crop. #IDs refer to original YOLO proposals; sixth panel is diagnostic only.',font=font,fill='black')
    rows=[None,dict(masks=gt,source_proposal_ids=list(range(1,len(gt)+1))),coarse,raw,final,weak]
    titles=['Original image','Independent annotation reference','YOLO masks: confidence >= 0.20',
            'U-Net masks before cleanup','Current final masks: confidence-first','Lower-confidence U-Net masks: 0.05–0.20']
    visible={}
    for i,(row,title) in enumerate(zip(rows,titles)):
        cut=image.crop(crop).convert('RGBA');layer=np.zeros((crop[3]-crop[1],crop[2]-crop[0],4),np.uint8);tags=[]
        if row:
            order=sorted(range(len(row['masks'])),key=lambda j: (j==case.get('target')) if i==1 else row['source_proposal_ids'][j]==case['source'])
            for j in order:
                source,rle=row['source_proposal_ids'][j],row['masks'][j]
                mask=decode_mask(rle).astype(bool)[crop[1]:crop[3],crop[0]:crop[2]]
                if not mask.any():continue
                selected=(j==case.get('target')) if i==1 else source==case['source']
                color=(255,220,20) if i==1 and selected else ((0,225,240) if selected else (100,155,255))
                layer[mask]=[*color,140]
                yy,xx=np.nonzero(mask);tags.append((source,int(xx.min()),int(yy.min()),color))
                if i!=1:
                    entry=visible.setdefault(source,dict(source=source,confidence=row['confidence'][j],panels=[]))
                    entry['panels'].append(title)
        cut.alpha_composite(Image.fromarray(layer));cut=cut.convert('RGB').resize((side,side),Image.Resampling.LANCZOS)
        cd=ImageDraw.Draw(cut)
        for source,x,y,color in tags:
            text=f'#{source}';w=int(cd.textlength(text,font=label))+8
            x=max(1,min(side-w-1,round(x*side/(crop[2]-crop[0]))));y=max(1,min(side-23,round(y*side/(crop[3]-crop[1]))-22))
            cd.rectangle((x,y,x+w,y+22),fill='#15212d');cd.text((x+4,y),text,font=label,fill=color)
        x=(i%3)*side;y=top+(i//3)*(side+30)
        draw.text((x+5,y-23),title,font=font,fill='black');sheet.paste(cut,(x,y))
    sheet.save(path,quality=91)
    case['crop']=crop;case['visible_candidates']=sorted(visible.values(),key=lambda r:-r['confidence'])


def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True);(OUT/'sheets').mkdir(exist_ok=True)
    raw,final,low=cache(RAW),cache(FINAL),cache(LOW);v=ValidationPQ();errors=[];controls=[];totals=Counter()
    for filename,records in v.records.items():
        r=subset(raw[filename],.20);f=final[filename];w=subset(raw[filename],.05,.20)
        for rec in records:
            gt=rec['gt'];m=rle_iou(gt,f['masks']);mr=rle_iou(gt,r['masks']);mw=rle_iou(gt,w['masks'])
            hit=(m>.5).any(axis=1);pred_hit=(m>.5).any(axis=0)
            totals.update(TP=int(hit.sum()),FN=int((~hit).sum()),FP=int((~pred_hit).sum()))
            for t in np.flatnonzero(~hit):
                t=int(t);best=float(m[t].max(initial=0));rawbest=float(mr[t].max(initial=0));weakbest=float(mw[t].max(initial=0))
                group=('FN: lost during cleanup' if rawbest>.5 else
                       'FN: qualifying lower-confidence mask' if weakbest>.5 else
                       'FN: partial final overlap' if best>0 else 'FN: no final overlap')
                source=(r['source_proposal_ids'][int(mr[t].argmax())] if rawbest>.5 else
                        w['source_proposal_ids'][int(mw[t].argmax())] if weakbest>.5 else
                        f['source_proposal_ids'][int(m[t].argmax())] if best else
                        r['source_proposal_ids'][int(mr[t].argmax())] if rawbest else None)
                area=int(decode_mask(gt[t]).sum())
                errors.append(dict(id=f"{rec['image_id']}_gt{t}",kind='FN',group=group,record=rec['image_id'],
                    file_name=filename,target=t,prediction=None,source=source,iou=best,raw_iou=rawbest,weak_iou=weakbest,area=area))
            for p in np.flatnonzero(~pred_hit):
                p=int(p);scores=m[:,p];best=float(scores.max(initial=0));overlap_count=int((scores>0).sum())
                group=('FP: no annotation overlap' if best==0 else 'FP: overlaps multiple references' if overlap_count>1 else 'FP: partial single-reference mask')
                errors.append(dict(id=f"{rec['image_id']}_pred{f['source_proposal_ids'][p]}",kind='FP',group=group,
                    record=rec['image_id'],file_name=filename,target=int(scores.argmax()) if best else None,
                    prediction=p,source=f['source_proposal_ids'][p],iou=best,overlapping_references=overlap_count,
                    area=int(decode_mask(f['masks'][p]).sum())))
            for t in np.flatnonzero(hit):
                t=int(t);p=int(m[t].argmax());controls.append(dict(id=f"{rec['image_id']}_control{t}",kind='TP',group='TP: matched controls',
                    record=rec['image_id'],file_name=filename,target=t,prediction=p,source=f['source_proposal_ids'][p],iou=float(m[t,p]),area=int(decode_mask(gt[t]).sum())))
    assert dict(totals)==dict(TP=945,FN=577,FP=520),totals
    counts=Counter(e['group'] for e in errors);counts['TP: matched controls']=len(controls)
    chosen=[]
    for group in counts:
        pool=sorted([e for e in errors+controls if e['group']==group],key=lambda c:(c['area'],c['id']))
        n=min(8 if group.startswith('TP:') else 12,len(pool))
        chosen.extend(pool[i] for i in sorted(set(np.linspace(0,len(pool)-1,n,dtype=int))))
    for i,c in enumerate(chosen):
        rec=next(r for r in v.records[c['file_name']] if r['image_id']==c['record'])
        with Image.open(v.reader.image_dir/c['file_name']) as im:image=im.convert('RGB')
        c['sheet']=f'sheets/case_{i:03d}.jpg'
        render(c,image,rec['gt'],subset(low[c['file_name']],.20),subset(raw[c['file_name']],.20),final[c['file_name']],subset(raw[c['file_name']],.05,.20),OUT/c['sheet'])
        if (i+1)%15==0:print(f'Rendered {i+1}/{len(chosen)}',flush=True)
    for c in chosen:
        with Image.open(OUT/c['sheet']) as im:im.verify()
    summary=dict(totals=totals,groups=counts,sampled=Counter(c['group'] for c in chosen),
        images=146,records=234,local_pq=.43123711722488506,official_displayed_score=.35,
        script_sha256=sha(__file__),all_sheets_verified=True,no_training=True,no_new_inference=True,
        note='Groups are annotation-specific geometry, not confirmed physical error causes. Multiple-reference overlap does not prove a merge. Area-stratified samples are not prevalence estimates.')
    write_json(OUT/'all_errors.json',errors);write_json(OUT/'cases.json',chosen);write_json(OUT/'summary.json',summary)
    (OUT/'data.js').write_text('const DATA='+json.dumps(dict(cases=chosen,summary=summary))+';',encoding='utf-8')
    (OUT/'index.html').write_text(browser_html(),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


HTML='''<!doctype html><meta charset="utf-8"><title>Current pipeline error review — official 0.35</title>
<style>body{font:16px system-ui;background:#f3f5f8;color:#173044;margin:22px}main{max-width:1260px;margin:auto}select,button,textarea{font:inherit;padding:9px;margin:5px}img{width:100%;height:auto}#detail,.box{background:white;padding:15px;line-height:1.6}textarea{width:96%;min-height:85px}td,th{padding:6px 14px;text-align:left;border-bottom:1px solid #ddd}h1{margin-bottom:8px}</style><main>
<h1>Review the current 0.35 pipeline</h1><p>Basic U-Net + YOLO11s, cutoff 0.20, confidence-first cleanup. Local PQ 0.43124: TP 945 · FP 520 · FN 577. Official displayed score: 0.35.</p>
<p>Review an area-stratified sample of remaining errors and matched controls. Reference masks appear only in the annotation panel. No models or predictions were changed.</p>
<label>Group <select id="group"></select></label><button id="prev">← Previous</button><select id="picker" aria-label="Review case"></select><button id="next">Next →</button>
<div id="detail"></div><img id="sheet" alt="Six aligned panels: original, independent reference, YOLO masks, refined masks, final masks, lower-confidence diagnostic masks">
<details><summary>Visible candidate IDs and confidence</summary><table><thead><tr><th>ID</th><th>YOLO confidence</th><th>Appears in</th></tr></thead><tbody id="candidates"></tbody></table></details>
<h2>Your observations</h2><p>Describe what you see: faint structure, disconnected pieces, wrong grouping, incomplete boundary, duplicate, or an uncertain annotation. These are hypotheses to check together.</p>
<textarea id="note" aria-label="Your observations"></textarea><button id="export">Export notes</button>
<p class="box">FN means no final mask exceeds IoU 0.5 for that annotation. FP means a final mask fails to match under that annotation record. An FP may be a fragment, a poor boundary, or supported by another annotator. Overlapping several references does not by itself prove an incorrect merge. The last panel shows refined proposals below our cutoff; it is not submitted output.</p>
<p><a href="all_errors.json">Full error metadata</a> · <a href="summary.json">Audit and group counts</a></p></main>
<script src="data.js"></script><script>const $=id=>document.getElementById(id),key='solar-current-conf-cleanup-review-20261027';let notes=JSON.parse(localStorage.getItem(key)||'{}'),items=[],current;
for(const [g,n] of Object.entries(DATA.summary.groups)){let o=document.createElement('option');o.value=g;o.textContent=g+' ('+n+' total)';$('group').append(o)}
function show(){current=items[Number($('picker').value)];const c=current;if(!c)return;$('sheet').src=c.sheet;$('note').value=notes[c.id]||'';$('detail').textContent=c.record+' | '+(c.target===null?'No overlapping reference':'Reference #'+(c.target+1))+' | focus proposal '+(c.source===null?'none':'#'+c.source)+' | final IoU '+c.iou.toFixed(3)+(c.kind==='FN'?' | before-cleanup IoU '+c.raw_iou.toFixed(3)+' | best lower-confidence IoU '+c.weak_iou.toFixed(3):'')+' | '+items.length+' sampled from '+DATA.summary.groups[c.group];$('candidates').replaceChildren(...c.visible_candidates.map(v=>{let tr=document.createElement('tr');for(const t of ['#'+v.source,v.confidence.toFixed(3),v.panels.join('; ')]){let td=document.createElement('td');td.textContent=t;tr.append(td)}return tr}));$('prev').disabled=Number($('picker').value)===0;$('next').disabled=Number($('picker').value)===items.length-1;}
function filter(){items=DATA.cases.filter(c=>c.group===$('group').value);$('picker').replaceChildren(...items.map((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=(i+1)+'/'+items.length+' · '+c.record;return o}));show()}
$('group').onchange=filter;$('picker').onchange=show;for(const [id,step] of [['prev',-1],['next',1]])$(id).onclick=()=>{$('picker').value=Number($('picker').value)+step;show()};$('note').oninput=()=>{if(current){notes[current.id]=$('note').value;localStorage.setItem(key,JSON.stringify(notes))}};$('export').onclick=()=>{let a=document.createElement('a'),u=URL.createObjectURL(new Blob([JSON.stringify({schema:'current-cleanup-review-v1',notes},null,2)],{type:'application/json'}));a.href=u;a.download='current_cleanup_review_notes.json';a.click();URL.revokeObjectURL(u)};filter();</script>'''


def browser_html():
    return HTML.replace('src="data.js"', 'src="data.js?v=2"').replace("src=c.sheet;", "src=c.sheet+'?v=2';")


if __name__=='__main__':main()
