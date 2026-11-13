"""Bounded image/group/action CNN pilot; inference never receives annotations."""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import random
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from pycocotools import mask as mu

from common import ROOT, OUT, read_json, write_json, sha, decode_mask
from group_actions import DEST as DATA, candidates, replay, signature, outcome, utility, REFERENCE_PQ
from decision_selector import training_records, stats
from ranker import RANK, predict
from refiner import DEFAULT_PROJECT, FilamentDataset, PQStats
from build_visual_review import cache
from evaluate_merge_recovery import compare
from validation_pq import rle_iou
from sweep_confidence import assert_disjoint
from official_notebook_diagnostics import official_functions

DEST=ROOT/'quality_selection/outputs/group_cnn_20261103'
CHECKS=[0,10,20,40]
THRESHOLDS=[None,0.,.01,.02,.05,.10]
SEED=53
KINDS=['unchanged','prefer','reject','allocate','merge']
SIDE=256
LATEST=ROOT/'local_refiner/outputs/realistic_matched_2500_20261103'


class GroupActionCNN(nn.Module):
    def __init__(self):
        super().__init__(); layers=[]; channels=10
        for width in [16,32,64,64]:
            layers.extend([nn.Conv2d(channels,width,3,2,1,bias=False),
                           nn.GroupNorm(4 if width==16 else 8,width),nn.SiLU()])
            channels=width
        self.encoder=nn.Sequential(*layers,nn.AdaptiveAvgPool2d((4,4)))
        self.head=nn.Sequential(nn.Linear(64*16+13,64),nn.SiLU(),nn.Dropout(.2),nn.Linear(64,1))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self,pixels,geometry):
        return self.head(torch.cat([self.encoder(pixels).flatten(1),geometry],1)).flatten()


def normalize(pixels):
    x=pixels.float()/255.
    return torch.cat([x[:,:1]*2-1,x[:,1:]],1)


def union_masks(rles,shape):
    return decode_mask(mu.merge(rles)).astype(bool) if rles else np.zeros(shape,bool)


def grouped_output(pred,region):
    """Two highest-confidence intersecting instances + union of the rest.

    Separate channels preserve instance grouping even if foreground unions match.
    Ordering uses predictions only; no annotation-dependent instance selection.
    """
    shape=region.shape; relevant=[]
    for i,rle in enumerate(pred['masks']):
        mask=decode_mask(rle).astype(bool)
        if (mask & region).any(): relevant.append((i,mask))
    relevant.sort(key=lambda item:(-pred['confidence'][item[0]],item[0]))
    output=[m for _,m in relevant[:2]]
    while len(output)<2: output.append(np.zeros(shape,bool))
    output.append(np.logical_or.reduce([m for _,m in relevant[2:]]) if len(relevant)>2 else np.zeros(shape,bool))
    return output,len(relevant)


def action_input(image,row,control,alternative,action):
    """Registered image and nine mask channels, from prediction-only inputs."""
    shape=image.shape; members=action['members'] if action else list(range(len(row['masks'])))
    region=union_masks([row['masks'][i] for i in members],shape)
    if not region.any(): region=np.ones(shape,bool)
    ids=action['ids'] if action else []
    raw=[decode_mask(row['masks'][i]).astype(bool) for i in ids[:2]]
    while len(raw)<2: raw.append(np.zeros(shape,bool))
    before,nb=grouped_output(control,region); after,na=grouped_output(alternative,region)
    yy,xx=np.nonzero(region); height,width=shape
    extent=min(max(shape),max(96,int(xx.max()-xx.min()+1),int(yy.max()-yy.min()+1))+64)
    left=int(np.clip((int(xx.min())+int(xx.max())+1-extent)//2,0,max(0,width-extent)))
    top=int(np.clip((int(yy.min())+int(yy.max())+1-extent)//2,0,max(0,height-extent)))
    right,bottom=min(width,left+extent),min(height,top+extent)
    cut=(slice(top,bottom),slice(left,right)); box=[left,top,right,bottom]
    pixels=[cv2.resize(image[cut],(SIDE,SIDE),interpolation=cv2.INTER_LINEAR)]
    for mask in raw+[region]+before+after:
        crop=mask[cut].astype('float32')
        pixels.append(np.rint(cv2.resize(crop,(SIDE,SIDE),interpolation=cv2.INTER_AREA if max(crop.shape)>SIDE else cv2.INTER_NEAREST)*255).astype('uint8'))
    kind='unchanged' if action is None else action['kind']
    conf=[row['confidence'][i] for i in ids[:2]]
    while len(conf)<2: conf.append(0.)
    geometry=np.asarray([float(kind==k) for k in KINDS]+conf+
                        [len(members)/20,nb/10,na/10,np.log1p(raw[0].sum())/10,
                         np.log1p(raw[1].sum())/10,extent/512],dtype='float32')
    return np.stack(pixels),geometry,box


def protected():
    manifests=[ROOT/'quality_selection/current_candidate.json',ROOT/'local_refiner/current_best_pipeline.json',
               LATEST/'selected_confidence_diagnostic.json']
    paths=list(manifests)
    for p in [manifests[0],manifests[2]]:
        m=read_json(p)
        paths.extend(ROOT/m[k]['path'] for k in ['detector','refiner','mask_ranking'])
    return {str(p.relative_to(ROOT)):sha(p) for p in set(paths)}


def check_protected():
    for path,digest in read_json(DEST/'plan.json')['protected'].items(): assert sha(ROOT/path)==digest,path


def prepare():
    DEST.mkdir(parents=True,exist_ok=True)
    if (DEST/'prepared.json').exists(): check_protected(); return
    assert read_json(DATA/'completed.json')['complete']
    plan=dict(hypothesis='Expanded full-output actions and explicit instance partitions allow a small CNN to predict net utility better than old pair swaps.',
              architecture='Scratch10-channel CNN16/32/64/64 stride2 GroupNorm SiLU,4x4 spatial pool,13scalars,64hidden dropout0.2,zero output initialization.',
              channels=['grayscale','rawA','rawB','group_union','control1','control2','control_rest_union','alternative1','alternative2','alternative_rest_union'],
              data='All2299distinct training alternatives plus561unchanged controls; original fit/tune day split. Equal physical-image total fitting weight.',
              target='Per-record mean change in full-image matchedIoU - fixed0.448878046*(TP+0.5FP+0.5FN). Preserve source per-record outcomes.',
              training=dict(seed=SEED,epochs=40,batch=16,lr=.0003,weight_decay=.01,precision='FP32',loss='image-balanced MSE',checks=CHECKS,
                            augmentation='shared flips and90-degree rotations on all10channels',gradient_clip=5),
              selection=dict(checkpoint='Minimum reserved-training-day weighted MSE; epoch0 eligible, earliest tie.',thresholds=THRESHOLDS,
                             policy='Highest predicted action above threshold; at most1group action/image; unchanged utility0 always available; earliest/off tie.'),
              evaluation='Freeze epoch/threshold before validation. Evaluate epoch6MLP and transferred update2500confidence-only separately; no promotion.',
              limitations='Single seed, in-sample upstream predictions, reused validation. Transfer to update2500 was not trained or tuned on that checkpoint.',
              dataset_plan_sha256=sha(DATA/'plan.json'),protected=protected(),
              sources={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'quality_selection/group_actions.py']})
    if (DEST/'plan.json').exists(): assert read_json(DEST/'plan.json')==plan
    else: write_json(DEST/'plan.json',plan)
    reader=FilamentDataset(DEFAULT_PROJECT,'train'); folder=DEST/'crops';folder.mkdir(exist_ok=True)
    index=[]; hashes={}
    for k,path in enumerate(sorted((DATA/'images').glob('*.json'))):
        labeled=read_json(path); row=read_json(ROOT/labeled['raw_path']);control=labeled['alternatives'][0]
        hashes[str(path.relative_to(ROOT))]=sha(path)
        with Image.open(reader.image_dir/row['file_name']) as im: image=np.asarray(im.convert('L'))
        images=[];geometry=[];labels=[]
        for j,alt in enumerate(labeled['alternatives']):
            patch,g,box=action_input(image,row,control['prediction'],alt['prediction'],alt['action'])
            delta=(utility(alt['outcome'],REFERENCE_PQ)-utility(control['outcome'],REFERENCE_PQ))/len(alt['outcome']['records'])
            images.append(patch);geometry.append(g);labels.append(delta)
            index.append(dict(file_name=row['file_name'],partition=labeled['partition'],label_file=path.name,
                              alternative=j,crop=box,action=alt['action'],weight=1/len(labeled['alternatives'])))
        np.savez_compressed(folder/path.with_suffix('.npz').name,images=np.stack(images),geometry=np.stack(geometry),labels=np.asarray(labels,dtype='float32'))
        if (k+1)%50==0: print(f'CNN training crops {k+1}/561',flush=True)
    assert len(index)==2860
    assert not {r['file_name'][:8] for r in index if r['partition']=='fit'} & {r['file_name'][:8] for r in index if r['partition']=='tune'}
    write_json(DEST/'index.json',index);write_json(DEST/'label_hashes.json',hashes)
    write_json(DEST/'prepared.json',dict(examples=len(index),images=561,source_day_split_disjoint=True,
               crop_hashes={str(p.name):sha(p) for p in folder.glob('*.npz')}))


def load_data():
    values=defaultdict(list)
    for path in sorted((DEST/'crops').glob('*.npz')):
        with np.load(path) as data:
            for key in data.files: values[key].append(data[key])
    result={k:np.concatenate(v) for k,v in values.items()}
    index=read_json(DEST/'index.json')
    result['tune']=np.asarray([r['partition']=='tune' for r in index])
    result['weight']=np.asarray([r['weight'] for r in index],dtype='float32')
    return result


def batch_predict(net,images,geometry):
    result=[]; device=next(net.parameters()).device;net.eval()
    with torch.inference_mode():
        for start in range(0,len(images),16):
            result.append(net(normalize(torch.as_tensor(images[start:start+16],device=device)),
                              torch.as_tensor(geometry[start:start+16],device=device)).cpu().numpy())
    return np.concatenate(result) if result else np.zeros(0)


def train():
    if (DEST/'trained.json').exists(): check_protected(); return
    assert not (DEST/'last.pt').exists(),'Interrupted state preserved; inspect before restarting.'
    assert torch.cuda.is_available(),'GPU unavailable; do not silently change the protocol.'
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    calculation=torch.arange(1024,device='cuda',dtype=torch.float32).square().sum()
    assert float(calculation)==357389824.
    data=load_data();fit=~data['tune'];tune=data['tune']
    mean=data['geometry'][fit].mean(0);std=np.maximum(data['geometry'][fit].std(0),1e-3)
    geo=np.clip((data['geometry']-mean)/std,-5,5).astype('float32')
    # Keep the image archive in host memory; only current batches occupy GPU RAM.
    fit_ids=np.flatnonzero(fit);net=GroupActionCNN().cuda();opt=torch.optim.AdamW(net.parameters(),lr=.0003,weight_decay=.01)
    weight=data['weight'][fit]; weight=weight/weight.mean();steps=0;history=[];best=float('inf');start=time.monotonic()
    write_json(DEST/'environment.json',dict(interpreter=__import__('sys').executable,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
        gpu_calculation=float(calculation),parameters=sum(p.numel() for p in net.parameters()),precision='FP32'))
    for epoch in range(41):
        loss_sum=0.
        if epoch:
            net.train()
            for positions in torch.randperm(len(fit_ids)).split(16):
                positions=positions.numpy();ids=fit_ids[positions]
                x=normalize(torch.as_tensor(data['images'][ids],device='cuda'))
                if torch.rand(()).item()<.5: x=x.flip(-1)
                if torch.rand(()).item()<.5: x=x.flip(-2)
                x=torch.rot90(x,int(torch.randint(4,()).item()),dims=(-2,-1))
                g=torch.as_tensor(geo[ids],device='cuda');y=torch.as_tensor(data['labels'][ids],device='cuda')
                w=torch.as_tensor(weight[positions],device='cuda')
                opt.zero_grad(set_to_none=True);loss=((net(x,g)-y).square()*w).mean()
                assert torch.isfinite(loss),'Nonfinite loss; preserve last.pt and stop.'
                loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);opt.step();steps+=1
                loss_sum+=float(loss.detach())*len(ids)/len(fit_ids)
        if epoch in CHECKS:
            values=batch_predict(net,data['images'][tune],geo[tune]);mse=float(np.average((values-data['labels'][tune])**2,weights=data['weight'][tune]))
            fit_values=batch_predict(net,data['images'][fit],geo[fit]);fit_mse=float(np.average((fit_values-data['labels'][fit])**2,weights=data['weight'][fit]))
            state=dict(model={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},mean=mean,std=std,epoch=epoch,optimizer_steps=steps)
            torch.save(state,DEST/f'epoch_{epoch:02d}.pt')
            if mse<best: best=mse;torch.save(state,DEST/'best.pt')
            history.append(dict(epoch=epoch,fit_mse=fit_mse,tune_mse=mse,augmented_loss=loss_sum,optimizer_steps=steps,seconds=time.monotonic()-start))
            write_json(DEST/'history.json',history); print(json.dumps(history[-1]),flush=True)
        state=dict(model={k:v.detach().cpu().clone() for k,v in net.state_dict().items()},mean=mean,std=std,epoch=epoch,
                   optimizer_steps=steps,optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state())
        torch.save(state,DEST/'last.pt');write_json(DEST/'status.json',dict(stage='training',epoch=epoch,cap=40,optimizer_steps=steps))
    selected=torch.load(DEST/'best.pt',map_location='cpu',weights_only=False)
    write_json(DEST/'trained.json',dict(selected_epoch=selected['epoch'],tune_mse=best,actual_optimizer_steps=steps,
                                     fit_examples=int(fit.sum()),peak_gpu_bytes=torch.cuda.max_memory_allocated()))
    check_protected()


def load_model():
    state=torch.load(DEST/'best.pt',map_location='cpu',weights_only=False)
    net=GroupActionCNN().cuda().eval();net.load_state_dict(state['model']);return net,state


def choose():
    if (DEST/'selection.json').exists(): return
    net,state=load_model();data=load_data();index=read_json(DEST/'index.json')
    geo=np.clip((data['geometry']-state['mean'])/state['std'],-5,5).astype('float32')
    values=batch_predict(net,data['images'],geo)
    lookup={(r['label_file'],r['alternative']):float(value) for r,value in zip(index,values)}
    rows=[(p,read_json(p)) for p in sorted((DATA/'images').glob('*.json')) if read_json(p)['partition']=='tune'];results=[]
    for threshold in THRESHOLDS:
        total=PQStats();actions=Counter()
        for path,row in rows:
            j=0
            if threshold is not None and len(row['alternatives'])>1:
                options=[lookup[(path.name,k)] for k in range(1,len(row['alternatives']))];best=int(np.argmax(options))+1
                if options[best-1]>threshold: j=best
            s=row['alternatives'][j]['outcome'];total+=PQStats(**{k:s[k] for k in ['tp','fp','fn','iou_sum']})
            if j: actions[row['alternatives'][j]['action']['kind']]+=1
        results.append(dict(threshold=threshold,**asdict(total),pq=total.pq,actions=dict(actions)))
    best=max(results,key=lambda r:r['pq'])
    write_json(DEST/'selection.json',dict(selected_threshold=best['threshold'],head_epoch=state['epoch'],head_sha256=sha(DEST/'best.pt'),results=results,
                                        no_validation_selection=True))
    print(json.dumps(read_json(DEST/'selection.json'),indent=2),flush=True)


def select_image(row,scores,image,net,state,threshold,control):
    if threshold is None: return control,None
    masks=[decode_mask(m).astype(bool) for m in row['masks']]
    assert signature(replay(row,scores,None,masks))==signature(control)
    _,actions=candidates(row);seen={signature(control)};predictions=[];patches=[];geometry=[];kept=[];boxes=[]
    for action in actions:
        pred=replay(row,scores,action,masks);sig=signature(pred)
        if sig in seen: continue
        seen.add(sig);p,g,box=action_input(image,row,control,pred,action)
        predictions.append(pred);patches.append(p);geometry.append(g);kept.append(action);boxes.append(box)
    if not patches: return control,None
    g=np.clip((np.stack(geometry)-state['mean'])/state['std'],-5,5).astype('float32')
    values=batch_predict(net,np.stack(patches),g);j=int(np.argmax(values))
    if values[j]<=threshold: return control,None
    return predictions[j],dict(action=kept[j],predicted_gain=float(values[j]),crop=boxes[j])


def evaluate():
    if (DEST/'completed.json').exists(): print(json.dumps(read_json(DEST/'summary.json'),indent=2));return
    selection=read_json(DEST/'selection.json');threshold=selection['selected_threshold'];net,state=load_model()
    rank_state=torch.load(RANK/'best.pt',map_location='cpu',weights_only=False)
    records=defaultdict(list)
    for rec in read_json(ROOT/'local_refiner/outputs/fp_progress_three_options_20261028/validation_records.json'):records[rec['file_name']].append(rec)
    assert not {fn[:8] for fn in records}&{fn[:8] for fn in training_records()}
    audit={(c['record'],c['target_index']):c for c in read_json(ROOT/'local_refiner/outputs/pass6_error_audit_20261101/cases.json') if c['type']=='FN'}
    reader=FilamentDataset(DEFAULT_PROJECT,'val');official=official_functions();results={}
    modes=[('epoch6_mlp',OUT/'val',RANK/'nms_only/validation/nms_quality/predictions',True,.4488780461290956),
           ('update2500_confidence',LATEST/'control/update_02500_validation/raw_overlapping_diagnostic',
            LATEST/'control/update_02500_validation/confidence_cleanup',False,.44930662641554414)]
    for name,raw_folder,base_folder,ranked,expected in modes:
        baseline=cache(base_folder);raw_cache=cache(raw_folder)
        raw_paths={read_json(p)['file_name']:p for p in raw_folder.glob('*.json')} if ranked else {}
        assert set(raw_cache)==set(baseline)==set(records)
        totals={n:PQStats() for n in ['baseline','selector']};tables={n:[] for n in totals};changes=[];decisions=[]
        folder=DEST/name/'predictions';folder.mkdir(parents=True,exist_ok=True)
        for k,(fn,row) in enumerate(sorted(raw_cache.items())):
            control=baseline[fn];scores=np.asarray(row['confidence'])
            if ranked:
                path=raw_paths[fn]
                with np.load(path.with_suffix('.npz')) as data: scores=predict(rank_state,data['X'],row['confidence'],.5)
            if threshold is None: pred,decision=control,None
            else:
                with Image.open(reader.image_dir/fn) as im: image=np.asarray(im.convert('L'))
                pred,decision=select_image(row,scores,image,net,state,threshold,control)
            assert_disjoint(pred['masks']);write_json(folder/f'{k:04d}.json',pred)
            if decision: decisions.append(dict(file_name=fn,**decision))
            for c in compare(records[fn],control,pred):
                case=audit.get((c['record'],c['target_index']))
                changes.append(dict(audit_id=case['id'] if case else None,category=case['category'] if case else None,**c))
            for key,p in [('baseline',control),('selector',pred)]:
                for rec in records[fn]:
                    matrix=rle_iou(rec['gt'],p['masks']);totals[key]+=stats([rec],p)
                    tables[key].append(dict(iou_matrix=torch.from_numpy(matrix),n_gt=len(rec['gt']),n_pred=len(p['masks'])))
            if (k+1)%25==0: print(f'{name} validation {k+1}/146',flush=True)
            write_json(DEST/'status.json',dict(stage='validation',mode=name,images=k+1,total=146))
        assert abs(totals['baseline'].pq-expected)<1e-10
        for key,table in tables.items(): assert abs(float(official['get_pq_score'](pd.DataFrame(table)))-totals[key].pq)<1e-10
        reloaded=PQStats()
        for fn,p in cache(folder).items(): assert_disjoint(p['masks']);reloaded+=stats(records[fn],p)
        assert (reloaded.tp,reloaded.fp,reloaded.fn)==(totals['selector'].tp,totals['selector'].fp,totals['selector'].fn)
        assert abs(reloaded.iou_sum-totals['selector'].iou_sum)<1e-10
        result=dict(scores={key:dict(**asdict(s),pq=s.pq) for key,s in totals.items()},decisions=len(decisions),
                    gained=sum(c['group']=='Gained' for c in changes),lost=sum(c['group']=='Lost' for c in changes),
                    original50_new_recoveries=sum(c['group']=='Gained' and c['category']=='Qualifying refined candidate removed' for c in changes),
                    original50_new_losses=sum(c['group']=='Lost' and c['category']=='Qualifying refined candidate removed' for c in changes),
                    organizer_parity=True,disjoint=True,reloaded_exact=True,transfer_only=not ranked)
        results[name]=result;write_json(DEST/name/'summary.json',result);write_json(DEST/name/'changes.json',changes);write_json(DEST/name/'decisions.json',decisions)
    check_protected();write_json(DEST/'summary.json',dict(results=results,selected_epoch=state['epoch'],selected_threshold=threshold,
                   unchanged_selected=threshold is None or state['epoch']==0,protected_unchanged=True,no_validation_training=True))
    write_json(DEST/'completed.json',dict(complete=True));write_json(DEST/'status.json',dict(stage='complete'))
    print(json.dumps(read_json(DEST/'summary.json'),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','train','choose','evaluate','all'],default='all');args=p.parse_args()
    torch.set_num_threads(2);cv2.setNumThreads(1)
    for stage,operation in [('prepare',prepare),('train',train),('choose',choose),('evaluate',evaluate)]:
        if args.stage in ['all',stage]: operation()
