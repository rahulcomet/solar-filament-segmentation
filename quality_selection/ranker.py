"""Bounded residual ranking: directly learn which overlapping candidate is better."""
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import OUT,read_json,write_json,sha
from validation_pq import rle_iou

RANK=OUT/'residual_ranker'
GAMMAS=[0.,.5,1.]


class RankHead(nn.Module):
    def __init__(self,n):
        super().__init__();self.layers=nn.Sequential(nn.Linear(n,32),nn.ReLU(),nn.Dropout(.1),nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
        nn.init.zeros_(self.layers[-1].weight);nn.init.zeros_(self.layers[-1].bias)

    def forward(self,x):return .5*self.layers(x).tanh().flatten()


def pair_data(partition,records):
    xs=[];cs=[];pairs=[];weights=[];offset=0;images=0
    for path in sorted((OUT/'train').glob('*.json')):
        row=read_json(path)
        if row['partition']!=partition or not len(row['masks']):continue
        with np.load(path.with_suffix('.npz'),allow_pickle=False) as data:x=data['X']
        best=[]
        for rec in records[row['file_name']]:
            m=rle_iou(rec['gt'],row['masks']);best.append(m.max(axis=0) if len(rec['gt']) else np.zeros(len(row['masks'])))
        best=np.stack(best);utility=(best*(best>.5)).mean(axis=0)
        overlaps=rle_iou(row['masks'],row['masks']);local=[]
        for i in range(len(x)):
            for j in range(i):
                if overlaps[i,j]<.2 or abs(utility[i]-utility[j])<.1:continue
                a,b=(i,j) if utility[i]>utility[j] else (j,i)
                local.append((offset+a,offset+b))
        if local:images+=1;pairs.extend(local);weights.extend([1/len(local)]*len(local))
        xs.append(x);cs.extend(row['confidence']);offset+=len(x)
    w=np.array(weights,dtype='float32');w/=w.mean()
    return np.concatenate(xs),np.array(cs,dtype='float32'),np.array(pairs),w,images


def pair_loss(net,x,logits,pairs,weights):
    a,b=pairs[:,0],pairs[:,1];da,db=net(x[a]),net(x[b])
    ranking=F.softplus(-((logits[a]+da)-(logits[b]+db)))
    return ((ranking+.05*(da.square()+db.square()))*weights).mean()


def predict(state,x,confidence,gamma=1.):
    c=np.asarray(confidence,dtype=float)
    if gamma==0 or state['epoch']==0:return c.copy()
    net=RankHead(x.shape[1]);net.load_state_dict(state['model']);net.eval()
    norm=torch.from_numpy(np.clip((x-state['mean'])/state['std'],-5,5).astype('float32'))
    with torch.inference_mode():delta=net(norm).numpy()
    logit=np.log(np.clip(c,1e-6,1-1e-6)/(1-np.clip(c,1e-6,1-1e-6)))
    return 1/(1+np.exp(-(logit+gamma*delta)))


def main():
    RANK.mkdir(exist_ok=True)
    if (RANK/'trained.json').exists():print('Residual ranker already trained');return
    torch.set_num_threads(2);torch.manual_seed(43);rng=np.random.default_rng(43)
    from collections import defaultdict
    records=defaultdict(list)
    for rec in read_json(OUT/'training_records.json'):records[rec['file_name']].append(rec)
    x,c,p,w,fit_images=pair_data('fit',records);tx,tc,tp,tw,tune_images=pair_data('tune',records)
    mean=x.mean(axis=0);std=np.maximum(x.std(axis=0),1e-3)
    transform=lambda a:torch.from_numpy(np.clip((a-mean)/std,-5,5).astype('float32'))
    X,TX=transform(x),transform(tx);C=torch.from_numpy(np.log(c/(1-c)));TC=torch.from_numpy(np.log(tc/(1-tc)))
    P,TP=torch.from_numpy(p),torch.from_numpy(tp);W,TW=torch.from_numpy(w),torch.from_numpy(tw)
    net=RankHead(x.shape[1]);optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.001)
    plan=dict(reason='Absolute quality head lost against confidence on training-day tuning; it correctly ordered75/125relevant pairs versus86/125forconfidence.',
        target='Higher expected matchedIoU (mean maxIoU*indicator>0.50 across records) in overlapping same-image pairs. RawmaskpairIoU>=0.20, utility difference>=0.10.',
        trust_region='Score=sigmoid(logit(YOLOconfidence)+gamma*0.5*tanh(network(features))). Original confidences retained downstream.',
        fit_pairs=len(p),tune_pairs=len(tp),fit_images_with_pairs=fit_images,tune_images_with_pairs=tune_images,
        objective='Pairwise softplus loss plus0.05squaredcorrections; image-balanced pairs. CPU AdamW lr0.001 wd0.001,80epochs batch128 seed43.',
        selection='Initial zero correction plus epochs10,20,40,80, minimum held-out training-day pair loss. Then gamma[0,.5,1] by full tuning PQ only. One bounded follow-up; no further head variants.',
        base_features_sha256=sha(OUT/'features_complete.json'))
    write_json(RANK/'plan.json',plan);history=[];best=None
    for epoch in range(81):
        if epoch:
            net.train();order=rng.permutation(len(p))
            for start in range(0,len(p),128):
                ids=order[start:start+128];optimizer.zero_grad(set_to_none=True)
                value=pair_loss(net,X,C,P[ids],W[ids]);assert torch.isfinite(value)
                value.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5,error_if_nonfinite=True);optimizer.step()
        if epoch in [0,10,20,40,80]:
            net.eval()
            with torch.inference_mode():value=float(pair_loss(net,TX,TC,TP,TW));d=net(TX);accuracy=float(((TC[TP[:,0]]+d[TP[:,0]])>(TC[TP[:,1]]+d[TP[:,1]])).float().mean())
            row=dict(epoch=epoch,tune_loss=value,tune_pair_accuracy=accuracy);history.append(row)
            state=dict(model=copy.deepcopy(net.state_dict()),mean=mean,std=std,epoch=epoch)
            torch.save(state,RANK/f'epoch_{epoch:02d}.pt')
            if best is None or value<best['tune_loss']:best=row;torch.save(state,RANK/'best.pt')
            print(json.dumps(row),flush=True)
    write_json(RANK/'trained.json',dict(best=best,history=history,fit_pairs=len(p),tune_pairs=len(tp)));print('Residual ranking selected',best,flush=True)


if __name__=='__main__':main()
