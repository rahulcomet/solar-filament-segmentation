"""Prediction-only final-parent reconstruction and guarded remnant removal."""
from pycocotools import mask as mu
from component_merge import merge_components
from validation_pq import rle_iou
from copy import deepcopy


def parent_data(prepared,history):
    keep=[o['source'] for o in history['outcomes'] if o['status'] not in
        ['Removed by duplicate suppression','Removed by fragment protection']]
    owned=dict(file_name=prepared['file_name'],masks=[prepared['masks'][i] for i in keep],
        confidence=[prepared['confidence'][i] for i in keep],source_proposal_ids=keep)
    owned=merge_components(owned,history['component_transfers'])
    lookup={s:j for j,s in enumerate(owned['source_proposal_ids'])}
    result=[]
    for j,mask in enumerate(history['masks']):
        sources=[o['source'] for o in history['outcomes'] if j in o['final_indices']]
        # Unmerged outputs keep their original raw parent in final_filters.
        # A synthetic early-union parent is built from component-adjusted
        # ownership candidates. Those are different histories to preserve.
        parent=(prepared['masks'][sources[0]] if len(sources)==1 else
            mu.merge([owned['masks'][lookup[s]] for s in sources]))
        result.append(dict(parent=parent,sources=sources,
            confidence=max(prepared['confidence'][s] for s in sources),
            retained=float(mu.area(mask)/mu.area(parent))))
    return result


def select(prepared,history,guarded=True,protect_supported=False,confidence_limit=.50):
    masks=history['masks'];data=parent_data(prepared,history);drop=[];accepted=set()
    for j in sorted(range(len(data)),key=lambda j:(-data[j]['confidence'],j)):
        d=data[j]
        if d['confidence']>=confidence_limit or d['retained']>=.65:
            accepted.add(j);continue
        if not guarded:
            drop.append(dict(prediction=j,retained=d['retained'],confidence=d['confidence'],reason='retained below0.65'));continue
        candidates=[]
        for k in accepted:
            other=data[k]
            if other['confidence']-d['confidence']<.10:continue
            shared=float(mu.area(mu.merge([d['parent'],masks[k]],intersect=True)))
            if shared/float(mu.area(masks[k]))<.90:continue
            if shared/float(mu.area(d['parent']))<.30:continue
            explained=(shared+float(mu.area(masks[j])))/float(mu.area(d['parent']))
            if explained<.90:continue
            candidates.append((shared,k,explained))
        if candidates:
            _,k,explained=max(candidates)
            if protect_supported:
                excluded=set(d['sources']+data[k]['sources'])
                alternatives=[i for i in range(len(prepared['masks'])) if i not in excluded]
                agreement=rle_iou([masks[j]],[prepared['masks'][i] for i in alternatives])[0]
                if agreement.max(initial=0)>=.50:
                    d['protected_by_proposal']=alternatives[int(agreement.argmax())]
                    d['proposal_agreement']=float(agreement.max())
                    accepted.add(j);continue
            drop.append(dict(prediction=j,retained=d['retained'],confidence=d['confidence'],
                competitor=k,confidence_gap=data[k]['confidence']-d['confidence'],
                explained_fraction=explained,reason='weak clipped remainder around stronger contained instance'))
        else:accepted.add(j)
    rejected={d['prediction'] for d in drop}
    return [m for j,m in enumerate(masks) if j not in rejected],drop,data


def predict(prepared):
    from contained_extension_selection import propose,priorities
    from hybrid_lineage import trace
    rank,_=priorities(prepared,propose(prepared))
    history=trace(prepared,rank)
    return filter_history(prepared,history)['masks']


def filter_history(prepared,history,confidence_limit=.50):
    masks,drop,_=select(prepared,history,guarded=True,protect_supported=True,confidence_limit=confidence_limit)
    rejected={d['prediction'] for d in drop}
    kept=[i for i in range(len(history['masks'])) if i not in rejected]
    mapping={old:new for new,old in enumerate(kept)}
    result=deepcopy(history);result['masks']=masks
    for outcome in result['outcomes']:
        old_indices=outcome['final_indices']
        outcome['final_indices']=[mapping[i] for i in old_indices if i in mapping]
        if old_indices and not outcome['final_indices']:outcome['status']='Removed by supported remnant filter'
    if 'parent_masks' in result:result['parent_masks']=[result['parent_masks'][i] for i in kept]
    result['remnant_drops']=drop
    return result
