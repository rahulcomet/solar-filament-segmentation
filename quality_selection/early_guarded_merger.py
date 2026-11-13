"""Use existing merge guards in an early, single-stage grouping decision."""
from copy import deepcopy
from pycocotools import mask as mu
from component_merge import merge_components
from selection_policies import apply_policy,Policy
from merge_recovery import edges,accept


def allowed(edge,hybrid=False,curved_extension=False):
    accepted=accept(edge,'majority_veto_with_continuation')
    if not accepted:
        accepted=bool(curved_extension and edge['guarded'] and edge['raw_iou']>=.20 and
            edge['angle'] is not None and 25<edge['angle']<=30 and edge['width_ratio']<=1.5 and
            all(v is not None and v>=3 for v in edge['axis_ratios']) and min(edge['coherence'])>=.90)
    if not accepted:return False
    if edge['raw_iou']>=.20:return True
    return bool(hybrid and edge['raw_iou']>=.10 and edge['angle'] is not None and
        edge['angle']<=25 and edge['width_ratio']<=1.5 and
        all(v is not None and v>=3 for v in edge['axis_ratios']) and
        min(edge['coherence'])>=.8)


def group_candidates(owned,row,selected):
    """Give early unions their own parent mask for subsequent fragment filters.

    row is the disposable per-evaluation copy used by repair, never the cache.
    Appending a union parent preserves both sources' foreground in the area
    denominator, rather than treating the union as just its stronger member.
    """
    ids=owned['source_proposal_ids'];used={s for p in selected for s in p['sources']}
    keep=[j for j,s in enumerate(ids) if s not in used]
    result={k:([v[j] for j in keep] if isinstance(v,list) else v) for k,v in owned.items()}
    for p in selected:
        source=p['sources'];positions=[ids.index(s) for s in source]
        union=mu.merge([owned['masks'][j] for j in positions])
        synthetic=len(row['masks'])
        row['masks'].append(union)
        if 'coarse_masks' in row:
            row['coarse_masks'].append(mu.merge([row['coarse_masks'][s] for s in source]))
        row['confidence'].append(max(row['confidence'][s] for s in source))
        row['source_proposal_ids'].append(synthetic)
        result['masks'].append(union)
        result['confidence'].append(max(owned['confidence'][j] for j in positions))
        result['source_proposal_ids'].append(synthetic)
        p['union_source']=synthetic
    return result


def combine_guarded(ownership,components,row,hybrid=False,curved_extension=False):
    owned=merge_components(ownership,components)
    preview,_=apply_policy(owned,Policy('preview',cutoff=0,mask_nms=False))
    preview['confidence']=[row['confidence'][s] for s in preview['source_proposal_ids']]
    measured=edges(row,preview,min_iou=.10 if hybrid else .20)
    eligible=[]
    lookup={s:j for j,s in enumerate(owned['source_proposal_ids'])}
    for e in measured:
        if not allowed(e,hybrid,curved_extension):continue
        sa,sb=e['sources'];a,b=lookup[sa],lookup[sb]
        if (owned['confidence'][a],-sa)<(owned['confidence'][b],-sb):a,b=b,a
        high,low=[owned['masks'][j] for j in [a,b]]
        shared=float(mu.area(mu.merge([high,low],intersect=True)))
        ah,al=float(mu.area(high)),float(mu.area(low));added=al-shared
        e=dict(e,score=(shared/min(ah,al))*added/(ah+added),added_to_stronger=added)
        eligible.append(e)
    # First arm preserves old IoU ordering. Hybrid rewards useful extension
    # among pairs that have passed the old guards / narrow low-IoU exception.
    eligible.sort(key=lambda e:(-e['score'] if hybrid else -e['raw_iou'],e['sources']))
    used=set();selected=[]
    for e in eligible:
        if any(s in used for s in e['sources']):continue
        used.update(e['sources']);selected.append(e)
    result=group_candidates(owned,row,selected)
    return result,selected


def predict(prepared,hybrid=True,selection_priorities=None,curved_extension=False):
    """Run the selected cleanup on an isolated copy of prepared candidates."""
    from focused_ownership_rules import find_actions
    from selection_policies import features
    from test_component_ownership import repair
    row=deepcopy(prepared);measured=features(row)
    ranked=dict(row,confidence=selection_priorities) if selection_priorities is not None else row
    _,audit=apply_policy(ranked,Policy('nms',cutoff=.18),cached=measured)
    survivors=[i for i in range(len(row['masks'])) if i not in audit['suppressed']]
    components=find_actions(row,measured,survivors,include_tip=False)['component']
    return repair(row,geometry_guard=True,direction_limit=30,whole_group_priority=True,
        agreement_weight=False,merge_policy='none',selection_priorities=selection_priorities,
        ownership_transform=lambda owned:combine_guarded(owned,components,row,hybrid=hybrid,curved_extension=curved_extension)[0])[0]
