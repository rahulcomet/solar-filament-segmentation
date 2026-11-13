"""Replay current hybrid while retaining source-to-final-mask history."""
from copy import deepcopy
from selection_policies import features,apply_policy,Policy
from focused_ownership_rules import find_actions
from early_guarded_merger import combine_guarded
from test_component_ownership import repair
from pycocotools import mask as mu


def trace(prepared,selection_priorities=None,curved_extension=False):
    row=deepcopy(prepared);f=features(row)
    ranked=dict(row,confidence=selection_priorities) if selection_priorities is not None else row
    _,audit=apply_policy(ranked,Policy('nms',cutoff=.18),cached=f)
    survivors=[i for i in range(len(row['masks'])) if i not in audit['suppressed']]
    components=find_actions(row,f,survivors,include_tip=False)['component'];groups=[]
    def modify(owned):
        result,chosen=combine_guarded(owned,components,row,hybrid=True,curved_extension=curved_extension)
        groups.extend(chosen);return result
    masks,fragment,meta=repair(row,geometry_guard=True,direction_limit=30,whole_group_priority=True,
        agreement_weight=False,merge_policy='none',ownership_transform=modify,
        return_metadata=True,selection_priorities=selection_priorities)
    parent_members={g['union_source']:g['sources'] for g in groups}
    finals={i:[] for i in range(len(prepared['masks']))}
    for p,sources in enumerate(meta['final']['source_groups']):
        for s in sources:
            for member in parent_members.get(s,[s]):finals[member].append(p)
    outcomes=[]
    for i in range(len(prepared['masks'])):
        blockers=[j for j in survivors if f['pair_iou'][i,j]>=.5] if i in audit['suppressed'] else []
        if finals[i]:status='Final output' if not any(i in g['sources'] for g in groups) else 'Early merged group'
        elif i in audit['suppressed']:status='Removed by duplicate suppression'
        elif i not in meta['permitted']:status='Removed by fragment protection'
        else:
            identity=next((g['union_source'] for g in groups if i in g['sources']),i)
            status='Removed during ownership' if identity not in meta['ownership']['source_proposal_ids'] else 'Removed by final filters'
        outcomes.append(dict(source=i,status=status,final_indices=finals[i],nms_blockers=blockers))
    parents=[mu.merge([row['masks'][s] for s in sources]) for sources in meta['final']['source_groups']]
    parents=[dict(m,counts=m['counts'].decode('ascii')) if isinstance(m['counts'],bytes) else m for m in parents]
    return dict(masks=masks,outcomes=outcomes,early_groups=groups,component_transfers=components,fragment_actions=fragment,parent_masks=parents)
