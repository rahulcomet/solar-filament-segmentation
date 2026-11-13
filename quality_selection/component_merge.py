"""Join a relevant secondary component without joining its unrelated owner."""
from pycocotools import mask as mu
from refiner import encode_mask


def merge_components(ownership,actions):
    result={k:list(v) if isinstance(v,list) else v for k,v in ownership.items()}
    ids=result['source_proposal_ids']
    for action in actions:
        if action['owner'] not in ids or action['target'] not in ids:continue
        owner=ids.index(action['owner']);target=ids.index(action['target'])
        part=mu.decode(action['component_mask']).astype(bool)
        current=mu.decode(result['masks'][owner]).astype(bool)
        # A component can only be transferred once, and the unrelated main
        # component must survive. Never substitute a whole-candidate union.
        if not (current&part).any() or not (current&~part).any():continue
        receiver=mu.decode(result['masks'][target]).astype(bool)
        result['masks'][owner]=encode_mask((current&~part).astype('uint8'))
        result['masks'][target]=encode_mask((receiver|part).astype('uint8'))
    return result
