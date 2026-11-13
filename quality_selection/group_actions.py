"""Prediction-only conflict actions and training-only full-output labels.

No detector/refiner inference, network fitting, or validation-based action design.
The annotation-assisted ceiling is a diagnostic, never a deployable selector.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mu

from common import ROOT, OUT, read_json, write_json, sha, decode_mask, encode_mask
from ranker import RANK, predict
from decision_selector import orders, counterfactual, finish, cached_prune, training_records
from merge_recovery import edges, selected_edges, group
from reaudit_fp_progress import parent_union
from refiner import PQStats, pq_stats_from_iou
from validation_pq import rle_iou
from sweep_confidence import assert_disjoint
from official_notebook_diagnostics import official_functions

DEST = ROOT / 'quality_selection/outputs/group_actions_20261103'
REFERENCE_PQ = .4488780461290956


def signature(pred):
    return tuple(sorted(m['counts'] for m in pred['masks']))


def candidates(row):
    """Enumerate without labels: overlap components, all members, bounded pairs.

    Every admitted candidate can be preferred or rejected, regardless of confidence
    gap. Pair actions use at most six most-overlapping pairs per component. This
    cap bounds inference cost without truncating individual candidate coverage.
    """
    n = len(row['masks'])
    assert all(c >= .20 for c in row['confidence'])
    matrix = rle_iou(row['masks'], row['masks'])
    unseen = set(range(n)); components = []
    while unseen:
        todo = [min(unseen)]; unseen.remove(todo[0]); found = []
        while todo:
            a = todo.pop(); found.append(a)
            neighbors = sorted(b for b in unseen if matrix[a, b] > 0)
            unseen.difference_update(neighbors); todo.extend(neighbors)
        if len(found) > 1:
            components.append(sorted(found))
    result = []
    for gid, members in enumerate(components):
        for a in members:
            result.extend([dict(group=gid, kind='prefer', members=members, ids=[a]),
                           dict(group=gid, kind='reject', members=members, ids=[a])])
        pairs = sorted(((float(matrix[a, b]), a, b) for j, a in enumerate(members)
                        for b in members[j+1:] if matrix[a, b] > 0), reverse=True)[:6]
        for overlap, a, b in pairs:
            for kind, ids in [('allocate', [a, b]), ('allocate', [b, a])]:
                result.append(dict(group=gid, kind=kind, members=members, ids=ids))
            if overlap >= .20:
                result.append(dict(group=gid, kind='merge', members=members, ids=[a, b]))
    return components, result


def replay(row, scores, action, masks):
    """Apply one action before ownership, then replay every downstream filter."""
    if action is None:
        return counterfactual(row, scores, None, masks)
    rank, _, pair = orders(row, scores)
    ids = action['ids']; kind = action['kind']
    if kind == 'reject':
        rank = [i for i in rank if i != ids[0]]
        kept = []
    elif kind in ['prefer', 'allocate', 'merge']:
        rank = [i for i in rank if i not in ids]
        kept = list(ids)
    else:
        raise ValueError(kind)
    for i in rank:
        if not kept or not (pair[i, kept] >= .50).any():
            kept.append(i)
    if kind == 'reject':
        ownership = sorted(kept, key=lambda i: (-row['confidence'][i], i))
    else:
        ownership = list(ids) + sorted((i for i in kept if i not in ids),
                                     key=lambda i: (-row['confidence'][i], i))
    # Identical occupied-pixel and >=10%-remaining semantics to production.
    shape = masks[0].shape if masks else (2048, 2048)
    occupied = np.zeros(shape, bool); saved = {}
    for i in ownership:
        x, y, w, h = map(int, mu.toBbox(row['masks'][i]))
        cut = masks[i][y:y+h, x:x+w]; claimed = occupied[y:y+h, x:x+w]
        remain = cut & ~claimed; claimed |= cut
        if remain.any() and remain.sum() / cut.sum() >= .10:
            if remain.sum() == cut.sum():
                saved[i] = row['masks'][i]
            else:
                native = np.zeros(shape, bool); native[y:y+h, x:x+w] = remain
                saved[i] = encode_mask(native)
    ordered = sorted(saved)
    clean = dict(file_name=row['file_name'], masks=[saved[i] for i in ordered],
                 confidence=[row['confidence'][i] for i in ordered],
                 source_proposal_ids=[row['source_proposal_ids'][i] for i in ordered])
    if kind != 'merge' or not all(i in saved for i in ids):
        return finish(clean, row)
    # Force the proposed pair into the merge stage; leave all nonincident merge
    # decisions and the tiny-satellite/remnant filters unchanged.
    a, b = [ordered.index(i) for i in ids]
    chosen = [dict(a=a, b=b)] + [e for e in selected_edges(edges(row, clean), 'majority_veto')
                               if not {e['a'], e['b']} & {a, b}]
    merged = group(clean, chosen); keep = []; pruned = []
    for j, rle in enumerate(merged['masks']):
        rle = cached_prune(rle['counts'], *rle['size'])
        if rle is None:
            continue
        if merged['confidence'][j] < .5 and float(mu.area(rle)) / float(mu.area(parent_union(row, merged['source_groups'][j]))) < .5:
            continue
        keep.append(j); pruned.append(rle)
    return dict(file_name=row['file_name'], masks=pruned,
                **{k: [merged[k][j] for j in keep] for k in ['confidence', 'source_proposal_ids', 'source_groups']})


def outcome(records, pred):
    per_record = []; total = PQStats()
    for rec in records:
        matrix = rle_iou(rec['gt'], pred['masks'])
        s = pq_stats_from_iou(matrix); total += s
        per_record.append(dict(record=rec['record'], **asdict(s),
                               hits=np.flatnonzero((matrix > .5).any(axis=1)).tolist()))
    return dict(**asdict(total), records=per_record)


def utility(s, q):
    return s['iou_sum'] - q * (s['tp'] + .5 * (s['fp'] + s['fn']))


def choose(rows):
    """Optimize pooled PQ over the finite one-change-per-image action space.

    Fractional-programming updates choose full-image contributions at a shared
    global q. A tie retains control. This is an annotation-assisted ceiling only.
    """
    q = REFERENCE_PQ
    for _ in range(30):
        chosen = [max(range(len(r['alternatives'])), key=lambda j:
                      utility(r['alternatives'][j]['outcome'], q)) for r in rows]
        total = PQStats()
        for row, j in zip(rows, chosen):
            s = row['alternatives'][j]['outcome']
            total += PQStats(**{k: s[k] for k in ['tp', 'fp', 'fn', 'iou_sum']})
        if abs(total.pq - q) < 1e-12:
            break
        q = total.pq
    return chosen, dict(**asdict(total), pq=total.pq)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0, help='Smoke only; zero runs every training image')
    args = parser.parse_args(); torch.set_num_threads(2); cv2.setNumThreads(1)
    dest = DEST if args.limit == 0 else DEST / f'smoke_{args.limit}'
    dest.mkdir(parents=True, exist_ok=True); folder = dest / 'images'; folder.mkdir(exist_ok=True)
    paths = sorted((OUT / 'train').glob('*.json'))
    if args.limit:
        paths = paths[:args.limit]
    protected = [ROOT/'quality_selection/current_candidate.json', ROOT/'local_refiner/current_best_pipeline.json',
                 ROOT/'local_refiner/outputs/realistic_matched_2500_20261103/selected_confidence_diagnostic.json']
    source_files = [Path(__file__), ROOT/'quality_selection/decision_selector.py', ROOT/'local_refiner/merge_recovery.py']
    plan = dict(admission=.20, data='Cached frozen epoch6 masks; original training images only; in-sample upstream errors.',
                checkpoint='Protected epoch6 + MLP NMS reference .448878; NOT update2500 raw masks.',
                actions='Overlap-connected groups; prefer/reject every member, both allocations for top6 overlapping pairs per group, union merge if pairIoU>=.20. No confidence-gap cap.',
                grouping='Positive pixel intersection only; disconnected nonoverlapping masks are outside this pilot.',
                alternatives='One group/action per image; unchanged always first. All masks/allocations prediction-only.',
                labels='Full-image downstream outcomes separately for every annotation; no averaged group PQ.',
                oracle='Finite-action pooled-PQ diagnostic, not learned performance. No CNN trained or validation oracle.',
                highest_valid_diagnostic=.44930662641554414,
                protected={str(p.relative_to(ROOT)): sha(p) for p in protected},
                source_sha256={str(p.relative_to(ROOT)): sha(p) for p in source_files},
                inputs={str(p.relative_to(ROOT)): sha(p) for p in paths + [p.with_suffix('.npz') for p in paths] + [OUT/'training_records.json', RANK/'best.pt']})
    if (dest/'plan.json').exists():
        assert read_json(dest/'plan.json') == plan, 'Inputs changed; use a new output folder.'
    else:
        write_json(dest/'plan.json', plan)
    records = training_records()
    state = torch.load(RANK/'best.pt', map_location='cpu', weights_only=False)
    started = time.perf_counter()
    for k, path in enumerate(paths):
        target = folder/path.name
        if target.exists():
            continue
        row = read_json(path)
        with np.load(path.with_suffix('.npz')) as data:
            scores = predict(state, data['X'], row['confidence'], .5) if len(data['X']) else np.zeros(0)
        masks = [decode_mask(m).astype(bool) for m in row['masks']]
        components, actions = candidates(row)
        baseline = replay(row, scores, None, masks); assert_disjoint(baseline['masks'])
        if row['partition'] == 'tune':
            assert signature(baseline) == signature(read_json(RANK/'nms_only/tuning/gamma_0.50'/path.name))
        alternatives = [dict(action=None, prediction=baseline, outcome=outcome(records[row['file_name']], baseline))]
        seen = {signature(baseline)}; duplicate_actions = 0
        for action in actions:
            pred = replay(row, scores, action, masks); sig = signature(pred)
            if sig in seen:
                duplicate_actions += 1; continue
            seen.add(sig); assert_disjoint(pred['masks'])
            alternatives.append(dict(action=action, prediction=pred, outcome=outcome(records[row['file_name']], pred)))
        result = dict(file_name=row['file_name'], partition=row['partition'], raw_path=str(path.relative_to(ROOT)),
                      groups=components, enumerated_actions=len(actions), duplicate_outputs=duplicate_actions, alternatives=alternatives)
        # Never mark a partly written image complete.
        tmp = target.with_suffix('.tmp'); write_json(tmp, result); tmp.replace(target)
        write_json(dest/'status.json', dict(stage='training_action_labels', images=k+1, total=len(paths),
                                         seconds=time.perf_counter()-started))
        if (k+1) % 10 == 0:
            print(f'Full-output action labels {k+1}/{len(paths)}', flush=True)
    rows = [read_json(folder/p.name) for p in paths]
    official = official_functions(); partitions = {}
    for part in ['fit', 'tune']:
        selected = [r for r in rows if r['partition'] == part]
        choices, upper = choose(selected); base = PQStats(); table = []
        gained = lost = conflicting_records = useful = no_help = 0; counts = Counter()
        for row, j in zip(selected, choices):
            b = row['alternatives'][0]['outcome']; winner = row['alternatives'][j]['outcome']
            base += PQStats(**{k: b[k] for k in ['tp','fp','fn','iou_sum']})
            useful += j != 0; no_help += j == 0
            if j: counts[row['alternatives'][j]['action']['kind']] += 1
            deltas = [utility(a['outcome'], REFERENCE_PQ)-utility(b, REFERENCE_PQ) for a in row['alternatives'][1:]]
            for alt, delta in zip(row['alternatives'][1:], deltas):
                changes = [utility(a, REFERENCE_PQ)-utility(z, REFERENCE_PQ)
                           for a,z in zip(alt['outcome']['records'], b['records'])]
                conflicting_records += bool(any(d > 1e-9 for d in changes) and any(d < -1e-9 for d in changes))
            for rec, old in zip(winner['records'], b['records']):
                gained += len(set(rec['hits'])-set(old['hits'])); lost += len(set(old['hits'])-set(rec['hits']))
            rec_by_id = {r['record']:r for r in records[row['file_name']]}
            for rec in winner['records']:
                gt = rec_by_id[rec['record']]['gt']; pred = row['alternatives'][j]['prediction']['masks']
                table.append(dict(iou_matrix=torch.from_numpy(rle_iou(gt,pred)), n_gt=len(gt), n_pred=len(pred)))
        official_pq = float(official['get_pq_score'](pd.DataFrame(table)))
        assert abs(official_pq-upper['pq']) < 1e-10
        partitions[part] = dict(images=len(selected), baseline=dict(**asdict(base),pq=base.pq), oracle=upper,
                                changed_images=useful, unchanged_images=no_help, gained=gained,lost=lost,
                                actions=dict(counts), alternatives_with_opposite_annotator_effects=conflicting_records,
                                choices=[dict(file_name=r['file_name'], alternative=j) for r,j in zip(selected,choices)])
    for path, digest in plan['protected'].items(): assert sha(ROOT/path) == digest
    summary = dict(partitions=partitions, groups=sum(len(r['groups']) for r in rows),
                   alternatives=sum(len(r['alternatives'])-1 for r in rows),
                   images_without_conflicts=sum(not r['groups'] for r in rows),
                   cnn_trained=False, validation_used=False, organizer_parity=True, disjoint=True,
                   protected_unchanged=True, complete_training_pool=args.limit==0)
    write_json(dest/'summary.json', summary)
    write_json(dest/'status.json',dict(stage='complete',images=len(rows)))
    write_json(dest/'completed.json',dict(complete=True,cnn_trained=False))
    print(json.dumps({p:{k:v for k,v in s.items() if k!='choices'} for p,s in partitions.items()},indent=2),flush=True)


if __name__ == '__main__':
    main()
