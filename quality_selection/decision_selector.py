"""Learn the net value of one local ordering change, using training records only.

The incumbent ranker still controls NMS. A separate regularized linear model
estimates the final-pipeline benefit of a bounded NMS or ownership swap. No
annotation or counterfactual score is available to selection at inference.
"""
from collections import defaultdict, Counter
from dataclasses import asdict
import argparse
from functools import lru_cache
import json
import cv2
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mu
from common import OUT, ROOT, WEIGHTS, read_json, write_json, sha, encode_mask, decode_mask
from ranker import RANK, predict
from merge_recovery import edges, group, selected_edges
from reaudit_fp_progress import parent_union
from validation_pq import rle_iou
from refiner import PQStats, pq_stats_from_iou
from build_visual_review import cache
from evaluate_merge_recovery import compare
from sweep_confidence import assert_disjoint
from official_notebook_diagnostics import official_functions

DEST = OUT / 'decision_selector'
REFERENCE_PQ = .4488780461290956
LAMBDAS = [1., 10., 100., 1000.]
THRESHOLDS = [None, .02, .10]


@lru_cache(maxsize=2048)
def cached_prune(counts, height, width):
    """Equivalent tiny-component rule on the foreground ROI, cached across swaps."""
    rle = dict(counts=counts, size=[height, width])
    x, y, w, h = map(int, mu.toBbox(rle))
    if not w or not h:
        return None
    mask = decode_mask(rle).astype(bool)
    cut = mask[y:y+h, x:x+w]
    _, labels, stats, _ = cv2.connectedComponentsWithStats(cut.astype('uint8'), connectivity=8)
    area = stats[1:, cv2.CC_STAT_AREA]
    largest = int(area.argmax()) + 1
    remove = [j+1 for j, a in enumerate(area) if j+1 != largest and a < 64 and a < .10 * area.sum()]
    if not remove:
        return dict(counts=counts.decode('ascii') if isinstance(counts, bytes) else counts, size=[height, width])
    mask[y:y+h, x:x+w] &= ~np.isin(labels, remove)
    return encode_mask(mask)


def finish(clean, row):
    merged = group(clean, selected_edges(edges(row, clean), 'majority_veto'))
    keep = []
    masks = []
    for j, rle in enumerate(merged['masks']):
        rle = cached_prune(rle['counts'], *rle['size'])
        if rle is None:
            continue
        if merged['confidence'][j] < .5 and float(mu.area(rle)) / float(mu.area(parent_union(row, merged['source_groups'][j]))) < .5:
            continue
        keep.append(j); masks.append(rle)
    return dict(file_name=row['file_name'], masks=masks,
                **{key: [merged[key][j] for j in keep] for key in ['confidence', 'source_proposal_ids', 'source_groups']})


def orders(row, scores):
    rank = sorted(range(len(scores)), key=lambda i: (-scores[i], -row['confidence'][i], i))
    pair = rle_iou(row['masks'], row['masks'])
    kept = []
    for i in rank:
        if not kept or not (pair[i, kept] >= .5).any():
            kept.append(i)
    return rank, sorted(kept, key=lambda i: (-row['confidence'][i], i)), pair


def swap(order, a, b):
    result = list(order)
    i, j = result.index(a), result.index(b)
    result[i], result[j] = result[j], result[i]
    return result


def finish_order(row, ownership, masks):
    shape = masks[0].shape if masks else (2048, 2048)
    occupied = np.zeros(shape, bool)
    saved = {}
    for i in ownership:
        x, y, w, h = map(int, mu.toBbox(row['masks'][i]))
        cut = masks[i][y:y+h, x:x+w]
        claimed = occupied[y:y+h, x:x+w]
        remain = cut & ~claimed
        claimed |= cut
        original_area = int(cut.sum()); remaining_area = int(remain.sum())
        if remaining_area and remaining_area / original_area >= .10:
            if remaining_area == original_area:
                saved[i] = row['masks'][i]
            else:
                restored = np.zeros(shape, bool)
                restored[y:y+h, x:x+w] = remain
                saved[i] = encode_mask(restored)
    ids = sorted(saved)
    clean = dict(file_name=row['file_name'], masks=[saved[i] for i in ids],
                 confidence=[row['confidence'][i] for i in ids],
                 source_proposal_ids=[row['source_proposal_ids'][i] for i in ids])
    return finish(clean, row)


def counterfactual(row, scores, action, masks):
    rank, ownership, pair = orders(row, scores)
    if action is not None:
        kind, a, b = action
        if kind == 'nms':
            rank = swap(rank, a, b)
            kept = []
            for i in rank:
                if not kept or not (pair[i, kept] >= .5).any():
                    kept.append(i)
            ownership = sorted(kept, key=lambda i: (-row['confidence'][i], i))
        else:
            assert kind == 'ownership' and a in ownership and b in ownership
            ownership = swap(ownership, a, b)
    return finish_order(row, ownership, masks)


def action_features(row, x, scores):
    rank, ownership, pair = orders(row, scores)
    area = mu.area(row['masks']).astype(float) if row['masks'] else np.zeros(0)
    actions = []
    for kind, order in [('nms', rank), ('ownership', ownership)]:
        eligible = []
        for ai, a in enumerate(order):
            for bi in range(ai + 1, len(order)):
                b = order[bi]
                iou = pair[a, b]
                gap = abs(row['confidence'][a] - row['confidence'][b])
                if kind == 'nms' and not (iou >= .5 and gap <= .25):
                    continue
                if kind == 'ownership' and not (.02 <= iou < .5 and gap <= .20):
                    continue
                inter = iou * (area[a] + area[b]) / (1 + iou)
                small_overlap = inter / min(area[a], area[b])
                scalars = [float(kind == 'ownership'), iou, inter / area[a], inter / area[b],
                           np.log(area[a] / area[b]), gap, scores[a] - scores[b],
                           (bi - ai) / max(1, len(order)), len(order) / 20,
                           float((pair[a] > .02).sum()) / 10,
                           float((pair[b] > .02).sum()) / 10]
                features = np.r_[x[a] - x[b], (x[a] + x[b]) / 2, scalars].astype('float32')
                eligible.append((small_overlap, (kind, a, b), features))
        # Fixed inference-time budget, chosen without annotation information.
        eligible.sort(key=lambda z: (-z[0], z[1][1], z[1][2]))
        actions.extend(eligible[:3])
    return [(action, feature) for _, action, feature in actions]


def stats(records, pred):
    total = PQStats()
    for rec in records:
        total += pq_stats_from_iou(rle_iou(rec['gt'], pred['masks']))
    return total


def utility(s):
    return s.iou_sum - REFERENCE_PQ * (s.tp + .5 * (s.fp + s.fn))


def training_records():
    result = defaultdict(list)
    for rec in read_json(OUT / 'training_records.json'):
        result[rec['file_name']].append(rec)
    return result


def prepare():
    DEST.mkdir(exist_ok=True)
    if (DEST / 'prepared.json').exists():
        return
    plan = dict(hypothesis='Predict actual final disjoint utility of a local ordering change, rather than raw-mask quality.',
                intervention='At most one swap per image: NMS rank OR surviving-candidate ownership order; incumbent NMS ranker gamma0.5 otherwise retained.',
                features='303 frozen candidate-pair features: difference and average of146 features plus11 pair/context scalars.',
                candidate_budget='Three most-contained eligible pairs per stage. NMS IoU>=0.50/confidence gap<=0.25; ownership IoU[0.02,0.50)/gap<=0.20.',
                target='Change in image summed matchedIoU - referencePQ*(TP+0.5FP+0.5FN), averaged across independent annotation records.',
                fit='Image-balanced ridge regression. Lambda1,10,100,1000 selected by held-out training-day MSE; no-action baseline eligible.',
                policy_selection='Held-out training-day full PQ, thresholds off/0.02/0.10; frozen before validation.',
                caveat='Base models in-sample on training days, one reused research validation. Not out-of-fold evidence.',
                protected={str(p.relative_to(ROOT)): sha(p) for p in [WEIGHTS, RANK / 'best.pt', ROOT / 'quality_selection/current_candidate.json', ROOT / 'local_refiner/current_best_pipeline.json']})
    if (DEST / 'plan.json').exists():
        assert read_json(DEST / 'plan.json') == plan
    else:
        write_json(DEST / 'plan.json', plan)
    state = torch.load(RANK / 'best.pt', map_location='cpu', weights_only=False)
    records = training_records()
    folder = DEST / 'training'
    folder.mkdir(exist_ok=True)
    paths = sorted((OUT / 'train').glob('*.json'))
    for k, path in enumerate(paths):
        target = folder / path.name
        if target.exists():
            continue
        row = read_json(path)
        with np.load(path.with_suffix('.npz')) as data:
            x = data['X']
        scores = predict(state, x, row['confidence'], .5) if len(x) else np.zeros(0)
        masks = [decode_mask(m).astype(bool) for m in row['masks']]
        control = counterfactual(row, scores, None, masks)
        expected = read_json(RANK / 'nms_only/tuning/gamma_0.50' / path.name) if row['partition'] == 'tune' else None
        if expected is not None:
            assert control == expected, row['file_name']
        base = stats(records[row['file_name']], control)
        examples = []
        for action, features in action_features(row, x, scores):
            pred = counterfactual(row, scores, action, masks)
            assert_disjoint(pred['masks'])
            s = stats(records[row['file_name']], pred)
            examples.append(dict(action=action, features=features.tolist(),
                                 delta=(utility(s) - utility(base)) / len(records[row['file_name']]),
                                 stats=asdict(s)))
            if row['partition'] == 'tune':
                (DEST / 'tuning_actions' / path.stem).mkdir(parents=True, exist_ok=True)
                write_json(DEST / 'tuning_actions' / path.stem / f'{len(examples)-1}.json', pred)
        write_json(target, dict(file_name=row['file_name'], partition=row['partition'], baseline=asdict(base), examples=examples))
        write_json(DEST / 'status.json', dict(stage='training_counterfactuals', images=k+1, total=len(paths)))
        if (k+1) % 25 == 0:
            print(f'Training counterfactuals {k+1}/{len(paths)}', flush=True)
    rows = [read_json(p) for p in folder.glob('*.json')]
    summary = {part: dict(images=sum(r['partition'] == part for r in rows),
                         actions=sum(len(r['examples']) for r in rows if r['partition'] == part),
                         positive=sum(e['delta'] > 1e-9 for r in rows if r['partition'] == part for e in r['examples']),
                         negative=sum(e['delta'] < -1e-9 for r in rows if r['partition'] == part for e in r['examples'])) for part in ['fit', 'tune']}
    write_json(DEST / 'prepared.json', summary)
    print(json.dumps(summary), flush=True)


def predict_action(model, feature):
    z = np.clip((np.asarray(feature) - model['mean']) / model['std'], -5, 5)
    return float(z @ model['coef'] + model['intercept'])


def fit():
    if (DEST / 'selection.json').exists():
        return
    rows = [read_json(p) for p in sorted((DEST / 'training').glob('*.json'))]
    arrays = {}
    for part in ['fit', 'tune']:
        examples = [(r, e) for r in rows if r['partition'] == part for e in r['examples']]
        x = np.asarray([e['features'] for r, e in examples], dtype=float)
        y = np.asarray([e['delta'] for r, e in examples])
        w = np.asarray([1 / len(r['examples']) for r, e in examples])
        w /= w.mean()
        arrays[part] = x, y, w
    x, y, w = arrays['fit']; tx, ty, tw = arrays['tune']
    mean, std = x.mean(0), np.maximum(x.std(0), 1e-3)
    z = np.c_[np.clip((x - mean) / std, -5, 5), np.ones(len(x))]
    tz = np.c_[np.clip((tx - mean) / std, -5, 5), np.ones(len(tx))]
    candidates = []
    for lam in LAMBDAS:
        penalty = np.eye(z.shape[1]) * lam; penalty[-1, -1] = 0
        coef = np.linalg.solve(z.T @ (z * w[:, None]) + penalty, z.T @ (y * w))
        loss = float(np.average((tz @ coef - ty)**2, weights=tw))
        candidates.append((loss, lam, coef))
    loss, lam, coef = min(candidates, key=lambda c: (c[0], -c[1]))
    model = dict(mean=mean, std=std, coef=coef[:-1], intercept=coef[-1])
    np.savez(DEST / 'head.npz', **model)
    selection = []
    for threshold in THRESHOLDS:
        total = PQStats(); actions = Counter()
        for r in rows:
            if r['partition'] != 'tune':
                continue
            chosen = None
            if threshold is not None and r['examples']:
                values = [predict_action(model, e['features']) for e in r['examples']]
                j = int(np.argmax(values))
                if values[j] > threshold:
                    chosen = r['examples'][j]
            total += PQStats(**(r['baseline'] if chosen is None else chosen['stats']))
            if chosen is not None:
                actions[chosen['action'][0]] += 1
        selection.append(dict(threshold=threshold, **asdict(total), pq=total.pq, actions=dict(actions)))
    best = max(selection, key=lambda r: r['pq'])
    result = dict(selected_threshold=best['threshold'], ridge_lambda=lam, tune_mse=loss,
                  losses=[dict(mse=a, ridge_lambda=b) for a, b, _ in candidates], results=selection,
                  no_validation_selection=True)
    write_json(DEST / 'selection.json', result)
    print(json.dumps(result, indent=2), flush=True)


def evaluate():
    if (DEST / 'completed.json').exists():
        print(json.dumps(read_json(DEST / 'summary.json'), indent=2)); return
    selection = read_json(DEST / 'selection.json'); threshold = selection['selected_threshold']
    with np.load(DEST / 'head.npz') as data:
        model = dict(data)
    state = torch.load(RANK / 'best.pt', map_location='cpu', weights_only=False)
    baseline = cache(RANK / 'nms_only/validation/nms_quality/predictions')
    records = defaultdict(list)
    for rec in read_json(ROOT / 'local_refiner/outputs/fp_progress_three_options_20261028/validation_records.json'):
        records[rec['file_name']].append(rec)
    assert not {fn[:8] for fn in records} & {fn[:8] for fn in training_records()}
    audit = {(c['record'], c['target_index']): c for c in read_json(ROOT / 'local_refiner/outputs/pass6_error_audit_20261101/cases.json') if c['type'] == 'FN'}
    totals = {n: PQStats() for n in ['incumbent', 'selector']}; tables = {n: [] for n in totals}
    changes = []; decisions = []
    folder = DEST / 'validation/predictions'; folder.mkdir(parents=True, exist_ok=True)
    for k, path in enumerate(sorted((OUT / 'val').glob('*.json'))):
        row = read_json(path); fn = row['file_name']; control = baseline[fn]; pred = control
        if threshold is not None:
            with np.load(path.with_suffix('.npz')) as data:
                x = data['X']
            scores = predict(state, x, row['confidence'], .5) if len(x) else np.zeros(0)
            options = action_features(row, x, scores)
            values = [predict_action(model, f) for a, f in options]
            if values and max(values) > threshold:
                j = int(np.argmax(values)); action = options[j][0]
                masks = [decode_mask(m).astype(bool) for m in row['masks']]
                assert counterfactual(row, scores, None, masks) == control
                pred = counterfactual(row, scores, action, masks)
                decisions.append(dict(file_name=fn, action=action, predicted_gain=values[j]))
        assert_disjoint(pred['masks']); write_json(folder / path.name, pred)
        for c in compare(records[fn], control, pred):
            item = audit.get((c['record'], c['target_index']))
            changes.append(dict(audit_id=None if item is None else item['id'], category=None if item is None else item['category'], **c))
        for name, p in [('incumbent', control), ('selector', pred)]:
            for rec in records[fn]:
                m = rle_iou(rec['gt'], p['masks'])
                totals[name] += pq_stats_from_iou(m)
                tables[name].append(dict(iou_matrix=torch.from_numpy(m), n_gt=len(rec['gt']), n_pred=m.shape[1]))
        if (k+1) % 25 == 0:
            print(f'Frozen selector validation {k+1}/{len(baseline)}', flush=True)
    official = official_functions()
    for name, table in tables.items():
        assert abs(float(official['get_pq_score'](pd.DataFrame(table))) - totals[name].pq) < 1e-10
    reloaded = PQStats()
    saved = cache(folder); assert set(saved) == set(records)
    for fn, pred in saved.items():
        assert_disjoint(pred['masks']); reloaded += stats(records[fn], pred)
    assert asdict(reloaded) == asdict(totals['selector'])
    for path, digest in read_json(DEST / 'plan.json')['protected'].items():
        assert sha(ROOT / path) == digest
    summary = dict(results=[dict(policy=name, **asdict(s), pq=s.pq) for name, s in totals.items()],
                   selected_threshold=threshold, changed_decisions=len(decisions),
                   gained=sum(c['group'] == 'Gained' for c in changes), lost=sum(c['group'] == 'Lost' for c in changes),
                   removed50_additional_recovered=sum(c['group'] == 'Gained' and c['category'] == 'Qualifying refined candidate removed' for c in changes),
                   organizer_parity=True, reload_exact=True, disjoint=True, protected_hashes_unchanged=True,
                   unchanged_control_selected=threshold is None)
    write_json(DEST / 'changes.json', changes); write_json(DEST / 'decisions.json', decisions)
    write_json(DEST / 'summary.json', summary); write_json(DEST / 'completed.json', dict(complete=True))
    write_json(DEST / 'status.json', dict(stage='complete'))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--stage', choices=['prepare', 'fit', 'evaluate', 'all'], default='all')
    args = parser.parse_args(); torch.set_num_threads(2); cv2.setNumThreads(1)
    for name, operation in [('prepare', prepare), ('fit', fit), ('evaluate', evaluate)]:
        if args.stage in ['all', name]: operation()
