"""Evaluate frozen YOLO + basic U-Net at five confidence cutoffs; no training."""
from dataclasses import asdict
import csv
import json
import time
import numpy as np
import pandas as pd
from PIL import Image
from pycocotools import mask as mask_utils
import torch
from refiner import (HERE, WORKSPACE, read_json, write_json, sha, runtime,
                     InstanceUNet, CropTransform, make_input, restore_prediction,
                     encode_mask, decode_mask, PQStats, pq_stats_from_iou)
from validation_pq import ValidationPQ, rle_iou
from predict_submission import resolve_overlaps
from official_notebook_diagnostics import official_functions, REFERENCE

CUTS = (.25, .20, .15, .10, .05)
OUT = HERE / 'outputs/confidence_sweep_basic_unet_20261026'
LOW = HERE / 'outputs/low_confidence_candidates_20261019'
HIGH = HERE / 'outputs/full_train_image43_10ep_validation/predictions'
LEGAL = HERE / 'outputs/leaderboard_yolo_unet_disjoint_20261018/validation/predictions'


def select_candidates(row, cutoff):
    ids = [i for i, confidence in enumerate(row['confidence']) if confidence >= cutoff]
    return dict(file_name=row['file_name'], masks=[row['masks'][i] for i in ids],
                confidence=[row['confidence'][i] for i in ids],
                source_proposal_ids=[row.get('source_proposal_ids', list(range(len(row['masks']))))[i] for i in ids])


def refine_one(network, image, rle, settings):
    coarse = decode_mask(rle)
    geometry = CropTransform.from_mask(coarse, size=settings['canvas'], padding=settings['crop_padding'])
    x = torch.from_numpy(make_input(geometry.extract(image), geometry.extract(coarse, mask=True),
                                  geometry.valid(), settings['input_mode']))[None].cuda()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        probability = network(x).float().sigmoid()[0, 0].cpu().numpy()
    mask, empty = restore_prediction(probability, geometry, coarse, settings['threshold'])
    return encode_mask(mask), bool(empty)


def assert_disjoint(masks):
    if masks:
        areas = mask_utils.area(masks)
        assert (areas > 0).all()
        assert int(areas.sum()) == int(mask_utils.area(mask_utils.merge(masks)))


def main():
    environment = runtime(43)
    reference = read_json(HERE / 'working_reference.json')
    settings = reference['refiner']
    detector = WORKSPACE / reference['detector']['path']
    checkpoint_path = WORKSPACE / settings['path']
    assert sha(detector) == reference['detector']['sha256']
    assert sha(checkpoint_path) == settings['sha256']
    low_plan = read_json(LOW / 'plan.json')
    assert low_plan['checkpoint_sha256'] == sha(detector)
    assert low_plan['inference'] == dict(reference['detector']['inference'], conf=.01)
    OUT.mkdir(exist_ok=True)
    (OUT / 'refined_candidates').mkdir(exist_ok=True)
    sources = {str(p.relative_to(WORKSPACE)): sha(p) for folder in [LOW / 'predictions', HIGH, LEGAL]
               for p in sorted(folder.glob('*.json'))}
    signature = dict(reference=reference, cutoffs=CUTS, source_hashes=sources,
                     script_sha256=sha(__file__), notebook_sha256=sha(REFERENCE),
                     support_sha256={p.name: sha(p) for p in [HERE / n for n in
                         ['refiner.py', 'model.py', 'geometry.py', 'validation_pq.py', 'predict_submission.py']]})
    signature = json.loads(json.dumps(signature))
    if (OUT / 'plan.json').exists():
        assert read_json(OUT / 'plan.json')['signature'] == signature, 'Inputs changed; use a fresh output folder'
    write_json(OUT / 'plan.json', dict(signature=signature, environment=environment,
        hypothesis='Lower confidence may recover useful masks but false positives can offset recovery.',
        scope='Full held-out validation, already used for model selection; not hidden test.',
        no_training=True, no_new_yolo_inference=True,
        refiner_inference='Reuse exact saved high-confidence masks; refine new candidates once, identically across cuts.'))
    validator = ValidationPQ()
    high = {read_json(p)['file_name']: read_json(p) for p in HIGH.glob('*.json')}
    legal_reference = {read_json(p)['file_name']: read_json(p) for p in LEGAL.glob('*.json')}
    low = {read_json(p)['file_name']: read_json(p) for p in (LOW / 'predictions').glob('*.json')}
    assert set(low) == set(high) == set(legal_reference) == set(validator.records)
    assert len(low) == 146 and sum(map(len, validator.records.values())) == 234
    network = InstanceUNet().cuda().eval()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert checkpoint['epoch'] == settings['epoch'] and checkpoint['input_mode'] == settings['input_mode']
    network.load_state_dict(checkpoint['model'])
    # Recompute examples to confirm the current inference path reproduces saved refinement.
    parity = []
    for filename in [sorted(high)[i] for i in [0, len(high)//2, len(high)-1]]:
        if not high[filename]['masks']:
            continue
        image = np.asarray(Image.open(validator.reader.image_dir / filename).convert('L'))
        actual, _ = refine_one(network, image, validator.caches[filename]['masks'][0], settings)
        assert actual == high[filename]['masks'][0], filename
        parity.append(dict(file_name=filename, proposal=0, exact=True))
    write_json(OUT / 'refiner_parity.json', parity)
    totals = {cut: PQStats() for cut in CUTS}
    tables = {cut: [] for cut in CUTS}
    changes = {cut: dict(gained=0, lost=0, zero_overlap_gt=0, physical_predictions=0) for cut in CUTS}
    records_out = []
    started = time.perf_counter()
    for number, (filename, records) in enumerate(validator.records.items()):
        row = low[filename]
        ids_high = [i for i, c in enumerate(row['confidence']) if c >= .25]
        assert [row['masks'][i] for i in ids_high] == validator.caches[filename]['masks']
        assert [row['confidence'][i] for i in ids_high] == high[filename]['confidence']
        assert row['returned_detections'] < 100, 'Cached inference hit max_det; threshold filtering might differ'
        path = OUT / 'refined_candidates' / f'{number:04d}.json'
        if path.exists():
            refined = read_json(path)
            expected = select_candidates(row, min(CUTS))
            assert refined['file_name'] == filename and refined['source_proposal_ids'] == expected['source_proposal_ids']
            assert refined['confidence'] == expected['confidence']
            assert len(refined['masks']) == len(expected['masks'])
        else:
            image = np.asarray(Image.open(validator.reader.image_dir / filename).convert('L'))
            existing = dict(zip(ids_high, high[filename]['masks']))
            encoded, confidence, ids, empty = [], [], [], 0
            for i, c in enumerate(row['confidence']):
                if c < min(CUTS):
                    continue
                if i in existing:
                    result = existing[i]
                else:
                    result, was_empty = refine_one(network, image, row['masks'][i], settings)
                    empty += int(was_empty)
                encoded.append(result); confidence.append(c); ids.append(i)
            refined = dict(file_name=filename, masks=encoded, confidence=confidence,
                           source_proposal_ids=ids, new_candidates_empty_retained=empty)
            temporary = path.with_suffix('.tmp')
            write_json(temporary, refined)
            temporary.replace(path)
        baseline_hits = {}
        for record in records:
            baseline_hits[record['image_id']] = (rle_iou(record['gt'], legal_reference[filename]['masks']) > .5).any(axis=1)
        for cut in CUTS:
            legal, audit = resolve_overlaps(select_candidates(refined, cut))
            assert_disjoint(legal['masks'])
            if cut == .25:
                assert legal['masks'] == legal_reference[filename]['masks'], 'Baseline output changed'
                assert legal['confidence'] == legal_reference[filename]['confidence']
            changes[cut]['physical_predictions'] += len(legal['masks'])
            for record in records:
                matrix = rle_iou(record['gt'], legal['masks'])
                score = pq_stats_from_iou(matrix)
                totals[cut] += score
                hits = (matrix > .5).any(axis=1)
                before = baseline_hits[record['image_id']]
                gained, lost = int((~before & hits).sum()), int((before & ~hits).sum())
                changes[cut]['gained'] += gained
                changes[cut]['lost'] += lost
                changes[cut]['zero_overlap_gt'] += int((~(matrix > 0).any(axis=1)).sum())
                tables[cut].append(dict(iou_matrix=torch.from_numpy(matrix), n_gt=len(record['gt']), n_pred=len(legal['masks'])))
                records_out.append(dict(cutoff=cut, image_id=record['image_id'], file_name=filename,
                    **asdict(score), pq=score.pq, gained=gained, lost=lost,
                    gained_target_indices=np.flatnonzero(~before & hits).tolist(),
                    lost_target_indices=np.flatnonzero(before & ~hits).tolist()))
        if (number + 1) % 10 == 0 or number + 1 == len(low):
            progress = dict(images=number+1, total=len(low), elapsed_seconds=time.perf_counter()-started)
            write_json(OUT / 'progress.json', progress)
            print(json.dumps(progress), flush=True)
    official = official_functions()
    results = []
    for cut in CUTS:
        score = totals[cut]
        official_pq = float(official['get_pq_score'](pd.DataFrame(tables[cut])))
        assert abs(official_pq - score.pq) < 1e-10
        results.append(dict(cutoff=cut, **asdict(score), pq=score.pq,
            delta_pq=score.pq-totals[.25].pq, **changes[cut],
            precision=score.tp/(score.tp+score.fp), recall=score.tp/(score.tp+score.fn),
            sq=score.iou_sum/score.tp, rq=score.tp/(score.tp+.5*(score.fp+score.fn)),
            organizer_pq=official_pq))
    expected = reference['submission_postprocessing']['validation']
    for key in ['tp', 'fp', 'fn']:
        assert results[0][key] == expected[key]
    assert abs(results[0]['pq'] - expected['pq']) < 1e-12
    summary = dict(results=results, physical_images=146, independent_records=234,
        refiner='basic ResNet34 U-Net epoch 10', detector='YOLO11s-seg epoch 26',
        baseline_exact_disjoint_parity_images=146, all_masks_disjoint=True,
        organizer_notebook_pq_verified=True, reference_unchanged=True,
        no_training=True, no_test_inference=True, elapsed_seconds=time.perf_counter()-started,
        candidate_cache_sha256={p.name: sha(p) for p in sorted((OUT/'refined_candidates').glob('*.json'))})
    write_json(OUT / 'summary.json', summary)
    (OUT / 'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records_out), encoding='utf-8')
    with (OUT / 'results.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(results[0]))
        writer.writeheader(); writer.writerows(results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
