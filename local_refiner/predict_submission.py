"""Generate and validate leaderboard CSV from the approved frozen YOLO + U-Net."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
import torch
from refiner import (HERE, WORKSPACE, DEFAULT_PROJECT, InstanceUNet, CropTransform,
    make_input, restore_prediction, encode_mask, decode_mask, read_json, write_json,
    sha, runtime)

FIELDS = ['filament_id','segmentation_rle']


def resolve_overlaps(row,shape=(2048,2048)):
    """Highest detector confidence owns contested pixels; original order breaks ties."""
    count = len(row['masks'])
    confidence = np.asarray(row['confidence'],dtype=float)
    if confidence.shape!=(count,) or not np.isfinite(confidence).all():
        raise ValueError('Each mask requires a finite detector confidence')
    occupied = np.zeros(shape,dtype=bool)
    overlapping = np.zeros(shape,dtype=bool)
    corrected = [None]*count
    changed,removed,assigned_pixels = 0,0,0
    for index in sorted(range(count),key=lambda i:(-confidence[i],i)):
        mask = decode_mask(row['masks'][index]).astype(bool)
        if mask.shape!=shape or not mask.any():
            raise ValueError('Invalid input mask for overlap resolution')
        collision = mask & occupied
        overlapping |= collision
        lost = int(collision.sum())
        changed += int(lost>0); removed += lost
        kept = mask & ~occupied
        assigned_pixels += int(mask.sum())-lost
        occupied |= mask
        if kept.any():
            corrected[index] = encode_mask(kept)
    kept_indices = [i for i,rle in enumerate(corrected) if rle is not None]
    if assigned_pixels!=int(occupied.sum()):
        raise RuntimeError('Overlap resolution changed foreground union area')
    source_ids = row.get('source_proposal_ids',row.get('proposal_ids',list(range(count))))
    result = dict(row,masks=[corrected[i] for i in kept_indices],
        confidence=[row['confidence'][i] for i in kept_indices],
        source_proposal_ids=[source_ids[i] for i in kept_indices],proposal_ids=list(range(len(kept_indices))))
    if 'coarse_masks' in row:
        result['coarse_masks'] = [row['coarse_masks'][i] for i in kept_indices]
    audit = dict(input_instances=count,output_instances=len(kept_indices),changed_instances=changed,
        overlapping_pixels=int(overlapping.sum()),removed_pixel_assignments=removed,
        dropped_source_proposal_ids=[source_ids[i] for i in range(count) if corrected[i] is None],
        foreground_union_preserved=True,policy='higher YOLO confidence owns overlap; original proposal order breaks ties')
    return result,audit


def expected_rows(predictions, shape=(2048,2048)):
    stems = [Path(name).stem for name in predictions]
    if len(stems)!=len(set(stems)):
        raise ValueError('Image stems must be unique')
    for filename in sorted(predictions):
        occupied = np.zeros(shape,dtype=bool)
        for index,rle in enumerate(predictions[filename]['masks'],1):
            mask = decode_mask(rle)
            if mask.shape!=shape or not mask.any():
                raise ValueError('Every filament row requires a nonempty original-size mask')
            if encode_mask(mask)!=rle:
                raise ValueError('RLE must be canonical compressed COCO counts')
            if np.any(occupied & mask.astype(bool)):
                raise ValueError(f'Instance masks overlap within image: {filename}')
            occupied |= mask.astype(bool)
            yield dict(filament_id=f'{Path(filename).stem}_{index}',segmentation_rle=rle['counts'])


def validate_csv(path,predictions,shape=(2048,2048)):
    expected = {row['filament_id']:row['segmentation_rle'] for row in expected_rows(predictions,shape)}
    seen = set()
    with path.open(newline='',encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames!=FIELDS:
            raise ValueError('CSV columns must be filament_id,segmentation_rle')
        for row in reader:
            key = row['filament_id']
            if key in seen or key not in expected or row['segmentation_rle']!=expected[key]:
                raise ValueError('Duplicate/unknown ID or mask changed in CSV')
            decoded = decode_mask(dict(size=list(shape),counts=row['segmentation_rle']))
            if decoded.shape!=shape or not decoded.any():
                raise ValueError('Invalid CSV mask')
            seen.add(key)
    if seen!=set(expected):
        raise ValueError('CSV has missing predictions')
    return dict(images_processed=len(predictions),filament_rows=len(seen),
        images_without_detections=[name for name,row in predictions.items() if not row['masks']],
        all_masks_roundtrip_exact=True,all_masks_disjoint=True,columns=FIELDS)


def export_csv(path,predictions,shape=(2048,2048)):
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w',newline='',encoding='utf-8') as stream:
        writer = csv.DictWriter(stream,fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(expected_rows(predictions,shape))
    audit = validate_csv(temporary,predictions,shape)
    temporary.replace(path)
    return audit


def predict_image(detector,network,path,reference,*,track_empty=False):
    with Image.open(path) as source:
        if source.size!=(2048,2048):
            raise ValueError(f'Unexpected image size: {path.name}')
        image = np.array(source.convert('L'),dtype=np.uint8)
    result = detector.predict(str(path),device='0',save=False,verbose=False,
        **reference['detector']['inference'])[0]
    coarse = [] if result.masks is None else list(result.masks.data.cpu().numpy().astype(np.uint8))
    confidence = [] if result.masks is None else result.boxes.conf.cpu().tolist()
    encoded,original,empty = [],[],0
    empty_indices = []
    settings = reference['refiner']
    for mask in coarse:
        if mask.shape!=(2048,2048) or not mask.any():
            raise ValueError('Detector emitted an invalid instance mask')
        original.append(encode_mask(mask))
        geometry = CropTransform.from_mask(mask,size=settings['canvas'],padding=settings['crop_padding'])
        x = torch.from_numpy(make_input(geometry.extract(image),geometry.extract(mask,mask=True),
            geometry.valid(),settings['input_mode']))[None].cuda()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
            probability = network(x).float().sigmoid()[0,0].cpu().numpy()
        prediction,was_empty = restore_prediction(probability,geometry,mask,settings['threshold'])
        empty += int(was_empty)
        if was_empty:
            empty_indices.append(len(encoded))
        rle = encode_mask(prediction)
        if not np.array_equal(decode_mask(rle),prediction):
            raise ValueError('Prediction changed during RLE encoding')
        encoded.append(rle)
    assert len(encoded)==len(confidence)==len(coarse)
    row = dict(file_name=path.name,confidence=confidence,proposal_ids=list(range(len(encoded))),
        masks=encoded,coarse_masks=original,empty_retained=empty)
    if track_empty:
        row['empty_indices'] = empty_indices
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',type=Path,default=DEFAULT_PROJECT/'data/test/images')
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    environment = runtime(43)
    reference = read_json(HERE/'working_reference.json')
    detector_path = WORKSPACE/reference['detector']['path']
    refiner_path = WORKSPACE/reference['refiner']['path']
    assert sha(detector_path)==reference['detector']['sha256']
    assert sha(refiner_path)==reference['refiner']['sha256']
    files = sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'})
    if not files:
        raise ValueError('No test images found')
    training_names = {p.name for p in (DEFAULT_PROJECT/'data/train/images').iterdir() if p.is_file()}
    if {p.name for p in files}&training_names:
        raise ValueError('Test input overlaps labeled training image filenames')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'predictions').mkdir()
    write_json(args.output/'plan.json',dict(reference=reference,environment=environment,
        image_directory=str(args.images.resolve()),images={p.name:sha(p) for p in files},
        reference_sha256=sha(HERE/'working_reference.json'),script_sha256=sha(Path(__file__)),
        scope='competition test inference; no hidden labels available'))
    from ultralytics import YOLO
    detector = YOLO(str(detector_path))
    checkpoint = torch.load(refiner_path,map_location='cpu',weights_only=False)
    assert checkpoint['epoch']==reference['refiner']['epoch']
    assert checkpoint['input_mode']==reference['refiner']['input_mode']
    network = InstanceUNet().cuda().eval()
    network.load_state_dict(checkpoint['model'])
    saved_folder = HERE/'outputs/full_train_image43_10ep_validation/predictions'
    expected = {read_json(p)['file_name']:read_json(p) for p in saved_folder.glob('*.json')}
    names = sorted(expected)
    checks = []
    for filename in [names[0],names[len(names)//2],names[-1]]:
        actual = predict_image(detector,network,DEFAULT_PROJECT/'data/train/images'/filename,reference)
        if actual['masks']!=expected[filename]['masks']:
            raise ValueError(f'End-to-end validation prediction mismatch: {filename}')
        np.testing.assert_allclose(actual['confidence'],expected[filename]['confidence'],rtol=0,atol=1e-6)
        checks.append(dict(file_name=filename,mask_count=len(actual['masks']),all_masks_exact=True))
    write_json(args.output/'reference_checks.json',checks)
    print(f'End-to-end reference checks passed on {len(checks)} validation images',flush=True)
    predictions = {}
    started = time.perf_counter()
    for i,path in enumerate(files):
        result = predict_image(detector,network,path,reference)
        write_json(args.output/'predictions'/f'{i:04d}.json',result)
        predictions[path.name] = result
        row = dict(processed=i+1,total=len(files),file_name=path.name,instances=len(result['masks']),
            elapsed_seconds=time.perf_counter()-started)
        with (args.output/'progress.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
    exclusive,overlap_audits = {},{}
    for filename,row in predictions.items():
        exclusive[filename],overlap_audits[filename] = resolve_overlaps(row)
    (args.output/'submission_predictions').mkdir()
    for i,row in enumerate(exclusive.values()):
        write_json(args.output/'submission_predictions'/f'{i:04d}.json',row)
    write_json(args.output/'overlap_resolution.json',overlap_audits)
    audit = export_csv(args.output/'submission.csv',exclusive)
    audit.update(csv_sha256=sha(args.output/'submission.csv'),reference_checks=checks,
        elapsed_seconds=time.perf_counter()-started,
        no_detection_policy='No filament rows for an image with no proposals; processed image retained in audit',
        kaggle_uploaded=False,kaggle_score=None)
    write_json(args.output/'audit.json',audit)
    print(json.dumps(audit,indent=2),flush=True)
    print('CSV: '+str((args.output/'submission.csv').resolve()),flush=True)


if __name__=='__main__':
    main()
