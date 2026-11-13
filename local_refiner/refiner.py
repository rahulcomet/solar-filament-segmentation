"""Prepare target-conditioned triplets, verify local learning, and score fixed proposals."""
import argparse
from collections import defaultdict
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
DEFAULT_PROJECT = WORKSPACE
sys.path.insert(0, str(DEFAULT_PROJECT))
os.environ.setdefault('YOLO_CONFIG_DIR', str(HERE / '.ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(HERE / '.matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')
from src.dataset import FilamentDataset
from src.masks import decode_mask, encode_mask, mask_iou
from src.metrics import PQStats, evaluate_record, pairwise_mask_iou, pq_stats_from_iou
from geometry import CropTransform, clear_associations, restore_prediction
from model import InstanceUNet, mask_loss

INFERENCE = dict(imgsz=2048, conf=.25, iou=.7, max_det=100, retina_masks=True)
FROZEN = WORKSPACE / 'completed_loss_comparison/bce'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def day(filename):
    return Path(filename).stem[:8]


def partition(filename):
    # Same physical image and all annotators always share a refiner partition.
    return 'tune' if int(hashlib.sha256(day(filename).encode()).hexdigest()[:8], 16) % 5 == 0 else 'fit'


def runtime(seed=42):
    torch.set_num_threads(2)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this local pilot; CPU fallback is disabled')
    x = torch.arange(1024, dtype=torch.float32, device='cuda')
    assert (x*x).sum().item() == 357389824.
    torch.cuda.synchronize()
    return dict(python=sys.version, executable=sys.executable,
                gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                packages={name: version(name) for name in ['torch','torchvision','ultralytics','numpy']})


def load_readers(project):
    readers = {s: FilamentDataset(project, s) for s in ['train', 'val']}
    train_days = {day(info['file_name']) for info in readers['train'].images}
    val_days = {day(info['file_name']) for info in readers['val'].images}
    if train_days & val_days:
        raise ValueError('Original train/validation observation days overlap')
    return readers


def data_hashes(project):
    return {name: sha(project / name) for name in ['data/train/annotations.json',
            'data/splits/train_ids.json', 'data/splits/val_ids.json']}


def check_frozen():
    summary = read_json(FROZEN / 'summary.json')
    if sha(FROZEN / 'checkpoint.pt') != summary['checkpoint_sha256']:
        raise ValueError('Frozen detector checkpoint hash changed')
    if summary['inference'] != INFERENCE:
        raise ValueError('Validation cache inference settings differ')
    return summary


def render_triplet(image, coarse, target, path, title, third_label='Independent annotation'):
    canvas = Image.new('RGB', (image.shape[1]*3, image.shape[0]+40), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (mask, label, color) in enumerate([(None,'Original',(0,0,0)),
                       (coarse,'YOLO target prompt',(255,100,0)),(target,third_label,(0,255,90))]):
        panel = np.repeat(image[...,None], 3, axis=2).astype(np.float32)
        if mask is not None:
            panel[mask.astype(bool)] = .4*panel[mask.astype(bool)] + .6*np.array(color)
        canvas.paste(Image.fromarray(panel.astype(np.uint8)), (i*image.shape[1],40))
        draw.text((i*image.shape[1]+5,22),label,fill='black')
    draw.text((5,4), title, fill='black')
    canvas.save(path)


def prepare(args):
    environment = runtime()
    frozen = check_frozen()
    readers = load_readers(args.project)
    if args.output.exists():
        raise FileExistsError('Choose a fresh triplet output directory')
    args.output.mkdir(parents=True)
    (args.output / 'triplets').mkdir()
    (args.output / 'gallery').mkdir()
    (args.output / 'prediction_cache').mkdir()
    from ultralytics import YOLO
    model = YOLO(str(FROZEN / 'checkpoint.pt'))
    train = readers['train']
    groups = defaultdict(list)
    for index, info in enumerate(train.images):
        groups[info['file_name']].append(index)
    names = sorted(groups, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
    tune_goal = max(4, args.pairs//5)
    goals = dict(fit=args.pairs-tune_goal, tune=tune_goal)
    counts = dict(fit=0, tune=0)
    rows, rejects = [], defaultdict(int)
    seen_targets = set()
    for filename in names:
        part = partition(filename)
        if counts[part] >= goals[part]:
            continue
        prediction = model.predict(str(train.image_dir / filename), device='0', verbose=False,
                                   save=False, **INFERENCE)[0]
        masks = [] if prediction.masks is None else list(prediction.masks.data.cpu().numpy().astype(np.uint8))
        scores = [] if prediction.masks is None else prediction.boxes.conf.cpu().tolist()
        write_json(args.output / 'prediction_cache' / (Path(filename).stem+'.json'),
                   dict(file_name=filename,masks=[encode_mask(m) for m in masks],confidence=scores))
        candidates = []
        samples = {}
        for index in groups[filename]:
            sample = train[index]
            samples[index] = sample
            ious = pairwise_mask_iou(sample['masks'], masks)
            ga = np.array([m.sum() for m in sample['masks']],dtype=float)
            pa = np.array([m.sum() for m in masks],dtype=float)
            intersection = ious*(ga[:,None]+pa[None,:])/(1+ious)
            coverage = np.divide(intersection, ga[:,None], out=np.zeros_like(ious), where=ga[:,None]>0)
            for proposal, target, overlap in clear_associations(ious, coverage):
                target_mask = sample['masks'][target]
                geometry = CropTransform.from_mask(masks[proposal])
                x0,y0,x1,y1 = geometry.box
                visible = float(target_mask[y0:y1,x0:x1].sum()/target_mask.sum())
                if visible < .9:
                    rejects['target_outside_crop'] += 1
                    continue
                candidates.append((index,proposal,target,overlap,geometry,visible))
        # One triplet per physical image in the first pass broadens day coverage.
        # Cycle among small/medium/large annotations instead of taking only high IoU.
        candidates.sort(key=lambda c: int(samples[c[0]]['masks'][c[2]].sum()))
        if candidates:
            choices = [candidates[0],candidates[len(candidates)//2],candidates[-1]]
            index,proposal,target,overlap,geometry,visible = choices[len(rows)%3]
            sample = samples[index]
            target_id = sample['annotations'][target]['id']
            assert (sample['image_id'],target_id) not in seen_targets
            seen_targets.add((sample['image_id'],target_id))
            image = geometry.extract(sample['image'])
            prompt = geometry.extract(masks[proposal],mask=True)
            truth = geometry.extract(sample['masks'][target],mask=True)
            if not prompt.any() or not truth.any():
                rejects['vanished_after_resize'] += 1
                continue
            restored = geometry.restore(prompt,mask=True)
            roundtrip = mask_iou(masks[proposal],restored)
            if geometry.resized == (geometry.box[3]-geometry.box[1],geometry.box[2]-geometry.box[0]):
                assert np.array_equal(masks[proposal],restored)
            key = f'{len(rows):04d}'
            np.savez_compressed(args.output / 'triplets' / (key+'.npz'),image=image,prompt=prompt,
                                target=truth,valid=geometry.valid())
            row = dict(key=key,partition=part,record_index=index,image_id=sample['image_id'],
                       annotation_id=target_id,file_name=filename,day=day(filename),proposal=proposal,
                       confidence=scores[proposal],original_iou=overlap,
                       source_gt_area=int(sample['masks'][target].sum()),visible_gt_fraction=visible,
                       roundtrip_iou=roundtrip,geometry=asdict(geometry),
                       image_sha256=sha(sample['image_path']))
            row['triplet_sha256'] = sha(args.output/'triplets'/(key+'.npz'))
            rows.append(row)
            counts[part] += 1
            if len(rows) <= 20:
                render_triplet(image,prompt,truth,args.output/'gallery'/f'{key}.png',
                               f'{part} | {sample["image_id"]} | IoU {overlap:.3f} | roundtrip {roundtrip:.3f}')
            print(f'Triplets {len(rows)}/{args.pairs}: {filename} ({part}, IoU {overlap:.3f})',flush=True)
        if all(counts[p]>=goals[p] for p in counts):
            break
    if counts != goals:
        raise RuntimeError(f'Insufficient clear pairs: {counts}, requested {goals}')
    assert not ({r['day'] for r in rows if r['partition']=='fit'} & {r['day'] for r in rows if r['partition']=='tune'})
    plan = dict(checkpoint_sha256=frozen['checkpoint_sha256'],inference=INFERENCE,
                project=str(args.project),metadata_sha256=data_hashes(args.project),environment=environment,
                counts=counts,rejected=dict(rejects),canvas=512,threshold=.5,empty_policy='retain_original',
                training_prompt_scope='in-sample to frozen YOLO',
                association='mutual-best IoU>=0.1; runner-up<half; at most one GT coverage>=0.25',rows=rows)
    write_json(args.output/'manifest.json',plan)
    links = ''.join(f'<h3>{r["key"]}: {r["partition"]}</h3><img width="100%" src="{r["key"]}.png">' for r in rows[:20])
    (args.output/'gallery/index.html').write_text('<meta charset="utf-8"><title>Training triplets</title>'+links,encoding='utf-8')
    print(json.dumps(dict(counts=counts,roundtrip_min=min(r['roundtrip_iou'] for r in rows),
                          below_half_iou=sum(r['original_iou']<.5 for r in rows)),indent=2))


def corrupt_prompt(prompt):
    result = prompt.copy()
    kind = random.randrange(5)
    kernel = np.ones((3,3),np.uint8)
    if kind == 1:
        result = cv2.erode(result,kernel)
    elif kind == 2:
        result = cv2.dilate(result,kernel)
    elif kind == 3:
        transform = np.float32([[1,0,random.randint(-4,4)],[0,1,random.randint(-4,4)]])
        result = cv2.warpAffine(result,transform,(result.shape[1],result.shape[0]),flags=cv2.INTER_NEAREST)
    elif kind == 4:
        yy,xx = np.nonzero(result)
        if len(xx):
            y,x = int(np.median(yy)),int(np.median(xx))
            result[max(0,y-8):y+8,max(0,x-16):x+16] = 0
    return result if result.any() else prompt


def make_input(image, prompt, valid, input_mode='image'):
    if input_mode not in {'image','mask_only'}:
        raise ValueError('Unknown input mode')
    channel = image.astype(np.float32)/255 if input_mode=='image' else .5*valid.astype(np.float32)
    return np.stack([channel,prompt.astype(np.float32)])


def tensors(path, augment=False, input_mode='image'):
    with np.load(path) as item:
        image,prompt,target,valid = (item[k].copy() for k in ['image','prompt','target','valid'])
    if augment:
        prompt = corrupt_prompt(prompt)
    x = make_input(image,prompt,valid,input_mode)
    return (torch.from_numpy(x)[None].cuda(),torch.from_numpy(target.astype(np.float32))[None,None].cuda(),
            torch.from_numpy(valid.astype(np.float32))[None,None].cuda())


def training_rounds(training, rounds, updates_per_round=None):
    remaining = []
    for _ in range(rounds):
        if updates_per_round is None:
            order = training.copy()
            random.shuffle(order)
        else:
            order = []
            while len(order)<updates_per_round:
                if not remaining:
                    remaining = training.copy()
                    random.shuffle(remaining)
                count = min(len(remaining),updates_per_round-len(order))
                order.extend(remaining[:count])
                remaining = remaining[count:]
        yield order


def fit(args):
    environment = runtime(args.seed)
    manifest = read_json(args.pairs/'manifest.json')
    if data_hashes(Path(manifest['project'])) != manifest['metadata_sha256']:
        raise ValueError('Source metadata changed since triplet generation')
    if args.output.exists():
        raise FileExistsError('Choose a fresh training output directory')
    args.output.mkdir(parents=True)
    training = [r for r in manifest['rows'] if r['partition']=='fit']
    tuning = [r for r in manifest['rows'] if r['partition']=='tune']
    triplet_hashes = {row['key']:sha(args.pairs/'triplets'/(row['key']+'.npz')) for row in manifest['rows']}
    for row in manifest['rows']:
        if row.get('triplet_sha256',triplet_hashes[row['key']]) != triplet_hashes[row['key']]:
            raise ValueError('Triplet bytes changed after preparation')
    if args.mode == 'overfit':
        training = training[:min(16,len(training))]
        tuning = training
    selection = getattr(args,'selection','tuning-loss')
    if not training or (selection=='tuning-loss' and not tuning):
        raise ValueError('Training examples are required; tuning-loss selection also requires tuning examples')
    validator = None
    if selection=='validation-pq':
        if args.mode!='pilot' or tuning:
            raise ValueError('Validation-PQ selection requires pilot mode and a full-training manifest with no tuning rows')
        from validation_pq import ValidationPQ
        validator = ValidationPQ(Path(manifest['project']))
    network = InstanceUNet()
    initial_digest = hashlib.sha256()
    for name,value in network.state_dict().items():
        initial_digest.update(name.encode())
        initial_digest.update(value.numpy().tobytes())
    network = network.cuda()
    optimizer = torch.optim.AdamW(network.parameters(),lr=args.lr,weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda',init_scale=128.)
    torch.cuda.reset_peak_memory_stats()
    updates = 0
    optimizer.register_step_post_hook(lambda *_: step_count())
    def step_count():
        nonlocal updates
        updates += 1
    best = float('inf')
    corruption_digest = hashlib.sha256()
    (args.output/'source_snapshot').mkdir()
    for name in ['refiner.py','geometry.py','model.py']:
        shutil.copyfile(HERE/name,args.output/'source_snapshot'/name)
    if validator:
        shutil.copyfile(HERE/'validation_pq.py',args.output/'source_snapshot/validation_pq.py')
    started = time.perf_counter()
    write_json(args.output/'plan.json',dict(mode=args.mode,epochs=args.epochs,lr=args.lr,seed=args.seed,
               input_mode=args.input_mode,selection=selection,max_updates=getattr(args,'max_updates',None),
               updates_per_round=getattr(args,'updates_per_round',None),initial_model_sha256=initial_digest.hexdigest(),
               constant_image_policy='0.5 in valid crop; 0 in padding' if args.input_mode=='mask_only' else None,
               architecture='ResNet34 U-Net; 2-channel; random initialization; GroupNorm',
               loss='valid-pixel BCE + soft Dice',threshold=.5,empty_policy='retain_original',
               pairs=str(args.pairs.resolve()),pairs_sha256=sha(args.pairs/'manifest.json'),
               triplet_sha256=triplet_hashes,
               implementation_sha256={name:sha(HERE/name) for name in ['refiner.py','geometry.py','model.py']},
               checkpoint_sha256=manifest['checkpoint_sha256'],environment=environment))
    orders = training_rounds(training,args.epochs,getattr(args,'updates_per_round',None))
    for epoch,order in enumerate(orders,1):
        network.train()
        losses = []
        for row in order:
            x,y,v = tensors(args.pairs/'triplets'/(row['key']+'.npz'),augment=args.mode=='pilot',input_mode=args.input_mode)
            corruption_digest.update(f'{epoch}:{row["key"]}:'.encode())
            corruption_digest.update(np.packbits(x[0,1].detach().cpu().numpy()>.5).tobytes())
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):
                loss = mask_loss(network(x),y,v)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(network.parameters(),5.,error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            losses.append(loss.item())
            if getattr(args,'max_updates',None) and updates >= args.max_updates:
                break
        network.eval()
        tune_loss,dice,coarse_dice = [],[],[]
        with torch.no_grad():
            for row in tuning:
                x,y,v = tensors(args.pairs/'triplets'/(row['key']+'.npz'),input_mode=args.input_mode)
                with torch.autocast('cuda',dtype=torch.float16):
                    logits = network(x)
                tune_loss.append(mask_loss(logits,y,v).item())
                for target_list,p in [(dice,(logits.sigmoid()>=.5).float()),(coarse_dice,x[:,1:2])]:
                    target_list.append(((2*(p*y*v).sum()+1)/((p*v).sum()+(y*v).sum()+1)).item())
        validation = validator.score(network,args.output/f'validation_epoch_{epoch:02d}',epoch,args.input_mode) if validator else None
        selection_loss = -validation['scores']['refined']['pq'] if validation else float(np.mean(tune_loss))
        row = dict(epoch=epoch,train_loss=float(np.mean(losses)),selection_loss=selection_loss,
                   diagnostic_dice=float(np.mean(dice)) if dice else None,coarse_dice=float(np.mean(coarse_dice)) if coarse_dice else None,
                   optimizer_updates=updates,scope='validation checkpoint selection' if validator else 'training-fit diagnostic' if args.mode=='overfit' else 'training-partition tuning days')
        if validation:
            row['validation_pq'] = validation['scores']['refined']['pq']
            with (args.output/'competition_pq.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(epoch=epoch,optimizer_updates=updates,**validation['scores']['refined'],
                    gained_gt=validation['gained_gt'],lost_gt=validation['lost_gt']))+'\n')
        row['order_and_corruption_sha256'] = corruption_digest.hexdigest()
        with (args.output/'epochs.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        checkpoint = dict(model=network.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                          epoch=epoch,mode=args.mode,pairs_sha256=sha(args.pairs/'manifest.json'),
                          input_mode=args.input_mode,seed=args.seed,
                          selection=selection,
                          detector_sha256=manifest['checkpoint_sha256'],threshold=.5)
        torch.save(checkpoint,args.output/'last.pt')
        if row['selection_loss'] < best:
            best = row['selection_loss']
            torch.save(checkpoint,args.output/'best.pt')
        if getattr(args,'max_updates',None) and updates >= args.max_updates:
            break
    if updates == 0:
        raise RuntimeError('No optimizer updates; this is not a successful training trial')
    with torch.no_grad():
        for item in tuning[:8]:
            x,y,v = tensors(args.pairs/'triplets'/(item['key']+'.npz'),input_mode=args.input_mode)
            prediction = (network(x).sigmoid()[0,0].cpu().numpy()>=.5) & v[0,0].cpu().numpy().astype(bool)
            # Four-panel training diagnostics, not held-out results.
            with np.load(args.pairs/'triplets'/(item['key']+'.npz')) as d:
                render_triplet(d['image'],d['prompt'],prediction,args.output/(item['key']+'_refined.png'),
                               'Image | original prompt | network output (target in pair gallery)',third_label='U-Net output')
    write_json(args.output/'completed.json',dict(optimizer_updates=updates,epochs=epoch,
               peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
               elapsed_seconds=time.perf_counter()-started,scope=row['scope'],final_diagnostic=row))


def evaluate(args):
    environment = runtime()
    frozen = check_frozen()
    checkpoint = torch.load(args.weights,map_location='cpu',weights_only=False)
    if checkpoint['mode'] != 'pilot':
        raise ValueError('Overfit checkpoints are training diagnostics; fit a pilot before held-out scoring')
    plan = read_json(args.weights.parent/'plan.json')
    input_mode = checkpoint.get('input_mode','image')
    if input_mode!=plan.get('input_mode','image'):
        raise ValueError('Checkpoint and training plan disagree on input mode')
    manifest_path = Path(plan['pairs'])/'manifest.json'
    if sha(manifest_path) != checkpoint['pairs_sha256']:
        raise ValueError('Training manifest changed')
    manifest = read_json(manifest_path)
    if data_hashes(args.project) != manifest['metadata_sha256']:
        raise ValueError('Evaluation source metadata differs from training')
    if checkpoint['detector_sha256'] != frozen['checkpoint_sha256']:
        raise ValueError('Refiner and cached detector checkpoints differ')
    if args.output.exists():
        raise FileExistsError('Choose a fresh evaluation output')
    args.output.mkdir(parents=True)
    (args.output/'predictions').mkdir()
    network = InstanceUNet().cuda().eval()
    network.load_state_dict(checkpoint['model'])
    from scripts.analyze_yolo_errors import diagnose_record
    validation = load_readers(args.project)['val']
    caches = {read_json(p)['file_name']:p for p in (FROZEN/'prediction_cache').glob('*.json')}
    if set(caches) != {info['file_name'] for info in validation.images}:
        raise ValueError('Frozen cache does not cover exactly the validation images')
    totals = {name:PQStats() for name in ['baseline','roundtrip','refined']}
    groups = defaultdict(list)
    for i,info in enumerate(validation.images):
        groups[info['file_name']].append(i)
    records,empty,gained,lost = [],0,0,0
    topology = {name:dict(merge_candidates=0,fragment_candidates=0) for name in ['baseline','refined']}
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for number,(filename,indices) in enumerate(groups.items()):
        sample = validation[indices[0]]
        cache = read_json(caches[filename])
        masks = [decode_mask(m) for m in cache['masks']]
        refined,roundtrip = [],[]
        for mask in masks:
            if not mask.any():
                refined.append(mask.copy())
                roundtrip.append(mask.copy())
                empty += 1
                continue
            geometry = CropTransform.from_mask(mask)
            prompt = geometry.extract(mask,mask=True)
            image = geometry.extract(sample['image'])
            x = torch.from_numpy(make_input(image,prompt,geometry.valid(),input_mode))[None].cuda()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                probability = network(x).float().sigmoid()[0,0].cpu().numpy()
            prediction,was_empty = restore_prediction(probability,geometry,mask,checkpoint['threshold'])
            empty += int(was_empty)
            refined.append(prediction)
            roundtrip.append(geometry.restore(prompt,mask=True))
        assert len(masks) == len(refined) == len(cache['confidence'])
        if number < 6 and masks:
            geometry = CropTransform.from_mask(masks[0]) if masks[0].any() else None
            if geometry is not None:
                render_triplet(geometry.extract(sample['image']),geometry.extract(masks[0],mask=True),
                               geometry.extract(refined[0],mask=True),args.output/f'example_{number:02d}.png',
                               f'{filename}: original | coarse mask | refined mask',third_label='U-Net refined mask')
        write_json(args.output/'predictions'/f'{number:04d}.json',dict(file_name=filename,
                   confidence=cache['confidence'],proposal_ids=list(range(len(masks))),
                   masks=[encode_mask(m) for m in refined]))
        for i in indices:
            record = validation[i]
            entry = dict(image_id=record['image_id'],file_name=filename,day=day(filename))
            overlaps = {}
            for name,predictions in [('baseline',masks),('roundtrip',roundtrip),('refined',refined)]:
                overlaps[name] = pairwise_mask_iou(record['masks'],predictions)
                stats = pq_stats_from_iou(overlaps[name])
                totals[name] += stats
                entry[name] = dict(**asdict(stats),pq=stats.pq)
                if name in topology and not args.skip_topology:
                    diagnostics,_,_ = diagnose_record(record['masks'],predictions,cache['confidence'])
                    topology[name]['merge_candidates'] += sum(r['merge_candidate'] for r in diagnostics)
                    topology[name]['fragment_candidates'] += sum(r['fragment_candidate'] for r in diagnostics)
            old = (overlaps['baseline']>.5).any(axis=1)
            new = (overlaps['refined']>.5).any(axis=1)
            entry.update(gained=int((~old & new).sum()),lost=int((old & ~new).sum()))
            gained += entry['gained']; lost += entry['lost']
            records.append(entry)
        print(f'Evaluated {number+1}/{len(groups)} images',flush=True)
    if (abs(totals['baseline'].pq-frozen['pq'])>1e-10 or
            any(getattr(totals['baseline'],key)!=frozen[key] for key in ['tp','fp','fn'])):
        raise RuntimeError('Cached baseline no longer reproduces frozen checkpoint score')
    with (args.output/'records.jsonl').open('w') as stream:
        for row in records:
            stream.write(json.dumps(row)+'\n')
    days = defaultdict(lambda:{name:PQStats() for name in ['baseline','refined']})
    for record in records:
        for name in ['baseline','refined']:
            days[record['day']][name] += PQStats(**{k:record[name][k] for k in ['tp','fp','fn','iou_sum']})
    arrays = {name:np.array([[getattr(values[name],k) for k in ['tp','fp','fn','iou_sum']]
                           for values in days.values()]) for name in ['baseline','refined']}
    rng = np.random.default_rng(42)
    indices = rng.integers(0,len(days),size=(2000,len(days)))
    boot = {}
    for name,values in arrays.items():
        counts = values[indices].sum(axis=1)
        boot[name] = counts[:,3]/(counts[:,0]+.5*counts[:,1]+.5*counts[:,2])
    interval = np.quantile(boot['refined']-boot['baseline'],[.025,.975]).tolist()
    report = dict(scores={name:dict(**asdict(s),pq=s.pq) for name,s in totals.items()},
                  gained_gt=gained,lost_gt=lost,empty_retained=empty,physical_images=len(groups),
                  independent_records=len(records),environment=environment,
                  detector_sha256=frozen['checkpoint_sha256'],refiner_sha256=sha(args.weights),
                  inference=INFERENCE,selection_plan=plan,
                  input_mode=input_mode,
                  topology_heuristics=None if args.skip_topology else topology,paired_day_bootstrap_95=interval,
                  bootstrap_scope='conditional on this trained refiner/checkpoint, not training-seed uncertainty',
                  threshold=checkpoint['threshold'],empty_policy='retain_original',
                  elapsed_seconds=time.perf_counter()-started,
                  peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
    write_json(args.output/'summary.json',report)
    print(json.dumps(report,indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--project',type=Path,default=DEFAULT_PROJECT)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=20)
    p.set_defaults(function=prepare)
    p = sub.add_parser('fit')
    p.add_argument('--pairs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['overfit','pilot'],required=True)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--input-mode',choices=['image','mask_only'],default='image')
    p.add_argument('--max-updates',type=int,help='Stop at this actual optimizer-update budget, including a partial epoch')
    p.add_argument('--updates-per-round',type=int,help='Tune after this many examples; shuffled full-data passes continue across rounds')
    p.add_argument('--selection',choices=['tuning-loss','validation-pq'],default='tuning-loss')
    p.set_defaults(function=fit)
    p = sub.add_parser('evaluate')
    p.add_argument('--project',type=Path,default=DEFAULT_PROJECT)
    p.add_argument('--weights',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--skip-topology',action='store_true',help='Skip optional merge/fragment diagnostics; official PQ is unchanged')
    p.set_defaults(function=evaluate)
    args = parser.parse_args()
    if getattr(args,'command',None)=='prepare' and args.pairs<8:
        parser.error('At least eight pairs are required for fit/tune separation')
    if hasattr(args,'epochs') and args.epochs<1:
        parser.error('epochs must be positive')
    if getattr(args,'max_updates',None) is not None and args.max_updates<1:
        parser.error('max-updates must be positive')
    if getattr(args,'updates_per_round',None) is not None and args.updates_per_round<1:
        parser.error('updates-per-round must be positive')
    args.function(args)


if __name__ == '__main__':
    main()
