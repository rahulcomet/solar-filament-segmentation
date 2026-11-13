"""Exact official PQ from cached independent GT RLE and fixed YOLO proposals."""
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
from refiner import (DEFAULT_PROJECT, FROZEN, load_readers, read_json, write_json,
    encode_mask, decode_mask, PQStats, pq_stats_from_iou, CropTransform,
    make_input, restore_prediction, check_frozen, day)


def rle_iou(gt, prediction):
    if not gt or not prediction:
        return np.zeros((len(gt),len(prediction)),dtype=np.float64)
    return mask_utils.iou(gt,prediction,[0]*len(prediction))


class ValidationPQ:
    def __init__(self, project=DEFAULT_PROJECT):
        frozen = check_frozen()
        self.reader = load_readers(project)['val']
        self.caches = {read_json(p)['file_name']:read_json(p) for p in (FROZEN/'prediction_cache').glob('*.json')}
        assert set(self.caches)=={r['file_name'] for r in self.reader.images}
        self.records = {}
        self.baseline = PQStats()
        for i,info in enumerate(self.reader.images):
            sample = self.reader[i]
            gt = [encode_mask(m) for m in sample['masks']]
            matrix = rle_iou(gt,self.caches[info['file_name']]['masks'])
            stats = pq_stats_from_iou(matrix)
            self.baseline += stats
            self.records.setdefault(info['file_name'],[]).append(dict(image_id=sample['image_id'],
                gt=gt,baseline=stats,hits=(matrix>.5).any(axis=1)))
        assert self.baseline.tp==frozen['tp'] and self.baseline.fp==frozen['fp'] and self.baseline.fn==frozen['fn']
        assert abs(self.baseline.pq-frozen['pq'])<1e-10

    def score_saved(self, folder):
        predictions = {read_json(p)['file_name']:read_json(p) for p in (folder/'predictions').glob('*.json')}
        assert set(predictions)==set(self.caches)
        total = PQStats()
        for filename,records in self.records.items():
            cache = predictions[filename]
            assert cache['confidence']==self.caches[filename]['confidence']
            assert cache['proposal_ids']==list(range(len(cache['masks'])))
            for record in records:
                total += pq_stats_from_iou(rle_iou(record['gt'],cache['masks']))
        return total

    def score(self, network, output, epoch, input_mode='image'):
        output.mkdir(exist_ok=False)
        (output/'predictions').mkdir()
        total,rows = PQStats(),[]
        gained,lost,empty = 0,0,0
        network.eval()
        for number,(filename,records) in enumerate(self.records.items()):
            cache = self.caches[filename]
            with Image.open(self.reader.image_dir/filename) as source:
                image = np.array(source.convert('L'),dtype=np.uint8)
            encoded = []
            for rle in cache['masks']:
                coarse = decode_mask(rle)
                if not coarse.any():
                    encoded.append(rle)
                    empty += 1
                    continue
                geometry = CropTransform.from_mask(coarse)
                prompt = geometry.extract(coarse,mask=True)
                x = torch.from_numpy(make_input(geometry.extract(image),prompt,geometry.valid(),input_mode))[None].cuda()
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16):
                    probability = network(x).float().sigmoid()[0,0].cpu().numpy()
                prediction,was_empty = restore_prediction(probability,geometry,coarse,.5)
                empty += int(was_empty)
                encoded.append(encode_mask(prediction))
            write_json(output/'predictions'/f'{number:04d}.json',dict(file_name=filename,
                confidence=cache['confidence'],proposal_ids=list(range(len(encoded))),masks=encoded))
            for record in records:
                matrix = rle_iou(record['gt'],encoded)
                stats = pq_stats_from_iou(matrix)
                total += stats
                hits = (matrix>.5).any(axis=1)
                gain = int((~record['hits']&hits).sum())
                loss = int((record['hits']&~hits).sum())
                gained += gain; lost += loss
                rows.append(dict(image_id=record['image_id'],file_name=filename,day=day(filename),
                    baseline=dict(**asdict(record['baseline']),pq=record['baseline'].pq),
                    refined=dict(**asdict(stats),pq=stats.pq),gained=gain,lost=loss))
            if (number+1)%30==0:
                print(f'Epoch {epoch}: PQ validation {number+1}/{len(self.records)} images',flush=True)
        with (output/'records.jsonl').open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row)+'\n')
        result = dict(epoch=epoch,scores=dict(baseline=dict(**asdict(self.baseline),pq=self.baseline.pq),
            refined=dict(**asdict(total),pq=total.pq)),gained_gt=gained,lost_gt=lost,empty_retained=empty,
            physical_images=len(self.records),independent_records=len(rows),
            scope='validation used for checkpoint selection; not an independent test')
        write_json(output/'summary.json',result)
        return result


if __name__=='__main__':
    from refiner import HERE, runtime
    runtime(43)
    validator = ValidationPQ()
    folder = HERE/'outputs/scale_image43_1600_validation'
    score = validator.score_saved(folder)
    expected = read_json(folder/'summary.json')['scores']['refined']
    assert asdict(score)=={k:expected[k] for k in ['tp','fp','fn','iou_sum']}
    print(json.dumps(dict(records=sum(len(v) for v in validator.records.values()),
        score=dict(**asdict(score),pq=score.pq),exact_saved_evaluation_match=True),indent=2))
