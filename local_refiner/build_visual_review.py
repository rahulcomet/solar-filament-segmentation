"""Build an offline visual review using organizer-notebook matrices and saved masks."""
from collections import Counter
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image
from refiner import HERE,FROZEN,DEFAULT_PROJECT,FilamentDataset,decode_mask,read_json,write_json,sha,CropTransform

OUT = HERE/'outputs/visual_review_20261019'
DIAGNOSTICS = HERE/'outputs/official_notebook_diagnostics_20261018'


def gt_groups(matrix):
    best = matrix.max(axis=1) if matrix.shape[1] else np.zeros(matrix.shape[0])
    degree = (matrix>0).sum(axis=1)
    return dict(complete=degree==0,partial=(degree>0)&(best<=.5),
        fragmented=degree>1,matched=best>.5)


def bbox(mask):
    y,x = np.nonzero(mask)
    return [int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1]


def view_box(box,margin=80):
    x0,y0,x1,y1 = box
    cx,cy = (x0+x1)/2,(y0+y1)/2
    side = min(2048,max(320,x1-x0+2*margin,y1-y0+2*margin))
    left,top = max(0,min(2048-side,int(cx-side/2))),max(0,min(2048-side,int(cy-side/2)))
    return [left,top,left+side,top+side]


def color(index):
    palette = [(0,220,255),(255,125,40),(120,245,90),(230,95,245),(255,215,60),(80,145,255)]
    return palette[index%len(palette)]


def overlay(masks,path,ids=None):
    rgba = np.zeros((2048,2048,4),dtype=np.uint8)
    for i,mask in enumerate(masks):
        mask = mask.astype(bool)
        rgba[mask,:3] = color(i if ids is None else ids[i])
        rgba[mask,3] = 110
    Image.fromarray(rgba).save(path)


def target_patch(mask,path):
    box = bbox(mask)
    x0,y0,x1,y1 = box
    cut = mask[y0:y1,x0:x1].astype(np.uint8)
    edge = cut-cv2.erode(cut,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
    rgba = np.zeros((*cut.shape,4),np.uint8)
    rgba[edge>0] = [255,230,50,255]
    Image.fromarray(rgba).save(path)
    return box


def cache(folder):
    return {read_json(p)['file_name']:read_json(p) for p in folder.glob('*.json')}


def main():
    OUT.mkdir(exist_ok=False)
    assets = OUT/'assets'; assets.mkdir()
    reader = FilamentDataset(DEFAULT_PROJECT,'val')
    original = cache(FROZEN/'prediction_cache')
    refined_raw = cache(HERE/'outputs/full_train_image43_10ep_validation/predictions')
    yolo = cache(HERE/'outputs/disjoint_pipeline_comparison_20261018/yolo_disjoint/predictions')
    refined = cache(HERE/'outputs/leaderboard_yolo_unet_disjoint_20261018/validation/predictions')
    order = read_json(DIAGNOSTICS/'yolo_unet_record_order.json')
    by_record = {row['annotator_image']:i for i,row in enumerate(order)}
    matrices = np.load(DIAGNOSTICS/'yolo_unet_overlap_matrices.npz')
    yolo_matrices = np.load(DIAGNOSTICS/'yolo_overlap_matrices.npz')
    cases,records,images = [],{},{}
    for record_no,info in enumerate(reader.images):
        sample = reader[record_no]
        filename,record_id = info['file_name'],info['id']
        index = by_record[record_id]
        assert order[index]['file_name']==filename
        matrix = matrices[f'iou_{index:03d}']
        yolo_matrix = yolo_matrices[f'iou_{index:03d}']
        assert matrix.shape==(len(sample['masks']),len(refined[filename]['masks']))
        if filename not in images:
            image_no = len(images)
            prefix = f'assets/image_{image_no:03d}'
            shutil.copyfile(sample['image_path'],OUT/(prefix+'.jpeg'))
            raw_masks = [decode_mask(r) for r in original[filename]['masks']]
            boxes = [bbox(m) for m in raw_masks]
            image = dict(file_name=filename,original=prefix+'.jpeg',boxes=boxes,
                crop_boxes=[list(CropTransform.from_mask(m).box) for m in raw_masks],
                confidence=original[filename]['confidence'])
            for key,row in [('yolo_raw',original[filename]),('yolo',yolo[filename]),
                            ('unet_raw',refined_raw[filename]),('unet',refined[filename])]:
                path = prefix+'_'+key+'.png'
                overlay([decode_mask(r) for r in row['masks']],OUT/path,row.get('source_proposal_ids'))
                image[key] = path
            images[filename] = image
        gt_path = f'assets/record_{record_no:03d}_gt.png'
        overlay(sample['masks'],OUT/gt_path)
        records[record_id] = dict(file_name=filename,gt=gt_path)
        groups = gt_groups(matrix)
        for target,mask in enumerate(sample['masks']):
            ident = f'gt_{record_no:03d}_{target:02d}'
            patch = 'assets/'+ident+'.png'
            box = target_patch(mask,OUT/patch)
            best = float(matrix[target].max()) if matrix.shape[1] else 0.
            best_yolo = float(yolo_matrix[target].max()) if yolo_matrix.shape[1] else 0.
            center = np.array([(box[0]+box[2])/2,(box[1]+box[3])/2])
            boxes = images[filename]['boxes']
            nearest = min(range(len(boxes)),key=lambda j:np.sum((center-np.array([(boxes[j][0]+boxes[j][2])/2,(boxes[j][1]+boxes[j][3])/2]))**2)) if boxes else None
            proposal = int(yolo_matrix[target].argmax()) if best_yolo>0 else None
            source_id = yolo[filename]['source_proposal_ids'][proposal] if proposal is not None else nearest
            cases.append(dict(id=ident,kind='annotation',record=record_id,file_name=filename,
                annotation_id=sample['annotations'][target]['id'],target_index=target,
                groups=[name for name,selection in groups.items() if selection[target]],
                patch=patch,box=box,crop=view_box(box),area=int(mask.sum()),
                iou=best,yolo_iou=best_yolo,degree=int((matrix[target]>0).sum()),
                source_id=source_id,candidate_basis='best overlapping YOLO mask' if proposal is not None else 'nearest mask bounding-box centre; not an established association'))
        for proposal,rle in enumerate(refined[filename]['masks']):
            degree = int((matrix[:,proposal]>0).sum())
            if degree==1:
                continue
            mask = decode_mask(rle)
            ident = f'pred_{record_no:03d}_{proposal:02d}'
            patch = 'assets/'+ident+'.png'
            box = target_patch(mask,OUT/patch)
            cases.append(dict(id=ident,kind='prediction',record=record_id,file_name=filename,
                proposal_index=proposal,source_id=refined[filename]['source_proposal_ids'][proposal],
                groups=['background' if degree==0 else 'merged'],patch=patch,box=box,crop=view_box(box),
                area=int(mask.sum()),iou=float(matrix[:,proposal].max()) if matrix.shape[0] else 0.,
                yolo_iou=None,degree=degree,candidate_basis='selected final U-Net prediction'))
        if (record_no+1)%20==0:
            print(f'Gallery assets {record_no+1}/{len(reader)} records',flush=True)
    counts = Counter(group for case in cases for group in case['groups'])
    assert counts['complete']==381 and counts['partial']==250
    assert counts['fragmented']==212 and counts['background']==190 and counts['merged']==72
    payload = dict(cases=cases,images=images,records=records,counts=counts)
    (OUT/'data.js').write_text('const REVIEW = '+json.dumps(payload)+';',encoding='utf-8')
    shutil.copyfile(HERE/'visual_review.html',OUT/'index.html')
    write_json(OUT/'manifest.json',dict(counts=counts,cases=len(cases),physical_images=len(images),
        independent_records=len(records),annotation_instances=sum(c['kind']=='annotation' for c in cases),
        script_sha256=sha(__file__),template_sha256=sha(HERE/'visual_review.html'),
        notebook_summary_sha256=sha(DIAGNOSTICS/'summary.json'),
        box_scope='All boxes are reconstructed from saved masks, not saved detector boxes',
        framing='Annotation-derived review crops only; not model inputs or new inference',
        no_retraining=True,no_new_inference=True))
    print(json.dumps(dict(counts=counts,cases=len(cases),gallery=str(OUT/'index.html')),indent=2),flush=True)


if __name__=='__main__':
    main()
