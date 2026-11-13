"""Run the organizer notebook's actual metric and plotting functions on validation."""
import ast
from collections import Counter
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pycocotools import mask as mask_util
from refiner import HERE,DEFAULT_PROJECT,read_json,write_json,sha
from validation_pq import ValidationPQ

REFERENCE = DEFAULT_PROJECT/'outputs/reference/self-evaluation-notebook.ipynb'
OUT = HERE/'outputs/official_notebook_diagnostics_20261018'


def official_functions():
    names = {'get_overlap_matrices','fp_count_hit','fn_count_hit','rles_to_layers',
        'process_entry','get_overlap_df','get_pq_score','plot_distribution','plot_m2n_counts'}
    nodes = []
    for cell in read_json(REFERENCE)['cells']:
        if cell['cell_type']=='code':
            source = cell['source']
            for node in ast.parse(''.join(source) if isinstance(source,list) else source).body:
                if isinstance(node,ast.FunctionDef) and node.name in names:
                    nodes.append(node)
    assert {node.name for node in nodes}==names
    scope = dict(torch=torch,np=np,pd=pd,mask_util=mask_util,plt=plt,Counter=Counter,Patch=Patch)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(REFERENCE),'exec'),scope)
    return scope


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=False)
    official = official_functions()
    validator = ValidationPQ()
    comparison = read_json(HERE/'outputs/disjoint_pipeline_comparison_20261018/comparison.json')
    folders = dict(yolo=HERE/'outputs/disjoint_pipeline_comparison_20261018/yolo_disjoint',
        yolo_unet=HERE/'outputs/leaderboard_yolo_unet_disjoint_20261018/validation')
    summaries,tables = {},{}
    for name,folder in folders.items():
        predictions = {read_json(p)['file_name']:read_json(p) for p in (folder/'predictions').glob('*.json')}
        assert set(predictions)==set(validator.records)
        rows = []
        for filename,records in validator.records.items():
            pred = predictions[filename]['masks']
            pred_layers = torch.from_numpy(official['rles_to_layers']([r['counts'] for r in pred]))
            for record in records:
                gt = record['gt']
                gt_layers = torch.from_numpy(official['rles_to_layers']([r['counts'] for r in gt]))
                iou,dice = official['get_overlap_matrices'](gt_layers,pred_layers)
                rows.append(dict(annotator_image=record['image_id'],file_name=filename,
                    iou_matrix=iou,dice_matrix=dice,n_gt=len(gt),n_pred=len(pred)))
            if len(rows)%20< len(records):
                print(f'{name}: {len(rows)}/234 records evaluated by notebook functions',flush=True)
        table = pd.DataFrame(rows)
        tables[name] = table
        pq = official['get_pq_score'](table)
        expected = comparison['disjoint'][name+'_disjoint']['pq']
        assert abs(pq-expected)<1e-7,(pq,expected)
        metrics = dict(pq=pq,existing_pq=expected,pq_parity_delta=pq-expected)
        for key in ['iou','dice']:
            values = np.concatenate([m.numpy().ravel() for m in table[key+'_matrix']])
            values = values[values>0]
            metrics[key+'_positive_pair_distribution'] = dict(count=len(values),mean=float(values.mean()),
                median=float(np.median(values)),q25=float(np.quantile(values,.25)),q75=float(np.quantile(values,.75)))
        gt_degrees,pred_degrees = [],[]
        for row in rows:
            hit = row['iou_matrix']>0
            gt_degrees.extend(hit.sum(dim=1).tolist())
            pred_degrees.extend(hit.sum(dim=0).tolist())
        metrics['overlap_relations_iou_gt_zero'] = dict(
            gt_degree_histogram=dict(Counter(gt_degrees)),pred_degree_histogram=dict(Counter(pred_degrees)),
            gt_with_multiple_predictions=sum(x>1 for x in gt_degrees),
            predictions_overlapping_multiple_gt=sum(x>1 for x in pred_degrees),
            gt_without_any_overlap=gt_degrees.count(0),pred_without_any_overlap=pred_degrees.count(0))
        summaries[name] = metrics
        np.savez_compressed(OUT/(name+'_overlap_matrices.npz'),
            **{f'{key}_{i:03d}':row[key+'_matrix'].numpy() for i,row in enumerate(rows) for key in ['iou','dice']})
        write_json(OUT/(name+'_record_order.json'),[dict(annotator_image=r['annotator_image'],file_name=r['file_name'],n_gt=r['n_gt'],n_pred=r['n_pred']) for r in rows])
    fig = plt.figure(figsize=(19,10))
    for i,(name,table) in enumerate(tables.items()):
        plt.subplot(2,3,3*i+1)
        official['plot_distribution'](table['iou_matrix'],name+' — nonzero pairwise IoU','IoU')
        plt.subplot(2,3,3*i+2)
        official['plot_distribution'](table['dice_matrix'],name+' — nonzero pairwise Dice','Dice')
        plt.subplot(2,3,3*i+3)
        official['plot_m2n_counts'](table,name+' — overlap relations (IoU > 0)')
    fig.tight_layout()
    fig.savefig(OUT/'notebook_diagnostics.png',dpi=160)
    fig.savefig(OUT/'notebook_diagnostics.pdf')
    write_json(OUT/'summary.json',dict(notebook=str(REFERENCE),notebook_sha256=sha(REFERENCE),
        script_sha256=sha(__file__),physical_images=146,independent_records=234,
        actual_notebook_functions_executed=True,no_retraining=True,
        distribution_scope='All nonzero GT/prediction pairs, as in organizer notebook; not only PQ matches',
        relation_scope='IoU > 0 geometric overlaps, not IoU > 0.5 PQ matches or confirmed merge/split errors',
        models=summaries))
    print(json.dumps(summaries,indent=2),flush=True)


if __name__=='__main__':
    main()
