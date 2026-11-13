"""Run YOLO11s and the pass-6 refiner on H-alpha images."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / 'quality_selection'), str(ROOT / 'local_refiner'), str(ROOT)]
os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / '.ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.matplotlib'))
os.environ.setdefault('MPLBACKEND', 'Agg')

import cv2
import torch
from ultralytics import YOLO

from current_masks import prepare
from final_evidence_filter import predict
from predict_submission import export_csv, predict_image
from refiner import InstanceUNet, read_json, runtime, sha
from boundary_vote import vote_from_views
from isolated_admission import admit
from mirror_inference import append_isolated, high_resolution_coarse, mirror_coarse, refine, supported


def load():
    recipe = read_json(ROOT / 'recipe.json')
    paths = {key: ROOT / recipe[key]['path'] for key in ('detector', 'refiner')}
    for key, path in paths.items():
        if sha(path) != recipe[key]['sha256']:
            raise ValueError(f'Unexpected {key} checkpoint')
    runtime(43)
    detector = YOLO(str(paths['detector']))
    checkpoint = torch.load(paths['refiner'], map_location='cpu', weights_only=False)
    network = InstanceUNet().cuda().eval().requires_grad_(False)
    network.load_state_dict(checkpoint['model'])
    return recipe, detector, network


def reference_masks(path, recipe, detector, network, *, mirror):
    final, mirrored, _ = reference_views(path, recipe, detector, network, mirror=mirror)
    return final, mirrored


def reference_views(path, recipe, detector, network, *, mirror):
    proposal_recipe = dict(recipe, detector=dict(recipe['detector'], inference=dict(
        recipe['detector']['inference'], conf=.18)))
    original = predict_image(detector, network, path, proposal_recipe)
    indexes = [i for i, score in enumerate(original['confidence']) if score >= .20]
    admitted = dict(file_name=path.name, **{key: [original[key][i] for i in indexes]
                    for key in ('masks', 'coarse_masks', 'confidence')})
    final = predict(prepare(admitted, image_path=path))
    if not mirror:
        return final, None, original
    mirrored = refine(network, mirror_coarse(detector, path), path)
    final = append_isolated(final, supported(original, mirrored), .24)
    final = append_isolated(final, zip(mirrored['masks'], mirrored['confidence']), .35)
    return final, mirrored, original


def apply_high_resolution_vote(path, detector, network, anchor, mirrored):
    scaled = refine(network, high_resolution_coarse(detector, path), path)
    return vote_from_views(anchor, mirrored, scaled)


def run_image(path, recipe, detector, network, variant):
    final, mirrored, original = reference_views(path, recipe, detector, network, mirror=variant != 'incumbent')
    if variant == 'boundary-vote':
        return apply_high_resolution_vote(path, detector, network, final, mirrored)
    if variant == 'boundary-vote-admission':
        scaled = refine(network, high_resolution_coarse(detector, path), path)
        voted = vote_from_views(final, mirrored, scaled)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f'Cannot read {path}')
        return admit(voted, (original, mirrored, scaled), image)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--variant', choices=('incumbent', 'mirror', 'boundary-vote', 'boundary-vote-admission'),
                        default='incumbent')
    args = parser.parse_args()
    paths = sorted(path for path in args.images.iterdir()
                   if path.is_file() and path.suffix.lower() in {'.jpeg', '.jpg', '.png'})
    if not paths:
        parser.error('No JPEG or PNG images found')
    if args.output.exists():
        parser.error('Select a new output file')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    recipe, detector, network = load()
    predictions = {}
    for index, path in enumerate(paths, 1):
        masks = run_image(path, recipe, detector, network, args.variant)
        predictions[path.name] = dict(file_name=path.name, masks=masks)
        if index % 20 == 0 or index == len(paths):
            print(f'{index}/{len(paths)} images', flush=True)
    print(export_csv(args.output, predictions), flush=True)


if __name__ == '__main__':
    main()
