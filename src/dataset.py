"""Model-independent reader for MAGFiLO annotation records."""

from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.masks import polygon_to_mask


class FilamentDataset:
    """Read one sample per image record, preserving repeated filenames.

    ``root`` is the project directory containing ``data/``. Metadata is loaded
    once per reader; images and masks are loaded on demand at original size.
    Samples follow annotation JSON order, independently of manifest order.
    """

    def __init__(self, root, split="train"):
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.root = Path(root)
        self.split = split
        annotation_path = self.root / "data" / "train" / "annotations.json"
        split_path = self.root / "data" / "splits" / f"{split}_ids.json"
        with annotation_path.open(encoding="utf-8") as file:
            data = json.load(file)
        with split_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
        if not isinstance(manifest, list):
            raise ValueError("Split manifest must be a list of image record IDs.")
        selected_ids = set(manifest)
        if len(selected_ids) != len(manifest):
            raise ValueError("Split manifest contains duplicate image record IDs.")

        self.images = [image for image in data["images"] if image["id"] in selected_ids]
        matched_ids = {image["id"] for image in self.images}
        if matched_ids != selected_ids or len(self.images) != len(selected_ids):
            raise ValueError("Split IDs do not match the annotation records.")

        self.annotations = defaultdict(list)
        for annotation in data["annotations"]:
            image_id = annotation["image_id"]
            if image_id in selected_ids:
                self.annotations[image_id].append(annotation)
        self.image_dir = self.root / "data" / "train" / "images"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """Return grayscale pixels, a list of instance masks, and metadata.

        ``image`` and each mask have shape (height, width) and dtype uint8.
        ``masks[i]`` corresponds to ``annotations[i]``; images without filaments
        return empty lists. No resizing or normalization is applied.
        """
        info = self.images[index]
        image_path = self.image_dir / info["file_name"]
        with Image.open(image_path) as source:
            if source.size != (info["width"], info["height"]):
                raise ValueError(
                    f"Image dimensions disagree with annotations for {info['id']}: "
                    f"{source.size} != {(info['width'], info['height'])}"
                )
            image = np.array(source.convert("L"), dtype=np.uint8)
        annotations = list(self.annotations[info["id"]])
        masks = [
            polygon_to_mask(annotation["segmentation"], info["height"], info["width"])
            for annotation in annotations
        ]
        return {
            "image": image,
            "masks": masks,
            "annotations": annotations,
            "image_id": info["id"],
            "file_name": info["file_name"],
            "image_path": image_path,
            "image_info": dict(info),
        }
