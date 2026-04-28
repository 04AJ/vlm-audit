"""
Flickr30k Entities — Data Loader
----------------------------------
Images    : nlphuji/flickr30k on HuggingFace (used for images only)
Captions  : local Sentences/ directory  — one .txt per image, 5 sentences each
            Format: [/EN#id/class phrase] words ...
Boxes     : local Annotations/ directory — one .xml per image
            <name> contains the entity ID; class comes from the sentence file.

Split     : filtered to IDs listed in split_file (e.g. test.txt).
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Callable, Dict, List, Optional, Set, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from core.config import AuditConfig


class Flickr30kDataset(Dataset):
    """
    PyTorch Dataset for Flickr30k Entities.

    HuggingFace is used only to supply PIL images.
    Captions and bounding boxes are loaded from local annotation files.

    Each __getitem__ returns:
    {
        "image"      : Tensor (C, H, W)  or PIL if processor is None,
        "caption"    : str  — clean caption (entity markers stripped),
        "boxes"      : List[Dict]  — [{"label": class, "phrase": str,
                                        "box": [x1, y1, x2, y2]}, ...],
        "filename"   : str,
        "image_size" : (H, W) in pixels,
    }
    """

    def __init__(
        self,
        config: AuditConfig,
        processor=None,
        caption_idx: int = 4,
        transform: Optional[Callable] = None,
    ) -> None:
        self.config = config
        self.processor = processor
        self.caption_idx = caption_idx
        self.transform = transform

        self._hf_dataset = None
        self._load()

    # Loading

    def _load(self) -> None:
        """Stream images from HuggingFace, keeping only IDs in split_file."""
        import datasets as hf_datasets

        split_ids: Optional[Set[str]] = None
        if self.config.split_file:
            with open(self.config.split_file) as f:
                split_ids = {line.strip() for line in f if line.strip()}

        def in_split(row) -> bool:
            return os.path.splitext(row["filename"])[0] in split_ids

        stream = hf_datasets.load_dataset(
            self.config.dataset_name,
            split=self.config.dataset_split,
            trust_remote_code=True,
            streaming=True,
        )

        if split_ids is not None:
            stream = stream.filter(in_split)

        if self.config.max_samples is not None:
            stream = stream.take(self.config.max_samples)

        rows = list(stream)
        self._hf_dataset = hf_datasets.Dataset.from_list(rows, features=stream.features)

    def __len__(self) -> int:
        return len(self._hf_dataset)

    def __getitem__(self, idx: int) -> Dict:
        row = self._hf_dataset[idx]
        filename = row.get("filename", "")

        pil_image = row["image"]
        img_w, img_h = pil_image.size  # PIL gives (width, height)

        # Image preprocessing
        if self.processor is not None:
            inputs = self.processor(images=pil_image, return_tensors="pt")
            image = inputs["pixel_values"].squeeze(0)  # (C, H, W)
        else:
            image = pil_image

        if self.transform is not None:
            image = self.transform(image)

        # Caption + entity class map from local sentences file
        caption, entity_class_map = self._load_sentence(filename)

        # Bounding boxes from local XML, labelled with class from sentence
        boxes = self._load_xml_boxes(filename, entity_class_map)

        return {
            "image":      image,
            "caption":    caption,
            "boxes":      boxes,
            "filename":   filename,
            "image_size": (img_h, img_w),  # (H, W) for grounding evaluator
        }

    # Annotation helpers

    def _load_sentence(self, filename: str) -> Tuple[str, Dict[str, str]]:
        """
        Parse the Flickr30k Entities sentence file for one image.

        Sentence format:  [/EN#id/class phrase words] rest of sentence ...

        Returns
        -------
        caption          : clean caption string (entity markers stripped)
        entity_class_map : {entity_id -> class_label}
        """
        if not self.config.sentences_dir:
            return "", {}

        stem = os.path.splitext(filename)[0]
        sent_path = os.path.join(self.config.sentences_dir, stem + ".txt")

        if not os.path.isfile(sent_path):
            return "", {}

        with open(sent_path) as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]

        line = lines[self.caption_idx] if self.caption_idx < len(lines) else lines[0]

        # Build entity id → class map from all tags in the line
        entity_class_map: Dict[str, str] = {}
        pattern = re.compile(r'\[/EN#(\d+)/(\w+)\s+([^\]]+)\]')
        for m in pattern.finditer(line):
            entity_id, cls = m.group(1), m.group(2)
            entity_class_map[entity_id] = cls

        # Strip markers to produce a clean caption
        clean = pattern.sub(lambda m: m.group(3), line)
        clean = re.sub(r'\s+', ' ', clean).strip()

        return clean, entity_class_map

    def _load_xml_boxes(
        self,
        filename: str,
        entity_class_map: Dict[str, str],
    ) -> List[Dict]:
        """
        Parse a Flickr30k Entities XML file.

        Each <object> has one or more <name> tags containing an entity ID,
        plus a single <bndbox>.  The class label is looked up from
        entity_class_map (built from the sentence file).

        Boxes are already in pixel coordinates.

        Returns
        -------
        List of {"label": class, "phrase": entity_id, "box": [x1,y1,x2,y2]}
        """
        if not self.config.annotations_dir:
            return []

        stem = os.path.splitext(filename)[0]
        xml_path = os.path.join(self.config.annotations_dir, stem + ".xml")

        if not os.path.isfile(xml_path):
            return []

        root = ET.parse(xml_path).getroot()
        boxes: List[Dict] = []

        for obj in root.findall("object"):
            bndbox_el = obj.find("bndbox")
            if bndbox_el is None:
                continue

            x1 = float(bndbox_el.findtext("xmin", 0))
            y1 = float(bndbox_el.findtext("ymin", 0))
            x2 = float(bndbox_el.findtext("xmax", 0))
            y2 = float(bndbox_el.findtext("ymax", 0))

            # One bounding box can belong to multiple entity IDs
            for name_el in obj.findall("name"):
                entity_id = (name_el.text or "").strip()
                label = entity_class_map.get(entity_id, "unknown")
                if label in ("notvisual", "unknown"):
                    continue
                boxes.append({
                    "label":  label,
                    "phrase": entity_id,
                    "box":    [x1, y1, x2, y2],
                })

        return boxes

    # Collation

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Custom collate so DataLoader can stack variable-length box lists.
        Images are stacked if tensors; boxes and captions are kept as lists.
        """
        if isinstance(batch[0]["image"], torch.Tensor):
            stacked_images = torch.stack([item["image"] for item in batch])
        else:
            stacked_images = [item["image"] for item in batch]

        return {
            "image":      stacked_images,
            "caption":    [item["caption"]    for item in batch],
            "boxes":      [item["boxes"]      for item in batch],
            "filename":   [item["filename"]   for item in batch],
            "image_size": [item["image_size"] for item in batch],
        }


def get_dataloader(
    config: AuditConfig,
    processor=None,
    batch_size: int = 8,
    shuffle: bool = False,
    num_workers: int = 2,
) -> DataLoader:
    """
    Convenience factory that builds Flickr30kDataset + DataLoader.

    Parameters
    ----------
    config      : AuditConfig
    processor   : HuggingFace processor (pass model.processor)
    batch_size  : samples per batch
    shuffle     : whether to shuffle (set False for reproducible evaluation)
    num_workers : DataLoader worker count

    Returns
    -------
    torch.utils.data.DataLoader
    """
    dataset = Flickr30kDataset(config=config, processor=processor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=Flickr30kDataset.collate_fn,
    )
