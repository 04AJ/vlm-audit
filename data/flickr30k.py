"""
Flickr30k Entities — HuggingFace Loader
-----------------------------------------
Dataset: nlphuji/flickr30k  (Flickr30k Entities)
  - Images + 5 reference captions per image
  - Bounding-box annotations per noun phrase  <-- used by Grounding Module

This module handles all HuggingFace I/O so the rest of the codebase
never imports `datasets` directly.

"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from core.config import AuditConfig


# Expected columns in nlphuji/flickr30k after loading:
#   "image"       : PIL.Image
#   "caption"     : List[str]  (5 captions)
#   "entities"    : List[dict] with keys "phrase", "boxes" (x1,y1,x2,y2 normalised)
#   "filename"    : str


class Flickr30kDataset(Dataset):
    """
    PyTorch Dataset wrapping the HuggingFace Flickr30k Entities split.

    Parameters
    ----------
    config      : AuditConfig — supplies split name, max_samples, etc.
    processor   : HuggingFace processor (from VLMAuditModel) for image/text
                  preprocessing.  Pass None to get raw PIL + str.
    caption_idx : which of the 5 reference captions to use (default: 0).
    transform   : optional additional torchvision transform applied after
                  the processor.

    Each __getitem__ returns a dict:
    {
        "image"    : Tensor (C, H, W)  or PIL if processor is None,
        "caption"  : str,
        "boxes"    : List[Dict]  — [{phrase, box:[x1,y1,x2,y2]}, ...],
        "filename" : str,
    }
    """

    def __init__(
        self,
        config: AuditConfig,
        processor=None,
        caption_idx: int = 0,
        transform: Optional[Callable] = None,
    ) -> None:
        self.config = config
        self.processor = processor
        self.caption_idx = caption_idx
        self.transform = transform

        self._hf_dataset = None   # TODO: load via datasets.load_dataset(...)
        # self._load()

    def _load(self) -> None:
        """
        Pull the dataset from HuggingFace hub.

        Steps
        -----
        1. import datasets
        2. datasets.load_dataset(self.config.dataset_name,
                                 split=self.config.dataset_split)
        3. Optionally slice to self.config.max_samples.
        """
        # TODO: implement
        raise NotImplementedError

    def __len__(self) -> int:
        # TODO: return len(self._hf_dataset)
        raise NotImplementedError

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns a single sample dict.  See class docstring for schema.

        Bounding boxes are returned in pixel coordinates after denormalising
        against the original image size so they are compatible with the
        Grounding Module.
        """
        # TODO: fetch row, preprocess image with self.processor,
        #       select caption, parse entity bounding boxes
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Custom collate so DataLoader can stack variable-length box lists.
        Images and captions are stacked normally; boxes are kept as a list.
        """
        # TODO: implement
        raise NotImplementedError


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
