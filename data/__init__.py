"""
Data Module
===========
Owns: HuggingFace dataset loading, preprocessing, and batch collation.
Public interface:
  - Flickr30kDataset : torch Dataset wrapping the HF flickr30k split
  - get_dataloader   : convenience factory for DataLoader
"""

from data.flickr30k import Flickr30kDataset, get_dataloader

__all__ = ["Flickr30kDataset", "get_dataloader"]
