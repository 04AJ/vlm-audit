"""
Tests for Extraction Layer  (extraction/)
"""

import pytest
import torch
from core.config import AuditConfig
from extraction.attention import AttentionExtractor


def test_attention_extractor_init():
    cfg = AuditConfig()
    extractor = AttentionExtractor(cfg, patch_grid=(24, 24), image_size=(384, 384))
    assert extractor.patch_grid == (24, 24)


# TODO: add tests once extraction methods are implemented

# def test_fuse_heads_mean():
#     cfg = AuditConfig(attention_head_fusion="mean")
#     extractor = AttentionExtractor(cfg, (24, 24), (384, 384))
#     weights = torch.rand(2, 8, 10, 576)   # (B, heads, T, N)
#     fused   = extractor._fuse_heads(weights)
#     assert fused.shape == (2, 10, 576)

# def test_full_pipeline_shape():
#     """End-to-end: cache → heatmap shape matches image_size."""
#     cfg   = AuditConfig()
#     cache = {0: torch.rand(2, 8, 10, 576)}
#     extractor = AttentionExtractor(cfg, (24, 24), (384, 384))
#     heatmaps  = extractor.extract(cache)
#     assert heatmaps[0].shape == (2, 384, 384)
