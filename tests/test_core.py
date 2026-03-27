"""
Tests for Core Module  (core/)
"""

import pytest
from core.config import AuditConfig


def test_default_config():
    cfg = AuditConfig()
    assert cfg.model_name == "Salesforce/blip-itm-base-coco"
    assert cfg.iou_threshold == 0.5


def test_config_custom_layers():
    cfg = AuditConfig(target_layers=[6, 7, 8])
    assert cfg.target_layers == [6, 7, 8]


# TODO: add tests once VLMAuditModel._load_model() is implemented
# def test_model_loads():
#     cfg   = AuditConfig(device="cpu", max_samples=1)
#     model = VLMAuditModel(cfg)
#     assert model.num_layers > 0

# TODO: test that forward pass populates attention cache
# def test_attention_cache_populated():
#     ...
