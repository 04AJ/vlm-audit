"""
Tests for Evaluation Suite  (evaluation/)
"""

import pytest
from evaluation.results import EvalResults, LayerGroundingResult, LayerFaithfulnessResult


def test_eval_results_empty():
    r = EvalResults()
    assert r.best_grounding_layer() is None
    assert r.best_faithfulness_layer() is None


def test_eval_results_best_grounding():
    r = EvalResults(grounding=[
        LayerGroundingResult(layer_idx=6, pointing_game_accuracy=0.6, mean_iou=0.3),
        LayerGroundingResult(layer_idx=8, pointing_game_accuracy=0.7, mean_iou=0.5),
    ])
    assert r.best_grounding_layer().layer_idx == 8


def test_eval_results_summary():
    r = EvalResults(grounding=[
        LayerGroundingResult(layer_idx=7, pointing_game_accuracy=0.65, mean_iou=0.42),
    ])
    assert "mIoU" in r.summary()


# TODO: add integration tests once evaluators are implemented

# def test_grounding_evaluator_smoke():
#     cfg = AuditConfig()
#     ev  = GroundingEvaluator(cfg)
#     ev.reset()
#     # feed dummy heatmaps + boxes, check results() runs without error
