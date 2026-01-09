"""
R3MES Pipeline Module

Contains pipelines for:
- GradientSubmissionPipeline: Auto-submit gradients to IPFS + Blockchain
- ModelUpdatePipeline: Model update and aggregation
- TrainingPipeline: End-to-end training orchestration
"""

from pipeline.gradient_submission import GradientSubmissionPipeline
from pipeline.model_update import ModelUpdatePipeline

__all__ = [
    "GradientSubmissionPipeline",
    "ModelUpdatePipeline",
]
