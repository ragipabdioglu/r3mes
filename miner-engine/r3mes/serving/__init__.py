"""
R3MES Serving Node Module

Provides serving node functionality for AI model inference.

Components:
- ServingEngine: Production serving engine with blockchain integration
- InferencePipeline: BitNet + DoRA + RAG inference pipeline
- StreamingInferencePipeline: Streaming variant for real-time output
"""

__version__ = "0.2.0"

from .inference_pipeline import (
    InferencePipeline,
    StreamingInferencePipeline,
    PipelineConfig,
    PipelineResult,
    PipelineMetrics,
    PipelineStage,
    create_pipeline,
    create_streaming_pipeline,
    quick_inference,
)

# Lazy imports for ServingEngine to avoid dependency issues
def __getattr__(name):
    if name == "ServingEngine":
        from .engine import ServingEngine
        return ServingEngine
    elif name == "EngineState":
        from .engine import EngineState
        return EngineState
    elif name == "EngineHealth":
        from .engine import EngineHealth
        return EngineHealth
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # Engine (lazy loaded)
    "ServingEngine",
    "EngineState",
    "EngineHealth",
    # Pipeline
    "InferencePipeline",
    "StreamingInferencePipeline",
    "PipelineConfig",
    "PipelineResult",
    "PipelineMetrics",
    "PipelineStage",
    "create_pipeline",
    "create_streaming_pipeline",
    "quick_inference",
]


def get_serving_engine():
    """Lazy import for ServingEngine to avoid dependency issues."""
    from .engine import ServingEngine
    return ServingEngine

