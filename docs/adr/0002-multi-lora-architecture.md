# ADR-0002: Multi-LoRA Architecture

**Status**: Accepted  
**Date**: 2024-01-15  
**Deciders**: ML Team, Backend Team

## Context

We need to support multiple domain-specific fine-tuned models (e.g., coding, legal, medical) without:
- Loading multiple full models (too much VRAM)
- Retraining the base model for each domain (too slow)
- Maintaining separate model instances (too expensive)

LoRA (Low-Rank Adaptation) allows fine-tuning with minimal parameters, but we need a system to:
- Load base model once
- Dynamically load/unload LoRA adapters
- Switch between adapters efficiently
- Support streaming inference

## Decision

We will implement a **Multi-LoRA Architecture** where:
1. Base model (BitNet) is loaded once at startup
2. LoRA adapters are loaded on-demand and cached
3. Adapters can be swapped dynamically per request
4. System supports multiple adapters simultaneously (with memory limits)

## Consequences

### Positive
- **Memory Efficient**: Only one base model in memory, adapters are small
- **Fast Switching**: Adapters can be loaded/unloaded quickly
- **Scalable**: Can add new domains without retraining base model
- **Cost Effective**: Reduced VRAM requirements compared to multiple full models
- **Flexible**: Easy to add/remove adapters without redeployment

### Negative
- **Complexity**: More complex model management logic
- **Adapter Loading Time**: First request per adapter has loading overhead
- **Memory Management**: Need to track and limit adapter memory usage
- **Compatibility**: Adapters must be compatible with base model version

### Neutral
- Requires PEFT (Parameter-Efficient Fine-Tuning) library
- Adapter files need to be stored and versioned
- Need fallback mechanism if adapter loading fails

## Implementation Details

- Use `peft.PeftModel` for adapter management
- Implement adapter cache with LRU eviction
- Support both semantic and keyword-based adapter selection
- Monitor adapter memory usage and enforce limits
- Provide adapter health checks and versioning

## Alternatives Considered

1. **Multiple Full Models**:
   - Pros: Simple, no adapter management
   - Cons: High VRAM usage, slow to load, expensive

2. **Single Fine-Tuned Model**:
   - Pros: Simple, single model
   - Cons: Poor performance on specialized domains, retraining needed for new domains

3. **Model Ensembles**:
   - Pros: Better accuracy
   - Cons: Very high VRAM usage, slow inference

---

**Related ADRs**: 
- ADR-0003: Semantic Router for Adapter Selection

