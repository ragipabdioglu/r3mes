# ADR-0003: Semantic Router for Adapter Selection

**Status**: Accepted  
**Date**: 2024-02-01  
**Deciders**: ML Team, Backend Team

## Context

We need to automatically select the appropriate LoRA adapter based on user input. Initial implementation used keyword matching, but this has limitations:
- Keyword matching is brittle and can misclassify
- Complex queries don't match well with simple keywords
- Requires manual keyword list maintenance
- Doesn't understand semantic similarity

## Decision

We will implement a **Semantic Router** that:
1. Uses sentence embeddings to understand query semantics
2. Compares query embeddings with adapter descriptions
3. Selects adapter with highest similarity score

**Update (2025-01-14):** Semantic Router is now mandatory. The keyword router (Router) has been deprecated and removed. If SemanticRouter initialization fails, the application will raise an error.

## Consequences

### Positive
- **Better Accuracy**: Understands semantic meaning, not just keywords
- **Handles Complex Queries**: Works with natural language, not just keywords
- **Automatic**: No manual keyword list maintenance
- **Flexible**: Can adapt to new query patterns
- **Mandatory**: Semantic router is now required (keyword router deprecated)

### Negative
- **Dependency**: Requires sentence-transformers library
- **Initialization Time**: Embedding model needs to be loaded
- **Memory**: Embedding model uses additional memory
- **Latency**: Embedding computation adds small overhead
- **Threshold Tuning**: Requires tuning similarity threshold

### Neutral
- Embedding model can be cached
- **Update (2025-01-14):** Semantic router is now mandatory and cannot be disabled

## Implementation Details

- Use `sentence-transformers` for embeddings
- Pre-compute adapter description embeddings
- Use cosine similarity for matching
- Configurable similarity threshold (default: 0.7)
- Cache query embeddings for repeated queries

## Alternatives Considered

1. **Keyword Router Only**:
   - Pros: Simple, fast, no dependencies
   - Cons: Poor accuracy, manual maintenance, brittle

2. **ML Classification Model**:
   - Pros: High accuracy, can be fine-tuned
   - Cons: Requires training data, more complex, slower

3. **Rule-Based System**:
   - Pros: Explicit, debuggable
   - Cons: Doesn't scale, manual maintenance

---

**Related ADRs**: 
- ADR-0002: Multi-LoRA Architecture

