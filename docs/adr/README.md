# Architecture Decision Records (ADR)

This directory contains Architecture Decision Records for the R3MES project.

## What is an ADR?

An Architecture Decision Record (ADR) is a document that captures an important architectural decision made along with its context and consequences.

## ADR Format

Each ADR follows this structure:

1. **Title**: Short descriptive title
2. **Status**: Proposed, Accepted, Deprecated, Superseded
3. **Context**: The issue motivating this decision
4. **Decision**: The change that we're proposing or have agreed to implement
5. **Consequences**: What becomes easier or more difficult to do because of this change

## ADR Index

### 0001 - Use FastAPI for Backend API
- **Status**: Accepted
- **Date**: 2024-01-01
- **Decision**: Use FastAPI framework for the backend inference service
- **See**: [ADR-0001-fastapi-backend.md](./0001-fastapi-backend.md)

### 0002 - Multi-LoRA Architecture
- **Status**: Accepted
- **Date**: 2024-01-15
- **Decision**: Implement multi-LoRA adapter system for domain-specific fine-tuning
- **See**: [ADR-0002-multi-lora-architecture.md](./0002-multi-lora-architecture.md)

### 0003 - Semantic Router for Adapter Selection
- **Status**: Accepted
- **Date**: 2024-02-01
- **Decision**: Use embedding-based semantic routing instead of keyword matching
- **See**: [ADR-0003-semantic-router.md](./0003-semantic-router.md)

### 0004 - Cosmos SDK Blockchain Integration
- **Status**: Accepted
- **Date**: 2024-02-15
- **Decision**: Use Cosmos SDK for blockchain layer
- **See**: [ADR-0004-cosmos-sdk-blockchain.md](./0004-cosmos-sdk-blockchain.md)

### 0005 - Credit System for API Access
- **Status**: Accepted
- **Date**: 2024-03-01
- **Decision**: Implement credit-based access control for API endpoints
- **See**: [ADR-0005-credit-system.md](./0005-credit-system.md)

### 0006 - Redis Caching Strategy
- **Status**: Accepted
- **Date**: 2024-03-15
- **Decision**: Use Redis for distributed caching with tag-based invalidation
- **See**: [ADR-0006-redis-caching.md](./0006-redis-caching.md)

### 0007 - OpenTelemetry for Distributed Tracing
- **Status**: Accepted
- **Date**: 2024-12-24
- **Decision**: Use OpenTelemetry for distributed tracing and log correlation
- **See**: [ADR-0007-opentelemetry-tracing.md](./0007-opentelemetry-tracing.md)

### 0008 - Prometheus and Grafana Monitoring
- **Status**: Accepted
- **Date**: 2024-12-24
- **Decision**: Use Prometheus for metrics collection and Grafana for visualization
- **See**: [ADR-0008-prometheus-monitoring.md](./0008-prometheus-monitoring.md)

## How to Write an ADR

1. Create a new file: `NNNN-short-title.md`
2. Use the template below
3. Update this README with the new ADR
4. Submit for review

## ADR Template

```markdown
# ADR-NNNN: Short Title

**Status**: Proposed | Accepted | Deprecated | Superseded
**Date**: YYYY-MM-DD
**Deciders**: [List of people involved]

## Context

Describe the issue motivating this decision.

## Decision

Describe the change that we're proposing or have agreed to implement.

## Consequences

### Positive
- What becomes easier or better

### Negative
- What becomes harder or worse

### Neutral
- Other consequences
```

---

**Son Güncelleme**: 2025-12-24

