# ADR-0007: OpenTelemetry for Distributed Tracing

**Status**: Accepted  
**Date**: 2024-12-24  
**Deciders**: DevOps Team, Backend Team

## Context

As the system grows with multiple services (backend, blockchain node, miner engine), we need:
- End-to-end request tracing across services
- Performance bottleneck identification
- Error correlation across services
- Log correlation with traces
- Service dependency visualization

Options include:
- OpenTelemetry (standard, vendor-neutral)
- Jaeger (popular, but vendor-specific)
- Zipkin (older, less feature-rich)
- Custom solution (complex, maintenance burden)

## Decision

We will use **OpenTelemetry** for distributed tracing with:
1. Jaeger as the trace backend (visualization)
2. OTLP (OpenTelemetry Protocol) for trace export
3. Automatic instrumentation for FastAPI, HTTP clients, databases
4. Manual instrumentation for custom spans
5. Log correlation via trace IDs

## Consequences

### Positive
- **Standard**: OpenTelemetry is the industry standard
- **Vendor-Neutral**: Can switch backends without code changes
- **Automatic**: Automatic instrumentation reduces boilerplate
- **Comprehensive**: Supports traces, metrics, and logs
- **Observability**: Full observability stack
- **Future-Proof**: Widely adopted, active development

### Negative
- **Overhead**: Adds small performance overhead
- **Complexity**: More complex than no tracing
- **Dependencies**: Additional dependencies and services
- **Learning Curve**: Team needs to learn OpenTelemetry

### Neutral
- Can be disabled in development
- Supports sampling to reduce overhead
- Can use different backends (Jaeger, Tempo, etc.)

## Implementation Details

- Use OpenTelemetry Python SDK
- Instrument FastAPI, HTTP clients, databases automatically
- Export traces via OTLP to collector
- Use Jaeger for trace visualization
- Include trace IDs in logs for correlation
- Support manual span creation for custom operations

## Alternatives Considered

1. **Jaeger Direct**:
   - Pros: Simple, direct integration
   - Cons: Vendor lock-in, less flexible

2. **Zipkin**:
   - Pros: Simple, lightweight
   - Cons: Older, less features, smaller ecosystem

3. **Custom Solution**:
   - Pros: Full control
   - Cons: Maintenance burden, not standard

---

**Related ADRs**: 
- ADR-0008: Prometheus and Grafana Monitoring

