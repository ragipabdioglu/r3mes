# Distributed Tracing and Log Correlation Guide

This document describes the distributed tracing and log correlation setup for R3MES.

## Overview

R3MES uses **OpenTelemetry** for distributed tracing and **Jaeger** for trace visualization. Log correlation is achieved by including trace IDs and span IDs in all log messages.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Backend   │────▶│ OTLP Collector│────▶│   Jaeger    │
│ (traces)    │     │  (aggregate) │     │ (visualize) │
└─────────────┘     └──────────────┘     └─────────────┘
      │                     │
      │                     ▼
      │              ┌─────────────┐
      │              │    Loki     │
      │              │ (log store) │
      └─────────────▶└─────────────┘
      (logs with trace_id)
```

## Components

### 1. OpenTelemetry

OpenTelemetry provides:
- **Automatic instrumentation** for FastAPI, HTTP clients, databases
- **Manual instrumentation** for custom spans
- **Trace context propagation** across services

### 2. Jaeger

Jaeger provides:
- **Trace visualization** and analysis
- **Service dependency graphs**
- **Performance analysis** (latency, bottlenecks)

### 3. Loki + Promtail

Loki provides:
- **Log aggregation** from all services
- **Log correlation** with traces via trace IDs
- **LogQL queries** for log analysis

## Quick Start

### 1. Start Tracing Stack

```bash
cd docker
docker-compose -f docker-compose.yml -f docker-compose.tracing.yml up -d
```

### 2. Configure Backend

Set environment variables:

```bash
# Jaeger endpoint (HTTP)
export JAEGER_ENDPOINT=http://jaeger:14268/api/traces

# OR OTLP endpoint (preferred)
export OTLP_ENDPOINT=http://otel-collector:4317

# Enable console exporter for debugging
export ENABLE_CONSOLE_TRACING=true
```

### 3. Access Services

- **Jaeger UI**: http://localhost:16686
- **Loki**: http://localhost:3100
- **Grafana** (with Loki datasource): http://localhost:3001

## Trace ID in Logs

All log messages automatically include trace ID and span ID:

```
2025-12-24 10:30:45 - backend.app.main - INFO - [trace_id=abc123...] [span_id=def456...] - Request processed
```

This enables:
- **Log correlation**: Find all logs for a specific trace
- **Request tracking**: Follow a request across services
- **Debugging**: Trace errors back to their source

## Manual Instrumentation

### Creating Custom Spans

```python
from opentelemetry import trace
from .opentelemetry_setup import get_tracer

tracer = get_tracer()

# Create a span
with tracer.start_as_current_span("custom_operation") as span:
    span.set_attribute("user_id", user_id)
    span.set_attribute("operation", "process_payment")
    
    # Your code here
    result = process_payment(user_id, amount)
    
    span.set_attribute("result", result.status)
```

### Adding Attributes to Spans

```python
span = trace.get_current_span()
if span:
    span.set_attribute("key", "value")
    span.add_event("Event description", {"key": "value"})
```

## Log Correlation

### Automatic Correlation

The `TraceMiddleware` automatically:
1. Extracts trace ID from OpenTelemetry context
2. Adds trace ID to request state
3. Includes trace ID in all log messages
4. Adds trace ID to response headers (`X-Trace-ID`)

### Querying Logs by Trace ID

In Loki (via Grafana):

```logql
{container="r3mes-backend"} |= "trace_id=abc123..."
```

Or search for all logs with a specific trace ID:

```logql
{container=~"r3mes-.*"} | json | trace_id="abc123..."
```

## Service-to-Service Tracing

### Propagating Trace Context

When making HTTP requests, trace context is automatically propagated:

```python
import httpx

# Trace context is automatically added to headers
async with httpx.AsyncClient() as client:
    response = await client.get("http://other-service/api")
```

### Manual Propagation

```python
from opentelemetry import trace
from opentelemetry.propagate import inject

headers = {}
inject(headers)  # Adds trace context to headers

# Use headers in your request
response = requests.get(url, headers=headers)
```

## Grafana Integration

### Adding Loki Datasource

1. Go to Grafana → Configuration → Data Sources
2. Add Loki datasource
3. URL: `http://loki:3100`

### Creating Log Dashboards

Create dashboards with:
- **Log volume** over time
- **Error rate** by service
- **Trace ID search** for debugging
- **Service dependency** graphs

### Example LogQL Queries

```logql
# Error rate by service
sum(rate({container=~"r3mes-.*"} |= "ERROR" [5m])) by (container)

# Logs for a specific trace
{container="r3mes-backend"} | json | trace_id="abc123..."

# Top error messages
topk(10, sum(count_over_time({container=~"r3mes-.*"} |= "ERROR" [1h])) by (message))
```

## Jaeger Queries

### Finding Traces

1. **Service**: Select service (e.g., `r3mes-backend`)
2. **Operation**: Select operation (e.g., `GET /api/chat`)
3. **Tags**: Filter by tags (e.g., `error=true`)
4. **Time Range**: Select time range

### Analyzing Traces

- **Trace Timeline**: See span durations
- **Service Map**: Visualize service dependencies
- **Trace Comparison**: Compare similar traces
- **Statistics**: View latency percentiles

## Best Practices

### 1. Span Naming

Use consistent naming:
- **Good**: `GET /api/users/{id}`, `database.query.select`
- **Bad**: `operation1`, `do_stuff`

### 2. Attributes

Add meaningful attributes:
- **User ID**: `user_id`, `wallet_address`
- **Request ID**: `request_id`, `trace_id`
- **Operation**: `operation`, `endpoint`
- **Status**: `status_code`, `error_type`

### 3. Error Handling

Always record errors in spans:

```python
try:
    result = risky_operation()
except Exception as e:
    span.record_exception(e)
    span.set_status(Status(StatusCode.ERROR, str(e)))
    raise
```

### 4. Sampling

Configure sampling for high-traffic services:

```python
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

sampler = TraceIdRatioBased(0.1)  # Sample 10% of traces
tracer_provider = TracerProvider(sampler=sampler)
```

## Production Deployment

### Environment Variables

```bash
# OTLP endpoint (preferred)
OTLP_ENDPOINT=https://otel-collector.r3mes.network:4317

# Jaeger endpoint (fallback)
JAEGER_ENDPOINT=https://jaeger.r3mes.network:14268/api/traces

# Disable console exporter in production
ENABLE_CONSOLE_TRACING=false
```

### Security

1. **TLS**: Use TLS for OTLP/Jaeger endpoints in production
2. **Authentication**: Configure authentication for Jaeger UI
3. **Network**: Restrict access to tracing services

### Performance

1. **Sampling**: Use sampling to reduce overhead
2. **Batch Export**: Use batch span processor
3. **Async Export**: Use async exporters for better performance

## Troubleshooting

### Traces Not Appearing

1. Check OpenTelemetry setup: `ENABLE_CONSOLE_TRACING=true`
2. Verify endpoint configuration
3. Check Jaeger/OTLP collector logs
4. Verify network connectivity

### Logs Missing Trace IDs

1. Verify `TraceMiddleware` is added
2. Check log formatter includes trace_id
3. Verify OpenTelemetry context is propagated

### High Overhead

1. Enable sampling
2. Reduce span attributes
3. Use batch export
4. Consider async export

---

**Son Güncelleme**: 2025-12-24

