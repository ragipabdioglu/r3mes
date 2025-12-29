# R3MES Monitoring and Observability

This document describes the monitoring and observability infrastructure for R3MES.

## Overview

R3MES uses Prometheus for metrics collection, Grafana for visualization, and Alertmanager for alerting.

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Backend   │────▶│  Prometheus  │────▶│   Grafana   │
│   (FastAPI) │     │   (Metrics)  │     │ (Dashboards)│
└─────────────┘     └──────────────┘     └─────────────┘
                            │
                            ▼
                    ┌──────────────┐
                    │ Alertmanager │
                    │  (Alerts)    │
                    └──────────────┘
```

## Metrics Endpoint

### Backend API

The backend exposes Prometheus metrics at:

```
GET /metrics
```

**Response**: Prometheus text format

**Example**:
```
# HELP api_requests_total Total number of API requests
# TYPE api_requests_total counter
api_requests_total{method="GET",endpoint="/status",status_code="200"} 1234
```

## Available Metrics

### API Metrics

- `api_requests_total` - Total API requests (labels: method, endpoint, status_code)
- `api_request_duration_seconds` - API request duration histogram

### Cache Metrics

- `cache_hits_total` - Total cache hits
- `cache_misses_total` - Total cache misses

### Database Metrics

- `database_connections_active` - Active database connections
- `database_query_duration_seconds` - Query duration histogram

### Model Inference Metrics

- `model_inference_duration_seconds` - Inference duration histogram
- `model_inference_requests_total` - Total inference requests

### System Metrics

- `system_memory_usage_bytes` - Memory usage in bytes
- `system_cpu_usage_percent` - CPU usage percentage

### GPU Metrics

- `gpu_utilization_percent` - GPU utilization (per GPU)
- `gpu_memory_usage_bytes` - GPU memory usage (per GPU)
- `gpu_temperature_celsius` - GPU temperature (per GPU)

## Grafana Dashboards

### Backend Metrics Dashboard

Location: `monitoring/grafana/dashboards/r3mes-backend.json`

**Panels**:
1. API Requests Rate
2. API Request Duration (p95)
3. Cache Hit Rate
4. System Memory Usage
5. CPU Usage
6. GPU Utilization
7. Model Inference Duration
8. Database Query Duration

### Importing Dashboards

1. Open Grafana UI
2. Go to Dashboards → Import
3. Upload `monitoring/grafana/dashboards/r3mes-backend.json`
4. Select Prometheus data source
5. Click Import

## Prometheus Configuration

Configuration file: `monitoring/prometheus/prometheus.yml`

### Scrape Targets

- **Backend API**: `backend:8000/metrics` (every 10s)
- **Blockchain Node**: `remesd:26660` (every 30s)
- **Prometheus**: `localhost:9090` (self-monitoring)

## Alerting Rules

Configuration file: `monitoring/prometheus/alerts.yml`

### Alert Rules

1. **HighAPIErrorRate** - API error rate > 10% for 5 minutes
2. **HighAPILatency** - 95th percentile latency > 2s for 5 minutes
3. **LowCacheHitRate** - Cache hit rate < 50% for 10 minutes
4. **HighMemoryUsage** - Memory usage > 14 GB for 5 minutes
5. **HighCPUUsage** - CPU usage > 80% for 10 minutes
6. **DatabaseConnectionFailure** - No active connections for 2 minutes
7. **HighInferenceLatency** - 95th percentile > 10s for 5 minutes
8. **GPUOverheating** - GPU temperature > 85°C for 5 minutes

### Alert Severity Levels

- **critical**: Immediate action required
- **warning**: Attention needed, but not urgent

## Alertmanager Configuration

### Notification Channels

Configure in `monitoring/alertmanager/config.yml`:

- **Email**: Send alerts to team email
- **Slack**: Post to Slack channel
- **PagerDuty**: Escalate critical alerts
- **Webhook**: Custom webhook integration

## Docker Compose Setup

Add to `docker/docker-compose.yml`:

```yaml
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'

  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  alertmanager:
    image: prom/alertmanager:latest
    volumes:
      - ./monitoring/alertmanager:/etc/alertmanager
    ports:
      - "9093:9093"
```

## Best Practices

1. **Set appropriate scrape intervals** - Balance between freshness and load
2. **Use histogram buckets** - For latency metrics, use appropriate buckets
3. **Label metrics properly** - Use consistent label names
4. **Monitor alert fatigue** - Avoid too many alerts
5. **Document alert runbooks** - Include resolution steps
6. **Regular dashboard reviews** - Update dashboards as system evolves

## Troubleshooting

### Metrics Not Appearing

1. Check `/metrics` endpoint is accessible
2. Verify Prometheus can reach the target
3. Check Prometheus logs for scrape errors
4. Verify metric names match Prometheus naming conventions

### High Cardinality

- Avoid high-cardinality labels (e.g., user IDs)
- Use aggregation where possible
- Consider sampling for high-volume metrics

### Alert Not Firing

1. Check alert rule syntax
2. Verify Prometheus is evaluating rules
3. Check Alertmanager configuration
4. Verify notification channel settings

## Additional Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)

