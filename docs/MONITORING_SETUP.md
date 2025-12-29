# Monitoring Setup Guide

## Production Monitoring

For production deployment, use the production monitoring stack:

```bash
./scripts/setup_monitoring.sh
```

This deploys:
- Prometheus (with production configuration)
- Grafana (with pre-configured dashboards)
- Alertmanager (with notification channels)

## Production Configuration

Production monitoring uses:
- `monitoring/prometheus/prometheus.prod.yml` - Production Prometheus config
- `monitoring/prometheus/alerts.prod.yml` - Production alert rules
- `docker/alertmanager/alertmanager.prod.yml` - Production Alertmanager config
- `docker/docker-compose.monitoring.prod.yml` - Production Docker Compose

## Health Check Monitoring

Health check metrics are exported via Prometheus:
- `health_check_status` - Health check status (1=healthy, 0=unhealthy)
- `health_check_duration_seconds` - Health check duration
- `health_check_last_success` - Last successful check timestamp
- `health_check_last_failure` - Last failed check timestamp

Alerts are configured for:
- Health check failures
- Slow health checks
- Service downtime

# Monitoring Setup Guide

This document describes the monitoring setup for R3MES using Prometheus, Grafana, and Alertmanager.

## Overview

The R3MES monitoring stack includes:

- **Prometheus**: Metrics collection and storage
- **Grafana**: Visualization and dashboards
- **Alertmanager**: Alert routing and notification
- **Node Exporter**: System metrics (CPU, memory, disk)
- **Redis Exporter**: Redis metrics
- **PostgreSQL Exporter**: Database metrics

## Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Backend   │────▶│ Prometheus │────▶│   Grafana   │
│  (metrics)  │     │  (collect) │     │ (visualize) │
└─────────────┘     └──────────────┘     └─────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Alertmanager │
                    │  (alerts)    │
                    └──────────────┘
```

## Quick Start

### 1. Start Monitoring Stack

```bash
cd docker
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d
```

### 2. Access Services

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001 (admin/admin)
- **Alertmanager**: http://localhost:9093

### 3. Import Dashboards

Grafana dashboards are automatically provisioned from `docker/grafana/dashboards/`.

## Metrics Endpoints

### Backend API

- **Endpoint**: `http://backend:8000/metrics`
- **Format**: Prometheus text format
- **Scrape Interval**: 10 seconds

### Metrics Exposed

#### API Metrics
- `api_requests_total`: Total API requests (by method, endpoint, status)
- `api_request_duration_seconds`: Request latency histogram

#### Cache Metrics
- `cache_hits_total`: Total cache hits
- `cache_misses_total`: Total cache misses

#### Database Metrics
- `database_connections_active`: Active database connections
- `database_query_duration_seconds`: Query duration histogram

#### Model Metrics
- `model_inference_duration_seconds`: Inference duration histogram
- `model_inference_requests_total`: Total inference requests

#### System Metrics
- `system_memory_usage_bytes`: Memory usage in bytes
- `system_cpu_usage_percent`: CPU usage percentage
- `gpu_utilization_percent`: GPU utilization (if available)
- `gpu_memory_usage_bytes`: GPU memory usage (if available)
- `gpu_temperature_celsius`: GPU temperature (if available)

## Prometheus Configuration

### Scrape Targets

Prometheus is configured to scrape:

1. **Backend API** (`r3mes-backend`): `/metrics` endpoint
2. **Node Exporter**: System metrics
3. **Redis Exporter**: Redis metrics
4. **PostgreSQL Exporter**: Database metrics

### Configuration File

Location: `docker/prometheus/prometheus.yml`

Key settings:
- **Scrape Interval**: 15 seconds
- **Evaluation Interval**: 15 seconds
- **Retention**: 30 days

## Alert Rules

### Alert Categories

1. **Backend Alerts**:
   - High error rate (> 0.1 errors/second)
   - High latency (95th percentile > 2s)
   - Low cache hit rate (< 50%)
   - Slow database queries (95th percentile > 1s)
   - Slow model inference (95th percentile > 10s)
   - High memory usage (> 8 GB)
   - High CPU usage (> 80%)
   - Service down

2. **Infrastructure Alerts**:
   - Database down
   - Redis down
   - High disk usage (> 90%)

### Alert Configuration

Location: `docker/prometheus/alerts/r3mes-alerts.yml`

Alert severity levels:
- **critical**: Immediate action required
- **warning**: Attention needed

## Grafana Dashboards

### Pre-configured Dashboards

1. **R3MES Backend Metrics**:
   - API request rate
   - API error rate
   - API latency (95th percentile)
   - Cache hit rate
   - Database query duration
   - Model inference duration
   - System memory usage
   - System CPU usage

### Dashboard Location

Location: `docker/grafana/dashboards/r3mes-backend.json`

Dashboards are automatically provisioned on Grafana startup.

### Creating Custom Dashboards

1. Create dashboard JSON file in `docker/grafana/dashboards/`
2. Restart Grafana container
3. Dashboard will be automatically imported

## Alertmanager Configuration

### Notification Channels

Alertmanager supports multiple notification channels:

- **Webhook**: HTTP webhook for custom integrations
- **Slack**: Slack notifications (configure via `SLACK_WEBHOOK_URL`)
- **Email**: Email notifications (configure in `alertmanager.yml`)

### Configuration File

Location: `docker/alertmanager/alertmanager.yml`

### Routing Rules

- **Critical alerts**: Sent to `critical-alerts` receiver
- **Warning alerts**: Sent to `warning-alerts` receiver
- **Default**: Sent to `default` receiver

## System Metrics Collection

The backend automatically collects system metrics:

- **Interval**: 10 seconds
- **Metrics**: CPU, memory, GPU (if available)
- **Implementation**: `backend/app/system_metrics_collector.py`

## Integration with Cache Metrics

Cache metrics are automatically exported to Prometheus:

- Cache hits/misses are recorded via `record_cache_hit()` and `record_cache_miss()`
- Metrics are available at `/metrics` endpoint

## Production Deployment

### Environment Variables

Set the following in production:

```bash
# Grafana
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=<secure-password>
GRAFANA_ROOT_URL=https://grafana.r3mes.network

# Prometheus
PROMETHEUS_PORT=9090

# Alertmanager
ALERTMANAGER_PORT=9093
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### Security Considerations

1. **Grafana Authentication**: Change default admin password
2. **Prometheus Access**: Restrict access to internal network
3. **Alertmanager**: Secure webhook endpoints
4. **TLS**: Use HTTPS for all external access

### Scaling

For high-traffic deployments:

1. **Prometheus**: Increase retention period or use remote storage
2. **Grafana**: Use Grafana Cloud or self-hosted with load balancing
3. **Alertmanager**: Use clustering for high availability

## Troubleshooting

### Metrics Not Appearing

1. Check Prometheus targets: http://localhost:9090/targets
2. Verify backend `/metrics` endpoint is accessible
3. Check Prometheus logs: `docker logs r3mes-prometheus`

### Alerts Not Firing

1. Check alert rules: http://localhost:9090/alerts
2. Verify Alertmanager is running: `docker ps | grep alertmanager`
3. Check Alertmanager logs: `docker logs r3mes-alertmanager`

### Grafana Dashboards Not Loading

1. Check datasource configuration: Grafana → Configuration → Data Sources
2. Verify Prometheus is accessible from Grafana
3. Check dashboard JSON syntax

## Best Practices

1. **Retention**: Set appropriate retention period based on storage capacity
2. **Alerts**: Keep alert rules focused and actionable
3. **Dashboards**: Create dashboards for different user roles (ops, dev, business)
4. **Documentation**: Document custom metrics and alert rules
5. **Testing**: Test alert notifications regularly
6. **Backup**: Backup Grafana dashboards and Prometheus configuration

---

**Son Güncelleme**: 2025-12-24

