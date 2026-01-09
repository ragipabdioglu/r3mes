# Log Storage and Retention Guide

This guide explains log storage, archiving, and retention policies for R3MES.

## Overview

R3MES uses a multi-tier log storage strategy:
- **Local Storage**: Recent logs (30 days)
- **Archive Storage**: Long-term logs in S3 (90 days)
- **Centralized Logging**: Loki for real-time log aggregation

## Log Locations

### Application Logs

- **Backend**: `/var/log/r3mes/backend*.log`
- **Frontend**: `/var/log/r3mes/frontend*.log`
- **Nginx**: `/var/log/nginx/*.log`
- **PostgreSQL**: `/var/log/postgresql/*.log`
- **Redis**: `/var/log/redis/*.log`

### Log Rotation

Logs are automatically rotated by:
- **logrotate**: System log rotation
- **Application**: Built-in rotation (if configured)

## Retention Policy

### Local Storage

- **Retention**: 30 days
- **Location**: `/var/log/r3mes/`
- **Cleanup**: Daily at 3 AM

### Archive Storage (S3)

- **Retention**: 90 days
- **Location**: `s3://r3mes-logs/logs/`
- **Format**: Compressed (gzip)
- **Naming**: `YYYYMMDD_logname.log.gz`

## Log Archiving

### Manual Archive

```bash
./scripts/archive_logs.sh /var/log/r3mes
```

### Automated Archive

Archiving happens automatically before log deletion:
- Triggered by cleanup script
- Uploads to S3
- Removes local files after successful upload

## Log Cleanup

### Manual Cleanup

```bash
# Cleanup with archiving
RETENTION_DAYS=30 ARCHIVE_BEFORE_DELETE=true ./scripts/cleanup_logs.sh

# Cleanup without archiving
RETENTION_DAYS=30 ARCHIVE_BEFORE_DELETE=false ./scripts/cleanup_logs.sh
```

### Automated Cleanup

Systemd timer runs daily:
```bash
# Enable timer
sudo systemctl enable r3mes-log-cleanup.timer
sudo systemctl start r3mes-log-cleanup.timer
```

## Centralized Logging (Loki)

### Setup

```bash
# Start Loki stack
docker-compose -f docker-compose.yml -f docker-compose.logging.yml up -d
```

### Configuration

- **Loki**: `monitoring/loki/loki-config.yml`
- **Promtail**: `monitoring/promtail/promtail-config.yml`
- **Retention**: 30 days in Loki
- **Storage**: Filesystem (can be configured for S3)

### Querying Logs

Access via Grafana:
1. Open Grafana
2. Go to Explore
3. Select Loki datasource
4. Query logs using LogQL

Example queries:
```
# Backend errors
{job="r3mes-backend"} |= "ERROR"

# High latency requests
{job="r3mes-backend"} | json | latency > 2

# Specific service
{service="backend"} | json
```

## S3 Configuration

### Bucket Setup

```bash
# Create S3 bucket
aws s3 mb s3://r3mes-logs --region us-east-1

# Set lifecycle policy
aws s3api put-bucket-lifecycle-configuration \
  --bucket r3mes-logs \
  --lifecycle-configuration file://lifecycle-policy.json
```

### Lifecycle Policy

```json
{
  "Rules": [
    {
      "Id": "LogLifecycle",
      "Status": "Enabled",
      "Transitions": [
        {
          "Days": 30,
          "StorageClass": "GLACIER"
        }
      ],
      "Expiration": {
        "Days": 90
      }
    }
  ]
}
```

## Best Practices

1. **Regular Cleanup**: Run cleanup daily
2. **Archive Before Delete**: Always archive before deleting
3. **Monitor Storage**: Monitor log storage usage
4. **Retention Policy**: Adjust retention based on requirements
5. **Centralized Logging**: Use Loki for real-time log analysis

## Troubleshooting

### Logs Not Archiving

1. Check S3 bucket configuration
2. Verify AWS credentials
3. Check network connectivity
4. Review archive script logs

### High Storage Usage

1. Review retention policy
2. Check for log rotation issues
3. Verify cleanup script is running
4. Consider reducing retention period

### Loki Not Collecting Logs

1. Check Promtail status
2. Verify log file paths
3. Check Loki connectivity
4. Review Promtail configuration

