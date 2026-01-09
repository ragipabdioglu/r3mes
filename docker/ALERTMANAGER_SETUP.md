# Alertmanager Setup Guide

## Overview

Alertmanager handles alerts from Prometheus and routes them to various notification channels (Slack, Email, PagerDuty, etc.).

## Quick Start

### 1. Configure Environment Variables

Add alert configuration to `docker/.env.production`:

```bash
# Slack (recommended for team notifications)
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL

# Email (for critical alerts)
ALERT_EMAIL_TO=alerts@your-domain.com
ALERT_EMAIL_FROM=alerts@your-domain.com
SMTP_HOST=smtp.gmail.com:587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

### 2. Restart Alertmanager

```bash
cd docker
docker-compose -f docker-compose.prod.yml restart alertmanager
```

### 3. Verify Configuration

```bash
# Check Alertmanager logs
docker logs r3mes-alertmanager-prod

# Access Alertmanager UI (if exposed via Nginx)
# http://your-domain.com:9093
```

## Notification Channels

### Slack Setup

1. **Create Slack Webhook**:
   - Go to https://api.slack.com/apps
   - Create a new app or use existing
   - Go to "Incoming Webhooks"
   - Activate Incoming Webhooks
   - Add New Webhook to Workspace
   - Choose channel (e.g., `#r3mes-alerts-critical`)
   - Copy webhook URL

2. **Configure in .env.production**:
   ```bash
   SLACK_WEBHOOK_URL=https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX
   ```

3. **Create Channels** (recommended):
   - `#r3mes-alerts-critical` - For critical alerts
   - `#r3mes-alerts-warning` - For warning alerts

4. **Restart Alertmanager**:
   ```bash
   docker-compose -f docker-compose.prod.yml restart alertmanager
   ```

### Email Setup

#### Gmail (Recommended for Testing)

1. **Enable 2FA** on your Google account

2. **Generate App Password**:
   - Go to: https://myaccount.google.com/apppasswords
   - Select "Mail" and your device
   - Copy the generated 16-character password

3. **Configure in .env.production**:
   ```bash
   ALERT_EMAIL_TO=your-email@gmail.com
   ALERT_EMAIL_FROM=your-email@gmail.com
   SMTP_HOST=smtp.gmail.com:587
   SMTP_USERNAME=your-email@gmail.com
   SMTP_PASSWORD=xxxx xxxx xxxx xxxx  # 16-char app password (no spaces)
   ```

#### SendGrid

```bash
ALERT_EMAIL_TO=alerts@your-domain.com
ALERT_EMAIL_FROM=alerts@your-domain.com
SMTP_HOST=smtp.sendgrid.net:587
SMTP_USERNAME=apikey
SMTP_PASSWORD=your-sendgrid-api-key
```

#### Mailgun

```bash
ALERT_EMAIL_TO=alerts@your-domain.com
ALERT_EMAIL_FROM=alerts@your-domain.com
SMTP_HOST=smtp.mailgun.org:587
SMTP_USERNAME=your-mailgun-username
SMTP_PASSWORD=your-mailgun-password
```

#### AWS SES

```bash
ALERT_EMAIL_TO=alerts@your-domain.com
ALERT_EMAIL_FROM=alerts@your-domain.com
SMTP_HOST=email-smtp.us-east-1.amazonaws.com:587
SMTP_USERNAME=your-aws-ses-smtp-username
SMTP_PASSWORD=your-aws-ses-smtp-password
```

### PagerDuty Setup (Optional)

1. **Create PagerDuty Service**:
   - Go to PagerDuty dashboard
   - Create new service
   - Choose "Prometheus" integration
   - Copy service key

2. **Configure in .env.production**:
   ```bash
   PAGERDUTY_SERVICE_KEY=your-pagerduty-service-key
   ```

3. **Uncomment in alertmanager.prod.yml**:
   ```yaml
   pagerduty_configs:
     - service_key: '${PAGERDUTY_SERVICE_KEY}'
   ```

## Alert Routing

### Alert Severities

- **Critical**: Service down, data loss, security breach
- **Warning**: High resource usage, degraded performance

### Alert Routes

Alerts are automatically routed based on severity:

- **Critical alerts** → `critical-alerts` receiver
  - Slack: `#r3mes-alerts-critical`
  - Email: Sent to `ALERT_EMAIL_TO`
  - PagerDuty: (if configured)

- **Warning alerts** → `warning-alerts` receiver
  - Slack: `#r3mes-alerts-warning`
  - Email: (optional, commented out by default)

### Custom Routes

Edit `docker/alertmanager/alertmanager.prod.yml` to add custom routes:

```yaml
routes:
  - match:
      alertname: 'DatabaseDown'
    receiver: 'critical-alerts'
    continue: false
  - match:
      severity: critical
    receiver: 'critical-alerts'
    continue: true
```

## Testing Alerts

### Test Slack Notification

```bash
# Send test alert via Alertmanager API
curl -X POST http://localhost:9093/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[{
    "labels": {
      "alertname": "TestAlert",
      "severity": "critical",
      "service": "backend"
    },
    "annotations": {
      "summary": "This is a test alert",
      "description": "Testing Slack notification"
    }
  }]'
```

### Test Email Notification

```bash
# Send test alert
curl -X POST http://localhost:9093/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[{
    "labels": {
      "alertname": "TestEmailAlert",
      "severity": "critical"
    },
    "annotations": {
      "summary": "Test email alert",
      "description": "Testing email notification"
    }
  }]'
```

### Verify in Prometheus

1. Access Prometheus: `http://your-domain.com:9090`
2. Go to "Alerts" tab
3. Check if test alert appears
4. Verify it's sent to Alertmanager

## Alert Examples

### Pre-configured Alerts

The following alerts are pre-configured in `docker/prometheus/alerts/r3mes-alerts.yml`:

- **BackendDown**: Backend service is down
- **HighAPIErrorRate**: API error rate is too high
- **DatabaseConnectionPoolExhausted**: Database connection pool is full
- **RedisMemoryHigh**: Redis memory usage is high
- **DiskSpaceLow**: Disk space is running low
- **HighCPUUsage**: CPU usage is too high
- **HighMemoryUsage**: Memory usage is too high

### Viewing Alerts

```bash
# View active alerts in Prometheus
curl http://localhost:9090/api/v1/alerts

# View alerts in Alertmanager
curl http://localhost:9093/api/v1/alerts
```

## Troubleshooting

### Alerts Not Sending

1. **Check Alertmanager logs**:
   ```bash
   docker logs r3mes-alertmanager-prod
   ```

2. **Verify environment variables**:
   ```bash
   docker exec r3mes-alertmanager-prod env | grep -E "SLACK|EMAIL|SMTP"
   ```

3. **Test Slack webhook**:
   ```bash
   curl -X POST "$SLACK_WEBHOOK_URL" \
     -H "Content-Type: application/json" \
     -d '{"text":"Test message"}'
   ```

4. **Test SMTP connection**:
   ```bash
   docker exec r3mes-alertmanager-prod sh -c "
     echo 'Subject: Test' | \
     sendmail -v -S smtp.gmail.com:587 \
       -au $SMTP_USERNAME \
       -ap $SMTP_PASSWORD \
       $ALERT_EMAIL_TO
   "
   ```

### Email Not Working

**Common Issues**:

1. **Gmail App Password**: Make sure you're using app password, not regular password
2. **2FA Required**: Gmail requires 2FA to be enabled
3. **SMTP Port**: Use port 587 (TLS) or 465 (SSL)
4. **Firewall**: Ensure port 587/465 is not blocked

**Debug**:
```bash
# Check SMTP settings
docker exec r3mes-alertmanager-prod env | grep SMTP

# Test SMTP manually
docker exec -it r3mes-alertmanager-prod sh
# Then inside container:
telnet smtp.gmail.com 587
```

### Slack Not Working

**Common Issues**:

1. **Invalid Webhook URL**: Check webhook URL format
2. **Channel Not Found**: Ensure channel exists and webhook has access
3. **Webhook Expired**: Regenerate webhook if it's old

**Debug**:
```bash
# Test webhook directly
curl -X POST "$SLACK_WEBHOOK_URL" \
  -H "Content-Type: application/json" \
  -d '{"text":"Test from R3MES"}'
```

## Best Practices

### 1. Use Multiple Channels

- **Slack**: For team notifications (real-time)
- **Email**: For critical alerts (persistent)
- **PagerDuty**: For on-call escalation

### 2. Alert Grouping

Alerts are automatically grouped by:
- `alertname`
- `cluster`
- `service`
- `severity`

### 3. Alert Inhibition

Duplicate alerts are automatically suppressed:
- If `BackendDown` is firing, `HighAPIErrorRate` is suppressed
- If critical alert is firing, warning alerts are suppressed

### 4. Alert Frequency

- **Group Wait**: 10s (wait before sending first alert)
- **Group Interval**: 10s (wait between alert groups)
- **Repeat Interval**: 12h (resend if alert persists)

### 5. Testing

- Test alerts regularly (monthly)
- Verify all notification channels work
- Document alert response procedures

## Configuration Files

- **Alertmanager Config**: `docker/alertmanager/alertmanager.prod.yml`
- **Prometheus Alerts**: `docker/prometheus/alerts/r3mes-alerts.yml`
- **Environment Variables**: `docker/.env.production`

## Additional Resources

- [Alertmanager Documentation](https://prometheus.io/docs/alerting/latest/alertmanager/)
- [Slack Incoming Webhooks](https://api.slack.com/messaging/webhooks)
- [Gmail App Passwords](https://support.google.com/accounts/answer/185833)

---

**Last Updated**: 2025-01-14  
**Maintained by**: R3MES Development Team
