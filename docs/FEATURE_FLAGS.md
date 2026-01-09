# Feature Flags Documentation

## Overview

Feature flags allow you to enable/disable features and configure behavior without code changes. This document lists all available feature flags and their configuration.

## Backend Feature Flags

### Semantic Router

**Environment Variable:** `USE_SEMANTIC_ROUTER`

**Default:** `true`

**Description:** Enables embedding-based intelligent routing for LoRA adapter selection.

**Values:**
- `true`: Use semantic router (embedding-based similarity)
- `false`: Use keyword-based router (rule-based)

**Configuration:**
- `SEMANTIC_ROUTER_THRESHOLD`: Similarity threshold (0.0-1.0, default: `0.7`)

**Status:** Semantic router is mandatory. If initialization fails, the application will raise an error (no fallback).

**Dependencies:**
- `sentence-transformers` package (CPU-only, CUDA not required)

### Notification Channels

**Environment Variable:** `NOTIFICATION_CHANNELS`

**Default:** `email,slack`

**Description:** Comma-separated list of notification channels to use.

**Values:**
- `email`: Email notifications via SMTP
- `slack`: Slack notifications via webhook
- `in_app`: In-app notifications (stored in database)

**Configuration:**
- Email: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `ALERT_EMAIL`
- Slack: `SLACK_WEBHOOK_URL`, `SLACK_CHANNEL`

### Error Rate Monitoring

**Environment Variable:** `ERROR_RATE_THRESHOLD`

**Default:** `0.1` (10%)

**Description:** Error rate threshold for triggering alerts.

**Configuration:**
- `ERROR_RATE_THRESHOLD`: Error rate threshold (0.0-1.0, default: `0.1`)
- `ERROR_RATE_CHECK_INTERVAL`: Check interval in seconds (default: `60`)
- `ERROR_RATE_MIN_REQUESTS`: Minimum requests before alerting (default: `100`)

### Faucet

**Environment Variable:** `FAUCET_ENABLED`

**Default:** `true`

**Description:** Enable/disable the token faucet.

**Values:**
- `true`: Faucet is enabled
- `false`: Faucet is disabled

**Configuration:**
- `FAUCET_AMOUNT`: Default amount per claim (default: `1000000uremes` = 1 REMES)
- `FAUCET_DAILY_LIMIT`: Maximum amount per day (default: `5000000uremes` = 5 REMES)

### Blockchain Indexer

**Environment Variable:** `INDEXER_BATCH_SIZE`

**Default:** `10`

**Description:** Number of blocks to process in a batch for indexing.

**Note:** Indexer is automatically enabled when using PostgreSQL. No explicit flag needed.

## Frontend Feature Flags

### Analytics

**Environment Variable:** `NEXT_PUBLIC_GA_ID`

**Description:** Google Analytics tracking ID (optional).

**Note:** Analytics are only enabled if this variable is set.

### Sentry

**Environment Variable:** `NEXT_PUBLIC_SENTRY_DSN`

**Description:** Sentry DSN for error tracking (optional).

**Note:** Error tracking is only enabled if this variable is set.

## Docker Feature Flags

### CUDA Support

**Build Argument:** `CUDA_AVAILABLE`

**Default:** `false`

**Description:** Indicates whether CUDA is available for GPU operations.

**Note:** Semantic router does not require CUDA. CUDA is only needed for bitsandbytes (quantization).

## Feature Flag Best Practices

1. **Default to Safe**: Feature flags should default to safe/stable values
2. **Documentation**: All feature flags should be documented
3. **Environment Variables**: Use environment variables for configuration
4. **Validation**: Validate feature flag values on startup
5. **Monitoring**: Monitor feature flag usage and impact

## Configuration Examples

### Production (All Features Enabled)

```bash
USE_SEMANTIC_ROUTER=true
SEMANTIC_ROUTER_THRESHOLD=0.7
NOTIFICATION_CHANNELS=email,slack
ERROR_RATE_THRESHOLD=0.1
FAUCET_ENABLED=true
INDEXER_BATCH_SIZE=10
```

### Development (Minimal Features)

```bash
USE_SEMANTIC_ROUTER=false
NOTIFICATION_CHANNELS=in_app
ERROR_RATE_THRESHOLD=0.2
FAUCET_ENABLED=true
```

### Testing (No External Services)

```bash
USE_SEMANTIC_ROUTER=false
NOTIFICATION_CHANNELS=
ERROR_RATE_THRESHOLD=0.5
FAUCET_ENABLED=false
```

## Feature Flag Status

| Feature | Default | Required | Notes |
|---------|---------|----------|-------|
| Semantic Router | `true` | Yes | Mandatory - no fallback available |
| Notification Service | `email,slack` | No | Can be disabled by setting empty |
| Error Rate Monitoring | `0.1` | No | Can be disabled by setting high threshold |
| Faucet | `true` | No | Can be disabled for production |
| Blockchain Indexer | Auto | No | Auto-enabled with PostgreSQL |

## Troubleshooting

### Semantic Router Not Working

1. Check `USE_SEMANTIC_ROUTER` is set to `true`
2. Verify `sentence-transformers` is installed: `pip install sentence-transformers`
3. Check logs for initialization errors
4. Semantic router is mandatory - ensure `sentence-transformers` is installed

### Notifications Not Sending

1. Check `NOTIFICATION_CHANNELS` includes desired channels
2. Verify SMTP/Slack configuration is correct
3. Check logs for notification errors
4. Test notification channels during deployment

### Error Rate Alerts Too Frequent

1. Increase `ERROR_RATE_THRESHOLD` (e.g., `0.2` for 20%)
2. Increase `ERROR_RATE_MIN_REQUESTS` to avoid false positives
3. Adjust `ERROR_RATE_CHECK_INTERVAL` for less frequent checks

