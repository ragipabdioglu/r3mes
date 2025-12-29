# R3MES Logging Policy

## Overview

This document outlines the logging policy for the R3MES project, including sensitive data protection, log rotation, and retention policies.

## Sensitive Data Protection

### Automatic Masking

All loggers in the R3MES project automatically filter and mask sensitive information:

- **Passwords**: `password=***MASKED***`
- **Private Keys**: `private_key=***MASKED***`
- **API Keys**: `api_key=***MASKED***`
- **Mnemonics**: `mnemonic=***MASKED***`
- **Authorization Tokens**: `authorization=***MASKED***`
- **Secrets**: `secret=***MASKED***`
- **Tokens**: `token=***MASKED***`

### Implementation

- **Backend**: `backend/app/setup_logging.py` - `SensitiveDataFilter` class
- **Miner Engine**: `miner-engine/utils/logger.py` - `SensitiveDataFilter` class
- **Sentry**: `backend/app/sentry.py` and `remes/x/remes/keeper/sentry.go` - BeforeSend filters

### Best Practices

1. **Never log sensitive data directly**:
   ```python
   # BAD
   logger.info(f"User password: {password}")
   
   # GOOD
   logger.info("User authentication attempted")
   ```

2. **Use structured logging**:
   ```python
   # BAD
   logger.info(f"API key: {api_key}")
   
   # GOOD
   logger.info("API request received", extra={"has_api_key": bool(api_key)})
   ```

3. **Mask in error messages**:
   ```python
   # BAD
   logger.error(f"Failed to connect with key: {private_key}")
   
   # GOOD
   logger.error("Failed to connect", exc_info=True)
   ```

## Log Rotation Policy

### Backend Logs

- **Location**: `~/.r3mes/logs/` (configurable via `R3MES_LOG_DIR`)
- **Main Log**: `r3mes_backend.log`
- **Error Log**: `r3mes_backend_errors.log`
- **Max File Size**: 10MB (configurable via `max_bytes` parameter)
- **Backup Count**: 5 files (configurable via `backup_count` parameter)
- **Rotation**: Automatic when file size exceeds max_bytes

### Miner Engine Logs

- **Location**: Configurable via `log_file` parameter
- **Max File Size**: 10MB (default)
- **Backup Count**: 5 files (default)
- **Rotation**: Automatic when file size exceeds max_bytes

### Log Retention

- **Production**: Keep last 5 rotated files (~50MB total per log file)
- **Development**: Keep last 3 rotated files (~30MB total per log file)
- **Archival**: Old logs should be archived to external storage (S3, etc.) after 30 days

## Log Levels

### Standard Levels

- **TRACE** (5): Very detailed debugging information
- **DEBUG** (10): Detailed debugging information
- **INFO** (20): General informational messages
- **WARNING** (30): Warning messages for potential issues
- **ERROR** (40): Error messages for failures

### Production Recommendations

- **Console**: INFO level and above
- **File**: DEBUG level and above (captures everything)
- **Error File**: ERROR level only
- **Sentry**: ERROR level and above (with sensitive data filtering)

## Configuration

### Environment Variables

```bash
# Log directory
R3MES_LOG_DIR=/var/log/r3mes

# Log level
LOG_LEVEL=INFO

# JSON logs (for log aggregation systems)
USE_JSON_LOGS=true
```

### Backend Configuration

```python
from app.setup_logging import setup_logging

setup_logging(
    log_dir="/var/log/r3mes",
    log_level="INFO",
    enable_file_logging=True,
    max_bytes=10 * 1024 * 1024,  # 10MB
    backup_count=5
)
```

### Miner Engine Configuration

```python
from utils.logger import setup_logger

logger = setup_logger(
    "r3mes.miner",
    level=logging.INFO,
    use_json=True,
    log_file="/var/log/r3mes/miner.log",
    max_bytes=10 * 1024 * 1024,  # 10MB
    backup_count=5
)
```

## Log Aggregation

### Recommended Tools

- **ELK Stack** (Elasticsearch, Logstash, Kibana)
- **Loki + Grafana**
- **CloudWatch** (AWS)
- **Datadog**
- **Splunk**

### JSON Format

For production log aggregation, enable JSON logging:

```bash
USE_JSON_LOGS=true
```

JSON logs include:
- `timestamp`: ISO 8601 format
- `level`: Log level
- `logger`: Logger name
- `message`: Log message (with sensitive data masked)
- `module`: Python module name
- `function`: Function name
- `line`: Line number
- `exception`: Exception details (if present)

## Monitoring and Alerts

### Key Metrics to Monitor

- **Error Rate**: Number of ERROR level logs per minute
- **Log Volume**: Total log size per hour
- **Sensitive Data Leaks**: Alerts if unmasked sensitive data detected
- **Log Rotation Failures**: Alerts if rotation fails

### Alert Thresholds

- **Error Rate**: > 10 errors/minute
- **Log Volume**: > 1GB/hour
- **Sensitive Data**: Any unmasked sensitive data detected

## Compliance

### GDPR Compliance

- Logs containing user data must be anonymized
- Log retention must comply with data retention policies
- Users have the right to request log deletion

### Security Compliance

- All sensitive data must be masked in logs
- Log files must be encrypted at rest
- Log access must be restricted to authorized personnel

## Troubleshooting

### Logs Not Rotating

1. Check file permissions
2. Verify disk space
3. Check `max_bytes` configuration
4. Verify `backup_count` is set correctly

### Sensitive Data in Logs

1. Check if `SensitiveDataFilter` is added to handlers
2. Verify filter patterns match your data format
3. Review code for direct sensitive data logging
4. Use structured logging instead of string formatting

### High Log Volume

1. Increase log level (INFO → WARNING)
2. Reduce `max_bytes` for faster rotation
3. Implement log sampling for high-volume operations
4. Use log aggregation to offload storage

## References

- [Python Logging Documentation](https://docs.python.org/3/library/logging.html)
- [Log Rotation Best Practices](https://www.loggly.com/ultimate-guide/python-logging-basics/)
- [GDPR Logging Requirements](https://gdpr.eu/data-protection-by-design-and-by-default/)

