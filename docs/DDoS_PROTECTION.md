# DDoS Protection Guide

This guide explains DDoS protection strategies for R3MES production deployment.

## Overview

DDoS (Distributed Denial of Service) attacks can overwhelm servers with traffic. This guide covers protection strategies.

## Protection Layers

### 1. Network Layer (Cloud Provider)

#### AWS Shield

If using AWS:
- Enable AWS Shield Standard (free)
- Consider AWS Shield Advanced for advanced protection
- Configure WAF rules

#### Cloudflare

If using Cloudflare:
- Enable DDoS protection (automatic)
- Configure rate limiting rules
- Enable Bot Fight Mode

### 2. Application Layer (Nginx)

#### Rate Limiting

Configure in `nginx/nginx.prod.conf`:

```nginx
# Rate limiting zones
limit_req_zone $binary_remote_addr zone=api_limit:10m rate=30r/s;
limit_req_zone $binary_remote_addr zone=dashboard_limit:10m rate=10r/s;

# Apply rate limiting
limit_req zone=api_limit burst=50 nodelay;
limit_req zone=dashboard_limit burst=20 nodelay;
```

#### Connection Limiting

```nginx
# Limit connections per IP
limit_conn_zone $binary_remote_addr zone=conn_limit:10m;
limit_conn conn_limit 10;
```

### 3. Application Layer (Backend)

#### Rate Limiting Middleware

Already implemented in FastAPI:
- Per-endpoint rate limits
- IP-based throttling
- Configurable limits

### 4. IP Whitelisting/Blacklisting

#### Nginx Configuration

```nginx
# Whitelist trusted IPs
geo $is_trusted {
    default 0;
    192.168.1.0/24 1;  # Internal network
}

# Block blacklisted IPs
geo $is_blocked {
    default 0;
    # Add blocked IPs here
}

server {
    if ($is_blocked) {
        return 403;
    }
    # ... rest of config
}
```

## Monitoring and Detection

### Metrics to Monitor

- Request rate per IP
- Connection count
- Error rate (4xx, 5xx)
- Response time
- Bandwidth usage

### Alerting

Set up alerts for:
- Unusual traffic spikes
- High error rates
- Connection exhaustion
- Bandwidth saturation

## Response Procedures

### 1. Detection

- Monitor traffic patterns
- Identify attack signatures
- Classify attack type (volumetric, protocol, application)

### 2. Mitigation

- Enable rate limiting
- Block malicious IPs
- Scale resources (if needed)
- Contact cloud provider support

### 3. Recovery

- Monitor traffic normalization
- Gradually remove mitigations
- Document incident
- Review and improve defenses

## Best Practices

1. **Multi-Layer Defense**: Use multiple protection layers
2. **Monitoring**: Continuously monitor traffic patterns
3. **Automation**: Automate response procedures
4. **Documentation**: Document response procedures
5. **Testing**: Regularly test DDoS response procedures

## Tools

- **Cloudflare**: CDN with DDoS protection
- **AWS Shield**: AWS-native DDoS protection
- **Nginx**: Rate limiting and connection limiting
- **Fail2ban**: IP blocking based on patterns

## Configuration Examples

### Nginx Rate Limiting

```nginx
# Define rate limit zones
limit_req_zone $binary_remote_addr zone=api:10m rate=30r/s;
limit_req_zone $binary_remote_addr zone=dashboard:10m rate=10r/s;

# Apply to API
location /api/ {
    limit_req zone=api burst=50 nodelay;
    # ... proxy config
}

# Apply to dashboard
location / {
    limit_req zone=dashboard burst=20 nodelay;
    # ... proxy config
}
```

### Fail2ban Configuration

```ini
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 5

[nginx-limit-req]
enabled = true
filter = nginx-limit-req
logpath = /var/log/nginx/error.log
maxretry = 3
```

