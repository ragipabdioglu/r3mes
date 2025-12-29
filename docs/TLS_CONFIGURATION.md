# TLS Configuration Guide

This guide explains the TLS/SSL configuration for R3MES production deployment.

## Overview

R3MES uses TLS 1.2 and TLS 1.3 for secure communication:
- Frontend (Web Dashboard): HTTPS on port 443
- Backend API: HTTPS on port 443
- gRPC (mTLS): TLS with mutual authentication

## TLS Configuration

### Supported Protocols

- **TLS 1.3**: Modern, secure protocol (preferred)
- **TLS 1.2**: Legacy support (required for older clients)

### Cipher Suites

Only strong cipher suites are enabled:
- ECDHE-ECDSA-AES128-GCM-SHA256
- ECDHE-RSA-AES128-GCM-SHA256
- ECDHE-ECDSA-AES256-GCM-SHA384
- ECDHE-RSA-AES256-GCM-SHA384
- ECDHE-ECDSA-CHACHA20-POLY1305
- ECDHE-RSA-CHACHA20-POLY1305
- DHE-RSA-AES128-GCM-SHA256
- DHE-RSA-AES256-GCM-SHA384

### Security Features

1. **OCSP Stapling**: Reduces latency and improves privacy
2. **HSTS**: Forces HTTPS for 2 years (max-age=63072000)
3. **Session Tickets**: Disabled for better security
4. **Perfect Forward Secrecy**: Enabled via ECDHE cipher suites

## Configuration Files

### Nginx Configuration

- Frontend: `nginx/nginx.prod.conf`
- Backend: `nginx/nginx-backend.conf`

### TLS Settings

```nginx
ssl_protocols TLSv1.2 TLSv1.3;
ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:...';
ssl_prefer_server_ciphers off;
ssl_session_cache shared:SSL:10m;
ssl_session_timeout 10m;
ssl_session_tickets off;
ssl_stapling on;
ssl_stapling_verify on;
```

## Testing TLS Configuration

### SSL Labs Test

Test your configuration at: https://www.ssllabs.com/ssltest/

### Command Line Test

```bash
# Test TLS 1.3 support
openssl s_client -connect r3mes.network:443 -tls1_3

# Test cipher suites
openssl s_client -connect r3mes.network:443 -cipher 'ECDHE-RSA-AES128-GCM-SHA256'
```

## Best Practices

1. **Use TLS 1.3**: Enable TLS 1.3 for modern clients
2. **Strong Ciphers**: Only use strong cipher suites
3. **OCSP Stapling**: Always enable OCSP stapling
4. **HSTS**: Use HSTS with preload for maximum security
5. **Certificate Monitoring**: Monitor certificate expiration
6. **Regular Updates**: Keep Nginx and OpenSSL updated

## Troubleshooting

### TLS Handshake Failures

Check Nginx error logs:
```bash
sudo tail -f /var/log/nginx/error.log
```

### Certificate Chain Issues

Verify certificate chain:
```bash
openssl s_client -connect r3mes.network:443 -showcerts
```

### Cipher Suite Mismatch

Test with specific cipher:
```bash
openssl s_client -connect r3mes.network:443 -cipher 'ECDHE-RSA-AES128-GCM-SHA256'
```

