# SSL/TLS Certificate Management Guide

This guide explains how to manage SSL/TLS certificates for R3MES production deployment.

## Overview

R3MES uses Let's Encrypt for SSL/TLS certificates with automatic renewal via Certbot.

## Setup

### 1. Install Nginx Configuration

```bash
sudo ./scripts/setup_nginx.sh
```

### 2. Obtain SSL Certificates

```bash
sudo ./scripts/setup_letsencrypt.sh r3mes.network api.r3mes.network
```

This will:
- Install Certbot (if not already installed)
- Obtain certificates for specified domains
- Configure Nginx to use the certificates

### 3. Set Up Automatic Renewal

```bash
# Copy systemd service and timer
sudo cp scripts/systemd/certbot-renew.service /etc/systemd/system/
sudo cp scripts/systemd/certbot-renew.timer /etc/systemd/system/

# Enable and start timer
sudo systemctl daemon-reload
sudo systemctl enable certbot-renew.timer
sudo systemctl start certbot-renew.timer
```

## Certificate Locations

Certificates are stored in:
- `/etc/letsencrypt/live/{domain}/fullchain.pem` - Certificate chain
- `/etc/letsencrypt/live/{domain}/privkey.pem` - Private key
- `/etc/letsencrypt/live/{domain}/chain.pem` - Intermediate certificate

## Manual Renewal

```bash
sudo ./scripts/renew_certificates.sh
```

Or use Certbot directly:
```bash
sudo certbot renew
```

## Verification

### Check Certificate Status

```bash
sudo certbot certificates
```

### Test HTTPS Connection

```bash
curl -I https://r3mes.network
```

### Verify Certificate Details

```bash
openssl s_client -connect r3mes.network:443 -servername r3mes.network < /dev/null
```

## Troubleshooting

### Certificate Renewal Failing

1. Check Nginx is running:
   ```bash
   sudo systemctl status nginx
   ```

2. Check domain validation:
   ```bash
   curl http://r3mes.network/.well-known/acme-challenge/test
   ```

3. Check Certbot logs:
   ```bash
   sudo journalctl -u certbot-renew.service
   ```

### Certificate Expiring Soon

Let's Encrypt certificates are valid for 90 days. Renewal should happen automatically 30 days before expiry.

To force renewal:
```bash
sudo certbot renew --force-renewal
```

## Best Practices

1. **Automatic Renewal**: Always enable automatic renewal
2. **Monitoring**: Monitor certificate expiration dates
3. **Backup**: Backup certificate files before major changes
4. **Testing**: Test renewal process in staging before production
5. **Multiple Domains**: Use one certificate per domain for better security

## Security Considerations

- Private keys are stored in `/etc/letsencrypt/archive/` with restricted permissions
- Certificates are automatically renewed before expiration
- HSTS headers are configured for maximum security
- TLS 1.3 is enabled for modern clients

