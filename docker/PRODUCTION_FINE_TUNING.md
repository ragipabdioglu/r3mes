# R3MES Production Fine-Tuning Guide

**Date**: 2025-01-14  
**Status**: Production Ready with Fine-Tuning Applied

---

## ✅ Completed Fine-Tuning

### 1. ✅ Nginx Domain Dynamic Configuration

**Problem**: Domain was hardcoded in nginx.conf

**Solution**: 
- Updated `docker/nginx/nginx.conf` to use `${DOMAIN}` placeholder
- Updated `docker/nginx/Dockerfile` to use `envsubst` for template processing
- Added `DOMAIN` environment variable to nginx service in `docker-compose.prod.yml`

**Files Changed**:
- `docker/nginx/nginx.conf` - Domain paths now use `${DOMAIN}`
- `docker/nginx/Dockerfile` - Added `gettext` package and template processing
- `docker/docker-compose.prod.yml` - Added `DOMAIN` environment variable to nginx service

**Usage**:
```bash
# Set domain in .env.production
DOMAIN=your-domain.com

# Nginx will automatically use this domain for SSL certificates
```

### 2. ✅ Log Rotation Configuration

**Problem**: Container logs could grow indefinitely

**Solution**: Added log rotation to all services in `docker-compose.prod.yml`

**Configuration**:
```yaml
logging:
  driver: "json-file"
  options:
    max-size: "10m"
    max-file: "3"
```

**Applied to**:
- PostgreSQL
- Redis
- IPFS
- Blockchain (remesd)
- Backend
- Frontend
- Nginx
- Miner
- Prometheus
- Grafana
- Alertmanager
- Node Exporter
- Redis Exporter
- PostgreSQL Exporter
- Postgres Backup

**Result**: Each container keeps maximum 3 log files of 10MB each (30MB total per container)

### 3. ✅ Firewall Configuration

**Problem**: No firewall configuration guide or script

**Solution**: 
- Created `scripts/setup_firewall.sh` - Automated firewall setup script
- Created `docker/FIREWALL_GUIDE.md` - Comprehensive firewall documentation

**Features**:
- Automated UFW setup
- Interactive blockchain port configuration
- Security best practices
- Cloud provider specific instructions

**Usage**:
```bash
sudo bash scripts/setup_firewall.sh
```

### 4. ✅ Monitoring Alert Channels

**Problem**: Alertmanager configured but no documentation on how to set up notifications

**Solution**: 
- Created comprehensive Alertmanager setup guide
- Added alert configuration to `env.production.example`
- Updated README with alert setup instructions

**Files Created**:
- `docker/ALERTMANAGER_SETUP.md` - Complete Alertmanager setup guide
- Updated `docker/env.production.example` - Alert configuration examples
- Updated `docker/README_PRODUCTION.md` - Alert setup instructions

**Configuration**: `docker/alertmanager/alertmanager.prod.yml`

**Environment Variables** (set in `.env.production`):
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/YOUR/WEBHOOK/URL
ALERT_EMAIL_TO=alerts@your-domain.com
ALERT_EMAIL_FROM=alerts@your-domain.com
SMTP_HOST=smtp.gmail.com:587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

**Documentation**: See `docker/ALERTMANAGER_SETUP.md` for:
- Slack webhook setup
- Email configuration (Gmail, SendGrid, Mailgun, AWS SES)
- PagerDuty integration
- Testing alerts
- Troubleshooting

### 5. ✅ Backup Restore Test Script

**Problem**: No way to verify backup/restore functionality

**Solution**: Created `scripts/test_backup_restore.sh`

**Features**:
- Creates test data
- Creates backup
- Deletes test data
- Restores backup
- Verifies data integrity
- Cleans up

**Usage**:
```bash
bash scripts/test_backup_restore.sh
```

---

## 📋 Production Readiness Checklist

### Critical (Must Complete Before Production)

- [x] Docker Secrets configured
- [x] Environment variables set
- [x] Domain configured in `.env.production`
- [x] SSL certificates (automatic via Let's Encrypt)
- [x] Firewall configured
- [x] Log rotation enabled
- [x] Monitoring stack running

### Important (Complete Within First Week)

- [ ] Alert channels configured (Slack/Email)
- [ ] Backup restore tested
- [ ] Monitoring dashboards reviewed
- [ ] Performance baseline established

### Optional (Optimization)

- [ ] CDN integration
- [ ] Performance tuning (PostgreSQL, Redis)
- [ ] Additional monitoring integrations
- [ ] Disaster recovery plan tested

---

## 🔧 Configuration Summary

### Environment Variables

**Required in `.env.production`**:
```bash
DOMAIN=your-domain.com
EMAIL=your-email@example.com
CORS_ALLOWED_ORIGINS=https://your-domain.com
NEXT_PUBLIC_API_URL=https://your-domain.com/api
```

**Optional (for alerts)**:
```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
ALERT_EMAIL_TO=alerts@your-domain.com
SMTP_HOST=smtp.gmail.com:587
SMTP_USERNAME=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

### Firewall Rules

**Required**:
- Port 22 (SSH)
- Port 80 (HTTP)
- Port 443 (HTTPS)

**Optional** (only if needed):
- Port 26656 (Blockchain P2P)
- Port 26657 (Blockchain RPC)
- Port 9090 (Blockchain gRPC)
- Port 1317 (Blockchain REST)

### Log Rotation

All containers configured with:
- Max log size: 10MB
- Max log files: 3
- Total per container: 30MB

---

## 🚀 Deployment Steps

1. **Create Docker Secrets**:
   ```bash
   bash scripts/create_secrets.sh
   ```

2. **Configure Environment**:
   ```bash
   cd docker
   cp env.production.example .env.production
   nano .env.production  # Set DOMAIN, EMAIL, etc.
   ```

3. **Setup Firewall**:
   ```bash
   sudo bash scripts/setup_firewall.sh
   ```

4. **Deploy Stack**:
   ```bash
   bash scripts/deploy_production_docker.sh
   ```

5. **Test Backup/Restore**:
   ```bash
   bash scripts/test_backup_restore.sh
   ```

6. **Configure Alerts** (optional):
   - Set Slack webhook URL in `.env.production`
   - Set email SMTP settings in `.env.production`
   - Restart alertmanager: `docker restart r3mes-alertmanager-prod`

---

## 📊 Monitoring

### Access Monitoring

- **Grafana**: `http://your-domain.com:3001` (configure via Nginx if needed)
- **Prometheus**: Internal only (configure via Nginx if needed)
- **Alertmanager**: Internal only

### Key Metrics to Monitor

- Backend API response times
- Database connection pool usage
- Redis memory usage
- Disk space usage
- Container resource usage
- SSL certificate expiration

---

## 🔒 Security

### Completed

- ✅ Docker Secrets for passwords
- ✅ SSL/HTTPS with Let's Encrypt
- ✅ Security headers (HSTS, CSP, etc.)
- ✅ Rate limiting
- ✅ CORS configuration
- ✅ Firewall configuration
- ✅ Log rotation

### Recommendations

- Use SSH key authentication
- Disable SSH password authentication
- Install fail2ban for SSH protection
- Regularly update Docker images
- Monitor firewall logs
- Review security alerts

---

## 📝 Files Created/Updated

### New Files

- `scripts/setup_firewall.sh` - Firewall setup script
- `scripts/test_backup_restore.sh` - Backup restore test script
- `docker/FIREWALL_GUIDE.md` - Firewall documentation
- `docker/PRODUCTION_FINE_TUNING.md` - This file

### Updated Files

- `docker/nginx/nginx.conf` - Dynamic domain configuration
- `docker/nginx/Dockerfile` - Template processing
- `docker/docker-compose.prod.yml` - Log rotation, domain env var

---

## ✅ Production Status

**Status**: ✅ **Production Ready**

All critical fine-tuning has been completed:
- ✅ Domain configuration is dynamic
- ✅ Log rotation is configured
- ✅ Firewall guide and script provided
- ✅ Backup restore testing available
- ✅ Alert channels documented

**Next Steps**:
1. Deploy to production server
2. Configure alert channels
3. Test backup restore
4. Monitor and optimize

---

**Last Updated**: 2025-01-14  
**Maintained by**: R3MES Development Team

