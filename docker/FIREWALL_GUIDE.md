# R3MES Production Firewall Guide

## Overview

This guide explains how to configure firewall rules for R3MES production deployment. Proper firewall configuration is critical for security.

## Recommended Firewall Rules

### Required Ports (Must Open)

- **22 (SSH)** - Server administration
- **80 (HTTP)** - Let's Encrypt certificate validation
- **443 (HTTPS)** - Web traffic

### Optional Ports (Only if needed)

- **26656 (Blockchain P2P)** - Only if you need external peers
- **26657 (Blockchain RPC)** - Only if you need external RPC access
- **9090 (Blockchain gRPC)** - Only if you need external gRPC access
- **1317 (Blockchain REST)** - Only if you need external REST access

**⚠️ WARNING**: Opening blockchain ports exposes your node to the internet. Only do this if necessary.

## UFW (Uncomplicated Firewall) Setup

### Automated Setup

Use the provided script:

```bash
sudo bash scripts/setup_firewall.sh
```

The script will:
1. Install UFW if not present
2. Set default policies (deny incoming, allow outgoing)
3. Allow SSH (port 22)
4. Allow HTTP (port 80) and HTTPS (port 443)
5. Optionally allow blockchain ports (if you choose)
6. Enable the firewall

### Manual Setup

```bash
# Install UFW (if not installed)
sudo apt-get update
sudo apt-get install -y ufw

# Set default policies
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH (IMPORTANT: Do this first!)
sudo ufw allow 22/tcp comment 'SSH'

# Allow HTTP and HTTPS
sudo ufw allow 80/tcp comment 'HTTP - Let'\''s Encrypt'
sudo ufw allow 443/tcp comment 'HTTPS'

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status verbose
```

### Allow Blockchain Ports (Optional)

**Only if you need external access:**

```bash
sudo ufw allow 26656/tcp comment 'Blockchain P2P'
sudo ufw allow 26657/tcp comment 'Blockchain RPC'
sudo ufw allow 9090/tcp comment 'Blockchain gRPC'
sudo ufw allow 1317/tcp comment 'Blockchain REST'
```

## Internal Services (No Firewall Rules Needed)

These services are only accessible within the Docker network:

- **PostgreSQL (5432)** - Internal only
- **Redis (6379)** - Internal only
- **IPFS (5001, 4001)** - Internal only
- **Backend (8000)** - Internal only (accessed via Nginx)
- **Frontend (3000)** - Internal only (accessed via Nginx)
- **Prometheus (9090)** - Internal only
- **Grafana (3001)** - Internal only (can be exposed via Nginx if needed)
- **Alertmanager (9093)** - Internal only

## Firewall Best Practices

### 1. SSH Security

- Use SSH key authentication (disable password auth)
- Change default SSH port (optional, but recommended)
- Use fail2ban to prevent brute force attacks

```bash
# Install fail2ban
sudo apt-get install -y fail2ban

# Configure fail2ban for SSH
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 2. Rate Limiting

UFW doesn't support rate limiting directly, but you can use:

- **Nginx rate limiting** (already configured in `docker/nginx/nginx.conf`)
- **fail2ban** for SSH protection

### 3. Monitoring

Monitor firewall logs:

```bash
# View UFW logs
sudo tail -f /var/log/ufw.log

# View blocked connections
sudo grep "UFW BLOCK" /var/log/ufw.log
```

### 4. Testing

Test firewall rules:

```bash
# Test SSH (should work)
ssh user@your-server

# Test HTTP (should work)
curl http://your-domain.com

# Test HTTPS (should work)
curl https://your-domain.com

# Test blocked port (should fail)
telnet your-server 5432
```

## Cloud Provider Firewalls

### Contabo VPS

Contabo VPS uses a web-based firewall. Configure it in the Contabo control panel:

1. Log in to Contabo control panel
2. Go to "Firewall" section
3. Add rules:
   - Allow TCP port 22 (SSH)
   - Allow TCP port 80 (HTTP)
   - Allow TCP port 443 (HTTPS)
4. Save and apply

### AWS EC2

Configure Security Groups:

1. Go to EC2 → Security Groups
2. Edit inbound rules:
   - Allow SSH (22) from your IP
   - Allow HTTP (80) from anywhere (0.0.0.0/0)
   - Allow HTTPS (443) from anywhere (0.0.0.0/0)
3. Save rules

### DigitalOcean

Configure Firewall in control panel:

1. Go to Networking → Firewalls
2. Create new firewall
3. Add inbound rules:
   - SSH (22)
   - HTTP (80)
   - HTTPS (443)
4. Apply to your droplet

## Troubleshooting

### Can't SSH After Enabling Firewall

**Problem**: Firewall blocked SSH access

**Solution**:
```bash
# If you have physical access or console access:
sudo ufw disable
sudo ufw allow 22/tcp
sudo ufw enable
```

### Let's Encrypt Certificate Fails

**Problem**: Port 80 is blocked, Let's Encrypt can't validate

**Solution**:
```bash
sudo ufw allow 80/tcp
```

### Services Not Accessible

**Problem**: Services are not accessible from outside

**Solution**:
1. Check if ports are open: `sudo ufw status`
2. Check if services are running: `docker ps`
3. Check if Nginx is proxying correctly: `docker logs r3mes-nginx-prod`

## Security Checklist

- [ ] UFW installed and enabled
- [ ] Only required ports are open (22, 80, 443)
- [ ] SSH key authentication enabled
- [ ] Password authentication disabled for SSH
- [ ] fail2ban installed and configured
- [ ] Firewall logs are monitored
- [ ] Blockchain ports are closed (unless needed)
- [ ] Internal services are not exposed externally

## Additional Resources

- [UFW Documentation](https://help.ubuntu.com/community/UFW)
- [fail2ban Documentation](https://www.fail2ban.org/wiki/index.php/Main_Page)
- [Docker Security Best Practices](https://docs.docker.com/engine/security/)

---

**Last Updated**: 2025-01-14  
**Maintained by**: R3MES Development Team

