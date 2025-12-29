# Dependency Management Guide

This guide explains dependency management and vulnerability scanning for R3MES.

## Overview

R3MES uses automated dependency management to:
- Keep dependencies up to date
- Identify security vulnerabilities
- Automate dependency updates
- Track dependency licenses

## Automated Dependency Updates

### Dependabot

Dependabot is configured via `.github/dependabot.yml` to:
- Check for updates weekly
- Create pull requests for updates
- Support multiple ecosystems (pip, npm, gomod, docker, github-actions)

### Update Frequency

- **Weekly**: Automatic checks every Monday
- **Security**: Immediate alerts for critical vulnerabilities
- **Limit**: Maximum 10 open PRs per ecosystem

## Manual Dependency Updates

### Python Dependencies

```bash
# Check for outdated packages
pip list --outdated

# Update requirements.txt
pip-compile --upgrade requirements.txt

# Update specific package
pip install --upgrade package-name
```

### Node.js Dependencies

```bash
# Check for outdated packages
npm outdated

# Update package.json
npm update

# Update specific package
npm install package-name@latest
```

### Go Dependencies

```bash
# Update all dependencies
go get -u ./...

# Update specific dependency
go get -u github.com/package/name

# Tidy go.mod
go mod tidy
```

## Security Scanning

### Automated Scanning

Security scans run automatically via GitHub Actions:
- **Python**: Bandit, Safety
- **Node.js**: npm audit
- **Docker**: Trivy
- **Dependencies**: Dependabot

### Manual Scanning

```bash
# Run security scan
./scripts/security_scan.sh

# Python only
./scripts/security_scan.sh python

# Node.js only
./scripts/security_scan.sh nodejs

# Docker only
./scripts/security_scan.sh docker
```

## Vulnerability Response

### Critical Vulnerabilities

1. **Immediate Action**:
   - Review vulnerability details
   - Check if exploit exists
   - Assess impact

2. **Remediation**:
   - Update to patched version
   - Test thoroughly
   - Deploy fix

3. **Documentation**:
   - Document vulnerability
   - Record remediation steps
   - Update security log

### High/Medium Vulnerabilities

1. **Assessment**:
   - Review vulnerability details
   - Assess impact on application
   - Plan remediation

2. **Scheduling**:
   - Schedule update in next release
   - Test in staging
   - Deploy with regular release

## Dependency Policies

### Version Pinning

- **Production**: Pin exact versions
- **Development**: Allow minor updates
- **Security**: Always update for security patches

### License Compliance

- Review all dependency licenses
- Ensure compatibility with project license
- Document license requirements

## Best Practices

1. **Regular Updates**: Update dependencies monthly
2. **Security First**: Prioritize security updates
3. **Testing**: Test all dependency updates
4. **Documentation**: Document major updates
5. **Monitoring**: Monitor for new vulnerabilities

## Tools

- **Dependabot**: Automated dependency updates
- **npm audit**: Node.js vulnerability scanning
- **Safety**: Python vulnerability scanning
- **Trivy**: Container vulnerability scanning
- **Snyk**: Additional security scanning (optional)

