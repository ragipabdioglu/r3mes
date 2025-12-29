# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.x.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously at R3MES. If you discover a security vulnerability, please report it responsibly.

### How to Report

**DO NOT** create a public GitHub issue for security vulnerabilities.

Instead, please report vulnerabilities via:

1. **Email**: security@r3mes.network
2. **PGP Key**: Available at [https://r3mes.network/.well-known/security.txt](https://r3mes.network/.well-known/security.txt)

### What to Include

Please include the following in your report:

- Type of vulnerability (e.g., XSS, SQL injection, authentication bypass)
- Full paths of affected source files
- Step-by-step instructions to reproduce
- Proof-of-concept or exploit code (if possible)
- Impact assessment
- Suggested fix (if any)

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Resolution Target**: Within 30 days (depending on severity)

### Severity Levels

| Level | Description | Response Time |
|-------|-------------|---------------|
| Critical | Remote code execution, private key exposure | 24 hours |
| High | Authentication bypass, data breach | 48 hours |
| Medium | Information disclosure, DoS | 7 days |
| Low | Minor issues, best practices | 30 days |

## Security Best Practices

### For Users

1. **Keep Software Updated**: Always run the latest version
2. **Secure Your Keys**: Never share private keys or mnemonics
3. **Verify Downloads**: Check checksums before installing
4. **Use Strong Passwords**: For any accounts or encrypted wallets
5. **Enable 2FA**: Where available

### For Validators

1. **Secure Server Access**: Use SSH keys, disable password auth
2. **Firewall Configuration**: Only expose necessary ports
3. **Regular Updates**: Keep OS and dependencies updated
4. **Monitor Logs**: Watch for suspicious activity
5. **Backup Keys**: Securely backup validator keys offline

### For Miners

1. **Verify Endpoints**: Only connect to official R3MES endpoints
2. **Secure Wallet**: Use hardware wallet for large holdings
3. **Monitor Resources**: Watch for unauthorized GPU usage
4. **Update Regularly**: Keep miner engine updated

## Known Security Measures

### Blockchain Security

- Ed25519 signatures for transactions
- BLS signatures for consensus
- Slashing for malicious validators
- Rate limiting on RPC endpoints

### Backend Security

- API key authentication with SHA-256 hashing
- Rate limiting (100 req/min default)
- Input validation and sanitization
- CORS protection
- SQL injection prevention (parameterized queries)

### Frontend Security

- Content Security Policy (CSP)
- XSS protection
- CSRF tokens
- Secure cookie handling

### Infrastructure Security

- Docker secrets for sensitive data
- TLS 1.3 for all connections
- Non-root container execution
- Network segmentation

## Bug Bounty Program

We offer rewards for responsibly disclosed vulnerabilities:

| Severity | Reward Range |
|----------|--------------|
| Critical | $5,000 - $25,000 |
| High | $1,000 - $5,000 |
| Medium | $250 - $1,000 |
| Low | $50 - $250 |

### Eligibility

- First reporter of the vulnerability
- Responsible disclosure (no public disclosure before fix)
- Not a current or recent employee
- Vulnerability must be in scope

### Scope

**In Scope:**
- R3MES blockchain (remes/)
- Backend API (backend/)
- Web Dashboard (web-dashboard/)
- Miner Engine (miner-engine/)
- Desktop Launcher (desktop-launcher-tauri/)
- Official SDKs (sdk/)

**Out of Scope:**
- Third-party dependencies (report to upstream)
- Social engineering attacks
- Physical attacks
- DoS attacks on production infrastructure

## Security Audits

We conduct regular security audits:

- **Internal Audits**: Quarterly
- **External Audits**: Annually
- **Penetration Testing**: Before major releases

Audit reports are available upon request for verified researchers.

## Contact

- **Security Team**: security@r3mes.network
- **General Inquiries**: info@r3mes.network
- **Discord**: [https://discord.gg/r3mes](https://discord.gg/r3mes)

Thank you for helping keep R3MES secure! 🔒
