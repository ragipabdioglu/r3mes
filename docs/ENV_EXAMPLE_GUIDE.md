# Environment Variables Example Files Guide

This document explains the `.env.example` files across all R3MES components and how to use them.

## Overview

Each component has its own `.env.example` file that serves as a template for environment configuration:

- **Backend**: `backend/.env.example` (development) and `backend/.env.production.example` (production)
- **Web Dashboard**: `web-dashboard/.env.example` (development) and `web-dashboard/env.production.example` (production)
- **Blockchain Node**: `remes/.env.example` (development) and `remes/.env.production.example` (production)
- **Miner Engine**: `miner-engine/.env.example` (development) and `miner-engine/env.production.example` (production)
- **Docker Compose**: `docker/.env.example` (for Docker deployments)

## Usage

### Development Setup

1. **Copy the example file**:
   ```bash
   # Backend
   cp backend/.env.example backend/.env
   
   # Web Dashboard
   cp web-dashboard/.env.example web-dashboard/.env.local
   
   # Blockchain Node
   cp remes/.env.example remes/.env
   
   # Miner Engine
   cp miner-engine/.env.example miner-engine/.env
   ```

2. **Fill in actual values**:
   - Replace placeholder values with actual configuration
   - For development, localhost URLs are acceptable
   - Update paths to match your local setup

3. **Never commit `.env` files**:
   - `.env` files are in `.gitignore`
   - Only `.env.example` files are committed to Git

### Production Setup

1. **Copy the production example file**:
   ```bash
   # Backend
   cp backend/.env.production.example backend/.env
   
   # Web Dashboard
   cp web-dashboard/env.production.example web-dashboard/.env.production
   
   # Blockchain Node
   cp remes/.env.production.example remes/.env
   
   # Miner Engine
   cp miner-engine/env.production.example miner-engine/.env.production
   ```

2. **Fill in production values**:
   - Use actual hostnames (no localhost)
   - Use HTTPS/WSS for all external URLs
   - Store sensitive values in a secret management service
   - Validate all environment variables using the validation system

3. **Use secret management**:
   - For production, use AWS Secrets Manager, HashiCorp Vault, or similar
   - Never store secrets in `.env` files
   - Inject secrets at runtime via environment variables

## Component-Specific Notes

### Backend

- **Development**: Uses SQLite by default, localhost URLs acceptable
- **Production**: Requires PostgreSQL, no localhost URLs, all required variables must be set
- **Validation**: Environment variables are validated on startup

### Web Dashboard

- **Development**: Uses localhost fallbacks for all services
- **Production**: All `NEXT_PUBLIC_*` variables must be set, no localhost
- **Build-time**: `NEXT_PUBLIC_*` variables are embedded at build time

### Blockchain Node

- **Development**: Uses localhost for all services
- **Production**: Requires actual hostnames, CORS must be configured
- **Validation**: Environment variables are validated on startup (Go validator)

### Miner Engine

- **Development**: Uses localhost for blockchain and backend connections
- **Production**: Requires actual hostnames, TLS should be enabled
- **Validation**: Environment variables are validated during initialization

## Environment Variable Categories

### Required in Production

These variables must be set in production:

- **Database**: `DATABASE_URL` (PostgreSQL)
- **Blockchain**: `BLOCKCHAIN_RPC_URL`, `BLOCKCHAIN_GRPC_URL`, `BLOCKCHAIN_REST_URL`
- **API**: `CORS_ALLOWED_ORIGINS`, `API_KEY_SECRET`
- **Network**: All service URLs (no localhost)

### Optional

These variables have defaults and are optional:

- **Logging**: `LOG_LEVEL` (default: INFO)
- **Ports**: `BACKEND_PORT` (default: 8000)
- **Feature Flags**: `FAUCET_ENABLED` (default: false)

### Sensitive

These variables contain sensitive data and should be stored in secret management:

- **API Keys**: `API_KEY_SECRET`, `JWT_SECRET`
- **Database Credentials**: `DATABASE_URL` (contains password)
- **Private Keys**: `PRIVATE_KEY` (miner engine)
- **TLS Certificates**: Certificate paths and keys

## Validation

All environment variables are validated on startup:

- **Backend**: Uses `backend/app/env_validator.py`
- **Blockchain Node**: Uses `remes/x/remes/keeper/env_validator.go`
- **Miner Engine**: Validates during initialization

Validation checks:
- Required variables are set (in production)
- URLs are valid and don't use localhost (in production)
- Ports are in valid range (1-65535)
- Boolean values are valid
- Secrets meet minimum length requirements

## Best Practices

1. **Use `.env.example` as template**: Always start from the example file
2. **Document custom variables**: If adding new variables, update the example file
3. **Validate before deployment**: Run validation checks before deploying
4. **Use secret management**: Store sensitive values in secret management services
5. **Environment-specific files**: Use different files for development, staging, and production
6. **Never commit secrets**: Ensure `.env` files are in `.gitignore`
7. **Rotate secrets regularly**: Change secrets periodically for security

## Troubleshooting

### Validation Errors

If you see validation errors on startup:

1. Check that all required variables are set
2. Verify URLs don't use localhost in production
3. Ensure ports are in valid range
4. Check boolean values are valid (true/false, 1/0, yes/no)

### Missing Variables

If a variable is missing:

1. Check the `.env.example` file for the variable name
2. Add it to your `.env` file with an appropriate value
3. For production, ensure it's set in your deployment configuration

### Localhost Errors in Production

If you see localhost errors in production:

1. Replace all localhost URLs with actual hostnames
2. Use HTTPS/WSS for external URLs
3. Verify DNS resolution for all hostnames

---

**Son Güncelleme**: 2025-12-24

