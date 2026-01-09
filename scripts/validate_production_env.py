#!/usr/bin/env python3
"""
Production Environment Validation Script

Validates all required environment variables and secret management configuration
before production deployment.

Usage:
    python scripts/validate_production_env.py

Exit codes:
    0: Validation passed
    1: Validation failed
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

from app.env_validator import validate_environment, get_env_validator
from app.secrets import get_secret_manager


def main():
    """Main validation function."""
    print("🔍 Validating production environment configuration...")
    print()
    
    # Set production mode
    os.environ["R3MES_ENV"] = "production"
    
    # Validate environment variables
    try:
        validate_environment()
        print("✅ Environment variables validated successfully")
    except Exception as e:
        print(f"❌ Environment variable validation failed: {e}")
        return 1
    
    # Validate secret management service
    print()
    print("🔐 Validating secret management service...")
    try:
        secret_manager = get_secret_manager()
        if secret_manager.test_connection():
            print(f"✅ Secret management service connection successful ({type(secret_manager).__name__})")
        else:
            print("❌ Secret management service connection failed")
            return 1
    except Exception as e:
        print(f"❌ Secret management service validation failed: {e}")
        return 1
    
    # Get detailed validation report
    print()
    print("📊 Detailed validation report:")
    validator = get_env_validator()
    report = validator.get_validation_report()
    
    print(f"  Environment: {report['environment']}")
    print(f"  Is Production: {report['is_production']}")
    print(f"  Valid: {report['valid']}")
    print(f"  Errors: {report['error_count']}")
    print(f"  Warnings: {report['warning_count']}")
    
    if report['errors']:
        print()
        print("❌ Errors:")
        for error in report['errors']:
            print(f"  - {error}")
    
    if report['warnings']:
        print()
        print("⚠️  Warnings:")
        for warning in report['warnings']:
            print(f"  - {warning}")
    
    if not report['valid']:
        return 1
    
    print()
    print("✅ All production environment validations passed!")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

