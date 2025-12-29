#!/bin/bash
# Database Initialization Script
# Runs Alembic migrations before starting the application

set -e

echo "=========================================="
echo "R3MES Database Initialization"
echo "=========================================="

# Wait for PostgreSQL to be ready
echo "Waiting for PostgreSQL to be ready..."
until pg_isready -h postgres -U ${POSTGRES_USER:-r3mes} -d ${POSTGRES_DB:-r3mes} > /dev/null 2>&1; do
  echo "PostgreSQL is unavailable - sleeping"
  sleep 2
done

echo "PostgreSQL is ready!"

# Check if we're using PostgreSQL
if [ "${DATABASE_TYPE:-postgresql}" = "postgresql" ]; then
    echo "Running Alembic migrations..."
    
    # Read PostgreSQL password from Docker secrets if available
    # Docker secrets are mounted at /run/secrets/postgres_password
    if [ -z "$POSTGRES_PASSWORD" ]; then
        # Try POSTGRES_PASSWORD_FILE first (environment variable)
        if [ -n "$POSTGRES_PASSWORD_FILE" ] && [ -f "$POSTGRES_PASSWORD_FILE" ]; then
            export POSTGRES_PASSWORD=$(cat "$POSTGRES_PASSWORD_FILE")
            echo "✅ Read PostgreSQL password from POSTGRES_PASSWORD_FILE: $POSTGRES_PASSWORD_FILE"
        # Try Docker secrets mount path
        elif [ -f "/run/secrets/postgres_password" ]; then
            export POSTGRES_PASSWORD=$(cat /run/secrets/postgres_password)
            echo "✅ Read PostgreSQL password from Docker secrets: /run/secrets/postgres_password"
        else
            echo "⚠️  WARNING: POSTGRES_PASSWORD not set and no secrets file found"
            echo "⚠️  Tried: POSTGRES_PASSWORD_FILE=$POSTGRES_PASSWORD_FILE, /run/secrets/postgres_password"
            echo "⚠️  Migration may fail if password is required"
        fi
    fi
    
    # Set DATABASE_URL if not already set
    if [ -z "$DATABASE_URL" ]; then
        if [ -n "$POSTGRES_PASSWORD" ]; then
            export DATABASE_URL="postgresql://${POSTGRES_USER:-r3mes}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB:-r3mes}"
        else
            echo "❌ ERROR: Cannot construct DATABASE_URL without POSTGRES_PASSWORD"
            exit 1
        fi
    fi
    
    # Run migrations
    cd /app
    alembic upgrade head
    
    if [ $? -eq 0 ]; then
        echo "✅ Database migrations completed successfully"
    else
        echo "❌ Database migrations failed"
        exit 1
    fi
else
    echo "⚠️  Using ${DATABASE_TYPE}, skipping migrations"
fi

echo "=========================================="
echo "Starting R3MES Backend Application"
echo "=========================================="

# Start the application
exec python -m app.main

