#!/bin/bash
# PostgreSQL Setup Script for R3MES

set -e

echo "🚀 Setting up PostgreSQL for R3MES..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running"
    exit 1
fi

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
# PostgreSQL Configuration
POSTGRES_USER=r3mes
POSTGRES_PASSWORD=$(openssl rand -base64 32)
POSTGRES_DB=r3mes
POSTGRES_PORT=5432

# Redis Configuration
REDIS_PASSWORD=$(openssl rand -base64 32)
REDIS_PORT=6379

# Backend Configuration
DATABASE_TYPE=postgresql
DATABASE_URL=postgresql://r3mes:\${POSTGRES_PASSWORD}@postgres:5432/r3mes
REDIS_URL=redis://:\${REDIS_PASSWORD}@redis:6379/0

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
NEXT_PUBLIC_SITE_URL=http://localhost:3000

# Environment
R3MES_ENV=development
EOF
    echo "✅ .env file created"
else
    echo "✅ .env file already exists"
fi

# Start PostgreSQL and Redis
echo "🐳 Starting PostgreSQL and Redis containers..."
cd docker
docker-compose up -d postgres redis

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
timeout=60
counter=0
until docker-compose exec -T postgres pg_isready -U r3mes > /dev/null 2>&1; do
    sleep 2
    counter=$((counter + 2))
    if [ $counter -ge $timeout ]; then
        echo "❌ Error: PostgreSQL did not become ready in time"
        exit 1
    fi
done

echo "✅ PostgreSQL is ready!"

# Run database initialization
echo "📊 Initializing database tables..."
docker-compose exec -T postgres psql -U r3mes -d r3mes << EOF
-- Database is initialized by AsyncPostgreSQL._init_database()
-- This script can be used for manual setup if needed
SELECT 'Database ready!' as status;
EOF

echo "✅ PostgreSQL setup completed!"
echo ""
echo "📋 Connection details:"
echo "   Host: localhost"
echo "   Port: 5432"
echo "   Database: r3mes"
echo "   User: r3mes"
echo "   Password: (check .env file)"
echo ""
echo "🔗 Connection string:"
echo "   postgresql://r3mes:\${POSTGRES_PASSWORD}@localhost:5432/r3mes"
echo ""
echo "📝 To migrate from SQLite:"
echo "   python scripts/migrate_sqlite_to_postgres.py"

