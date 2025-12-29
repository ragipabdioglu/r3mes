#!/bin/bash
set -e

echo "🚀 R3MES Production Mode Başlatılıyor..."

# Production environment variables
export R3MES_ENV=production
export R3MES_TEST_MODE=false
export CORS_ALLOWED_ORIGINS="http://localhost:3000,http://127.0.0.1:3000"
export CORS_ALLOW_ALL=false
export LOG_LEVEL=INFO
export ENABLE_FILE_LOGGING=true

# Backend'i production mode'da başlat
cd backend
source venv/bin/activate
python3 -m app.main > ../backend_prod.log 2>&1 &
BACKEND_PID=$!

# Frontend production build
cd ../web-dashboard
npm run build
NODE_ENV=production npm run start > ../frontend_prod.log 2>&1 &
FRONTEND_PID=$!

echo "✅ Production mode'da başlatıldı!"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
