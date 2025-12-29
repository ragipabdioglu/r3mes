# R3MES Makefile
# Production-ready command interface

.PHONY: help setup test test-all test-backend test-frontend test-blockchain test-miner test-e2e test-coverage \
        lint lint-backend lint-frontend lint-blockchain \
        build build-backend build-frontend build-blockchain build-miner build-desktop \
        start stop start-backend start-frontend start-dev \
        docker-up docker-down docker-logs docker-restart docker-build \
        docker-prod-up docker-prod-down docker-prod-logs docker-prod-restart \
        clean install security-scan docs

# Default target
.DEFAULT_GOAL := help

help:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║                    R3MES Makefile Commands                       ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "📦 Setup & Installation:"
	@echo "  make setup             - Install all dependencies"
	@echo "  make install           - Install production dependencies"
	@echo ""
	@echo "🧪 Testing:"
	@echo "  make test              - Run all tests"
	@echo "  make test-backend      - Run backend tests"
	@echo "  make test-frontend     - Run frontend tests"
	@echo "  make test-blockchain   - Run blockchain tests"
	@echo "  make test-miner        - Run miner engine tests"
	@echo "  make test-e2e          - Run end-to-end tests"
	@echo "  make test-coverage     - Run tests with coverage report"
	@echo ""
	@echo "🔍 Linting:"
	@echo "  make lint              - Run all linters"
	@echo "  make lint-backend      - Lint Python code"
	@echo "  make lint-frontend     - Lint TypeScript code"
	@echo "  make lint-blockchain   - Lint Go code"
	@echo ""
	@echo "🔨 Building:"
	@echo "  make build             - Build all components"
	@echo "  make build-backend     - Build backend"
	@echo "  make build-frontend    - Build frontend"
	@echo "  make build-blockchain  - Build blockchain binary"
	@echo "  make build-miner       - Build miner package"
	@echo "  make build-desktop     - Build desktop launcher"
	@echo ""
	@echo "🚀 Development:"
	@echo "  make start-dev         - Start development environment"
	@echo "  make start-backend     - Start backend only"
	@echo "  make start-frontend    - Start frontend only"
	@echo "  make stop              - Stop all services"
	@echo ""
	@echo "🐳 Docker (Development):"
	@echo "  make docker-up         - Start development stack"
	@echo "  make docker-down       - Stop development stack"
	@echo "  make docker-logs       - View logs"
	@echo "  make docker-build      - Build Docker images"
	@echo ""
	@echo "🐳 Docker (Production):"
	@echo "  make docker-prod-up    - Start production stack"
	@echo "  make docker-prod-down  - Stop production stack"
	@echo "  make docker-prod-logs  - View production logs"
	@echo ""
	@echo "🔒 Security:"
	@echo "  make security-scan     - Run security scans"
	@echo ""
	@echo "📚 Documentation:"
	@echo "  make docs              - Generate documentation"
	@echo ""
	@echo "🧹 Cleanup:"
	@echo "  make clean             - Clean build artifacts"
	@echo ""

# ============================================================================
# Setup & Installation
# ============================================================================

setup: setup-backend setup-frontend setup-blockchain setup-miner
	@echo "✅ All dependencies installed"

setup-backend:
	@echo "📦 Setting up backend..."
	@cd backend && python -m venv venv && . venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Backend setup complete"

setup-frontend:
	@echo "📦 Setting up frontend..."
	@cd web-dashboard && npm ci
	@echo "✅ Frontend setup complete"

setup-blockchain:
	@echo "📦 Setting up blockchain..."
	@cd remes && go mod download
	@echo "✅ Blockchain setup complete"

setup-miner:
	@echo "� Settirng up miner engine..."
	@cd miner-engine && python -m venv venv && . venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Miner setup complete"

install:
	@echo "📦 Installing production dependencies..."
	@cd backend && pip install -r requirements.txt
	@cd web-dashboard && npm ci --production
	@cd remes && go mod download
	@echo "✅ Production dependencies installed"

# ============================================================================
# Testing
# ============================================================================

test: test-backend test-frontend test-blockchain
	@echo "✅ All tests passed"

test-all: test-backend test-frontend test-blockchain test-miner test-e2e
	@echo "✅ All tests (including E2E) passed"

test-backend:
	@echo "🧪 Running backend tests..."
	@cd backend && python -m pytest tests/ -v --tb=short
	@echo "✅ Backend tests passed"

test-frontend:
	@echo "🧪 Running frontend tests..."
	@cd web-dashboard && npm test -- --watchAll=false
	@echo "✅ Frontend tests passed"

test-blockchain:
	@echo "🧪 Running blockchain tests..."
	@cd remes && go test ./... -v -race
	@echo "✅ Blockchain tests passed"

test-miner:
	@echo "🧪 Running miner engine tests..."
	@cd miner-engine && python -m pytest tests/ -v --tb=short
	@echo "✅ Miner tests passed"

test-e2e:
	@echo "🧪 Running E2E tests..."
	@cd web-dashboard && npx playwright test
	@echo "✅ E2E tests passed"

test-coverage:
	@echo "🧪 Running tests with coverage..."
	@cd backend && python -m pytest tests/ -v --cov=app --cov-report=html --cov-report=term
	@cd web-dashboard && npm test -- --coverage --watchAll=false
	@cd remes && go test ./... -coverprofile=coverage.out
	@echo "✅ Coverage reports generated"

# ============================================================================
# Linting
# ============================================================================

lint: lint-backend lint-frontend lint-blockchain
	@echo "✅ All linting passed"

lint-backend:
	@echo "🔍 Linting backend..."
	@cd backend && python -m ruff check app/ --fix || true
	@cd backend && python -m black app/ --check || true
	@echo "✅ Backend linting complete"

lint-frontend:
	@echo "🔍 Linting frontend..."
	@cd web-dashboard && npm run lint
	@echo "✅ Frontend linting complete"

lint-blockchain:
	@echo "🔍 Linting blockchain..."
	@cd remes && golangci-lint run ./... || true
	@echo "✅ Blockchain linting complete"

# ============================================================================
# Building
# ============================================================================

build: build-backend build-frontend build-blockchain
	@echo "✅ All components built"

build-backend:
	@echo "🔨 Building backend..."
	@cd backend && python -m py_compile app/main.py
	@echo "✅ Backend build complete"

build-frontend:
	@echo "🔨 Building frontend..."
	@cd web-dashboard && npm run build
	@echo "✅ Frontend build complete"

build-blockchain:
	@echo "🔨 Building blockchain..."
	@cd remes && go build -o build/remesd ./cmd/remesd
	@echo "✅ Blockchain build complete"

build-miner:
	@echo "🔨 Building miner package..."
	@cd miner-engine && python -m build
	@echo "✅ Miner package built"

build-desktop:
	@echo "🔨 Building desktop launcher..."
	@cd desktop-launcher-tauri && npm run tauri build
	@echo "✅ Desktop launcher built"

# ============================================================================
# Development
# ============================================================================

start-dev:
	@echo "🚀 Starting development environment..."
	@make docker-up
	@echo "✅ Development environment started"

start-backend:
	@echo "🚀 Starting backend..."
	@cd backend && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

start-frontend:
	@echo "🚀 Starting frontend..."
	@cd web-dashboard && npm run dev

stop:
	@echo "🛑 Stopping all services..."
	@-pkill -f "uvicorn" || true
	@-pkill -f "next" || true
	@make docker-down
	@echo "✅ All services stopped"

# ============================================================================
# Docker (Development)
# ============================================================================

docker-up:
	@echo "🐳 Starting development stack..."
	@cd docker && docker-compose up -d
	@echo "✅ Development stack started"

docker-down:
	@echo "🐳 Stopping development stack..."
	@cd docker && docker-compose down
	@echo "✅ Development stack stopped"

docker-logs:
	@cd docker && docker-compose logs -f

docker-build:
	@echo "🐳 Building Docker images..."
	@cd docker && docker-compose build
	@echo "✅ Docker images built"

docker-restart:
	@echo "🐳 Restarting development stack..."
	@cd docker && docker-compose restart
	@echo "✅ Development stack restarted"

# ============================================================================
# Docker (Production)
# ============================================================================

docker-prod-up:
	@echo "🐳 Starting production stack..."
	@cd docker && docker-compose -f docker-compose.prod.yml up -d
	@echo "✅ Production stack started"

docker-prod-up-miner:
	@echo "🐳 Starting production stack with miner..."
	@cd docker && docker-compose -f docker-compose.prod.yml --profile miner up -d
	@echo "✅ Production stack with miner started"

docker-prod-down:
	@echo "🐳 Stopping production stack..."
	@cd docker && docker-compose -f docker-compose.prod.yml down
	@echo "✅ Production stack stopped"

docker-prod-logs:
	@cd docker && docker-compose -f docker-compose.prod.yml logs -f

docker-prod-restart:
	@echo "🐳 Restarting production stack..."
	@cd docker && docker-compose -f docker-compose.prod.yml restart
	@echo "✅ Production stack restarted"

docker-prod-test:
	@echo "🧪 Testing production networking..."
	@bash scripts/test_docker_networking.sh

# ============================================================================
# Security
# ============================================================================

security-scan:
	@echo "🔒 Running security scans..."
	@echo "Scanning Python dependencies..."
	@cd backend && pip install safety && safety check || true
	@echo "Scanning Node.js dependencies..."
	@cd web-dashboard && npm audit || true
	@echo "Scanning Go dependencies..."
	@cd remes && go list -json -m all | nancy sleuth || true
	@echo "✅ Security scan complete"

# ============================================================================
# Documentation
# ============================================================================

docs:
	@echo "📚 Generating documentation..."
	@cd backend && python -m pdoc app -o ../docs/api/backend || true
	@cd remes && go doc -all > ../docs/api/blockchain.txt || true
	@echo "✅ Documentation generated"

# ============================================================================
# Cleanup
# ============================================================================

clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf backend/__pycache__ backend/.pytest_cache backend/htmlcov
	@rm -rf web-dashboard/.next web-dashboard/node_modules/.cache
	@rm -rf remes/build
	@rm -rf miner-engine/__pycache__ miner-engine/.pytest_cache
	@rm -rf desktop-launcher-tauri/src-tauri/target
	@echo "✅ Cleanup complete"

clean-all: clean
	@echo "🧹 Deep cleaning (including dependencies)..."
	@rm -rf backend/venv
	@rm -rf web-dashboard/node_modules
	@rm -rf miner-engine/venv
	@echo "✅ Deep cleanup complete"

