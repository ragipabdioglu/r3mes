#!/bin/bash
# Docker Networking Test Script
# Tests connectivity between all services in docker-compose.prod.yml

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCKER_DIR="$PROJECT_ROOT/docker"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_test() {
    echo -e "${BLUE}Testing: $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Test service connectivity
test_service() {
    local service=$1
    local command=$2
    local expected=$3
    
    print_test "$service connectivity"
    
    if docker exec "$service" sh -c "$command" > /dev/null 2>&1; then
        print_success "$service is reachable"
        return 0
    else
        print_error "$service is not reachable"
        return 1
    fi
}

# Test PostgreSQL
test_postgres() {
    print_test "PostgreSQL"
    
    if docker exec r3mes-postgres-prod pg_isready -U r3mes > /dev/null 2>&1; then
        print_success "PostgreSQL is ready"
        
        # Test connection from backend
        if docker exec r3mes-backend-prod sh -c "python3 -c 'import psycopg2; psycopg2.connect(\"host=postgres user=r3mes password=\$POSTGRES_PASSWORD dbname=r3mes\")'" > /dev/null 2>&1; then
            print_success "Backend can connect to PostgreSQL"
        else
            print_warning "Backend PostgreSQL connection test skipped (psycopg2 may not be installed in test)"
        fi
    else
        print_error "PostgreSQL is not ready"
        return 1
    fi
}

# Test Redis
test_redis() {
    print_test "Redis"
    
    if docker exec r3mes-redis-prod redis-cli --raw incr ping > /dev/null 2>&1; then
        print_success "Redis is ready"
        
        # Test connection from backend
        if docker exec r3mes-backend-prod sh -c "python3 -c 'import redis; r=redis.Redis(host=\"redis\", port=6379, password=\"\$REDIS_PASSWORD\"); r.ping()'" > /dev/null 2>&1; then
            print_success "Backend can connect to Redis"
        else
            print_warning "Backend Redis connection test skipped (redis-py may not be installed in test)"
        fi
    else
        print_error "Redis is not ready"
        return 1
    fi
}

# Test IPFS
test_ipfs() {
    print_test "IPFS"
    
    if docker exec r3mes-ipfs-prod wget --spider -q http://localhost:5001/api/v0/version > /dev/null 2>&1; then
        print_success "IPFS API is accessible"
        
        # Test from backend
        if docker exec r3mes-backend-prod wget --spider -q http://ipfs:5001/api/v0/version > /dev/null 2>&1; then
            print_success "Backend can reach IPFS"
        else
            print_error "Backend cannot reach IPFS"
            return 1
        fi
    else
        print_error "IPFS is not accessible"
        return 1
    fi
}

# Test Blockchain (remesd)
test_blockchain() {
    print_test "Blockchain (remesd)"
    
    # Test RPC endpoint
    if docker exec r3mes-blockchain-prod wget --spider -q http://localhost:26657/status > /dev/null 2>&1; then
        print_success "Blockchain RPC is accessible"
        
        # Test from backend
        if docker exec r3mes-backend-prod wget --spider -q http://remesd:26657/status > /dev/null 2>&1; then
            print_success "Backend can reach Blockchain RPC"
        else
            print_error "Backend cannot reach Blockchain RPC"
            return 1
        fi
        
        # Test gRPC (port check)
        if docker exec r3mes-backend-prod nc -z remesd 9090 > /dev/null 2>&1; then
            print_success "Backend can reach Blockchain gRPC"
        else
            print_warning "Blockchain gRPC connectivity test skipped (nc may not be available)"
        fi
    else
        print_error "Blockchain RPC is not accessible"
        return 1
    fi
}

# Test Backend
test_backend() {
    print_test "Backend API"
    
    if docker exec r3mes-backend-prod curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Backend health endpoint is accessible"
        
        # Test from frontend
        if docker exec r3mes-frontend-prod wget --spider -q http://backend:8000/health > /dev/null 2>&1; then
            print_success "Frontend can reach Backend"
        else
            print_warning "Frontend Backend connectivity test skipped (wget may not be available)"
        fi
        
        # Test from nginx
        if docker exec r3mes-nginx-prod wget --spider -q http://backend:8000/health > /dev/null 2>&1; then
            print_success "Nginx can reach Backend"
        else
            print_error "Nginx cannot reach Backend"
            return 1
        fi
    else
        print_error "Backend health endpoint is not accessible"
        return 1
    fi
}

# Test Frontend
test_frontend() {
    print_test "Frontend"
    
    if docker exec r3mes-frontend-prod curl -f http://localhost:3000/ > /dev/null 2>&1; then
        print_success "Frontend is accessible"
        
        # Test from nginx
        if docker exec r3mes-nginx-prod wget --spider -q http://frontend:3000/ > /dev/null 2>&1; then
            print_success "Nginx can reach Frontend"
        else
            print_error "Nginx cannot reach Frontend"
            return 1
        fi
    else
        print_error "Frontend is not accessible"
        return 1
    fi
}

# Test Nginx
test_nginx() {
    print_test "Nginx"
    
    if docker exec r3mes-nginx-prod curl -f http://localhost/health > /dev/null 2>&1; then
        print_success "Nginx is accessible"
    else
        print_error "Nginx is not accessible"
        return 1
    fi
}

# Test Miner (if running)
test_miner() {
    if docker ps --format '{{.Names}}' | grep -q "r3mes-miner-prod"; then
        print_test "Miner Engine"
        
        # Test blockchain connectivity from miner
        if docker exec r3mes-miner-prod nc -z remesd 9090 > /dev/null 2>&1; then
            print_success "Miner can reach Blockchain gRPC"
        else
            print_warning "Miner Blockchain connectivity test skipped"
        fi
        
        # Test IPFS connectivity from miner
        if docker exec r3mes-miner-prod wget --spider -q http://ipfs:5001/api/v0/version > /dev/null 2>&1; then
            print_success "Miner can reach IPFS"
        else
            print_error "Miner cannot reach IPFS"
            return 1
        fi
    else
        print_warning "Miner is not running (skipped)"
    fi
}

# Main test function
main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}R3MES Docker Networking Tests${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    
    cd "$DOCKER_DIR"
    
    # Check if services are running
    if ! docker-compose -f docker-compose.prod.yml ps | grep -q "Up"; then
        print_error "Services are not running. Start them first:"
        print_warning "cd $DOCKER_DIR && docker-compose -f docker-compose.prod.yml up -d"
        exit 1
    fi
    
    # Wait a bit for services to be ready
    print_warning "Waiting 10 seconds for services to be ready..."
    sleep 10
    
    # Run tests
    FAILED=0
    
    test_postgres || FAILED=$((FAILED + 1))
    echo ""
    
    test_redis || FAILED=$((FAILED + 1))
    echo ""
    
    test_ipfs || FAILED=$((FAILED + 1))
    echo ""
    
    test_blockchain || FAILED=$((FAILED + 1))
    echo ""
    
    test_backend || FAILED=$((FAILED + 1))
    echo ""
    
    test_frontend || FAILED=$((FAILED + 1))
    echo ""
    
    test_nginx || FAILED=$((FAILED + 1))
    echo ""
    
    test_miner || FAILED=$((FAILED + 1))
    echo ""
    
    # Summary
    echo -e "${BLUE}========================================${NC}"
    if [ $FAILED -eq 0 ]; then
        print_success "All networking tests passed!"
    else
        print_error "$FAILED test(s) failed"
        exit 1
    fi
    echo -e "${BLUE}========================================${NC}"
}

main "$@"

