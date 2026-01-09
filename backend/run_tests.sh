#!/bin/bash
# Test runner script for R3MES Backend

set -e

echo "🧪 Running R3MES Backend Tests..."
echo ""

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d "../venv" ]; then
    source ../venv/bin/activate
fi

# Install test dependencies
echo "📦 Installing test dependencies..."
pip install -q -r requirements-test.txt

# Run tests with coverage
echo ""
echo "🔬 Running unit tests..."
pytest tests/ -v --cov=app --cov-report=term-missing --cov-report=html

# Generate coverage report
echo ""
echo "📊 Coverage report generated in htmlcov/index.html"

# Run specific test suites
if [ "$1" == "adapter" ]; then
    echo ""
    echo "🔧 Running adapter sync tests..."
    pytest tests/test_adapter_sync_service.py -v
elif [ "$1" == "event" ]; then
    echo ""
    echo "📡 Running event listener tests..."
    pytest tests/test_blockchain_event_listener.py -v
elif [ "$1" == "all" ]; then
    echo ""
    echo "🚀 Running all tests..."
    pytest tests/ -v
fi

echo ""
echo "✅ Tests complete!"
