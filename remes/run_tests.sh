#!/bin/bash
# Test runner script for R3MES Blockchain (Go)

set -e

echo "🧪 Running R3MES Blockchain Tests..."
echo ""

# Run all tests
if [ "$1" == "all" ]; then
    echo "🚀 Running all tests..."
    go test ./... -v -race -coverprofile=coverage.out
    go tool cover -html=coverage.out -o coverage.html
    echo "📊 Coverage report generated in coverage.html"
    
# Run trap verification tests
elif [ "$1" == "trap" ]; then
    echo "🔒 Running trap verification tests..."
    go test ./x/remes/keeper/training -v -run TestTrapVerification
    
# Run training keeper tests
elif [ "$1" == "training" ]; then
    echo "🎓 Running training keeper tests..."
    go test ./x/remes/keeper/training -v
    
# Run economics keeper tests
elif [ "$1" == "economics" ]; then
    echo "💰 Running economics keeper tests..."
    go test ./x/remes/keeper/economics -v
    
# Run model keeper tests
elif [ "$1" == "model" ]; then
    echo "🤖 Running model keeper tests..."
    go test ./x/remes/keeper/model -v
    
# Run benchmarks
elif [ "$1" == "bench" ]; then
    echo "⚡ Running benchmarks..."
    go test ./x/remes/keeper/training -bench=. -benchmem
    
# Default: run tests with coverage
else
    echo "🔬 Running tests with coverage..."
    go test ./x/remes/keeper/... -v -coverprofile=coverage.out
    go tool cover -func=coverage.out
    echo ""
    echo "📊 For HTML coverage report, run: go tool cover -html=coverage.out"
fi

echo ""
echo "✅ Tests complete!"
