# R3MES CLI Makefile

# Build variables
VERSION ?= v0.1.0
BUILD_TIME := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_COMMIT := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
LDFLAGS := -X main.Version=$(VERSION) -X main.BuildTime=$(BUILD_TIME) -X main.GitCommit=$(GIT_COMMIT)

# Directories
BUILD_DIR := build
CLI_DIR := r3mes-cli

# Default target
.PHONY: all
all: clean deps build

# Clean build artifacts
.PHONY: clean
clean:
	@echo "🧹 Cleaning build artifacts..."
	@rm -rf $(BUILD_DIR)
	@cd $(CLI_DIR) && go clean

# Download dependencies
.PHONY: deps
deps:
	@echo "📦 Downloading dependencies..."
	@cd $(CLI_DIR) && go mod download && go mod tidy

# Build for current platform
.PHONY: build
build: deps
	@echo "🔨 Building R3MES CLI for current platform..."
	@mkdir -p $(BUILD_DIR)
	@cd $(CLI_DIR) && go build -ldflags="$(LDFLAGS)" -o ../$(BUILD_DIR)/r3mes-cli .
	@echo "✅ Build completed: $(BUILD_DIR)/r3mes-cli"

# Build for all platforms
.PHONY: build-all
build-all: deps
	@echo "🔨 Building R3MES CLI for all platforms..."
	@mkdir -p $(BUILD_DIR)
	@./build.sh

# Install locally
.PHONY: install
install: build
	@echo "📦 Installing R3MES CLI..."
	@sudo cp $(BUILD_DIR)/r3mes-cli /usr/local/bin/
	@echo "✅ Installed to /usr/local/bin/r3mes-cli"

# Run tests
.PHONY: test
test:
	@echo "🧪 Running tests..."
	@cd $(CLI_DIR) && go test -v ./...

# Run linter
.PHONY: lint
lint:
	@echo "🔍 Running linter..."
	@cd $(CLI_DIR) && golangci-lint run

# Format code
.PHONY: fmt
fmt:
	@echo "🎨 Formatting code..."
	@cd $(CLI_DIR) && go fmt ./...

# Check for security issues
.PHONY: security
security:
	@echo "🔒 Running security checks..."
	@cd $(CLI_DIR) && gosec ./...

# Development build (with debug info)
.PHONY: dev
dev: deps
	@echo "🔨 Building development version..."
	@mkdir -p $(BUILD_DIR)
	@cd $(CLI_DIR) && go build -gcflags="all=-N -l" -ldflags="$(LDFLAGS)" -o ../$(BUILD_DIR)/r3mes-cli-dev .
	@echo "✅ Development build completed: $(BUILD_DIR)/r3mes-cli-dev"

# Create release package
.PHONY: release
release: build-all
	@echo "📦 Creating release package..."
	@cd $(BUILD_DIR) && tar -czf r3mes-cli-$(VERSION).tar.gz r3mes-cli-* checksums.txt
	@echo "✅ Release package created: $(BUILD_DIR)/r3mes-cli-$(VERSION).tar.gz"

# Show help
.PHONY: help
help:
	@echo "R3MES CLI Build System"
	@echo ""
	@echo "Available targets:"
	@echo "  all        - Clean, download deps, and build for current platform"
	@echo "  build      - Build for current platform only"
	@echo "  build-all  - Build for all supported platforms"
	@echo "  clean      - Remove build artifacts"
	@echo "  deps       - Download Go dependencies"
	@echo "  dev        - Build development version with debug info"
	@echo "  fmt        - Format Go code"
	@echo "  help       - Show this help message"
	@echo "  install    - Install CLI to /usr/local/bin"
	@echo "  lint       - Run Go linter"
	@echo "  release    - Create release package"
	@echo "  security   - Run security checks"
	@echo "  test       - Run tests"
	@echo ""
	@echo "Variables:"
	@echo "  VERSION    - Version to build (default: $(VERSION))"
	@echo ""
	@echo "Examples:"
	@echo "  make build VERSION=v0.2.0"
	@echo "  make build-all"
	@echo "  make install"