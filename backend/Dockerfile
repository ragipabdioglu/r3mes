# Multi-stage build for production
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    bash \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app

# Copy application code (no volume mount for production)
COPY . .

# Copy init script
COPY scripts/init_db.sh /app/scripts/init_db.sh
RUN chmod +x /app/scripts/init_db.sh

# Create necessary directories
RUN mkdir -p checkpoints logs

# Keep running as root for now (alembic needs access to .local/bin)
# TODO: Fix permissions for non-root user later

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run database initialization and application
CMD ["/app/scripts/init_db.sh"]
