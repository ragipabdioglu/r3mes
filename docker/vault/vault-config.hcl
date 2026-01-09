# HashiCorp Vault Configuration for R3MES
# 
# This configuration is for development/testing.
# For production, use a proper Vault cluster with HA storage.

# Storage backend
storage "file" {
  path = "/vault/data"
}

# Listener configuration
listener "tcp" {
  address     = "0.0.0.0:8200"
  tls_disable = "true"  # Enable TLS in production!
}

# API address
api_addr = "http://127.0.0.1:8200"

# Cluster address (for HA)
cluster_addr = "https://127.0.0.1:8201"

# UI
ui = true

# Disable mlock (for Docker)
disable_mlock = true

# Telemetry
telemetry {
  prometheus_retention_time = "30s"
  disable_hostname          = true
}

# Max lease TTL
max_lease_ttl = "768h"

# Default lease TTL
default_lease_ttl = "768h"
