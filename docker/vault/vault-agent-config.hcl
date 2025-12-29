# Vault Agent Configuration for R3MES
# Automatically fetches and renews secrets for applications

pid_file = "/tmp/vault-agent.pid"

vault {
  address = "http://vault:8200"
}

# Auto-auth using AppRole
auto_auth {
  method "approle" {
    mount_path = "auth/approle"
    config = {
      role_id_file_path   = "/vault/config/role_id"
      secret_id_file_path = "/vault/config/secret_id"
      remove_secret_id_file_after_reading = false
    }
  }

  sink "file" {
    config = {
      path = "/vault/secrets/.vault-token"
      mode = 0644
    }
  }
}

# Cache for performance
cache {
  use_auto_auth_token = true
}

# Template for PostgreSQL credentials
template {
  source      = "/vault/templates/postgres.ctmpl"
  destination = "/vault/secrets/postgres_password"
  perms       = 0600
  command     = "echo 'PostgreSQL credentials updated'"
}

# Template for Redis credentials
template {
  source      = "/vault/templates/redis.ctmpl"
  destination = "/vault/secrets/redis_password"
  perms       = 0600
  command     = "echo 'Redis credentials updated'"
}

# Template for application secrets
template {
  source      = "/vault/templates/app.ctmpl"
  destination = "/vault/secrets/app_secrets.env"
  perms       = 0600
  command     = "echo 'Application secrets updated'"
}

# Template for database URL
template {
  source      = "/vault/templates/database_url.ctmpl"
  destination = "/vault/secrets/database_url"
  perms       = 0600
}

# Listener for API (optional)
listener "tcp" {
  address     = "127.0.0.1:8100"
  tls_disable = true
}
