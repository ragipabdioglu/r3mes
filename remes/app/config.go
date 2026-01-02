package app

import (
	"os"
	"strconv"
	"strings"
)

// Config holds application configuration
type Config struct {
	AccountAddressPrefix string
	ChainCoinType        uint32
	NodeHome             string
	Environment          string
	SentryDSN            string
	LogLevel             string
	EnableWebSocket      bool
	EnableDashboard      bool
}

// LoadConfig loads configuration from environment variables with defaults
func LoadConfig() *Config {
	config := &Config{
		AccountAddressPrefix: getEnvString("REMES_ADDRESS_PREFIX", "remes"),
		ChainCoinType:        getEnvUint32("REMES_COIN_TYPE", 9999),
		NodeHome:             getEnvString("REMES_NODE_HOME", ""),
		Environment:          getEnvString("REMES_ENVIRONMENT", "development"),
		SentryDSN:            getEnvString("SENTRY_DSN", ""),
		LogLevel:             getEnvString("LOG_LEVEL", "info"),
		EnableWebSocket:      getEnvBool("ENABLE_WEBSOCKET", true),
		EnableDashboard:      getEnvBool("ENABLE_DASHBOARD", true),
	}

	// Set default node home if not provided
	if config.NodeHome == "" {
		config.NodeHome = DefaultNodeHome
	}

	return config
}

// getEnvString gets string from environment with default
func getEnvString(key, defaultValue string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return defaultValue
}

// getEnvUint32 gets uint32 from environment with default
func getEnvUint32(key string, defaultValue uint32) uint32 {
	if value := os.Getenv(key); value != "" {
		if parsed, err := strconv.ParseUint(value, 10, 32); err == nil {
			return uint32(parsed)
		}
	}
	return defaultValue
}

// getEnvBool gets boolean from environment with default
func getEnvBool(key string, defaultValue bool) bool {
	if value := os.Getenv(key); value != "" {
		return strings.ToLower(value) == "true" || value == "1"
	}
	return defaultValue
}

// IsProduction returns true if running in production environment
func (c *Config) IsProduction() bool {
	return c.Environment == "production"
}

// IsDevelopment returns true if running in development environment
func (c *Config) IsDevelopment() bool {
	return c.Environment == "development"
}
