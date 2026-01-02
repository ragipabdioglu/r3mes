package keeper

import (
	"fmt"
	"os"
	"strings"
)

// ValidateProductionSecurity validates production security requirements
func ValidateProductionSecurity(ipfsAPIURL string) error {
	env := os.Getenv("R3MES_ENV")

	// Only validate in production environment
	if env != "production" {
		return nil
	}

	// Validate IPFS API URL is set in production
	if ipfsAPIURL == "" {
		return fmt.Errorf("IPFS API URL must be set in production")
	}

	// Validate IPFS API URL doesn't use localhost in production
	if strings.Contains(strings.ToLower(ipfsAPIURL), "localhost") ||
		strings.Contains(ipfsAPIURL, "127.0.0.1") ||
		strings.Contains(ipfsAPIURL, "::1") {
		return fmt.Errorf("IPFS API URL cannot use localhost in production: %s", ipfsAPIURL)
	}

	// Validate other production requirements
	if err := validateProductionEnvironment(); err != nil {
		return fmt.Errorf("production environment validation failed: %w", err)
	}

	return nil
}

// validateProductionEnvironment validates production environment variables
func validateProductionEnvironment() error {
	requiredEnvVars := []string{
		"R3MES_ENV",
	}

	for _, envVar := range requiredEnvVars {
		if os.Getenv(envVar) == "" {
			return fmt.Errorf("required environment variable %s is not set", envVar)
		}
	}

	// Validate no localhost usage in critical environment variables
	criticalEnvVars := []string{
		"MINER_STATS_HOST",
		"R3MES_VERIFICATION_SERVICE_ADDR",
	}

	for _, envVar := range criticalEnvVars {
		value := os.Getenv(envVar)
		if value != "" {
			if strings.Contains(strings.ToLower(value), "localhost") ||
				strings.Contains(value, "127.0.0.1") ||
				strings.Contains(value, "::1") {
				return fmt.Errorf("environment variable %s cannot use localhost in production: %s", envVar, value)
			}
		}
	}

	return nil
}
