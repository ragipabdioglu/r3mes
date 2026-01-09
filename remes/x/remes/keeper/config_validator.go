package keeper

import (
	"fmt"
	"net/url"
	"os"
	"strings"
)

// ProductionSecurityConfig holds production security requirements
type ProductionSecurityConfig struct {
	Environment      string
	IPFSAPIEndpoint  string
	AllowLocalhost   bool
	RequireHTTPS     bool
	MinStakeAmount   string
	MaxGradientSize  int64
}

// ValidateProductionSecurity validates production security requirements
func ValidateProductionSecurity(ipfsAPIURL string) error {
	env := os.Getenv("R3MES_ENV")
	if env == "" {
		env = "development"
	}

	config := ProductionSecurityConfig{
		Environment:      env,
		IPFSAPIEndpoint:  ipfsAPIURL,
		AllowLocalhost:   env != "production",
		RequireHTTPS:     env == "production",
		MinStakeAmount:   "1000000uremes", // 1 REMES
		MaxGradientSize:  100 * 1024 * 1024, // 100 MB
	}

	return config.Validate()
}

// Validate performs comprehensive security validation
func (c *ProductionSecurityConfig) Validate() error {
	// Validate environment
	validEnvs := map[string]bool{
		"development": true,
		"staging":     true,
		"testnet":     true,
		"production":  true,
	}

	if !validEnvs[c.Environment] {
		return fmt.Errorf("invalid environment: %s (must be one of: development, staging, testnet, production)", c.Environment)
	}

	// Validate IPFS endpoint
	if err := c.validateIPFSEndpoint(); err != nil {
		return fmt.Errorf("IPFS endpoint validation failed: %w", err)
	}

	// Production-specific validations
	if c.Environment == "production" {
		if err := c.validateProductionRequirements(); err != nil {
			return fmt.Errorf("production requirements not met: %w", err)
		}
	}

	return nil
}

// validateIPFSEndpoint validates IPFS API endpoint
func (c *ProductionSecurityConfig) validateIPFSEndpoint() error {
	if c.IPFSAPIEndpoint == "" {
		return fmt.Errorf("IPFS API endpoint cannot be empty")
	}

	// Parse URL
	parsedURL, err := url.Parse(c.IPFSAPIEndpoint)
	if err != nil {
		return fmt.Errorf("invalid IPFS URL format: %w", err)
	}

	// Check scheme
	if parsedURL.Scheme == "" {
		return fmt.Errorf("IPFS URL must include scheme (http:// or https://)")
	}

	// Production: require HTTPS
	if c.RequireHTTPS && parsedURL.Scheme != "https" {
		return fmt.Errorf("production environment requires HTTPS for IPFS endpoint")
	}

	// Check for localhost in production
	if !c.AllowLocalhost {
		hostname := strings.ToLower(parsedURL.Hostname())
		if hostname == "localhost" || hostname == "127.0.0.1" || hostname == "::1" {
			return fmt.Errorf("localhost IPFS endpoints not allowed in production")
		}
	}

	return nil
}

// validateProductionRequirements validates production-specific requirements
func (c *ProductionSecurityConfig) validateProductionRequirements() error {
	// Check for required environment variables
	requiredEnvVars := []string{
		"CHAIN_ID",
		"MONIKER",
	}

	for _, envVar := range requiredEnvVars {
		if os.Getenv(envVar) == "" {
			return fmt.Errorf("required environment variable %s not set", envVar)
		}
	}

	// Validate chain ID format
	chainID := os.Getenv("CHAIN_ID")
	if !strings.HasPrefix(chainID, "remes-") {
		return fmt.Errorf("invalid chain ID format: must start with 'remes-'")
	}

	// Check for testnet indicators in production
	if strings.Contains(strings.ToLower(chainID), "test") {
		return fmt.Errorf("testnet chain ID detected in production environment")
	}

	return nil
}

// GetSecurityConfig returns the current security configuration
func GetSecurityConfig() ProductionSecurityConfig {
	env := os.Getenv("R3MES_ENV")
	if env == "" {
		env = "development"
	}

	return ProductionSecurityConfig{
		Environment:      env,
		AllowLocalhost:   env != "production",
		RequireHTTPS:     env == "production",
		MinStakeAmount:   "1000000uremes",
		MaxGradientSize:  100 * 1024 * 1024,
	}
}

// IsProduction returns true if running in production environment
func IsProduction() bool {
	env := os.Getenv("R3MES_ENV")
	return env == "production"
}

// IsDevelopment returns true if running in development environment
func IsDevelopment() bool {
	env := os.Getenv("R3MES_ENV")
	return env == "" || env == "development"
}

// IsTestnet returns true if running in testnet environment
func IsTestnet() bool {
	env := os.Getenv("R3MES_ENV")
	return env == "testnet" || env == "staging"
}
