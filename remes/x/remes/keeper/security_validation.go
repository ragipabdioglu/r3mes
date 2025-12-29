package keeper

import (
	"fmt"
	"os"
	"strings"
)

// ValidateProductionSecurity performs security validations for production mode
// This should be called during keeper initialization
// CRITICAL: Test mode CANNOT be enabled in production environment
func ValidateProductionSecurity(ipfsAPIURL string) error {
	// Determine if we're in production
	r3mesEnv := strings.ToLower(os.Getenv("R3MES_ENV"))
	isProduction := r3mesEnv == "production" || r3mesEnv == "prod"

	// Check if test mode is enabled
	testMode := strings.ToLower(os.Getenv("R3MES_TEST_MODE")) == "true"

	// CRITICAL SECURITY CHECK: Test mode CANNOT be enabled in production
	if isProduction && testMode {
		return fmt.Errorf(
			"FATAL SECURITY ERROR: R3MES_TEST_MODE=true is set in production environment (R3MES_ENV=%s). "+
				"This is a critical security vulnerability that bypasses all security validations. "+
				"Test mode MUST be disabled in production. "+
				"Please set R3MES_TEST_MODE=false or unset it before running in production.",
			r3mesEnv,
		)
	}

	// If test mode is enabled (non-production), skip validation
	if testMode {
		// Log warning even in non-production
		fmt.Printf("WARNING: R3MES_TEST_MODE=true - Security validations are bypassed. " +
			"This should only be used for development/testing.\n")
		return nil
	}

	// PRODUCTION MODE: Strict validation

	// Check if IPFS manager is configured (required for production)
	if ipfsAPIURL == "" {
		return fmt.Errorf(
			"SECURITY ERROR: IPFS API URL is not configured. " +
				"IPFS manager is required for dataset verification in production. " +
				"Please set IPFS_API_URL environment variable or configure it in app configuration. " +
				"If you are testing, set R3MES_TEST_MODE=true to bypass this check.",
		)
	}

	return nil
}

// ValidateProductionSecurityWithKeeper validates security using an existing keeper
// This can be called after keeper initialization
func (k Keeper) ValidateProductionSecurity() error {
	// Determine if we're in production
	r3mesEnv := strings.ToLower(os.Getenv("R3MES_ENV"))
	isProduction := r3mesEnv == "production" || r3mesEnv == "prod"

	// Check if test mode is enabled
	testMode := strings.ToLower(os.Getenv("R3MES_TEST_MODE")) == "true"

	// CRITICAL SECURITY CHECK: Test mode CANNOT be enabled in production
	if isProduction && testMode {
		return fmt.Errorf(
			"FATAL SECURITY ERROR: R3MES_TEST_MODE=true is set in production environment (R3MES_ENV=%s). "+
				"This is a critical security vulnerability that bypasses all security validations. "+
				"Test mode MUST be disabled in production. "+
				"Please set R3MES_TEST_MODE=false or unset it before running in production.",
			r3mesEnv,
		)
	}

	// Check if IPFS manager is configured (required for production)
	if k.ipfsManager == nil && isProduction {
		return fmt.Errorf(
			"SECURITY ERROR: IPFS manager is not configured in production. " +
				"IPFS manager is required for dataset verification in production. " +
				"Please configure IPFS_API_URL in app configuration.",
		)
	}

	return nil
}

// IsTestModeAllowed checks if test mode can be safely enabled
// Returns false if in production environment
func IsTestModeAllowed() bool {
	r3mesEnv := strings.ToLower(os.Getenv("R3MES_ENV"))
	isProduction := r3mesEnv == "production" || r3mesEnv == "prod"
	return !isProduction
}
