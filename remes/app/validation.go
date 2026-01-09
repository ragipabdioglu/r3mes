package app

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"strconv"
	"strings"
)

// ValidationError represents a validation error
type ValidationError struct {
	Field   string
	Value   string
	Message string
}

func (ve ValidationError) Error() string {
	return fmt.Sprintf("validation error for field '%s' with value '%s': %s", ve.Field, ve.Value, ve.Message)
}

// ValidateEnvironment validates the application environment configuration
func ValidateEnvironment() []ValidationError {
	var errors []ValidationError

	// Validate address prefix
	if prefix := os.Getenv("REMES_ADDRESS_PREFIX"); prefix != "" {
		if len(prefix) < 2 || len(prefix) > 10 {
			errors = append(errors, ValidationError{
				Field:   "REMES_ADDRESS_PREFIX",
				Value:   prefix,
				Message: "must be between 2 and 10 characters",
			})
		}
		if !isAlphaNumeric(prefix) {
			errors = append(errors, ValidationError{
				Field:   "REMES_ADDRESS_PREFIX",
				Value:   prefix,
				Message: "must contain only alphanumeric characters",
			})
		}
	}

	// Validate coin type
	if coinType := os.Getenv("REMES_COIN_TYPE"); coinType != "" {
		if _, err := strconv.ParseUint(coinType, 10, 32); err != nil {
			errors = append(errors, ValidationError{
				Field:   "REMES_COIN_TYPE",
				Value:   coinType,
				Message: "must be a valid uint32 number",
			})
		}
	}

	// Validate node home directory
	if nodeHome := os.Getenv("REMES_NODE_HOME"); nodeHome != "" {
		if !isValidPath(nodeHome) {
			errors = append(errors, ValidationError{
				Field:   "REMES_NODE_HOME",
				Value:   nodeHome,
				Message: "must be a valid directory path",
			})
		}
	}

	// Validate environment
	if env := os.Getenv("REMES_ENVIRONMENT"); env != "" {
		validEnvs := []string{"development", "staging", "production", "test"}
		if !contains(validEnvs, env) {
			errors = append(errors, ValidationError{
				Field:   "REMES_ENVIRONMENT",
				Value:   env,
				Message: fmt.Sprintf("must be one of: %s", strings.Join(validEnvs, ", ")),
			})
		}
	}

	// Validate Sentry DSN if provided
	if sentryDSN := os.Getenv("SENTRY_DSN"); sentryDSN != "" {
		if _, err := url.Parse(sentryDSN); err != nil {
			errors = append(errors, ValidationError{
				Field:   "SENTRY_DSN",
				Value:   sentryDSN,
				Message: "must be a valid URL",
			})
		}
	}

	// Validate log level
	if logLevel := os.Getenv("LOG_LEVEL"); logLevel != "" {
		// Get valid levels from environment or use defaults
		validLevelsEnv := os.Getenv("VALID_LOG_LEVELS")
		var validLevels []string
		if validLevelsEnv != "" {
			validLevels = strings.Split(validLevelsEnv, ",")
		} else {
			// Default valid levels
			validLevels = []string{"debug", "info", "warn", "error", "fatal", "panic"}
		}

		if !contains(validLevels, strings.ToLower(logLevel)) {
			errors = append(errors, ValidationError{
				Field:   "LOG_LEVEL",
				Value:   logLevel,
				Message: fmt.Sprintf("must be one of: %s", strings.Join(validLevels, ", ")),
			})
		}
	}

	// Validate genesis reward amounts
	validateRewardAmount := func(envVar string) {
		if value := os.Getenv(envVar); value != "" {
			if _, err := strconv.ParseUint(value, 10, 64); err != nil {
				errors = append(errors, ValidationError{
					Field:   envVar,
					Value:   value,
					Message: "must be a valid uint64 number",
				})
			}
		}
	}

	validateRewardAmount("GENESIS_DEFAULT_REWARD")
	validateRewardAmount("GENESIS_MIN_REWARD")
	validateRewardAmount("GENESIS_MAX_REWARD")

	return errors
}

// ValidateNetworkConnectivity validates network connectivity requirements
func ValidateNetworkConnectivity() []ValidationError {
	var errors []ValidationError

	// Check if required ports are available
	requiredPorts := []string{"26656", "26657", "1317", "9090"}
	for _, port := range requiredPorts {
		if !isPortAvailable(port) {
			errors = append(errors, ValidationError{
				Field:   "network_port",
				Value:   port,
				Message: fmt.Sprintf("port %s is not available", port),
			})
		}
	}

	return errors
}

// Helper functions

func isAlphaNumeric(s string) bool {
	for _, r := range s {
		if !((r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')) {
			return false
		}
	}
	return true
}

func isValidPath(path string) bool {
	// Basic path validation - check if it's not empty and doesn't contain invalid characters
	if path == "" {
		return false
	}

	// Check for invalid characters (basic check)
	invalidChars := []string{"\x00", "<", ">", "|", "\""}
	for _, char := range invalidChars {
		if strings.Contains(path, char) {
			return false
		}
	}

	return true
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func isPortAvailable(port string) bool {
	ln, err := net.Listen("tcp", ":"+port)
	if err != nil {
		return false
	}
	ln.Close()
	return true
}

// PrintValidationErrors prints validation errors in a formatted way
func PrintValidationErrors(errors []ValidationError) {
	if len(errors) == 0 {
		fmt.Println("✅ Environment validation passed")
		return
	}

	fmt.Printf("❌ Environment validation failed with %d error(s):\n", len(errors))
	for i, err := range errors {
		fmt.Printf("  %d. %s\n", i+1, err.Error())
	}
}
