package keeper

import (
	"fmt"
	"net/url"
	"os"
	"strconv"
	"strings"
)

// EnvValidationError represents an environment variable validation error
type EnvValidationError struct {
	VarName string
	Message string
}

func (e *EnvValidationError) Error() string {
	return fmt.Sprintf("%s: %s", e.VarName, e.Message)
}

// EnvironmentValidator validates environment variables for the blockchain node
type EnvironmentValidator struct {
	isProduction bool
	errors       []*EnvValidationError
	warnings     []string
}

// NewEnvironmentValidator creates a new environment validator
func NewEnvironmentValidator() *EnvironmentValidator {
	env := os.Getenv("R3MES_ENV")
	isProduction := strings.ToLower(env) == "production"

	return &EnvironmentValidator{
		isProduction: isProduction,
		errors:       []*EnvValidationError{},
		warnings:     []string{},
	}
}

// ValidateURL validates a URL format
func (v *EnvironmentValidator) ValidateURL(value string, allowLocalhost bool) error {
	parsed, err := url.Parse(value)
	if err != nil {
		return fmt.Errorf("invalid URL format: %v", err)
	}

	if parsed.Scheme == "" {
		return fmt.Errorf("URL must include scheme (http:// or https://)")
	}

	if parsed.Host == "" {
		return fmt.Errorf("URL must include hostname")
	}

	if !allowLocalhost {
		// Use exact hostname matching (not substring)
		// Extract hostname from Host (which may include port)
		hostname := parsed.Hostname()
		hostnameLower := strings.ToLower(hostname)
		// Check for exact localhost matches (case-insensitive)
		if hostnameLower == "localhost" || hostnameLower == "127.0.0.1" || hostnameLower == "::1" {
			return fmt.Errorf("URL cannot use localhost or 127.0.0.1")
		}
		// Check for 127.x.x.x IP addresses
		if strings.HasPrefix(hostname, "127.") {
			return fmt.Errorf("URL cannot use 127.x.x.x IP addresses")
		}
	}

	return nil
}

// ValidatePort validates a port number
func (v *EnvironmentValidator) ValidatePort(value string) error {
	port, err := strconv.Atoi(value)
	if err != nil {
		return fmt.Errorf("port must be a valid integer")
	}

	if port < 1 || port > 65535 {
		return fmt.Errorf("port must be between 1 and 65535")
	}

	return nil
}

// ValidateRequired checks if a required variable is set
func (v *EnvironmentValidator) ValidateRequired(name string, value string, requiredInProduction bool) {
	if value == "" {
		if v.isProduction && requiredInProduction {
			v.errors = append(v.errors, &EnvValidationError{
				VarName: name,
				Message: "Required in production but not set",
			})
		} else if !v.isProduction {
			// In development, just warn
			v.warnings = append(v.warnings, fmt.Sprintf("%s: Not set (optional in development)", name))
		}
	}
}

// ValidateNoLocalhost checks if a value contains localhost in production
func (v *EnvironmentValidator) ValidateNoLocalhost(name string, value string) {
	if v.isProduction {
		if strings.Contains(value, "localhost") || strings.Contains(value, "127.0.0.1") {
			v.errors = append(v.errors, &EnvValidationError{
				VarName: name,
				Message: "Cannot use localhost or 127.0.0.1 in production",
			})
		}
	}
}

// ValidateAll validates all environment variables
func (v *EnvironmentValidator) ValidateAll() (bool, []*EnvValidationError, []string) {
	// R3MES_GRPC_ADDR
	grpcAddr := os.Getenv("R3MES_GRPC_ADDR")
	if v.isProduction {
		v.ValidateRequired("R3MES_GRPC_ADDR", grpcAddr, true)
		if grpcAddr != "" {
			v.ValidateNoLocalhost("R3MES_GRPC_ADDR", grpcAddr)
			if !strings.Contains(grpcAddr, ":") {
				v.errors = append(v.errors, &EnvValidationError{
					VarName: "R3MES_GRPC_ADDR",
					Message: "Must include hostname:port",
				})
			}
		}
	}

	// R3MES_TENDERMINT_RPC_ADDR
	tendermintRPC := os.Getenv("R3MES_TENDERMINT_RPC_ADDR")
	if v.isProduction {
		v.ValidateRequired("R3MES_TENDERMINT_RPC_ADDR", tendermintRPC, true)
		if tendermintRPC != "" {
			if err := v.ValidateURL(tendermintRPC, false); err != nil {
				v.errors = append(v.errors, &EnvValidationError{
					VarName: "R3MES_TENDERMINT_RPC_ADDR",
					Message: err.Error(),
				})
			}
		}
	}

	// CORS_ALLOWED_ORIGINS
	corsOrigins := os.Getenv("CORS_ALLOWED_ORIGINS")
	if v.isProduction {
		v.ValidateRequired("CORS_ALLOWED_ORIGINS", corsOrigins, true)
		if corsOrigins != "" {
			if strings.Contains(corsOrigins, "*") {
				v.errors = append(v.errors, &EnvValidationError{
					VarName: "CORS_ALLOWED_ORIGINS",
					Message: "Cannot use wildcard '*' in production",
				})
			}
			// Check each origin
			origins := strings.Split(corsOrigins, ",")
			for _, origin := range origins {
				origin = strings.TrimSpace(origin)
				if origin != "" {
					if err := v.ValidateURL(origin, false); err != nil {
						v.errors = append(v.errors, &EnvValidationError{
							VarName: "CORS_ALLOWED_ORIGINS",
							Message: fmt.Sprintf("Invalid origin '%s': %v", origin, err),
						})
					}
				}
			}
		}
	}

	// MINER_STATS_HOST
	minerStatsHost := os.Getenv("MINER_STATS_HOST")
	if v.isProduction {
		v.ValidateRequired("MINER_STATS_HOST", minerStatsHost, true)
		if minerStatsHost != "" {
			v.ValidateNoLocalhost("MINER_STATS_HOST", minerStatsHost)
		}
	}

	// MINER_STATS_PORT
	minerStatsPort := os.Getenv("MINER_STATS_PORT")
	if minerStatsPort != "" {
		if err := v.ValidatePort(minerStatsPort); err != nil {
			v.errors = append(v.errors, &EnvValidationError{
				VarName: "MINER_STATS_PORT",
				Message: err.Error(),
			})
		}
	}

	// R3MES_VERIFICATION_SERVICE_ADDR
	verificationAddr := os.Getenv("R3MES_VERIFICATION_SERVICE_ADDR")
	if v.isProduction {
		v.ValidateRequired("R3MES_VERIFICATION_SERVICE_ADDR", verificationAddr, true)
		if verificationAddr != "" {
			v.ValidateNoLocalhost("R3MES_VERIFICATION_SERVICE_ADDR", verificationAddr)
			if !strings.Contains(verificationAddr, ":") {
				v.errors = append(v.errors, &EnvValidationError{
					VarName: "R3MES_VERIFICATION_SERVICE_ADDR",
					Message: "Must include hostname:port",
				})
			}
		}
	}

	return len(v.errors) == 0, v.errors, v.warnings
}

// ValidateAndPanic validates environment variables and panics on errors
func (v *EnvironmentValidator) ValidateAndPanic() {
	isValid, errors, warnings := v.ValidateAll()

	// Log warnings
	for _, warning := range warnings {
		fmt.Printf("WARNING: %s\n", warning)
	}

	// Panic on errors
	if !isValid {
		errorMessages := make([]string, len(errors))
		for i, err := range errors {
			errorMessages[i] = err.Error()
		}
		panic(fmt.Sprintf("Environment validation failed:\n%s", strings.Join(errorMessages, "\n")))
	}
}
