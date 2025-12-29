package keeper

import (
	"fmt"
	"os"
	"strings"
)

// DebugLevel represents the verbosity level for debug mode
type DebugLevel string

const (
	DebugLevelMinimal  DebugLevel = "minimal"  // Only critical errors and performance metrics
	DebugLevelStandard DebugLevel = "standard" // Detailed logging, state inspection, performance profiling
	DebugLevelVerbose  DebugLevel = "verbose"  // All system internals, trace logs, internal state dumps
)

// DebugConfig holds debug configuration for the keeper
type DebugConfig struct {
	// Enabled indicates if debug mode is globally enabled
	Enabled bool

	// Level specifies the verbosity level
	Level DebugLevel

	// Components specifies which components should have debug enabled (comma-separated)
	// Valid values: blockchain,backend,miner,launcher,frontend
	// Empty or "*" means all components
	Components map[string]bool

	// Features
	Logging          bool // Enable enhanced logging
	Profiling        bool // Enable performance profiling
	StateInspection  bool // Enable state inspection
	Trace            bool // Enable distributed tracing

	// Logging configuration
	LogLevel string // TRACE, DEBUG, INFO, WARN, ERROR
	LogFormat string // json, text
	LogFile   string // Path to log file (optional)

	// Performance profiling configuration
	ProfileOutput   string // Directory for profile outputs
	ProfileInterval int    // Interval in seconds for periodic profiling

	// Trace configuration
	TraceEnabled   bool   // Enable trace collection
	TraceBufferSize int   // Maximum number of traces to keep in memory
	TraceExportPath string // Path to export traces
}

// IsComponentEnabled checks if debug is enabled for a specific component
func (dc *DebugConfig) IsComponentEnabled(component string) bool {
	if !dc.Enabled {
		return false
	}
	// If no components specified or "*" is present, enable all
	if len(dc.Components) == 0 || dc.Components["*"] {
		return true
	}
	return dc.Components[strings.ToLower(component)]
}

// IsBlockchainEnabled checks if debug is enabled for blockchain component
func (dc *DebugConfig) IsBlockchainEnabled() bool {
	return dc.IsComponentEnabled("blockchain")
}

// LoadDebugConfig loads debug configuration from environment variables
func LoadDebugConfig() (*DebugConfig, error) {
	config := &DebugConfig{
		Components: make(map[string]bool),
	}

	// Check if debug mode is enabled
	debugMode := os.Getenv("R3MES_DEBUG_MODE")
	config.Enabled = strings.ToLower(debugMode) == "true"

	if !config.Enabled {
		// Return default (disabled) config
		return config, nil
	}

	// Load debug level
	levelStr := strings.ToLower(os.Getenv("R3MES_DEBUG_LEVEL"))
	switch levelStr {
	case "minimal":
		config.Level = DebugLevelMinimal
	case "standard":
		config.Level = DebugLevelStandard
	case "verbose":
		config.Level = DebugLevelVerbose
	default:
		// Default to verbose if level is not specified
		config.Level = DebugLevelVerbose
	}

	// Load enabled components
	componentsStr := os.Getenv("R3MES_DEBUG_COMPONENTS")
	if componentsStr == "" {
		// Default to all components if not specified
		config.Components["*"] = true
	} else {
		components := strings.Split(componentsStr, ",")
		for _, comp := range components {
			comp = strings.TrimSpace(strings.ToLower(comp))
			if comp != "" {
				config.Components[comp] = true
			}
		}
	}

	// Load feature flags (default to true if debug mode is enabled)
	config.Logging = getEnvBool("R3MES_DEBUG_LOGGING", true)
	config.Profiling = getEnvBool("R3MES_DEBUG_PROFILING", true)
	config.StateInspection = getEnvBool("R3MES_DEBUG_STATE_INSPECTION", true)
	config.Trace = getEnvBool("R3MES_DEBUG_TRACE", true)

	// Load logging configuration
	config.LogLevel = strings.ToUpper(os.Getenv("R3MES_DEBUG_LOG_LEVEL"))
	if config.LogLevel == "" {
		config.LogLevel = "TRACE" // Default to TRACE in debug mode
	}
	// Validate log level
	validLogLevels := map[string]bool{
		"TRACE": true,
		"DEBUG": true,
		"INFO":  true,
		"WARN":  true,
		"ERROR": true,
	}
	if !validLogLevels[config.LogLevel] {
		return nil, fmt.Errorf("invalid R3MES_DEBUG_LOG_LEVEL: %s (valid: TRACE, DEBUG, INFO, WARN, ERROR)", config.LogLevel)
	}

	config.LogFormat = strings.ToLower(os.Getenv("R3MES_DEBUG_LOG_FORMAT"))
	if config.LogFormat == "" {
		config.LogFormat = "json" // Default to JSON for structured logging
	}
	if config.LogFormat != "json" && config.LogFormat != "text" {
		return nil, fmt.Errorf("invalid R3MES_DEBUG_LOG_FORMAT: %s (valid: json, text)", config.LogFormat)
	}

	config.LogFile = os.Getenv("R3MES_DEBUG_LOG_FILE")
	// Default log file location if not specified
	if config.LogFile == "" {
		homeDir, err := os.UserHomeDir()
		if err == nil {
			config.LogFile = fmt.Sprintf("%s/.r3mes/debug.log", homeDir)
		}
	}

	// Load performance profiling configuration
	config.ProfileOutput = os.Getenv("R3MES_DEBUG_PROFILE_OUTPUT")
	if config.ProfileOutput == "" {
		homeDir, err := os.UserHomeDir()
		if err == nil {
			config.ProfileOutput = fmt.Sprintf("%s/.r3mes/profiles", homeDir)
		}
	}
	config.ProfileInterval = getEnvInt("R3MES_DEBUG_PROFILE_INTERVAL", 60)

	// Load trace configuration
	config.TraceEnabled = getEnvBool("R3MES_DEBUG_TRACE_ENABLED", true)
	config.TraceBufferSize = getEnvInt("R3MES_DEBUG_TRACE_BUFFER_SIZE", 10000)
	config.TraceExportPath = os.Getenv("R3MES_DEBUG_TRACE_EXPORT_PATH")
	if config.TraceExportPath == "" {
		homeDir, err := os.UserHomeDir()
		if err == nil {
			config.TraceExportPath = fmt.Sprintf("%s/.r3mes/traces", homeDir)
		}
	}

	return config, nil
}

// ValidateDebugConfig validates debug configuration for security
// This should be called to ensure debug mode is not accidentally enabled in production
func ValidateDebugConfig(isProduction bool) error {
	if !isProduction {
		// In development/testing, debug mode is allowed
		return nil
	}

	// In production, check if debug mode is enabled
	debugMode := os.Getenv("R3MES_DEBUG_MODE")
	if strings.ToLower(debugMode) == "true" {
		return fmt.Errorf(
			"SECURITY ERROR: R3MES_DEBUG_MODE=true is set in production environment. " +
				"Debug mode should only be used in development/testing. " +
				"Please unset R3MES_DEBUG_MODE environment variable before running in production. " +
				"If you need production debugging, use R3MES_DEBUG_MODE=minimal with explicit component flags",
		)
	}

	// Check for verbose debug level in production (even if explicitly enabled)
	debugLevel := strings.ToLower(os.Getenv("R3MES_DEBUG_LEVEL"))
	if debugLevel == "verbose" {
		return fmt.Errorf(
			"SECURITY WARNING: R3MES_DEBUG_LEVEL=verbose is not recommended in production. " +
				"Consider using 'standard' or 'minimal' level instead",
		)
	}

	return nil
}

// getEnvBool reads a boolean environment variable
// Returns defaultValue if the variable is not set or cannot be parsed
func getEnvBool(key string, defaultValue bool) bool {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	return strings.ToLower(value) == "true"
}

// getEnvInt reads an integer environment variable
// Returns defaultValue if the variable is not set or cannot be parsed
func getEnvInt(key string, defaultValue int) int {
	value := os.Getenv(key)
	if value == "" {
		return defaultValue
	}
	var result int
	_, err := fmt.Sscanf(value, "%d", &result)
	if err != nil {
		return defaultValue
	}
	return result
}

// GetDebugConfig returns the global debug configuration
// This is a convenience function that caches the config (thread-safe for read-only access)
var globalDebugConfig *DebugConfig
var debugConfigInitialized bool

func GetDebugConfig() *DebugConfig {
	if !debugConfigInitialized {
		// Load config on first access (not thread-safe, but acceptable for initialization)
		config, err := LoadDebugConfig()
		if err != nil {
			// On error, return disabled config
			globalDebugConfig = &DebugConfig{
				Enabled:    false,
				Level:      DebugLevelStandard,
				Components: make(map[string]bool),
			}
		} else {
			globalDebugConfig = config
		}
		debugConfigInitialized = true
	}
	return globalDebugConfig
}
