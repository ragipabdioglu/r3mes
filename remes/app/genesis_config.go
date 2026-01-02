package app

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
)

// GenesisConfig holds genesis-specific configuration
type GenesisConfig struct {
	DefaultRewardAmount string `json:"default_reward_amount"`
	ModelVersion        string `json:"model_version"`
	MinRewardAmount     string `json:"min_reward_amount"`
	MaxRewardAmount     string `json:"max_reward_amount"`
}

// LoadGenesisConfig loads genesis configuration from environment or defaults
func LoadGenesisConfig() *GenesisConfig {
	config := &GenesisConfig{
		DefaultRewardAmount: getEnvString("GENESIS_DEFAULT_REWARD", "5000000"),
		ModelVersion:        getEnvString("GENESIS_MODEL_VERSION", "v1.0.0"),
		MinRewardAmount:     getEnvString("GENESIS_MIN_REWARD", "1000000"),
		MaxRewardAmount:     getEnvString("GENESIS_MAX_REWARD", "10000000"),
	}

	// Validate reward amounts are numeric
	if err := config.Validate(); err != nil {
		fmt.Printf("⚠️  Invalid genesis config: %v, using defaults\n", err)
		config = &GenesisConfig{
			DefaultRewardAmount: "5000000",
			ModelVersion:        "v1.0.0",
			MinRewardAmount:     "1000000",
			MaxRewardAmount:     "10000000",
		}
	}

	return config
}

// Validate validates the genesis configuration with comprehensive checks
func (gc *GenesisConfig) Validate() error {
	// Parse reward amounts
	defaultReward, err := strconv.ParseUint(gc.DefaultRewardAmount, 10, 64)
	if err != nil {
		return fmt.Errorf("invalid default_reward_amount: %w", err)
	}
	minReward, err := strconv.ParseUint(gc.MinRewardAmount, 10, 64)
	if err != nil {
		return fmt.Errorf("invalid min_reward_amount: %w", err)
	}
	maxReward, err := strconv.ParseUint(gc.MaxRewardAmount, 10, 64)
	if err != nil {
		return fmt.Errorf("invalid max_reward_amount: %w", err)
	}

	// Validate reward bounds
	if minReward >= maxReward {
		return fmt.Errorf("min_reward_amount (%d) must be less than max_reward_amount (%d)", minReward, maxReward)
	}

	if defaultReward < minReward || defaultReward > maxReward {
		return fmt.Errorf("default_reward_amount (%d) must be between min_reward_amount (%d) and max_reward_amount (%d)",
			defaultReward, minReward, maxReward)
	}

	// Validate reasonable bounds (prevent extreme values)
	const (
		absoluteMinReward = 100000    // 0.1 token (assuming 6 decimals)
		absoluteMaxReward = 100000000 // 100 tokens
	)

	if minReward < absoluteMinReward {
		return fmt.Errorf("min_reward_amount (%d) is below absolute minimum (%d)", minReward, absoluteMinReward)
	}
	if maxReward > absoluteMaxReward {
		return fmt.Errorf("max_reward_amount (%d) exceeds absolute maximum (%d)", maxReward, absoluteMaxReward)
	}

	// Validate model version format (semantic versioning)
	if gc.ModelVersion == "" {
		return fmt.Errorf("model_version cannot be empty")
	}

	// Basic semantic version validation (vX.Y.Z format)
	if len(gc.ModelVersion) < 5 || gc.ModelVersion[0] != 'v' {
		return fmt.Errorf("model_version must follow semantic versioning format (vX.Y.Z), got: %s", gc.ModelVersion)
	}

	return nil
}

// SaveToFile saves genesis config to a JSON file
func (gc *GenesisConfig) SaveToFile(filename string) error {
	data, err := json.MarshalIndent(gc, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal genesis config: %w", err)
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("failed to write genesis config file: %w", err)
	}

	return nil
}

// LoadFromFile loads genesis config from a JSON file
func LoadGenesisConfigFromFile(filename string) (*GenesisConfig, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read genesis config file: %w", err)
	}

	var config GenesisConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal genesis config: %w", err)
	}

	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid genesis config in file: %w", err)
	}

	return &config, nil
}

// GenesisVaultValidation validates genesis vault entries
type GenesisVaultValidation struct {
	MaxEntries      int     `json:"max_entries"`
	MaxDimensions   int     `json:"max_dimensions"`
	MinTrainingLoss float64 `json:"min_training_loss"`
	MaxTrainingLoss float64 `json:"max_training_loss"`
}

// DefaultGenesisVaultValidation returns default validation rules for genesis vault
func DefaultGenesisVaultValidation() *GenesisVaultValidation {
	return &GenesisVaultValidation{
		MaxEntries:      50,    // Maximum 50 genesis vault entries
		MaxDimensions:   50000, // Maximum 50k dimensions per gradient
		MinTrainingLoss: 0.0,   // Minimum training loss
		MaxTrainingLoss: 10.0,  // Maximum reasonable training loss
	}
}

// ValidateGenesisVaultEntry validates a single genesis vault entry
func (gvv *GenesisVaultValidation) ValidateGenesisVaultEntry(entry map[string]interface{}) error {
	// Check if gradient data exists
	gradientData, exists := entry["gradient_data"]
	if !exists {
		return fmt.Errorf("genesis vault entry missing gradient_data field")
	}

	// Validate gradient data is array
	gradientArray, ok := gradientData.([]interface{})
	if !ok {
		return fmt.Errorf("gradient_data must be an array")
	}

	// Check dimensions limit
	if len(gradientArray) > gvv.MaxDimensions {
		return fmt.Errorf("gradient_data dimensions (%d) exceed maximum allowed (%d)",
			len(gradientArray), gvv.MaxDimensions)
	}

	if len(gradientArray) == 0 {
		return fmt.Errorf("gradient_data cannot be empty")
	}

	// Validate training loss if present
	if trainingLoss, exists := entry["training_loss"]; exists {
		if loss, ok := trainingLoss.(float64); ok {
			if loss < gvv.MinTrainingLoss || loss > gvv.MaxTrainingLoss {
				return fmt.Errorf("training_loss (%.6f) must be between %.6f and %.6f",
					loss, gvv.MinTrainingLoss, gvv.MaxTrainingLoss)
			}
		}
	}

	return nil
}

// ValidateGenesisVaultEntries validates all genesis vault entries
func ValidateGenesisVaultEntries(entries []map[string]interface{}) error {
	validation := DefaultGenesisVaultValidation()

	// Check total entries limit
	if len(entries) > validation.MaxEntries {
		return fmt.Errorf("genesis vault entries (%d) exceed maximum allowed (%d)",
			len(entries), validation.MaxEntries)
	}

	// Validate each entry
	for i, entry := range entries {
		if err := validation.ValidateGenesisVaultEntry(entry); err != nil {
			return fmt.Errorf("genesis vault entry %d validation failed: %w", i, err)
		}
	}

	// Check for duplicate entries (basic check on gradient data length)
	dimensionCounts := make(map[int]int)
	for i, entry := range entries {
		if gradientData, exists := entry["gradient_data"]; exists {
			if gradientArray, ok := gradientData.([]interface{}); ok {
				dimensions := len(gradientArray)
				if count, exists := dimensionCounts[dimensions]; exists {
					// Allow up to 2 entries with same dimensions (for diversity)
					if count >= 2 {
						return fmt.Errorf("too many genesis vault entries (%d) with same dimensions (%d), entry %d",
							count+1, dimensions, i)
					}
					dimensionCounts[dimensions]++
				} else {
					dimensionCounts[dimensions] = 1
				}
			}
		}
	}

	return nil
}
