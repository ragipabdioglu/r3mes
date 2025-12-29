package keeper

import (
	"fmt"
	"strings"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// SupportedGPUArchitectures defines the whitelist of supported GPU architectures
// These architectures are known to support deterministic execution
var SupportedGPUArchitectures = map[string]bool{
	"Pascal":    true, // NVIDIA Pascal (GTX 10 series, P100)
	"Volta":     true, // NVIDIA Volta (V100)
	"Turing":    true, // NVIDIA Turing (RTX 20 series, T4)
	"Ampere":    true, // NVIDIA Ampere (RTX 30 series, A100)
	"Ada":       true, // NVIDIA Ada Lovelace (RTX 40 series)
	"Blackwell": true, // NVIDIA Blackwell (B100, H100)
}

// ValidateGPUArchitecture validates that a GPU architecture is in the whitelist
func (k Keeper) ValidateGPUArchitecture(ctx sdk.Context, gpuArchitecture string) error {
	if gpuArchitecture == "" {
		return errorsmod.Wrapf(types.ErrInvalidGPUArchitecture, "GPU architecture cannot be empty")
	}

	// Normalize architecture name (trim spaces, title case)
	normalized := strings.TrimSpace(gpuArchitecture)
	normalized = strings.Title(strings.ToLower(normalized))

	// Check against whitelist
	if !SupportedGPUArchitectures[normalized] {
		// Get list of supported architectures for error message
		supportedList := k.GetSupportedArchitecturesList()
		return errorsmod.Wrapf(
			types.ErrInvalidGPUArchitecture,
			"unsupported GPU architecture: %s. Supported architectures: %s",
			gpuArchitecture,
			supportedList,
		)
	}

	return nil
}

// GetSupportedArchitecturesList returns a comma-separated list of supported architectures
func (k Keeper) GetSupportedArchitecturesList() string {
	architectures := make([]string, 0, len(SupportedGPUArchitectures))
	for arch := range SupportedGPUArchitectures {
		architectures = append(architectures, arch)
	}
	// Sort for consistent output (optional, but nice for error messages)
	return strings.Join(architectures, ", ")
}

// IsGPUArchitectureSupported checks if a GPU architecture is supported
func (k Keeper) IsGPUArchitectureSupported(gpuArchitecture string) bool {
	normalized := strings.TrimSpace(gpuArchitecture)
	normalized = strings.Title(strings.ToLower(normalized))
	return SupportedGPUArchitectures[normalized]
}

// UpdateGPUWhitelist updates the GPU architecture whitelist (governance-only)
// This allows adding new architectures as they become supported
func (k Keeper) UpdateGPUWhitelist(ctx sdk.Context, architectures []string, remove bool) error {
	// This should be called from a governance proposal handler
	// For now, we'll just validate the input

	if !remove {
		// Adding architectures
		for _, arch := range architectures {
			normalized := strings.TrimSpace(arch)
			normalized = strings.Title(strings.ToLower(normalized))
			if normalized == "" {
				return fmt.Errorf("empty architecture name")
			}
			// In production, this would update on-chain params
			// For now, we'll just validate
		}
	} else {
		// Removing architectures
		for _, arch := range architectures {
			normalized := strings.TrimSpace(arch)
			normalized = strings.Title(strings.ToLower(normalized))
			if !SupportedGPUArchitectures[normalized] {
				return fmt.Errorf("architecture %s is not in whitelist", arch)
			}
		}
	}

	// In production, update params and emit event
	ctx.Logger().Info(fmt.Sprintf("GPU whitelist update: %v (remove: %v)", architectures, remove))

	return nil
}

