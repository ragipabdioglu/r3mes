package keeper

import (
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// ValidateExecutionEnvironment validates that a miner's execution environment matches approved specifications
func (k Keeper) ValidateExecutionEnvironment(
	ctx sdk.Context,
	minerAddress string,
	reportedEnvironment *types.ExecutionEnvironment,
) error {
	// Get active execution environment for the platform
	platform := reportedEnvironment.Platform
	activeEnv, err := k.GetActiveExecutionEnvironment(ctx, platform)
	if err != nil {
		// If no active environment, allow submission (for testing)
		ctx.Logger().Warn(fmt.Sprintf("No active execution environment for platform %s, allowing submission", platform))
		return nil
	}

	// Validate platform matches
	if activeEnv.Platform != platform {
		return errorsmod.Wrapf(
			types.ErrInvalidMiner,
			"platform mismatch: required %s, got %s",
			activeEnv.Platform,
			platform,
		)
	}

	// Validate Python version (strict matching)
	if activeEnv.PythonVersion != "" && reportedEnvironment.PythonVersion != activeEnv.PythonVersion {
		return errorsmod.Wrapf(
			types.ErrInvalidMiner,
			"Python version mismatch: required %s, got %s",
			activeEnv.PythonVersion,
			reportedEnvironment.PythonVersion,
		)
	}

	// Validate PyTorch version (strict matching)
	if activeEnv.PytorchVersion != "" && reportedEnvironment.PytorchVersion != activeEnv.PytorchVersion {
		return errorsmod.Wrapf(
			types.ErrInvalidMiner,
			"PyTorch version mismatch: required %s, got %s",
			activeEnv.PytorchVersion,
			reportedEnvironment.PytorchVersion,
		)
	}

	// Platform-specific validation
	if platform == "nvidia" {
		// Validate CUDA version
		if activeEnv.CudaVersion != "" && reportedEnvironment.CudaVersion != activeEnv.CudaVersion {
			return errorsmod.Wrapf(
				types.ErrInvalidMiner,
				"CUDA version mismatch: required %s, got %s",
				activeEnv.CudaVersion,
				reportedEnvironment.CudaVersion,
			)
		}

		// Validate cuDNN version
		if activeEnv.CudnnVersion != "" && reportedEnvironment.CudnnVersion != activeEnv.CudnnVersion {
			return errorsmod.Wrapf(
				types.ErrInvalidMiner,
				"cuDNN version mismatch: required %s, got %s",
				activeEnv.CudnnVersion,
				reportedEnvironment.CudnnVersion,
			)
		}

		// Validate CUBLAS workspace config
		if activeEnv.CublasWorkspaceConfig != "" && reportedEnvironment.CublasWorkspaceConfig != activeEnv.CublasWorkspaceConfig {
			return errorsmod.Wrapf(
				types.ErrInvalidMiner,
				"CUBLAS workspace config mismatch: required %s, got %s",
				activeEnv.CublasWorkspaceConfig,
				reportedEnvironment.CublasWorkspaceConfig,
			)
		}
	} else if platform == "amd" {
		// Validate ROCm version
		if activeEnv.RocmVersion != "" && reportedEnvironment.RocmVersion != activeEnv.RocmVersion {
			return errorsmod.Wrapf(
				types.ErrInvalidMiner,
				"ROCm version mismatch: required %s, got %s",
				activeEnv.RocmVersion,
				reportedEnvironment.RocmVersion,
			)
		}
	} else if platform == "intel" {
		// Validate Intel XPU version
		if activeEnv.IntelXpuVersion != "" && reportedEnvironment.IntelXpuVersion != activeEnv.IntelXpuVersion {
			return errorsmod.Wrapf(
				types.ErrInvalidMiner,
				"Intel XPU version mismatch: required %s, got %s",
				activeEnv.IntelXpuVersion,
				reportedEnvironment.IntelXpuVersion,
			)
		}
	}

	// Validate deterministic algorithms are enabled
	if activeEnv.DeterministicAlgorithmsEnabled && !reportedEnvironment.DeterministicAlgorithmsEnabled {
		return errorsmod.Wrap(
			types.ErrInvalidMiner,
			"deterministic algorithms must be enabled",
		)
	}

	// Validate floating point mode (hardware-specific consistency)
	if activeEnv.FloatingPointMode != "" && reportedEnvironment.FloatingPointMode != activeEnv.FloatingPointMode {
		return errorsmod.Wrapf(
			types.ErrInvalidMiner,
			"floating point mode mismatch: required %s, got %s",
			activeEnv.FloatingPointMode,
			reportedEnvironment.FloatingPointMode,
		)
	}

	return nil
}

// GetActiveExecutionEnvironment retrieves the active execution environment for a platform
func (k Keeper) GetActiveExecutionEnvironment(
	ctx sdk.Context,
	platform string,
) (*types.ExecutionEnvironment, error) {
	var activeEnv *types.ExecutionEnvironment
	found := false
	
	// Walk through approved environments to find active one for the platform
	err := k.ApprovedExecutionEnvironments.Walk(ctx, nil, func(key string, value types.ExecutionEnvironment) (stop bool, err error) {
		if value.Platform == platform && value.IsActive {
			activeEnv = &value
			found = true
			return true, nil
		}
		return false, nil
	})
	
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to get active execution environment")
	}
	
	if !found {
		return nil, fmt.Errorf("no active execution environment found for platform %s", platform)
	}
	
	return activeEnv, nil
}

// CheckEnvironmentViolation checks if a miner has violated environment requirements
func (k Keeper) CheckEnvironmentViolation(
	ctx sdk.Context,
	minerAddress string,
	reportedEnvironment *types.ExecutionEnvironment,
) (bool, string) {
	err := k.ValidateExecutionEnvironment(ctx, minerAddress, reportedEnvironment)
	if err != nil {
		return true, err.Error()
	}
	return false, ""
}

// RecordEnvironmentViolation records an environment violation for a miner
func (k Keeper) RecordEnvironmentViolation(
	ctx sdk.Context,
	minerAddress string,
	violationReason string,
) error {
	// Get mining contribution
	contribution, err := k.MiningContributions.Get(ctx, minerAddress)
	if err != nil {
		// Create new contribution with violation
		contribution = types.MiningContribution{
			MinerAddress:          minerAddress,
			TotalSubmissions:      0,
			SuccessfulSubmissions: 0,
			TrustScore:            "0.3", // Lower trust score for violations
			ReputationTier:        "new",
			SlashingEvents:        1, // Count violation as slashing event
		}
	} else {
		// Update existing contribution
		contribution.SlashingEvents++
		// Reduce trust score for environment violation
		// Use trust score update mechanism
		if err := k.UpdateTrustScore(ctx, minerAddress, "challenged"); err != nil {
			ctx.Logger().Error(fmt.Sprintf("Failed to update trust score for environment violation: %v", err))
			// Don't fail - trust score update is not critical
		}
	}

	// Store updated contribution
	if err := k.MiningContributions.Set(ctx, minerAddress, contribution); err != nil {
		return err
	}

	// Emit event
	ctx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeResourceViolationWarning,
			sdk.NewAttribute(types.AttributeKeyMiner, minerAddress),
			sdk.NewAttribute(types.AttributeKeyReason, violationReason),
		),
	)

	return nil
}

