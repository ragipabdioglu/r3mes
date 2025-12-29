package keeper

import (
	"context"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkmath "cosmossdk.io/math"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// ResourceUsage represents actual resource usage for a node
type ResourceUsage struct {
	CPUCores         uint32
	MemoryGB         uint32
	GPUCount         uint32
	GPUMemoryGB      uint32
	StorageGB        uint32
	NetworkBandwidth uint32
}

// ValidateResourceUsage validates that actual resource usage does not exceed allocated quotas
func (k Keeper) ValidateResourceUsage(
	ctx context.Context,
	nodeAddress string,
	actualUsage ResourceUsage,
	role types.NodeType,
) (bool, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Get node registration
	registration, err := k.NodeRegistrations.Get(sdkCtx, nodeAddress)
	if err != nil {
		return false, errorsmod.Wrap(err, "node not registered")
	}

	// Find role allocation for the specified role
	var roleQuota *types.ResourceQuota
	for _, allocation := range registration.RoleAllocations {
		if allocation.Role == role {
			quota := allocation.Quota
			roleQuota = &quota
			break
		}
	}

	if roleQuota == nil {
		return false, errorsmod.Wrap(err, "role allocation not found")
	}

	// Validate resource usage against quota
	if actualUsage.CPUCores > roleQuota.CpuCores {
		return false, nil
	}
	if actualUsage.MemoryGB > roleQuota.MemoryGb {
		return false, nil
	}
	if actualUsage.GPUCount > roleQuota.GpuCount {
		return false, nil
	}
	if actualUsage.GPUMemoryGB > roleQuota.GpuMemoryGb {
		return false, nil
	}

	return true, nil
}

// CheckResourceViolation checks for resource allocation violations and applies progressive penalties
func (k Keeper) CheckResourceViolation(
	ctx context.Context,
	nodeAddress string,
	actualUsage ResourceUsage,
	role types.NodeType,
) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Validate resource usage
	isValid, err := k.ValidateResourceUsage(ctx, nodeAddress, actualUsage, role)
	if err != nil {
		return err
	}

	if isValid {
		return nil // No violation
	}

	// Get node registration
	registration, err := k.NodeRegistrations.Get(sdkCtx, nodeAddress)
	if err != nil {
		return errorsmod.Wrap(err, "node not registered")
	}

	// Progressive penalty system
	slashingEvents := registration.SlashingEvents

	// First violation: Warning (no slashing, just increment counter)
	if slashingEvents == 0 {
		registration.SlashingEvents = 1
		if err := k.NodeRegistrations.Set(sdkCtx, nodeAddress, registration); err != nil {
			return err
		}

		// Emit warning event
		sdkCtx.EventManager().EmitEvent(
			sdk.NewEvent(
				types.EventTypeResourceViolationWarning,
				sdk.NewAttribute(types.AttributeKeyNodeAddress, nodeAddress),
				sdk.NewAttribute(types.AttributeKeyRole, role.String()),
			),
		)
		return nil
	}

	// Second violation: Partial slash (5% of stake)
	if slashingEvents == 1 {
		registration.SlashingEvents = 2
		registration.Status = types.NODE_STATUS_SUSPENDED

		// Slash 5% of stake
		slashFraction := sdkmath.LegacyMustNewDecFromStr("0.05") // 5%
		if err := k.SlashNode(ctx, nodeAddress, slashFraction, "resource_violation_second_offense"); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to slash node: %v", err))
			// Continue with status update even if slashing fails
		}

		if err := k.NodeRegistrations.Set(sdkCtx, nodeAddress, registration); err != nil {
			return err
		}

		// Emit slashing event
		sdkCtx.EventManager().EmitEvent(
			sdk.NewEvent(
				types.EventTypeResourceViolationSlash,
				sdk.NewAttribute(types.AttributeKeyNodeAddress, nodeAddress),
				sdk.NewAttribute(types.AttributeKeyRole, role.String()),
				sdk.NewAttribute(types.AttributeKeySlashAmount, "5%"),
			),
		)
		return nil
	}

	// Third+ violation: Full slash and role suspension
	registration.SlashingEvents++
	registration.Status = types.NODE_STATUS_SLASHED

	// Full slash (100% of stake)
	slashFraction := sdkmath.LegacyMustNewDecFromStr("1.0") // 100%
	if err := k.SlashNode(ctx, nodeAddress, slashFraction, "resource_violation_third_offense"); err != nil {
		sdkCtx.Logger().Error(fmt.Sprintf("Failed to slash node: %v", err))
		// Continue with status update even if slashing fails
	}

	if err := k.NodeRegistrations.Set(sdkCtx, nodeAddress, registration); err != nil {
		return err
	}

	// Emit full slashing event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeResourceViolationFullSlash,
			sdk.NewAttribute(types.AttributeKeyNodeAddress, nodeAddress),
			sdk.NewAttribute(types.AttributeKeyRole, role.String()),
		),
	)

	return nil
}

// MonitorResourceUsage monitors resource usage and validates against allocations
func (k Keeper) MonitorResourceUsage(
	ctx context.Context,
	nodeAddress string,
	actualUsage ResourceUsage,
	role types.NodeType,
) error {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Get node registration
	registration, err := k.NodeRegistrations.Get(sdkCtx, nodeAddress)
	if err != nil {
		return errorsmod.Wrap(err, "node not registered")
	}

	// Check if node is active
	if registration.Status != types.NODE_STATUS_ACTIVE {
		return errorsmod.Wrap(err, "node is not active")
	}

	// Validate and check for violations
	return k.CheckResourceViolation(ctx, nodeAddress, actualUsage, role)
}

