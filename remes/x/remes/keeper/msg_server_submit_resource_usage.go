package keeper

import (
	"context"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// SubmitResourceUsage handles MsgSubmitResourceUsage
func (k msgServer) SubmitResourceUsage(ctx context.Context, msg *types.MsgSubmitResourceUsage) (*types.MsgSubmitResourceUsageResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Validate node address
	_, err := sdk.AccAddressFromBech32(msg.NodeAddress)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid node address")
	}

	// Resource usage is required (proto ensures it's not nil)

	// Validate proof IPFS hash
	if msg.ProofIpfsHash != "" {
		if err := k.VerifyIPFSContentExists(sdkCtx, msg.ProofIpfsHash); err != nil {
			return nil, err
		}
	}

	// Convert ResourceSpec to ResourceUsage
	actualUsage := ResourceUsage{
		CPUCores:         msg.ResourceUsage.CpuCores,
		MemoryGB:         msg.ResourceUsage.MemoryGb,
		GPUCount:         msg.ResourceUsage.GpuCount,
		GPUMemoryGB:      msg.ResourceUsage.GpuMemoryGb,
		StorageGB:        msg.ResourceUsage.StorageGb,
		NetworkBandwidth: msg.ResourceUsage.NetworkBandwidthMbps,
	}

	// Monitor resource usage and check for violations
	err = k.MonitorResourceUsage(ctx, msg.NodeAddress, actualUsage, msg.Role)
	violationDetected := err != nil

	// Validate resource usage
	isValid, _ := k.ValidateResourceUsage(ctx, msg.NodeAddress, actualUsage, msg.Role)

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeSubmitResourceUsage,
			sdk.NewAttribute(types.AttributeKeyNodeAddress, msg.NodeAddress),
			sdk.NewAttribute(types.AttributeKeyRole, msg.Role.String()),
			sdk.NewAttribute(types.AttributeKeyIsValid, func() string {
				if isValid {
					return "true"
				}
				return "false"
			}()),
		),
	)

	return &types.MsgSubmitResourceUsageResponse{
		IsValid:          isValid,
		ViolationDetected: violationDetected,
	}, nil
}

