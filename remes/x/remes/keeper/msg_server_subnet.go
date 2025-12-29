package keeper

import (
	"context"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// CreateSubnet handles MsgCreateSubnet
func (k msgServer) CreateSubnet(ctx context.Context, msg *types.MsgCreateSubnet) (*types.MsgCreateSubnetResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Validate authority
	authority := sdk.AccAddress(k.GetAuthority()).String()
	if msg.Authority != authority {
		return nil, errorsmod.Wrapf(
			types.ErrInvalidSigner,
			"expected %s, got %s",
			authority,
			msg.Authority,
		)
	}

	// Create subnet
	if err := k.Keeper.CreateSubnet(
		sdkCtx,
		msg.SubnetId,
		msg.LayerRange,
		msg.NextSubnetId,
		msg.WindowId,
	); err != nil {
		return nil, errorsmod.Wrap(err, "failed to create subnet")
	}

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeCreateSubnet,
			sdk.NewAttribute(types.AttributeKeySubnetID, fmt.Sprintf("%d", msg.SubnetId)),
			sdk.NewAttribute(types.AttributeKeyWindowID, fmt.Sprintf("%d", msg.WindowId)),
			sdk.NewAttribute(types.AttributeKeyLayerRange, fmt.Sprintf("%d-%d", msg.LayerRange.StartLayer, msg.LayerRange.EndLayer)),
		),
	)

	return &types.MsgCreateSubnetResponse{
		SubnetId: msg.SubnetId,
	}, nil
}

// SubmitSubnetActivation handles MsgSubmitSubnetActivation
func (k msgServer) SubmitSubnetActivation(ctx context.Context, msg *types.MsgSubmitSubnetActivation) (*types.MsgSubmitSubnetActivationResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Validate proposer address
	proposer, err := sdk.AccAddressFromBech32(msg.Proposer)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid proposer address")
	}

	// Submit activation
	transmissionID, err := k.Keeper.SubmitSubnetActivation(
		sdkCtx,
		msg.SubnetId,
		msg.ActivationHash,
		msg.NextSubnetId,
		msg.Signature,
	)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to submit subnet activation")
	}

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeSubmitSubnetActivation,
			sdk.NewAttribute(types.AttributeKeySubnetID, fmt.Sprintf("%d", msg.SubnetId)),
			sdk.NewAttribute(types.AttributeKeyNextSubnetID, fmt.Sprintf("%d", msg.NextSubnetId)),
			sdk.NewAttribute(types.AttributeKeyActivationHash, msg.ActivationHash),
			sdk.NewAttribute(types.AttributeKeyTransmissionID, fmt.Sprintf("%d", transmissionID)),
			sdk.NewAttribute(types.AttributeKeyProposer, proposer.String()),
		),
	)

	return &types.MsgSubmitSubnetActivationResponse{
		TransmissionId: transmissionID,
	}, nil
}

// AssignMinerToSubnet handles MsgAssignMinerToSubnet
func (k msgServer) AssignMinerToSubnet(ctx context.Context, msg *types.MsgAssignMinerToSubnet) (*types.MsgAssignMinerToSubnetResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Validate miner address
	miner, err := sdk.AccAddressFromBech32(msg.Miner)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid miner address")
	}

	// Get subnet config to retrieve layer range
	subnetConfig, err := k.Keeper.GetSubnetConfig(sdkCtx, msg.SubnetId)
	if err != nil {
		return nil, errorsmod.Wrap(err, "subnet not found")
	}

	// Add miner to subnet
	if err := k.Keeper.AddMinerToSubnet(sdkCtx, msg.SubnetId, msg.Miner); err != nil {
		return nil, errorsmod.Wrap(err, "failed to assign miner to subnet")
	}

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeAssignMinerToSubnet,
			sdk.NewAttribute(types.AttributeKeySubnetID, fmt.Sprintf("%d", msg.SubnetId)),
			sdk.NewAttribute(types.AttributeKeyMiner, miner.String()),
			sdk.NewAttribute(types.AttributeKeyWindowID, fmt.Sprintf("%d", msg.WindowId)),
		),
	)

	return &types.MsgAssignMinerToSubnetResponse{
		SubnetId:    msg.SubnetId,
		LayerRange:  subnetConfig.LayerRange,
	}, nil
}

