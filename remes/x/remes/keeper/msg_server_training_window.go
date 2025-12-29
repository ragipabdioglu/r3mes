package keeper

import (
	"context"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// CreateTrainingWindow handles MsgCreateTrainingWindow
func (k msgServer) CreateTrainingWindow(ctx context.Context, msg *types.MsgCreateTrainingWindow) (*types.MsgCreateTrainingWindowResponse, error) {
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

	// Create training window
	if err := k.Keeper.CreateTrainingWindow(
		sdkCtx,
		msg.WindowId,
		msg.StartHeight,
		msg.AggregatorNode,
	); err != nil {
		return nil, errorsmod.Wrap(err, "failed to create training window")
	}

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeCreateTrainingWindow,
			sdk.NewAttribute(types.AttributeKeyWindowID, fmt.Sprintf("%d", msg.WindowId)),
			sdk.NewAttribute(types.AttributeKeyStartHeight, fmt.Sprintf("%d", msg.StartHeight)),
			sdk.NewAttribute(types.AttributeKeyAggregatorNode, msg.AggregatorNode),
		),
	)

	return &types.MsgCreateTrainingWindowResponse{
		WindowId: msg.WindowId,
	}, nil
}

// SubmitAsyncGradient handles MsgSubmitAsyncGradient
func (k msgServer) SubmitAsyncGradient(ctx context.Context, msg *types.MsgSubmitAsyncGradient) (*types.MsgSubmitAsyncGradientResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Validate miner address
	miner, err := sdk.AccAddressFromBech32(msg.Miner)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid miner address")
	}

	// Submit async gradient
	submissionID, err := k.Keeper.SubmitAsyncGradient(
		sdkCtx,
		msg.WindowId,
		msg.Miner,
		msg.GradientHash,
		msg.SubnetId,
		msg.LayerRange,
	)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to submit async gradient")
	}

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeSubmitAsyncGradient,
			sdk.NewAttribute(types.AttributeKeyWindowID, fmt.Sprintf("%d", msg.WindowId)),
			sdk.NewAttribute(types.AttributeKeyMiner, miner.String()),
			sdk.NewAttribute(types.AttributeKeyIPFSHash, msg.GradientHash),
			sdk.NewAttribute(types.AttributeKeySubnetID, fmt.Sprintf("%d", msg.SubnetId)),
			sdk.NewAttribute(types.AttributeKeySubmissionID, fmt.Sprintf("%d", submissionID)),
		),
	)

	return &types.MsgSubmitAsyncGradientResponse{
		SubmissionId: submissionID,
	}, nil
}

// SubmitLazyAggregation handles MsgSubmitLazyAggregation
func (k msgServer) SubmitLazyAggregation(ctx context.Context, msg *types.MsgSubmitLazyAggregation) (*types.MsgSubmitLazyAggregationResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Validate aggregator address
	aggregator, err := sdk.AccAddressFromBech32(msg.Aggregator)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid aggregator address")
	}

	// Submit lazy aggregation
	aggregationID, err := k.Keeper.SubmitLazyAggregation(
		sdkCtx,
		msg.WindowId,
		msg.Aggregator,
		msg.CollectedGradients,
		msg.AggregationMethod,
		msg.ResultHash,
		msg.MerkleRoot,
	)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to submit lazy aggregation")
	}

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeSubmitLazyAggregation,
			sdk.NewAttribute(types.AttributeKeyWindowID, fmt.Sprintf("%d", msg.WindowId)),
			sdk.NewAttribute(types.AttributeKeyAggregatorNode, aggregator.String()),
			sdk.NewAttribute(types.AttributeKeyAggregationID, fmt.Sprintf("%d", aggregationID)),
			sdk.NewAttribute(types.AttributeKeyIPFSHash, msg.ResultHash),
		),
	)

	return &types.MsgSubmitLazyAggregationResponse{
		AggregationId: aggregationID,
	}, nil
}

