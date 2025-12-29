package keeper

import (
	"context"
	"fmt"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// RegisterModel handles MsgRegisterModel
// Registers a new model configuration (governance-only)
func (k msgServer) RegisterModel(ctx context.Context, msg *types.MsgRegisterModel) (*types.MsgRegisterModelResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Verify authority (must be module authority or governance)
	authority := sdk.AccAddress(k.GetAuthority()).String()
	if msg.Authority != authority {
		return nil, errorsmod.Wrapf(
			types.ErrInvalidSigner,
			"expected %s, got %s",
			authority,
			msg.Authority,
		)
	}

	// 2. Validate model config
	if msg.Config.ModelType == types.ModelType_MODEL_TYPE_UNSPECIFIED {
		return nil, errorsmod.Wrapf(types.ErrInvalidModel, "model type cannot be unspecified")
	}

	if msg.Config.ModelVersion == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidModel, "model version cannot be empty")
	}

	if msg.Config.ArchitectureConfig == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidModel, "architecture config cannot be empty")
	}

	// 3. Register model using keeper function
	modelID, err := k.Keeper.RegisterModel(sdkCtx, msg.Config, msg.ProposalId)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to register model")
	}

	// 4. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeRegisterModel,
			sdk.NewAttribute(types.AttributeKeyModelID, fmt.Sprintf("%d", modelID)),
			sdk.NewAttribute(types.AttributeKeyModelType, msg.Config.ModelType.String()),
			sdk.NewAttribute(types.AttributeKeyModelVersion, msg.Config.ModelVersion),
			sdk.NewAttribute(types.AttributeKeyProposalID, fmt.Sprintf("%d", msg.ProposalId)),
		),
	)

	return &types.MsgRegisterModelResponse{
		ModelId: modelID,
	}, nil
}

// ActivateModel handles MsgActivateModel
// Activates a model for training (governance-only)
func (k msgServer) ActivateModel(ctx context.Context, msg *types.MsgActivateModel) (*types.MsgActivateModelResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Verify authority (must be module authority or governance)
	authority := sdk.AccAddress(k.GetAuthority()).String()
	if msg.Authority != authority {
		return nil, errorsmod.Wrapf(
			types.ErrInvalidSigner,
			"expected %s, got %s",
			authority,
			msg.Authority,
		)
	}

	// 2. Verify model exists
	registry, err := k.Keeper.GetModelConfig(sdkCtx, msg.ModelId)
	if err != nil {
		return nil, errorsmod.Wrapf(err, "model %d not found", msg.ModelId)
	}

	// 3. Activate model using keeper function
	if err := k.Keeper.ActivateModel(sdkCtx, msg.ModelId); err != nil {
		return nil, errorsmod.Wrap(err, "failed to activate model")
	}

	// 4. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeActivateModel,
			sdk.NewAttribute(types.AttributeKeyModelID, fmt.Sprintf("%d", msg.ModelId)),
			sdk.NewAttribute(types.AttributeKeyModelType, registry.Config.ModelType.String()),
			sdk.NewAttribute(types.AttributeKeyModelVersion, registry.Config.ModelVersion),
		),
	)

	return &types.MsgActivateModelResponse{}, nil
}

