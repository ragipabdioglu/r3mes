package keeper

import (
	"context"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// ProposeModelUpgrade handles MsgProposeModelUpgrade
func (k msgServer) ProposeModelUpgrade(ctx context.Context, msg *types.MsgProposeModelUpgrade) (*types.MsgProposeModelUpgradeResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Deposit is already in sdk.Coins format from proto
	deposit := msg.Deposit

	// Propose upgrade
	proposalID, err := k.Keeper.ProposeModelUpgrade(
		sdkCtx,
		msg.Proposer,
		msg.NewModelVersion,
		msg.NewModelHash,
		msg.NewModelId,
		msg.IpfsPath,
		msg.Architecture,
		msg.CompatibilityInfo,
		msg.MigrationWindow,
		deposit,
	)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to propose model upgrade")
	}

	return &types.MsgProposeModelUpgradeResponse{
		ProposalId: proposalID,
	}, nil
}

// VoteModelUpgrade handles MsgVoteModelUpgrade
func (k msgServer) VoteModelUpgrade(ctx context.Context, msg *types.MsgVoteModelUpgrade) (*types.MsgVoteModelUpgradeResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Vote on proposal
	if err := k.Keeper.VoteModelUpgrade(sdkCtx, msg.Voter, msg.ProposalId, msg.Vote); err != nil {
		return nil, errorsmod.Wrap(err, "failed to vote on model upgrade")
	}

	return &types.MsgVoteModelUpgradeResponse{}, nil
}

// ActivateModelUpgrade handles MsgActivateModelUpgrade
func (k msgServer) ActivateModelUpgrade(ctx context.Context, msg *types.MsgActivateModelUpgrade) (*types.MsgActivateModelUpgradeResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Verify authority
	authority := sdk.AccAddress(k.GetAuthority()).String()
	if msg.Authority != authority {
		return nil, errorsmod.Wrapf(
			types.ErrInvalidSigner,
			"expected %s, got %s",
			authority,
			msg.Authority,
		)
	}

	// Activate upgrade
	versionNumber, err := k.Keeper.ActivateModelUpgrade(sdkCtx, msg.ProposalId)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to activate model upgrade")
	}

	return &types.MsgActivateModelUpgradeResponse{
		VersionNumber: versionNumber,
	}, nil
}

// RollbackModelUpgrade handles MsgRollbackModelUpgrade
func (k msgServer) RollbackModelUpgrade(ctx context.Context, msg *types.MsgRollbackModelUpgrade) (*types.MsgRollbackModelUpgradeResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Verify authority
	authority := sdk.AccAddress(k.GetAuthority()).String()
	if msg.Authority != authority {
		return nil, errorsmod.Wrapf(
			types.ErrInvalidSigner,
			"expected %s, got %s",
			authority,
			msg.Authority,
		)
	}

	// Rollback upgrade
	if err := k.Keeper.RollbackModelUpgrade(sdkCtx, msg.VersionNumber, msg.Reason); err != nil {
		return nil, errorsmod.Wrap(err, "failed to rollback model upgrade")
	}

	return &types.MsgRollbackModelUpgradeResponse{}, nil
}

