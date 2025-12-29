package keeper

import (
	"context"
	"time"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// ProposeDataset handles MsgProposeDataset
func (k msgServer) ProposeDataset(ctx context.Context, msg *types.MsgProposeDataset) (*types.MsgProposeDatasetResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Validate proposer address
	_, err := sdk.AccAddressFromBech32(msg.Proposer)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid proposer address")
	}

	// Validate deposit
	_, err = sdk.ParseCoinsNormalized(msg.Deposit)
	if err != nil {
		return nil, errorsmod.Wrap(err, "invalid deposit")
	}

	// Validate IPFS hash
	if msg.DatasetIpfsHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "dataset IPFS hash is required")
	}

	// Validate dataset metadata
	if err := k.ValidateDatasetMetadata(msg.Metadata); err != nil {
		return nil, errorsmod.Wrap(err, "invalid dataset metadata")
	}

	// Verify IPFS content exists
	if err := k.VerifyIPFSContentExists(sdkCtx, msg.DatasetIpfsHash); err != nil {
		return nil, err
	}

	// Get next proposal ID
	proposalID, err := k.DatasetProposalID.Next(sdkCtx)
	if err != nil {
		return nil, err
	}

	// Get voting period (default: 7 days)
	votingPeriod := 7 * 24 * time.Hour // Default 7 days

	// Create proposal
	proposal := types.DatasetProposal{
		ProposalId:      proposalID,
		Proposer:        msg.Proposer,
		DatasetIpfsHash: msg.DatasetIpfsHash,
		Metadata:        msg.Metadata,
		Deposit:          msg.Deposit,
		VotingPeriodEnd: sdkCtx.BlockTime().Add(votingPeriod),
		Status:          "voting",
		SubmitTime:      sdkCtx.BlockTime(),
		TotalVotes:      "0",
		YesVotes:        "0",
		NoVotes:         "0",
		AbstainVotes:    "0",
	}

	// Store proposal
	if err := k.DatasetProposals.Set(sdkCtx, proposalID, proposal); err != nil {
		return nil, err
	}

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeProposeDataset,
			sdk.NewAttribute(types.AttributeKeyProposer, msg.Proposer),
			sdk.NewAttribute(types.AttributeKeyProposalID, string(rune(proposalID))),
			sdk.NewAttribute(types.AttributeKeyDatasetIPFSHash, msg.DatasetIpfsHash),
		),
	)

	return &types.MsgProposeDatasetResponse{
		ProposalId: proposalID,
	}, nil
}

