package keeper

import (
	"context"
	"strconv"

	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkmath "cosmossdk.io/math"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// VoteDataset handles MsgVoteDataset
func (k msgServer) VoteDataset(ctx context.Context, msg *types.MsgVoteDataset) (*types.MsgVoteDatasetResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Validate voter address
	voter, err := sdk.AccAddressFromBech32(msg.Voter)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid voter address")
	}

	// Get proposal
	proposal, err := k.DatasetProposals.Get(sdkCtx, msg.ProposalId)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrProposalNotFound, "proposal not found")
	}

	// Check if proposal is still in voting period
	if sdkCtx.BlockTime().After(proposal.VotingPeriodEnd) {
		return nil, errorsmod.Wrapf(types.ErrProposalNotFound, "voting period has ended")
	}

	// Validate vote option
	if msg.Option != "yes" && msg.Option != "no" && msg.Option != "abstain" {
		return nil, errorsmod.Wrapf(types.ErrInvalidVoteOption, "invalid vote option (must be yes, no, or abstain)")
	}

	// Get voter's voting power using configured voting method
	// Default: stake-weighted, can be configured to quadratic or simple
	votingPowerInt, err := k.CalculateVotingPower(sdkCtx, voter, "stake_weighted")
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to calculate voting power")
	}

	// Update proposal votes (using math.Int operations)
	totalVotesInt, _ := sdkmath.NewIntFromString(proposal.TotalVotes)
	totalVotesInt = totalVotesInt.Add(votingPowerInt)
	proposal.TotalVotes = totalVotesInt.String()

	switch msg.Option {
	case "yes":
		yesVotesInt, _ := sdkmath.NewIntFromString(proposal.YesVotes)
		proposal.YesVotes = yesVotesInt.Add(votingPowerInt).String()
	case "no":
		noVotesInt, _ := sdkmath.NewIntFromString(proposal.NoVotes)
		proposal.NoVotes = noVotesInt.Add(votingPowerInt).String()
	case "abstain":
		abstainVotesInt, _ := sdkmath.NewIntFromString(proposal.AbstainVotes)
		proposal.AbstainVotes = abstainVotesInt.Add(votingPowerInt).String()
	}

	// Store updated proposal
	if err := k.DatasetProposals.Set(sdkCtx, msg.ProposalId, proposal); err != nil {
		return nil, err
	}

	// Proposal finalization is now handled in EndBlocker
	// This ensures all votes are counted before finalization

	// Generate unique vote ID
	voteID, err := k.DatasetVoteID.Next(ctx)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to get next vote ID")
	}

	// Record individual vote
	vote := types.DatasetVote{
		VoteId:        voteID,
		ProposalId:    msg.ProposalId,
		Voter:         msg.Voter,
		Option:        msg.Option,
		VotingPower:   votingPowerInt.String(),
		QualityScore:  msg.QualityScore,
		RelevanceScore: msg.RelevanceScore,
		VotedAtHeight: sdkCtx.BlockHeight(),
		VotedAtTime:   sdkCtx.BlockTime(),
	}

	if err := k.DatasetVotes.Set(ctx, voteID, vote); err != nil {
		return nil, errorsmod.Wrap(err, "failed to store vote")
	}

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeVoteDataset,
			sdk.NewAttribute(types.AttributeKeyVoter, msg.Voter),
			sdk.NewAttribute(types.AttributeKeyProposalID, strconv.FormatUint(msg.ProposalId, 10)),
			sdk.NewAttribute(types.AttributeKeyVoteOption, msg.Option),
			sdk.NewAttribute("vote_id", strconv.FormatUint(voteID, 10)),
		),
	)

	return &types.MsgVoteDatasetResponse{
		VoteId: voteID,
	}, nil
}

