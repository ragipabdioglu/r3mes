package keeper

import (
	"context"
	"fmt"
	"time"

	sdk "github.com/cosmos/cosmos-sdk/types"
	sdkmath "cosmossdk.io/math"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// ProposeExecutionEnvironment handles MsgProposeExecutionEnvironment
func (k msgServer) ProposeExecutionEnvironment(ctx context.Context, msg *types.MsgProposeExecutionEnvironment) (*types.MsgProposeExecutionEnvironmentResponse, error) {
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

	// Validate environment specification
	if msg.Environment.EnvironmentId == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "environment ID is required")
	}

	if msg.Environment.Platform == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "platform is required")
	}

	// Validate platform-specific requirements
	if msg.Environment.Platform == "nvidia" {
		if msg.Environment.CudaVersion == "" {
			return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "CUDA version is required for NVIDIA platform")
		}
		if msg.Environment.PytorchVersion == "" {
			return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "PyTorch version is required")
		}
	} else if msg.Environment.Platform == "amd" {
		if msg.Environment.RocmVersion == "" {
			return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "ROCm version is required for AMD platform")
		}
	} else if msg.Environment.Platform == "intel" {
		if msg.Environment.IntelXpuVersion == "" {
			return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "Intel XPU version is required for Intel platform")
		}
	}

	// Cross-platform compatibility validation
	if err := k.ValidateCrossPlatformCompatibility(sdkCtx, &msg.Environment); err != nil {
		return nil, err
	}

	// Check if environment ID already exists in approved environments
	_, err = k.ApprovedExecutionEnvironments.Get(ctx, msg.Environment.EnvironmentId)
	if err == nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "environment with ID %s already exists", msg.Environment.EnvironmentId)
	}

	// Get next proposal ID
	proposalID, err := k.ExecutionEnvironmentProposalID.Next(ctx)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to generate proposal ID")
	}

	// Create proposal
	votingPeriod := 7 * 24 * time.Hour // Default 7 days
	proposal := types.ExecutionEnvironmentProposal{
		ProposalId:      proposalID,
		Proposer:        msg.Proposer,
		Environment:     msg.Environment,
		Deposit:         msg.Deposit,
		VotingPeriodEnd: sdkCtx.BlockTime().Add(votingPeriod),
		Status:          "voting",
		SubmitTime:      sdkCtx.BlockTime(),
		TotalVotes:      "0",
		YesVotes:        "0",
		NoVotes:         "0",
		AbstainVotes:    "0",
	}

	// Store proposal
	if err := k.ExecutionEnvironmentProposals.Set(ctx, proposalID, proposal); err != nil {
		return nil, errorsmod.Wrap(err, "failed to store execution environment proposal")
	}

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeProposeDataset, // Reuse event type for now
			sdk.NewAttribute(types.AttributeKeyProposer, msg.Proposer),
			sdk.NewAttribute("proposal_id", fmt.Sprintf("%d", proposalID)),
			sdk.NewAttribute("environment_id", msg.Environment.EnvironmentId),
			sdk.NewAttribute("platform", msg.Environment.Platform),
		),
	)

	return &types.MsgProposeExecutionEnvironmentResponse{
		ProposalId: proposalID,
	}, nil
}

// VoteExecutionEnvironment handles MsgVoteExecutionEnvironment
func (k msgServer) VoteExecutionEnvironment(ctx context.Context, msg *types.MsgVoteExecutionEnvironment) (*types.MsgVoteExecutionEnvironmentResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Validate voter address
	_, err := sdk.AccAddressFromBech32(msg.Voter)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid voter address")
	}

	// Get proposal
	proposal, err := k.ExecutionEnvironmentProposals.Get(ctx, msg.ProposalId)
	if err != nil {
		return nil, errorsmod.Wrap(err, "execution environment proposal not found")
	}

	// Check voting period
	if sdkCtx.BlockTime().After(proposal.VotingPeriodEnd) {
		return nil, errorsmod.Wrapf(types.ErrInvalidVoteOption, "voting period has ended")
	}

	// Validate vote option
	validOptions := map[string]bool{"yes": true, "no": true, "abstain": true}
	if !validOptions[msg.VoteOption] {
		return nil, errorsmod.Wrapf(types.ErrInvalidVoteOption, "invalid vote option")
	}

	// Get voter's voting power (stake)
	voter, err := sdk.AccAddressFromBech32(msg.Voter)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid voter address")
	}
	
	// Get voter's balance as voting power
	voterBalance := k.bankKeeper.GetBalance(sdkCtx, voter, "remes") // Use remes token for voting
	if voterBalance.IsZero() {
		// Fallback to stake token if remes balance is zero
		voterBalance = k.bankKeeper.GetBalance(sdkCtx, voter, "stake")
	}
	votingPower := voterBalance.Amount.String()

	// Update proposal votes (using math.Int operations)
	totalVotesInt, _ := sdkmath.NewIntFromString(proposal.TotalVotes)
	votingPowerInt, _ := sdkmath.NewIntFromString(votingPower)
	totalVotesInt = totalVotesInt.Add(votingPowerInt)
	proposal.TotalVotes = totalVotesInt.String()

	switch msg.VoteOption {
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

	// Check if proposal should be finalized (simple majority for now)
	yesVotesInt, _ := sdkmath.NewIntFromString(proposal.YesVotes)
	noVotesInt, _ := sdkmath.NewIntFromString(proposal.NoVotes)
	if yesVotesInt.GT(noVotesInt) && totalVotesInt.GT(sdkmath.NewInt(1000000)) { // Simple threshold
		// Proposal passed - approve environment
		env := proposal.Environment
		env.IsActive = false // Not active until explicitly activated
		env.ApprovedAtHeight = sdkCtx.BlockHeight()
		env.ApprovedBy = proposal.Proposer
		env.ApprovedAt = sdkCtx.BlockTime()
		
		if err := k.ApprovedExecutionEnvironments.Set(ctx, env.EnvironmentId, env); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to approve execution environment: %v", err))
		}
		
		proposal.Status = "approved"
	}

	// Store updated proposal
	if err := k.ExecutionEnvironmentProposals.Set(ctx, msg.ProposalId, proposal); err != nil {
		return nil, errorsmod.Wrap(err, "failed to update execution environment proposal")
	}

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeVoteDataset, // Reuse event type for now
			sdk.NewAttribute("voter", msg.Voter),
			sdk.NewAttribute("proposal_id", fmt.Sprintf("%d", msg.ProposalId)),
			sdk.NewAttribute("vote_option", msg.VoteOption),
		),
	)

	return &types.MsgVoteExecutionEnvironmentResponse{
		VoteId: 0, // Placeholder
	}, nil
}

// ActivateExecutionEnvironment handles MsgActivateExecutionEnvironment
func (k msgServer) ActivateExecutionEnvironment(ctx context.Context, msg *types.MsgActivateExecutionEnvironment) (*types.MsgActivateExecutionEnvironmentResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Validate authority
	if sdk.AccAddress(k.GetAuthority()).String() != msg.Authority {
		return nil, errorsmod.Wrapf(types.ErrInvalidSigner, "unauthorized")
	}

	// Get approved environment
	env, err := k.ApprovedExecutionEnvironments.Get(ctx, msg.EnvironmentId)
	if err != nil {
		return nil, errorsmod.Wrap(err, "execution environment not found or not approved")
	}

	// Deactivate current active environment for the same platform (if any)
	err = k.ApprovedExecutionEnvironments.Walk(ctx, nil, func(key string, value types.ExecutionEnvironment) (stop bool, err error) {
		if value.Platform == env.Platform && value.IsActive && value.EnvironmentId != msg.EnvironmentId {
			value.IsActive = false
			if err := k.ApprovedExecutionEnvironments.Set(ctx, key, value); err != nil {
				sdkCtx.Logger().Error(fmt.Sprintf("Failed to deactivate old environment: %v", err))
			}
		}
		return false, nil
	})
	if err != nil {
		sdkCtx.Logger().Error(fmt.Sprintf("Error walking execution environments: %v", err))
	}

	// Activate new environment
	env.IsActive = true
	env.ApprovedAtHeight = sdkCtx.BlockHeight()
	env.ApprovedBy = msg.Authority
	env.ApprovedAt = sdkCtx.BlockTime()

	if err := k.ApprovedExecutionEnvironments.Set(ctx, msg.EnvironmentId, env); err != nil {
		return nil, errorsmod.Wrap(err, "failed to activate execution environment")
	}

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeMarkDatasetOfficial, // Reuse event type for now
			sdk.NewAttribute("authority", msg.Authority),
			sdk.NewAttribute("environment_id", msg.EnvironmentId),
			sdk.NewAttribute("platform", env.Platform),
		),
	)

	return &types.MsgActivateExecutionEnvironmentResponse{}, nil
}

// ValidateCrossPlatformCompatibility validates cross-platform compatibility
func (k Keeper) ValidateCrossPlatformCompatibility(ctx sdk.Context, env *types.ExecutionEnvironment) error {
	// Ensure PyTorch version is consistent across platforms
	if env.PytorchVersion == "" {
		return errorsmod.Wrapf(types.ErrInvalidMiner, "PyTorch version must be specified for all platforms")
	}

	// Ensure Python version is consistent
	if env.PythonVersion == "" {
		return errorsmod.Wrapf(types.ErrInvalidMiner, "Python version must be specified for all platforms")
	}

	// Ensure floating point mode is consistent
	if env.FloatingPointMode == "" {
		return errorsmod.Wrapf(types.ErrInvalidMiner, "floating point mode must be specified")
	}

	// Ensure deterministic algorithms are enabled
	if !env.DeterministicAlgorithmsEnabled {
		return errorsmod.Wrapf(types.ErrInvalidMiner, "deterministic algorithms must be enabled")
	}

	return nil
}

