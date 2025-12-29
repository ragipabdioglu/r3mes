package keeper

import (
	"encoding/json"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// ProposeModelUpgrade creates a governance proposal for model upgrade
func (k Keeper) ProposeModelUpgrade(
	ctx sdk.Context,
	proposer string,
	newModelVersion uint64,
	newModelHash string,
	newModelID uint64,
	ipfsPath string,
	architecture string,
	compatibilityInfo string,
	migrationWindow int64,
	deposit sdk.Coins,
) (uint64, error) {
	// 1. Validate proposer address
	proposerAddr, err := k.addressCodec.StringToBytes(proposer)
	if err != nil {
		return 0, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid proposer address: %s", proposer)
	}
	_ = proposerAddr

	// 2. Validate new model exists
	_, err = k.ModelRegistries.Get(ctx, newModelID)
	if err != nil {
		return 0, errorsmod.Wrapf(err, "model %d not found", newModelID)
	}

	// 3. Validate version number doesn't exist
	_, err = k.ModelVersions.Get(ctx, newModelVersion)
	if err == nil {
		return 0, errorsmod.Wrapf(types.ErrInvalidModel, "model version %d already exists", newModelVersion)
	}

	// 4. Validate IPFS hash format
	if newModelHash == "" {
		return 0, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "model hash cannot be empty")
	}

	// 5. Validate migration window
	if migrationWindow <= 0 {
		return 0, errorsmod.Wrapf(types.ErrInvalidModel, "migration window must be positive")
	}

	// 6. Generate proposal ID
	proposalID, err := k.ModelUpgradeProposalID.Next(ctx)
	if err != nil {
		return 0, errorsmod.Wrap(err, "failed to generate proposal ID")
	}

	// 7. Create proposal
	votingPeriod := types.VotingPeriod{
		StartHeight: ctx.BlockHeight(),
		EndHeight:   ctx.BlockHeight() + 10000, // 7 days voting period (assuming 5s/block)
	}

	proposal := types.ModelUpgradeProposal{
		ProposalId:       proposalID,
		Proposer:         proposer,
		NewModelVersion:  newModelVersion,
		NewModelHash:     newModelHash,
		NewModelId:       newModelID,
		IpfsPath:         ipfsPath,
		Architecture:     architecture,
		CompatibilityInfo: compatibilityInfo,
		MigrationWindow:  migrationWindow,
		Status:           "voting",
		VotingPeriod:     votingPeriod,
		Deposit:          deposit.String(),
		CreatedAt:        ctx.BlockTime(),
	}

	// 8. Store proposal
	if err := k.ModelUpgradeProposals.Set(ctx, proposalID, proposal); err != nil {
		return 0, errorsmod.Wrap(err, "failed to store proposal")
	}

	// 9. Emit event
	ctx.EventManager().EmitEvent(
		sdk.NewEvent(
			"model_upgrade_proposed",
			sdk.NewAttribute("proposal_id", fmt.Sprintf("%d", proposalID)),
			sdk.NewAttribute("proposer", proposer),
			sdk.NewAttribute("new_version", fmt.Sprintf("%d", newModelVersion)),
			sdk.NewAttribute("model_hash", newModelHash),
		),
	)

	return proposalID, nil
}

// VoteModelUpgrade records a vote on a model upgrade proposal
func (k Keeper) VoteModelUpgrade(
	ctx sdk.Context,
	voter string,
	proposalID uint64,
	vote string,
) error {
	// 1. Validate voter address
	voterAddr, err := k.addressCodec.StringToBytes(voter)
	if err != nil {
		return errorsmod.Wrapf(types.ErrInvalidMiner, "invalid voter address: %s", voter)
	}
	_ = voterAddr

	// 2. Get proposal
	proposal, err := k.ModelUpgradeProposals.Get(ctx, proposalID)
	if err != nil {
		return errorsmod.Wrapf(err, "proposal %d not found", proposalID)
	}

	// 3. Validate proposal status
	if proposal.Status != "voting" {
		return errorsmod.Wrapf(types.ErrInvalidModel, "proposal %d is not in voting status (current: %s)", proposalID, proposal.Status)
	}

	// 4. Validate voting period
	if ctx.BlockHeight() < proposal.VotingPeriod.StartHeight || ctx.BlockHeight() > proposal.VotingPeriod.EndHeight {
		return errorsmod.Wrapf(types.ErrInvalidModel, "voting period has ended for proposal %d", proposalID)
	}

	// 5. Validate vote choice
	if vote != "yes" && vote != "no" && vote != "abstain" {
		return errorsmod.Wrapf(types.ErrInvalidModel, "invalid vote choice: %s (must be yes, no, or abstain)", vote)
	}

	// 6. Calculate voting power
	voterAcc, err := sdk.AccAddressFromBech32(voter)
	if err != nil {
		return errorsmod.Wrapf(types.ErrInvalidMiner, "invalid voter address: %s", voter)
	}
	votingPowerInt, err := k.CalculateVotingPower(ctx, voterAcc, "stake_weighted")
	if err != nil {
		return errorsmod.Wrap(err, "failed to calculate voting power")
	}

	// 7. Generate unique vote ID
	voteID, err := k.ModelUpgradeVoteID.Next(ctx)
	if err != nil {
		return errorsmod.Wrap(err, "failed to get next vote ID")
	}

	// 8. Record vote
	modelVote := types.ModelUpgradeVote{
		VoteId:        voteID,
		ProposalId:    proposalID,
		Voter:         voter,
		Vote:          vote,
		VotingPower:   votingPowerInt.String(),
		VotedAtHeight: ctx.BlockHeight(),
		VotedAtTime:   ctx.BlockTime(),
	}

	if err := k.ModelUpgradeVotes.Set(ctx, voteID, modelVote); err != nil {
		return errorsmod.Wrap(err, "failed to store vote")
	}

	// 9. Emit event
	ctx.EventManager().EmitEvent(
		sdk.NewEvent(
			"model_upgrade_voted",
			sdk.NewAttribute("proposal_id", fmt.Sprintf("%d", proposalID)),
			sdk.NewAttribute("voter", voter),
			sdk.NewAttribute("vote", vote),
			sdk.NewAttribute("vote_id", fmt.Sprintf("%d", voteID)),
		),
	)

	return nil
}

// ActivateModelUpgrade activates an approved model upgrade
// This function also triggers serving node model synchronization
func (k Keeper) ActivateModelUpgrade(
	ctx sdk.Context,
	proposalID uint64,
) (uint64, error) {
	// 1. Get proposal
	proposal, err := k.ModelUpgradeProposals.Get(ctx, proposalID)
	if err != nil {
		return 0, errorsmod.Wrapf(err, "proposal %d not found", proposalID)
	}

	// 2. Validate proposal status
	if proposal.Status != "approved" {
		return 0, errorsmod.Wrapf(types.ErrInvalidModel, "proposal %d is not approved (current: %s)", proposalID, proposal.Status)
	}

	// 3. Create new model version
	newVersion := types.ModelVersion{
		VersionNumber:        proposal.NewModelVersion,
		ModelId:              proposal.NewModelId,
		ModelHash:            proposal.NewModelHash,
		IpfsPath:             proposal.IpfsPath,
		Architecture:         proposal.Architecture,
		CompatibilityInfo:    proposal.CompatibilityInfo,
		Status:               "migration", // Migration window active
		ActivationHeight:     ctx.BlockHeight(),
		MigrationWindowStart: ctx.BlockHeight(),
		MigrationWindowEnd:   ctx.BlockHeight() + proposal.MigrationWindow,
		GovernanceProposalId:  proposalID,
		CreatedAt:            ctx.BlockTime(),
	}

	// 4. Store model version
	if err := k.ModelVersions.Set(ctx, newVersion.VersionNumber, newVersion); err != nil {
		return 0, errorsmod.Wrap(err, "failed to store model version")
	}

	// 5. Update active versions (dual model support during migration)
	activeVersions, err := k.ActiveModelVersions.Get(ctx)
	if err != nil {
		// Initialize if not exists
		activeVersions = types.ActiveModelVersions{
			VersionNumbers:        []uint64{},
			MigrationWindowActive: true,
			PrimaryVersion:        0,
		}
	}

	// Add new version to active versions
	activeVersions.VersionNumbers = append(activeVersions.VersionNumbers, newVersion.VersionNumber)
	activeVersions.MigrationWindowActive = true

	// Set primary version if this is the first version
	if activeVersions.PrimaryVersion == 0 {
		activeVersions.PrimaryVersion = newVersion.VersionNumber
	}

	// 6. Store active versions
	if err := k.ActiveModelVersions.Set(ctx, activeVersions); err != nil {
		return 0, errorsmod.Wrap(err, "failed to store active versions")
	}

	// 7. Update proposal status
	proposal.Status = "executed"
	if err := k.ModelUpgradeProposals.Set(ctx, proposalID, proposal); err != nil {
		return 0, errorsmod.Wrap(err, "failed to update proposal status")
	}

	// 8. Notify serving nodes of model update
	if err := k.NotifyServingNodesOfModelUpdate(ctx, newVersion.VersionNumber); err != nil {
		ctx.Logger().Error(fmt.Sprintf("Failed to notify serving nodes: %v", err))
		// Don't fail - model activation should succeed even if sync notification fails
	}

	// 9. Emit event
	ctx.EventManager().EmitEvent(
		sdk.NewEvent(
			"model_upgrade_activated",
			sdk.NewAttribute("proposal_id", fmt.Sprintf("%d", proposalID)),
			sdk.NewAttribute("version_number", fmt.Sprintf("%d", newVersion.VersionNumber)),
			sdk.NewAttribute("migration_window_start", fmt.Sprintf("%d", newVersion.MigrationWindowStart)),
			sdk.NewAttribute("migration_window_end", fmt.Sprintf("%d", newVersion.MigrationWindowEnd)),
		),
	)

	return newVersion.VersionNumber, nil
}

// RollbackModelUpgrade rolls back a failed model upgrade
func (k Keeper) RollbackModelUpgrade(
	ctx sdk.Context,
	versionNumber uint64,
	reason string,
) error {
	// 1. Get model version
	version, err := k.ModelVersions.Get(ctx, versionNumber)
	if err != nil {
		return errorsmod.Wrapf(err, "model version %d not found", versionNumber)
	}

	// 2. Update version status
	version.Status = "rollback"
	if err := k.ModelVersions.Set(ctx, versionNumber, version); err != nil {
		return errorsmod.Wrap(err, "failed to update version status")
	}

	// 3. Remove from active versions
	activeVersions, err := k.ActiveModelVersions.Get(ctx)
	if err == nil {
		// Remove version from active list
		newVersions := make([]uint64, 0)
		for _, v := range activeVersions.VersionNumbers {
			if v != versionNumber {
				newVersions = append(newVersions, v)
			}
		}
		activeVersions.VersionNumbers = newVersions

		// If no versions left, reset primary
		if len(activeVersions.VersionNumbers) == 0 {
			activeVersions.PrimaryVersion = 0
			activeVersions.MigrationWindowActive = false
		}

		if err := k.ActiveModelVersions.Set(ctx, activeVersions); err != nil {
			return errorsmod.Wrap(err, "failed to update active versions")
		}
	}

	// 4. Emit event
	ctx.EventManager().EmitEvent(
		sdk.NewEvent(
			"model_upgrade_rollback",
			sdk.NewAttribute("version_number", fmt.Sprintf("%d", versionNumber)),
			sdk.NewAttribute("reason", reason),
		),
	)

	return nil
}

// GetModelVersion retrieves a model version by version number
func (k Keeper) GetModelVersion(ctx sdk.Context, versionNumber uint64) (types.ModelVersion, error) {
	version, err := k.ModelVersions.Get(ctx, versionNumber)
	if err != nil {
		return types.ModelVersion{}, errorsmod.Wrapf(err, "model version %d not found", versionNumber)
	}
	return version, nil
}

// ListModelVersions lists all model versions
func (k Keeper) ListModelVersions(ctx sdk.Context) ([]types.ModelVersion, error) {
	versions := make([]types.ModelVersion, 0)
	err := k.ModelVersions.Walk(ctx, nil, func(key uint64, value types.ModelVersion) (stop bool, err error) {
		versions = append(versions, value)
		return false, nil
	})
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to iterate model versions")
	}
	return versions, nil
}

// GetActiveModelVersions retrieves active model versions
func (k Keeper) GetActiveModelVersions(ctx sdk.Context) (types.ActiveModelVersions, error) {
	activeVersions, err := k.ActiveModelVersions.Get(ctx)
	if err != nil {
		// Return default if not exists
		return types.ActiveModelVersions{
			VersionNumbers:        []uint64{1}, // Default to version 1
			MigrationWindowActive: false,
			PrimaryVersion:        1,
		}, nil
	}
	return activeVersions, nil
}

// ValidateModelVersion validates that a model version is active and compatible
func (k Keeper) ValidateModelVersion(ctx sdk.Context, versionNumber uint64) error {
	// 1. Get model version
	version, err := k.ModelVersions.Get(ctx, versionNumber)
	if err != nil {
		return errorsmod.Wrapf(err, "model version %d not found", versionNumber)
	}

	// 2. Check if version is active or in migration
	if version.Status != "active" && version.Status != "migration" {
		return errorsmod.Wrapf(types.ErrInvalidModel, "model version %d is not active (status: %s)", versionNumber, version.Status)
	}

	// 3. Check if migration window has expired
	if version.Status == "migration" && ctx.BlockHeight() > version.MigrationWindowEnd {
		// Migration window expired, mark as active
		version.Status = "active"
		if err := k.ModelVersions.Set(ctx, versionNumber, version); err != nil {
			return errorsmod.Wrap(err, "failed to update version status")
		}
	}

	return nil
}

// ProcessMigrationWindow processes migration window completion
func (k Keeper) ProcessMigrationWindow(ctx sdk.Context) error {
	// Get all model versions in migration status
	err := k.ModelVersions.Walk(ctx, nil, func(key uint64, value types.ModelVersion) (stop bool, err error) {
		if value.Status == "migration" && ctx.BlockHeight() >= value.MigrationWindowEnd {
			// Migration window completed, mark as active
			value.Status = "active"
			if err := k.ModelVersions.Set(ctx, key, value); err != nil {
				return false, errorsmod.Wrap(err, "failed to update version status")
			}

			// Update active versions
			activeVersions, err := k.ActiveModelVersions.Get(ctx)
			if err == nil {
				// If migration window ended, set primary version and disable migration
				if len(activeVersions.VersionNumbers) > 0 {
					activeVersions.PrimaryVersion = activeVersions.VersionNumbers[len(activeVersions.VersionNumbers)-1]
					activeVersions.MigrationWindowActive = false
					if err := k.ActiveModelVersions.Set(ctx, activeVersions); err != nil {
						return false, errorsmod.Wrap(err, "failed to update active versions")
					}
				}
			}

			// Emit event
			ctx.EventManager().EmitEvent(
				sdk.NewEvent(
					"model_migration_completed",
					sdk.NewAttribute("version_number", fmt.Sprintf("%d", key)),
				),
			)
		}
		return false, nil
	})
	return err
}

// CheckModelCompatibility checks if two model versions are compatible
func (k Keeper) CheckModelCompatibility(ctx sdk.Context, version1, version2 uint64) (bool, error) {
	v1, err := k.ModelVersions.Get(ctx, version1)
	if err != nil {
		return false, errorsmod.Wrapf(err, "model version %d not found", version1)
	}

	v2, err := k.ModelVersions.Get(ctx, version2)
	if err != nil {
		return false, errorsmod.Wrapf(err, "model version %d not found", version2)
	}

	// Parse compatibility info
	var compat1 map[string]interface{}
	if err := json.Unmarshal([]byte(v1.CompatibilityInfo), &compat1); err != nil {
		return false, errorsmod.Wrap(err, "failed to parse compatibility info for version 1")
	}

	var compat2 map[string]interface{}
	if err := json.Unmarshal([]byte(v2.CompatibilityInfo), &compat2); err != nil {
		return false, errorsmod.Wrap(err, "failed to parse compatibility info for version 2")
	}

	// Check if versions are compatible
	// Simple check: if compatible_with array contains the other version
	if compatibleWith, ok := compat1["compatible_with"].([]interface{}); ok {
		for _, v := range compatibleWith {
			if v == float64(version2) {
				return true, nil
			}
		}
	}

	if compatibleWith, ok := compat2["compatible_with"].([]interface{}); ok {
		for _, v := range compatibleWith {
			if v == float64(version1) {
				return true, nil
			}
		}
	}

	return false, nil
}

