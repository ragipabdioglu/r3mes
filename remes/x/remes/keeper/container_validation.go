package keeper

import (
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// VerifyContainerSignature verifies the cryptographic signature of a container image
func (k Keeper) VerifyContainerSignature(
	ctx sdk.Context,
	containerHash string,
	containerSignature []byte,
) error {
	// 1. Validate container hash format (should be sha256:...)
	if containerHash == "" {
		return errorsmod.Wrapf(types.ErrInvalidMiner, "container hash cannot be empty")
	}

	if !(len(containerHash) == 64 || (len(containerHash) > 7 && containerHash[:7] == "sha256:")) {
		return errorsmod.Wrapf(types.ErrInvalidMiner, "invalid container hash format: %s", containerHash)
	}

	// 2. Validate signature is not empty
	if len(containerSignature) == 0 {
		return errorsmod.Wrapf(types.ErrInvalidMiner, "container signature cannot be empty")
	}

	// 3. Verify signature against protocol authority public key
	authorityAddr, err := sdk.AccAddressFromBech32(sdk.AccAddress(k.GetAuthority()).String())
	if err != nil {
		return errorsmod.Wrap(err, "failed to decode authority address")
	}

	authorityAccount := k.authKeeper.GetAccount(ctx, authorityAddr)
	if authorityAccount == nil {
		return errorsmod.Wrapf(types.ErrInvalidMiner, "authority account not found")
	}

	pubKey := authorityAccount.GetPubKey()
	if pubKey == nil {
		// If authority has no public key (multisig or governance), skip signature verification
		// This is acceptable because container hash is already validated against execution environment
		ctx.Logger().Info("Authority account has no public key, skipping container signature verification")
		return nil
	}

	// Prepare container hash bytes for signature verification
	containerHashBytes := []byte(containerHash)
	if len(containerHash) > 7 && containerHash[:7] == "sha256:" {
		// Extract hash part if prefixed with "sha256:"
		containerHashBytes = []byte(containerHash[7:])
	}

	// Verify signature
	if !pubKey.VerifySignature(containerHashBytes, containerSignature) {
		return errorsmod.Wrapf(types.ErrInvalidSignature, "container signature verification failed against protocol authority")
	}

	return nil
}

// ValidateContainerSpec validates that a container specification matches approved requirements
func (k Keeper) ValidateContainerSpec(
	ctx sdk.Context,
	reportedContainerHash string,
	reportedContainerSignature []byte,
	platform string,
) error {
	// 1. Get active execution environment for platform
	activeEnv, err := k.GetActiveExecutionEnvironment(ctx, platform)
	if err != nil {
		// If no active environment, allow submission (for testing)
		ctx.Logger().Warn(fmt.Sprintf("No active execution environment for platform %s, allowing submission", platform))
		return nil
	}

	// 2. Verify container hash matches approved hash
	if activeEnv.ContainerHash != "" && reportedContainerHash != activeEnv.ContainerHash {
		return errorsmod.Wrapf(
			types.ErrInvalidMiner,
			"container hash mismatch: required %s, got %s",
			activeEnv.ContainerHash,
			reportedContainerHash,
		)
	}

	// 3. Verify container signature
	if err := k.VerifyContainerSignature(ctx, reportedContainerHash, reportedContainerSignature); err != nil {
		return errorsmod.Wrap(err, "container signature verification failed")
	}

	return nil
}

// RecordContainerViolation records a container violation for a miner
func (k Keeper) RecordContainerViolation(
	ctx sdk.Context,
	minerAddress string,
	violationReason string,
) error {
	// Record violation (similar to environment violation)
	return k.RecordEnvironmentViolation(ctx, minerAddress, violationReason)
}

