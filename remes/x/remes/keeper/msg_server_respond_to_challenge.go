package keeper

import (
	"context"
	"fmt"
	"strings"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"
	"remes/x/remes/types"
)

// RespondToChallenge handles MsgRespondToChallenge
// Responds to a data availability challenge by providing proof that the content is still accessible
func (k msgServer) RespondToChallenge(ctx context.Context, msg *types.MsgRespondToChallenge) (*types.MsgRespondToChallengeResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate node address
	nodeAddr, err := k.addressCodec.StringToBytes(msg.NodeAddress)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid node address: %s", msg.NodeAddress)
	}
	_ = nodeAddr // Address validated

	// 2. Verify challenge exists
	challenge, err := k.DataAvailabilityChallenges.Get(ctx, msg.ChallengeId)
	if err != nil {
		return nil, errorsmod.Wrapf(err, "challenge %d not found", msg.ChallengeId)
	}

	// 3. Verify challenge is still pending
	if challenge.Status != "pending" {
		return nil, errorsmod.Wrapf(
			types.ErrInvalidChallenge,
			"challenge %d is not pending (status: %s)",
			msg.ChallengeId,
			challenge.Status,
		)
	}

	// 4. Verify the node address matches the challenged node
	if challenge.NodeAddress != msg.NodeAddress {
		return nil, errorsmod.Wrapf(
			types.ErrInvalidMiner,
			"node address mismatch: expected %s, got %s",
			challenge.NodeAddress,
			msg.NodeAddress,
		)
	}

	// 5. Verify challenge hasn't expired
	if sdkCtx.BlockTime().After(challenge.ResponseDeadline) {
		// Challenge expired - mark as failed
		challenge.Status = "failed"
		resolutionTime := sdkCtx.BlockTime()
		challenge.ResolutionTime = &resolutionTime
		if err := k.DataAvailabilityChallenges.Set(ctx, msg.ChallengeId, challenge); err != nil {
			return nil, errorsmod.Wrap(err, "failed to update challenge status")
		}
		return nil, errorsmod.Wrapf(
			types.ErrInvalidChallenge,
			"challenge %d has expired (deadline: %s)",
			msg.ChallengeId,
			challenge.ResponseDeadline,
		)
	}

	// 6. Validate proof IPFS hash
	proofHash := strings.TrimSpace(msg.ProofIpfsHash)
	if proofHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "proof IPFS hash cannot be empty")
	}
	if !strings.HasPrefix(proofHash, "Qm") && !strings.HasPrefix(proofHash, "bafy") && !strings.HasPrefix(proofHash, "bafk") {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "invalid IPFS hash format: %s", proofHash)
	}

	// 7. Verify proof content exists in IPFS
	if err := k.VerifyIPFSContentExists(sdkCtx, proofHash); err != nil {
		return nil, errorsmod.Wrap(err, "proof IPFS content verification failed")
	}

	// 8. Verify the original challenged content still exists in IPFS
	if err := k.VerifyIPFSContentExists(sdkCtx, challenge.IpfsHash); err != nil {
		// Content no longer available - challenge failed
		challenge.Status = "failed"
		resolutionTime := sdkCtx.BlockTime()
		challenge.ResolutionTime = &resolutionTime
		if err := k.DataAvailabilityChallenges.Set(ctx, msg.ChallengeId, challenge); err != nil {
			return nil, errorsmod.Wrap(err, "failed to update challenge status")
		}
		return nil, errorsmod.Wrapf(
			types.ErrInvalidIPFSHash,
			"challenged content no longer available in IPFS: %s",
			challenge.IpfsHash,
		)
	}

	// 9. Mark challenge as resolved (content is available)
	challenge.Status = "resolved"
	resolutionTime := sdkCtx.BlockTime()
	challenge.ResolutionTime = &resolutionTime
	challenge.ResponseProof = proofHash
	if err := k.DataAvailabilityChallenges.Set(ctx, msg.ChallengeId, challenge); err != nil {
		return nil, errorsmod.Wrap(err, "failed to update challenge status")
	}

	// 10. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			"respond_to_challenge",
			sdk.NewAttribute(types.AttributeKeyChallengeID, fmt.Sprintf("%d", msg.ChallengeId)),
			sdk.NewAttribute(types.AttributeKeyNodeAddress, msg.NodeAddress),
			sdk.NewAttribute(types.AttributeKeyIPFSHash, challenge.IpfsHash),
			sdk.NewAttribute("proof_ipfs_hash", proofHash),
			sdk.NewAttribute("status", "resolved"),
		),
	)

	return &types.MsgRespondToChallengeResponse{
		IsValid: true,
	}, nil
}

