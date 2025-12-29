package keeper

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// ClaimTask handles MsgClaimTask
// Claims an available chunk from a task pool for mining
func (k msgServer) ClaimTask(ctx context.Context, msg *types.MsgClaimTask) (*types.MsgClaimTaskResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate miner address
	minerAddr, err := k.addressCodec.StringToBytes(msg.Miner)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid miner address: %s", msg.Miner)
	}
	_ = minerAddr // Address validated

	// 2. Claim task via keeper
	err = k.Keeper.ClaimTask(ctx, msg.Miner, msg.PoolId, msg.ChunkId)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to claim task")
	}

	// 3. Get transaction hash from context
	txHash := ""
	if txBytes := sdkCtx.TxBytes(); len(txBytes) > 0 {
		hash := sha256.Sum256(txBytes)
		txHash = hex.EncodeToString(hash[:])
	} else {
		// Fallback: calculate deterministic hash if TxBytes not available
		hashInput := fmt.Sprintf(
			"%d|%s|%d|%d|%d",
			sdkCtx.BlockHeight(),
			msg.Miner,
			msg.PoolId,
			msg.ChunkId,
			sdkCtx.BlockTime().Unix(),
		)
		hash := sha256.Sum256([]byte(hashInput))
		txHash = hex.EncodeToString(hash[:])
	}

	// 4. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeClaimTask,
			sdk.NewAttribute(types.AttributeKeyMiner, msg.Miner),
			sdk.NewAttribute(types.AttributeKeyPoolID, fmt.Sprintf("%d", msg.PoolId)),
			sdk.NewAttribute(types.AttributeKeyChunkID, fmt.Sprintf("%d", msg.ChunkId)),
		),
	)

	return &types.MsgClaimTaskResponse{
		Success: true,
		TxHash:  txHash,
	}, nil
}

// CompleteTask handles MsgCompleteTask
// Marks a claimed chunk as completed with gradient result
func (k msgServer) CompleteTask(ctx context.Context, msg *types.MsgCompleteTask) (*types.MsgCompleteTaskResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate miner address
	minerAddr, err := k.addressCodec.StringToBytes(msg.Miner)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid miner address: %s", msg.Miner)
	}
	_ = minerAddr // Address validated

	// 2. Validate gradient hash
	if msg.GradientHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidGradientHash, "gradient hash cannot be empty")
	}

	// 3. Validate IPFS hash format
	if msg.GradientIpfsHash != "" {
		ipfsHash := msg.GradientIpfsHash
		if !(len(ipfsHash) > 0 && (ipfsHash[:2] == "Qm" || ipfsHash[:4] == "bafy" || ipfsHash[:4] == "bafk")) {
			return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "invalid IPFS hash format: %s", ipfsHash)
		}
	}

	// 4. Complete task via keeper
	err = k.Keeper.CompleteTask(ctx, msg.Miner, msg.PoolId, msg.ChunkId, msg.GradientHash, msg.GradientIpfsHash, msg.MinerGpu)
	if err != nil {
		return nil, errorsmod.Wrap(err, "failed to complete task")
	}

	// 5. Get transaction hash from context
	txHash := ""
	if txBytes := sdkCtx.TxBytes(); len(txBytes) > 0 {
		hash := sha256.Sum256(txBytes)
		txHash = hex.EncodeToString(hash[:])
	} else {
		// Fallback: calculate deterministic hash if TxBytes not available
		hashInput := fmt.Sprintf(
			"%d|%s|%d|%d|%s|%s",
			sdkCtx.BlockHeight(),
			msg.Miner,
			msg.PoolId,
			msg.ChunkId,
			msg.GradientHash,
			sdkCtx.BlockTime().Unix(),
		)
		hash := sha256.Sum256([]byte(hashInput))
		txHash = hex.EncodeToString(hash[:])
	}

	// 6. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeCompleteTask,
			sdk.NewAttribute(types.AttributeKeyMiner, msg.Miner),
			sdk.NewAttribute(types.AttributeKeyPoolID, fmt.Sprintf("%d", msg.PoolId)),
			sdk.NewAttribute(types.AttributeKeyChunkID, fmt.Sprintf("%d", msg.ChunkId)),
			sdk.NewAttribute(types.AttributeKeyGradientHash, msg.GradientHash),
			sdk.NewAttribute(types.AttributeKeyIPFSHash, msg.GradientIpfsHash),
		),
	)

	return &types.MsgCompleteTaskResponse{
		Success: true,
		TxHash:  txHash,
	}, nil
}

