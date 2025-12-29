package keeper

import (
	"bytes"
	"context"
	"encoding/hex"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// RemoveDataset handles MsgRemoveDataset
func (k msgServer) RemoveDataset(ctx context.Context, msg *types.MsgRemoveDataset) (*types.MsgRemoveDatasetResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Verify authority
	authorityAddr, err := sdk.AccAddressFromBech32(msg.Authority)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidSigner, "invalid authority address")
	}
	if !bytes.Equal(k.GetAuthority(), authorityAddr) {
		return nil, errorsmod.Wrapf(types.ErrInvalidSigner, "unauthorized")
	}

	// Get approved dataset
	dataset, err := k.ApprovedDatasets.Get(sdkCtx, msg.DatasetId)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrDatasetNotFound, "dataset not found")
	}

	// Check if already removed
	if dataset.Status == "removed" {
		return nil, errorsmod.Wrap(err, "dataset already removed")
	}

	// Mark as removed
	dataset.Status = "removed"
	dataset.RemovalHeight = sdkCtx.BlockHeight()
	dataset.RemovalTxHash = hex.EncodeToString(sdkCtx.TxBytes())

	// Add audit trail entry
	auditEntry := &types.AuditTrailEntry{
		EntryType:   "removal",
		BlockHeight: sdkCtx.BlockHeight(),
		TxHash:      hex.EncodeToString(sdkCtx.TxBytes()),
		Actor:       msg.Authority,
		Description: "Dataset removed: " + msg.Reason,
		Timestamp:   sdkCtx.BlockTime(),
	}

	dataset.AuditTrail = append(dataset.AuditTrail, auditEntry)

	// Store updated dataset
	if err := k.ApprovedDatasets.Set(sdkCtx, msg.DatasetId, dataset); err != nil {
		return nil, err
	}

	// Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeRemoveDataset,
			sdk.NewAttribute(types.AttributeKeyDatasetID, string(rune(msg.DatasetId))),
			sdk.NewAttribute(types.AttributeKeyAuthority, msg.Authority),
			sdk.NewAttribute(types.AttributeKeyReason, msg.Reason),
		),
	)

	return &types.MsgRemoveDatasetResponse{}, nil
}

