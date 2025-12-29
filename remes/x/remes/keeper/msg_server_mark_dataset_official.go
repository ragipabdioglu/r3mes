package keeper

import (
	"bytes"
	"context"
	"encoding/hex"

	sdk "github.com/cosmos/cosmos-sdk/types"
	errorsmod "cosmossdk.io/errors"

	"remes/x/remes/types"
)

// MarkDatasetAsOfficial handles MsgMarkDatasetAsOfficial
func (k msgServer) MarkDatasetAsOfficial(ctx context.Context, msg *types.MsgMarkDatasetAsOfficial) (*types.MsgMarkDatasetAsOfficialResponse, error) {
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

	// Check if already marked as official
	if dataset.IsOfficialTrainingData {
		return nil, errorsmod.Wrap(err, "dataset already marked as official training data")
	}

	// Mark as official
	dataset.IsOfficialTrainingData = true

	// Add audit trail entry
	auditEntry := &types.AuditTrailEntry{
		EntryType:   "update",
		BlockHeight: sdkCtx.BlockHeight(),
		TxHash:      hex.EncodeToString(sdkCtx.TxBytes()),
		Actor:       msg.Authority,
		Description: "Dataset marked as Official Training Data",
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
			types.EventTypeMarkDatasetOfficial,
			sdk.NewAttribute(types.AttributeKeyDatasetID, string(rune(msg.DatasetId))),
			sdk.NewAttribute(types.AttributeKeyAuthority, msg.Authority),
		),
	)

	return &types.MsgMarkDatasetAsOfficialResponse{}, nil
}

