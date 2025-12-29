package keeper

import (
	"fmt"
	"time"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// PartitionDetectionThreshold is the number of blocks a participant can be offline
// before being considered partitioned
const PartitionDetectionThreshold = 100 // blocks

// PartitionRecoveryWindow is the number of blocks after partition detection
// during which a participant can recover
const PartitionRecoveryWindow = 200 // blocks

// DetectPartitions detects network partitions by checking participant sync states
// This is called periodically in EndBlocker
func (k Keeper) DetectPartitions(ctx sdk.Context) error {
	currentHeight := ctx.BlockHeight()
	partitionedCount := 0

	// Iterate through all participant sync states
	iter, err := k.ParticipantSyncStates.Iterate(ctx, nil)
	if err != nil {
		return fmt.Errorf("failed to iterate participant sync states: %w", err)
	}
	defer iter.Close()

	for ; iter.Valid(); iter.Next() {
		participantAddr, err := iter.Key()
		if err != nil {
			continue
		}

		syncState, err := iter.Value()
		if err != nil {
			continue
		}

		// Calculate time since last sync
		timeSinceLastSync := ctx.BlockTime().Sub(syncState.LastSyncTime)
		blocksSinceLastSync := currentHeight - syncState.LastSyncHeight

		// Detect partition if:
		// 1. More than PartitionDetectionThreshold blocks since last sync
		// 2. Or more than 1 hour since last sync (whichever comes first)
		isPartitioned := false
		if blocksSinceLastSync > PartitionDetectionThreshold {
			isPartitioned = true
		}
		if timeSinceLastSync > time.Hour {
			isPartitioned = true
		}

		if isPartitioned && !syncState.IsPartitioned {
			// Mark participant as partitioned
			syncState.IsPartitioned = true
			syncState.PartitionDetectedAt = ctx.BlockHeight()
			syncState.PartitionDetectedTime = ctx.BlockTime()

			if err := k.ParticipantSyncStates.Set(ctx, participantAddr, syncState); err != nil {
				ctx.Logger().Error(fmt.Sprintf("Failed to mark participant %s as partitioned: %v", participantAddr, err))
				continue
			}

			// Emit partition detection event
			ctx.EventManager().EmitEvent(
				sdk.NewEvent(
					types.EventTypePartitionDetected,
					sdk.NewAttribute(types.AttributeKeyParticipantAddress, participantAddr),
					sdk.NewAttribute(types.AttributeKeyPartitionHeight, fmt.Sprintf("%d", currentHeight)),
					sdk.NewAttribute(types.AttributeKeySyncLagBlocks, fmt.Sprintf("%d", blocksSinceLastSync)),
					sdk.NewAttribute(types.AttributeKeyParticipantType, syncState.ParticipantType),
				),
			)

			partitionedCount++
			ctx.Logger().Info(fmt.Sprintf("Detected partition for participant %s (lag: %d blocks)", participantAddr, blocksSinceLastSync))
		}
	}

	if partitionedCount > 0 {
		ctx.Logger().Info(fmt.Sprintf("Detected %d partitioned participants", partitionedCount))
	}

	return nil
}

// ProcessPartitionRecovery processes partition recovery for participants
// This is called when a participant reconnects after a partition
func (k Keeper) ProcessPartitionRecovery(
	ctx sdk.Context,
	participantAddress string,
) (*types.PartitionRecoveryInfo, error) {
	// Get participant sync state
	syncState, err := k.ParticipantSyncStates.Get(ctx, participantAddress)
	if err != nil {
		// No sync state - not partitioned
		return nil, nil
	}

	// Check if participant was partitioned
	if !syncState.IsPartitioned {
		// Not partitioned - no recovery needed
		return nil, nil
	}

	// Calculate partition duration
	partitionDuration := ctx.BlockHeight() - syncState.PartitionDetectedAt
	timeSincePartition := ctx.BlockTime().Sub(syncState.PartitionDetectedTime)

	// Check if recovery window has expired
	if partitionDuration > PartitionRecoveryWindow {
		// Recovery window expired - participant needs full catch-up
		return &types.PartitionRecoveryInfo{
			ParticipantAddress: participantAddress,
			WasPartitioned:      true,
			PartitionDuration:   partitionDuration,
			RecoveryType:        "full_catchup",
			RequiresFullSync:    true,
			SyncLagBlocks:       syncState.SyncLagBlocks,
		}, nil
	}

	// Within recovery window - can do incremental catch-up
	// Get catch-up information
	globalState, catchUpGradients, err := k.GetCatchUpInfoInternal(ctx, participantAddress)
	if err != nil {
		return nil, fmt.Errorf("failed to get catch-up info: %w", err)
	}

	// Mark participant as recovered
	syncState.IsPartitioned = false
	syncState.IsSynced = false // Will be synced after catch-up
	syncState.LastSyncHeight = ctx.BlockHeight()
	syncState.LastSyncTime = ctx.BlockTime()
	syncState.PartitionRecoveredAt = ctx.BlockHeight()
	syncState.PartitionRecoveredTime = ctx.BlockTime()

	if err := k.ParticipantSyncStates.Set(ctx, participantAddress, syncState); err != nil {
		return nil, fmt.Errorf("failed to update sync state: %w", err)
	}

	// Emit partition recovery event
	ctx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypePartitionRecovered,
			sdk.NewAttribute(types.AttributeKeyParticipantAddress, participantAddress),
			sdk.NewAttribute(types.AttributeKeyPartitionDuration, fmt.Sprintf("%d", partitionDuration)),
			sdk.NewAttribute(types.AttributeKeyRecoveryType, "incremental"),
			sdk.NewAttribute(types.AttributeKeySyncLagBlocks, fmt.Sprintf("%d", syncState.SyncLagBlocks)),
		),
	)

	ctx.Logger().Info(fmt.Sprintf("Partition recovery for participant %s (duration: %d blocks, %v)", 
		participantAddress, partitionDuration, timeSincePartition))

	return &types.PartitionRecoveryInfo{
		ParticipantAddress: participantAddress,
		WasPartitioned:     true,
		PartitionDuration:  partitionDuration,
		RecoveryType:       "incremental",
		RequiresFullSync:   false,
		SyncLagBlocks:      syncState.SyncLagBlocks,
		GlobalState:        *globalState,
		CatchUpGradients:    catchUpGradients,
	}, nil
}

// GetPartitionStatus returns the partition status for a participant
func (k Keeper) GetPartitionStatus(
	ctx sdk.Context,
	participantAddress string,
) (*types.PartitionStatus, error) {
	// Get participant sync state
	syncState, err := k.ParticipantSyncStates.Get(ctx, participantAddress)
	if err != nil {
		// No sync state - not partitioned
		return &types.PartitionStatus{
			IsPartitioned: false,
		}, nil
	}

	currentHeight := ctx.BlockHeight()
	blocksSinceLastSync := currentHeight - syncState.LastSyncHeight
	timeSinceLastSync := ctx.BlockTime().Sub(syncState.LastSyncTime)

	// Check if currently partitioned
	isPartitioned := syncState.IsPartitioned
	if !isPartitioned {
		// Check if should be partitioned
		if blocksSinceLastSync > PartitionDetectionThreshold || timeSinceLastSync > time.Hour {
			isPartitioned = true
		}
	}

	status := &types.PartitionStatus{
		IsPartitioned:        isPartitioned,
		SyncLagBlocks:        blocksSinceLastSync,
		TimeSinceLastSync:    timeSinceLastSync.String(),
		LastSyncHeight:       syncState.LastSyncHeight,
		LastSyncTime:         syncState.LastSyncTime,
		PartitionDetectedAt:  syncState.PartitionDetectedAt,
		PartitionRecoveredAt: syncState.PartitionRecoveredAt,
	}

	if isPartitioned {
		status.PartitionDuration = currentHeight - syncState.PartitionDetectedAt
	}

	return status, nil
}

