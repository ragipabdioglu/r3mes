package keeper

import (
	"context"
	"errors"
	"fmt"

	"cosmossdk.io/collections"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	sdk "github.com/cosmos/cosmos-sdk/types"
	querytypes "github.com/cosmos/cosmos-sdk/types/query"
	"remes/x/remes/types"
)

// GetParticipantSyncState queries a participant's synchronization state
func (k Keeper) GetParticipantSyncState(ctx context.Context, req *types.QueryGetParticipantSyncStateRequest) (*types.QueryGetParticipantSyncStateResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	syncState, err := k.ParticipantSyncStates.Get(ctx, req.ParticipantAddress)
	if err != nil {
		if errors.Is(err, collections.ErrNotFound) {
			return nil, status.Errorf(codes.NotFound, "participant sync state for %s not found", req.ParticipantAddress)
		}
		return nil, status.Error(codes.Internal, err.Error())
	}

	return &types.QueryGetParticipantSyncStateResponse{
		SyncState: syncState,
	}, nil
}

// ListParticipantSyncStates lists all participant sync states with pagination
func (k Keeper) ListParticipantSyncStates(ctx context.Context, req *types.QueryListParticipantSyncStatesRequest) (*types.QueryListParticipantSyncStatesResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	var syncStates []types.ParticipantSyncState
	pageRes := &querytypes.PageResponse{}

	// Simple pagination: collect all and slice
	err := k.ParticipantSyncStates.Walk(ctx, nil, func(key string, value types.ParticipantSyncState) (stop bool, err error) {
		syncStates = append(syncStates, value)
		return false, nil
	})
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}

	// Apply pagination
	offset := uint64(0)
	limit := uint64(100) // Default limit
	if req.Pagination != nil {
		if req.Pagination.Offset > 0 {
			offset = req.Pagination.Offset
		}
		if req.Pagination.Limit > 0 {
			limit = req.Pagination.Limit
		}
	}

	total := uint64(len(syncStates))
	start := offset
	end := offset + limit
	if start > total {
		start = total
	}
	if end > total {
		end = total
	}

	if start < total {
		syncStates = syncStates[start:end]
	} else {
		syncStates = []types.ParticipantSyncState{}
	}

	pageRes.Total = total
	if end < total {
		pageRes.NextKey = []byte{}
	}

	return &types.QueryListParticipantSyncStatesResponse{
		SyncStates: syncStates,
		Pagination: pageRes,
	}, nil
}

// GetGlobalModelState queries the current global model state
func (k Keeper) GetGlobalModelState(ctx context.Context, req *types.QueryGetGlobalModelStateRequest) (*types.QueryGetGlobalModelStateResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	state, err := k.GlobalModelState.Get(ctx)
	if err != nil {
		if errors.Is(err, collections.ErrNotFound) {
			// Return empty state if not found
			return &types.QueryGetGlobalModelStateResponse{
				State: types.GlobalModelState{},
			}, nil
		}
		return nil, status.Error(codes.Internal, err.Error())
	}

	return &types.QueryGetGlobalModelStateResponse{
		State: state,
	}, nil
}

// UpdateParticipantSyncState updates a participant's synchronization state
func (k Keeper) UpdateParticipantSyncState(
	ctx sdk.Context,
	participantAddress string,
	modelVersion string,
	participantType string,
) error {
	// Get current global model state
	globalState, err := k.GlobalModelState.Get(ctx)
	if err != nil {
		// If no global state exists, create a default sync state
		syncState := types.ParticipantSyncState{
			ParticipantAddress: participantAddress,
			CurrentModelVersion: modelVersion,
			LastSyncHeight:      ctx.BlockHeight(),
			LastSyncTime:        ctx.BlockTime(),
			IsSynced:            true,
			SyncLagBlocks:       0,
			ParticipantType:     participantType,
		}
		return k.ParticipantSyncStates.Set(ctx, participantAddress, syncState)
	}

	// Calculate sync lag
	syncLag := ctx.BlockHeight() - globalState.LastUpdatedHeight
	isSynced := modelVersion == globalState.ModelVersion && syncLag <= 10 // Allow 10 blocks lag

	// Update or create sync state
	syncState := types.ParticipantSyncState{
		ParticipantAddress:  participantAddress,
		CurrentModelVersion: modelVersion,
		LastSyncHeight:      ctx.BlockHeight(),
		LastSyncTime:        ctx.BlockTime(),
		IsSynced:            isSynced,
		SyncLagBlocks:       syncLag,
		ParticipantType:     participantType,
	}

	return k.ParticipantSyncStates.Set(ctx, participantAddress, syncState)
}

// HandleNetworkPartition handles network partition recovery for a participant
// This function is called when a participant reconnects after a disconnection
func (k Keeper) HandleNetworkPartition(
	ctx sdk.Context,
	participantAddress string,
) (*types.ParticipantSyncState, error) {
	// Get participant's current sync state
	syncState, err := k.ParticipantSyncStates.Get(ctx, participantAddress)
	if err != nil {
		// If no sync state exists, create a new one
		globalState, err := k.GlobalModelState.Get(ctx)
		if err != nil {
			// No global state - participant is up to date by default
			return nil, nil
		}
		
		// Create sync state indicating participant needs to catch up
		syncState = types.ParticipantSyncState{
			ParticipantAddress:  participantAddress,
			CurrentModelVersion: "", // Unknown - needs sync
			LastSyncHeight:      0,
			LastSyncTime:        ctx.BlockTime(),
			IsSynced:            false,
			SyncLagBlocks:       ctx.BlockHeight() - globalState.LastUpdatedHeight,
			ParticipantType:     "unknown",
		}
		return &syncState, nil
	}

	// Get current global model state
	globalState, err := k.GlobalModelState.Get(ctx)
	if err != nil {
		// No global state - participant is considered synced
		syncState.IsSynced = true
		syncState.SyncLagBlocks = 0
		return &syncState, nil
	}

	// Calculate sync lag
	syncLag := ctx.BlockHeight() - syncState.LastSyncHeight
	
	// Check if participant is significantly behind (more than 100 blocks)
	if syncLag > 100 {
		// Participant needs catch-up
		syncState.IsSynced = false
		syncState.SyncLagBlocks = syncLag
		syncState.CurrentModelVersion = "" // Needs to update
	} else {
		// Participant is relatively up to date
		syncState.IsSynced = syncState.CurrentModelVersion == globalState.ModelVersion
		syncState.SyncLagBlocks = syncLag
	}

	// Update last sync time
	syncState.LastSyncTime = ctx.BlockTime()
	syncState.LastSyncHeight = ctx.BlockHeight()

	// Save updated sync state
	if err := k.ParticipantSyncStates.Set(ctx, participantAddress, syncState); err != nil {
		return nil, err
	}

	return &syncState, nil
}

// GetCatchUpInfoInternal returns information needed for a participant to catch up (internal helper)
func (k Keeper) GetCatchUpInfoInternal(
	ctx sdk.Context,
	participantAddress string,
) (*types.GlobalModelState, []types.StoredGradient, error) {
	// Get global model state
	globalState, err := k.GlobalModelState.Get(ctx)
	if err != nil {
		return nil, nil, err
	}

	// Get participant's sync state
	syncState, err := k.ParticipantSyncStates.Get(ctx, participantAddress)
	if err != nil {
		// No sync state - return all recent gradients
		syncState = types.ParticipantSyncState{
			LastSyncHeight: 0,
		}
	}

	// Get gradients submitted since last sync (for catch-up)
	var catchUpGradients []types.StoredGradient
	err = k.StoredGradients.Walk(ctx, nil, func(key uint64, value types.StoredGradient) (stop bool, err error) {
		// Include gradients submitted after participant's last sync
		if value.SubmittedAtHeight > syncState.LastSyncHeight {
			catchUpGradients = append(catchUpGradients, value)
		}
		return false, nil
	})
	if err != nil {
		return nil, nil, err
	}

	return &globalState, catchUpGradients, nil
}

// GetCatchUpInfo queries catch-up information for a participant
func (k Keeper) GetCatchUpInfo(ctx context.Context, req *types.QueryGetCatchUpInfoRequest) (*types.QueryGetCatchUpInfoResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	sdkCtx := sdk.UnwrapSDKContext(ctx)
	
	// Get catch-up info
	globalState, catchUpGradients, err := k.GetCatchUpInfoInternal(sdkCtx, req.ParticipantAddress)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}

	// Get participant sync state
	syncState, err := k.ParticipantSyncStates.Get(ctx, req.ParticipantAddress)
	if err != nil {
		// If no sync state, create a default one
		syncState = types.ParticipantSyncState{
			ParticipantAddress: req.ParticipantAddress,
			IsSynced:           false,
		}
	}

	return &types.QueryGetCatchUpInfoResponse{
		GlobalState:      *globalState,
		CatchUpGradients: catchUpGradients,
		SyncState:        syncState,
	}, nil
}

// GetGlobalSeed returns the global seed for a training round
func (k Keeper) GetGlobalSeed(ctx context.Context, req *types.QueryGetGlobalSeedRequest) (*types.QueryGetGlobalSeedResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// Derive global seed
	globalSeed, err := k.DeriveGlobalSeed(sdkCtx, req.TrainingRoundId)
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}

	// Get block hash
	blockHash := sdkCtx.HeaderHash()
	blockHashHex := fmt.Sprintf("%x", blockHash)

	return &types.QueryGetGlobalSeedResponse{
		GlobalSeed:      globalSeed,
		BlockHash:       blockHashHex,
		TrainingRoundId: req.TrainingRoundId,
	}, nil
}

