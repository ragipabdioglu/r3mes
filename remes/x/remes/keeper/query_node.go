package keeper

import (
	"context"
	"errors"

	"cosmossdk.io/collections"
	"github.com/cosmos/cosmos-sdk/types/query"
	sdk "github.com/cosmos/cosmos-sdk/types"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"remes/x/remes/types"
)

// GetNodeRegistration queries a node registration by address
func (k Keeper) GetNodeRegistration(ctx context.Context, req *types.QueryGetNodeRegistrationRequest) (*types.QueryGetNodeRegistrationResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	sdkCtx := sdk.UnwrapSDKContext(ctx)

	registration, err := k.NodeRegistrations.Get(sdkCtx, req.NodeAddress)
	if err != nil {
		if errors.Is(err, collections.ErrNotFound) {
			return nil, status.Error(codes.NotFound, "node registration not found")
		}
		return nil, status.Error(codes.Internal, err.Error())
	}

	return &types.QueryGetNodeRegistrationResponse{
		Registration: registration,
	}, nil
}

// ListNodeRegistrations lists all node registrations
func (k Keeper) ListNodeRegistrations(ctx context.Context, req *types.QueryListNodeRegistrationsRequest) (*types.QueryListNodeRegistrationsResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	var registrations []types.NodeRegistration

	// Pagination parameters
	limit := uint64(100) // Default limit
	offset := uint64(0)
	if req.Pagination != nil {
		if req.Pagination.Limit > 0 {
			limit = req.Pagination.Limit
		}
		if req.Pagination.Offset > 0 {
			offset = req.Pagination.Offset
		}
	}

	// Track pagination state
	currentOffset := uint64(0)
	itemsCollected := uint64(0)
	var total uint64
	hasMore := false

	sdkCtx := sdk.UnwrapSDKContext(ctx)
	err := k.NodeRegistrations.Walk(sdkCtx, nil, func(key string, value types.NodeRegistration) (stop bool, err error) {
		total++

		// Skip items before offset
		if currentOffset < offset {
			currentOffset++
			return false, nil
		}

		// Collect items up to limit
		if itemsCollected < limit {
			registrations = append(registrations, value)
			itemsCollected++
		} else {
			hasMore = true
			return true, nil
		}

		return false, nil
	})
	if err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}

	// Build pagination response
	var pageResponse *query.PageResponse
	if req.Pagination != nil && req.Pagination.CountTotal {
		pageResponse = &query.PageResponse{
			Total: total,
		}
		if hasMore {
			pageResponse.NextKey = []byte("has_more")
		}
	} else if hasMore {
		pageResponse = &query.PageResponse{
			NextKey: []byte("has_more"),
		}
	}

	return &types.QueryListNodeRegistrationsResponse{
		Registrations: registrations,
		Pagination:    pageResponse,
	}, nil
}

