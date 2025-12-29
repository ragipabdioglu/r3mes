package keeper

import (
	"context"
	"errors"

	"cosmossdk.io/collections"
	"github.com/cosmos/cosmos-sdk/types/query"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"remes/x/remes/types"
)

// ListStoredGradient queries all stored gradients with pagination
func (q queryServer) ListStoredGradient(ctx context.Context, req *types.QueryListStoredGradientRequest) (*types.QueryListStoredGradientResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	var storedGradients []types.StoredGradient

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

	// Iterate with pagination
	err := q.k.StoredGradients.Walk(ctx, nil, func(key uint64, value types.StoredGradient) (stop bool, err error) {
		total++

		// Skip items before offset
		if currentOffset < offset {
			currentOffset++
			return false, nil
		}

		// Collect items up to limit
		if itemsCollected < limit {
			storedGradients = append(storedGradients, value)
			itemsCollected++
		} else {
			// We've collected enough items, but there might be more
			hasMore = true
			return true, nil // Stop iteration
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
			// Indicate there are more items (simplified - in production, encode next key)
			pageResponse.NextKey = []byte("has_more")
		}
	} else if hasMore {
		pageResponse = &query.PageResponse{
			NextKey: []byte("has_more"),
		}
	}

	return &types.QueryListStoredGradientResponse{
		StoredGradients: storedGradients,
		Pagination:      pageResponse,
	}, nil
}

// GetStoredGradient queries a specific stored gradient by ID
func (q queryServer) GetStoredGradient(ctx context.Context, req *types.QueryGetStoredGradientRequest) (*types.QueryGetStoredGradientResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	gradient, err := q.k.StoredGradients.Get(ctx, req.Id)
	if err != nil {
		if errors.Is(err, collections.ErrNotFound) {
			return nil, status.Errorf(codes.NotFound, "stored gradient %d not found", req.Id)
		}
		return nil, status.Error(codes.Internal, err.Error())
	}

	return &types.QueryGetStoredGradientResponse{
		StoredGradient: gradient,
	}, nil
}
