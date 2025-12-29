package keeper

import (
	"context"

	"cosmossdk.io/errors"
	"github.com/cosmos/cosmos-sdk/types/query"

	"remes/x/remes/types"
)

// GetInferenceRequest queries an inference request by ID
func (k Keeper) GetInferenceRequest(ctx context.Context, req *types.QueryGetInferenceRequestRequest) (*types.QueryGetInferenceRequestResponse, error) {
	request, err := k.InferenceRequests.Get(ctx, req.RequestId)
	if err != nil {
		return nil, errors.Wrap(err, "inference request not found")
	}

	return &types.QueryGetInferenceRequestResponse{
		Request: request,
	}, nil
}

// GetServingNodeStatus queries serving node status
func (k Keeper) GetServingNodeStatus(ctx context.Context, req *types.QueryGetServingNodeStatusRequest) (*types.QueryGetServingNodeStatusResponse, error) {
	status, err := k.ServingNodeStatuses.Get(ctx, req.NodeAddress)
	if err != nil {
		return nil, errors.Wrap(err, "serving node status not found")
	}

	return &types.QueryGetServingNodeStatusResponse{
		Status: status,
	}, nil
}

// ListServingNodes lists all serving nodes with pagination
func (k Keeper) ListServingNodes(ctx context.Context, req *types.QueryListServingNodesRequest) (*types.QueryListServingNodesResponse, error) {
	var servingNodes []types.ServingNodeStatus

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

	// Iterate with pagination
	err := k.ServingNodeStatuses.Walk(ctx, nil, func(key string, value types.ServingNodeStatus) (stop bool, err error) {
		total++

		// Skip items before offset
		if currentOffset < offset {
			currentOffset++
			return false, nil
		}

		// Collect items up to limit
		if itemsCollected < limit {
			servingNodes = append(servingNodes, value)
			itemsCollected++
		}

		// Stop if we've collected enough
		if itemsCollected >= limit {
			return true, nil
		}

		return false, nil
	})
	if err != nil {
		return nil, errors.Wrap(err, "failed to walk serving nodes")
	}

	// Create pagination response
	pageReq := &query.PageResponse{
		Total: total,
	}

	return &types.QueryListServingNodesResponse{
		ServingNodes: servingNodes,
		Pagination:   pageReq,
	}, nil
}

