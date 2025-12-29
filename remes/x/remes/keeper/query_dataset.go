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

// GetDatasetProposal queries a dataset proposal by ID
func (k Keeper) GetDatasetProposal(ctx context.Context, req *types.QueryGetDatasetProposalRequest) (*types.QueryGetDatasetProposalResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	sdkCtx := sdk.UnwrapSDKContext(ctx)

	proposal, err := k.DatasetProposals.Get(sdkCtx, req.ProposalId)
	if err != nil {
		if errors.Is(err, collections.ErrNotFound) {
			return nil, status.Error(codes.NotFound, "proposal not found")
		}
		return nil, status.Error(codes.Internal, err.Error())
	}

	return &types.QueryGetDatasetProposalResponse{
		Proposal: proposal,
	}, nil
}

// ListDatasetProposals lists all dataset proposals
func (k Keeper) ListDatasetProposals(ctx context.Context, req *types.QueryListDatasetProposalsRequest) (*types.QueryListDatasetProposalsResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	var proposals []types.DatasetProposal

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
	err := k.DatasetProposals.Walk(sdkCtx, nil, func(key uint64, value types.DatasetProposal) (stop bool, err error) {
		total++

		// Skip items before offset
		if currentOffset < offset {
			currentOffset++
			return false, nil
		}

		// Collect items up to limit
		if itemsCollected < limit {
			proposals = append(proposals, value)
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

	return &types.QueryListDatasetProposalsResponse{
		Proposals:  proposals,
		Pagination: pageResponse,
	}, nil
}

// GetApprovedDataset queries an approved dataset by ID
func (k Keeper) GetApprovedDataset(ctx context.Context, req *types.QueryGetApprovedDatasetRequest) (*types.QueryGetApprovedDatasetResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	sdkCtx := sdk.UnwrapSDKContext(ctx)

	dataset, err := k.ApprovedDatasets.Get(sdkCtx, req.DatasetId)
	if err != nil {
		if errors.Is(err, collections.ErrNotFound) {
			return nil, status.Error(codes.NotFound, "dataset not found")
		}
		return nil, status.Error(codes.Internal, err.Error())
	}

	return &types.QueryGetApprovedDatasetResponse{
		Dataset: dataset,
	}, nil
}

// ListApprovedDatasets lists all approved datasets
func (k Keeper) ListApprovedDatasets(ctx context.Context, req *types.QueryListApprovedDatasetsRequest) (*types.QueryListApprovedDatasetsResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	var datasets []types.ApprovedDataset

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
	err := k.ApprovedDatasets.Walk(sdkCtx, nil, func(key uint64, value types.ApprovedDataset) (stop bool, err error) {
		total++

		// Skip items before offset
		if currentOffset < offset {
			currentOffset++
			return false, nil
		}

		// Collect items up to limit
		if itemsCollected < limit {
			datasets = append(datasets, value)
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

	return &types.QueryListApprovedDatasetsResponse{
		Datasets:   datasets,
		Pagination: pageResponse,
	}, nil
}

