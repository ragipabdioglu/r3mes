package keeper

import (
	"context"
	"errors"

	"cosmossdk.io/collections"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"remes/x/remes/types"
)

// GetGradient queries a specific gradient by ID
func (q queryServer) GetGradient(ctx context.Context, req *types.QueryGetGradientRequest) (*types.QueryGetGradientResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	gradient, err := q.k.StoredGradients.Get(ctx, req.Id)
	if err != nil {
		if errors.Is(err, collections.ErrNotFound) {
			return nil, status.Errorf(codes.NotFound, "gradient %d not found", req.Id)
		}
		return nil, status.Error(codes.Internal, err.Error())
	}

	return &types.QueryGetGradientResponse{
		Gradient: gradient,
	}, nil
}

// GetModelParams queries the current global model parameters (IPFS hash)
func (q queryServer) GetModelParams(ctx context.Context, req *types.QueryGetModelParamsRequest) (*types.QueryGetModelParamsResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	globalModelState, err := q.k.GlobalModelState.Get(ctx)
	if err != nil {
		if errors.Is(err, collections.ErrNotFound) {
			// Return empty response if no model state exists yet
			return &types.QueryGetModelParamsResponse{
				ModelIpfsHash:     "",
				ModelVersion:      "",
				LastUpdatedHeight: 0,
			}, nil
		}
		return nil, status.Error(codes.Internal, err.Error())
	}

	return &types.QueryGetModelParamsResponse{
		ModelIpfsHash:     globalModelState.ModelIpfsHash,
		ModelVersion:      globalModelState.ModelVersion,
		LastUpdatedHeight: globalModelState.LastUpdatedHeight,
	}, nil
}

// GetAggregation queries a specific aggregation by ID
func (q queryServer) GetAggregation(ctx context.Context, req *types.QueryGetAggregationRequest) (*types.QueryGetAggregationResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	aggregation, err := q.k.AggregationRecords.Get(ctx, req.Id)
	if err != nil {
		if errors.Is(err, collections.ErrNotFound) {
			return nil, status.Errorf(codes.NotFound, "aggregation %d not found", req.Id)
		}
		return nil, status.Error(codes.Internal, err.Error())
	}

	return &types.QueryGetAggregationResponse{
		AggregationId:              aggregation.AggregationId,
		Proposer:                   aggregation.Proposer,
		AggregatedGradientIpfsHash: aggregation.AggregatedGradientIpfsHash,
		MerkleRoot:                 aggregation.MerkleRoot,
		ParticipantCount:           uint64(len(aggregation.ParticipantGradientIds)),
		TrainingRoundId:            aggregation.TrainingRoundId,
	}, nil
}

// GetMinerScore queries the reputation score and statistics for a miner
func (q queryServer) GetMinerScore(ctx context.Context, req *types.QueryGetMinerScoreRequest) (*types.QueryGetMinerScoreResponse, error) {
	if req == nil {
		return nil, status.Error(codes.InvalidArgument, "invalid request")
	}

	contribution, err := q.k.MiningContributions.Get(ctx, req.Miner)
	if err != nil {
		if errors.Is(err, collections.ErrNotFound) {
			// Return default values for new miners
			return &types.QueryGetMinerScoreResponse{
				Miner:                req.Miner,
				TrustScore:           "0.5",
				ReputationTier:       "new",
				TotalSubmissions:     0,
				SuccessfulSubmissions: 0,
				SlashingEvents:       0,
			}, nil
		}
		return nil, status.Error(codes.Internal, err.Error())
	}

	return &types.QueryGetMinerScoreResponse{
		Miner:                contribution.MinerAddress,
		TrustScore:           contribution.TrustScore,
		ReputationTier:       contribution.ReputationTier,
		TotalSubmissions:     contribution.TotalSubmissions,
		SuccessfulSubmissions: contribution.SuccessfulSubmissions,
		SlashingEvents:       contribution.SlashingEvents,
	}, nil
}

