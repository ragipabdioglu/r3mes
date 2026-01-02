package keeper

import (
	"context"

	"remes/x/remes/types"
)

// queryServer implements the Query gRPC service
type queryServer struct {
	k Keeper
}

// NewQueryServerImpl creates a new query server implementation
func NewQueryServerImpl(keeper Keeper) types.QueryServer {
	return &queryServer{k: keeper}
}

// Params returns the module parameters
func (q queryServer) Params(ctx context.Context, req *types.QueryParamsRequest) (*types.QueryParamsResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	params, err := q.k.GetParams(ctx)
	if err != nil {
		return nil, err
	}

	return &types.QueryParamsResponse{Params: params}, nil
}

// GetGradient returns a specific gradient by ID
func (q queryServer) GetGradient(ctx context.Context, req *types.QueryGetGradientRequest) (*types.QueryGetGradientResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	gradient, err := q.k.GetTrainingKeeper().GetGradient(ctx, req.Id)
	if err != nil {
		return nil, err
	}

	return &types.QueryGetGradientResponse{Gradient: gradient}, nil
}

// GetModelParams returns the current global model parameters
func (q queryServer) GetModelParams(ctx context.Context, req *types.QueryGetModelParamsRequest) (*types.QueryGetModelParamsResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	state, err := q.k.GetModelKeeper().GetGlobalModelState(ctx)
	if err != nil {
		return nil, err
	}

	return &types.QueryGetModelParamsResponse{
		ModelIpfsHash:     state.ModelIpfsHash,
		ModelVersion:      state.ModelVersion,
		LastUpdatedHeight: state.LastUpdatedHeight,
	}, nil
}

// GetAggregation returns a specific aggregation by ID
func (q queryServer) GetAggregation(ctx context.Context, req *types.QueryGetAggregationRequest) (*types.QueryGetAggregationResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	aggregation, err := q.k.GetTrainingKeeper().GetAggregation(ctx, req.Id)
	if err != nil {
		return nil, err
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

// GetMinerScore returns the reputation score and statistics for a miner
func (q queryServer) GetMinerScore(ctx context.Context, req *types.QueryGetMinerScoreRequest) (*types.QueryGetMinerScoreResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	// Get mining contribution from training keeper
	contribution, err := q.k.GetTrainingKeeper().GetMiningContribution(ctx, req.Miner)
	if err != nil {
		return nil, err
	}

	return &types.QueryGetMinerScoreResponse{
		Miner:            req.Miner,
		TotalSubmissions: contribution.TotalSubmissions,
	}, nil
}

// ListStoredGradient returns all stored gradients with pagination
func (q queryServer) ListStoredGradient(ctx context.Context, req *types.QueryListStoredGradientRequest) (*types.QueryListStoredGradientResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	return &types.QueryListStoredGradientResponse{
		StoredGradients: []types.StoredGradient{},
		Pagination:      nil,
	}, nil
}

// GetStoredGradient returns a specific stored gradient by ID
func (q queryServer) GetStoredGradient(ctx context.Context, req *types.QueryGetStoredGradientRequest) (*types.QueryGetStoredGradientResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	gradient, err := q.k.GetTrainingKeeper().GetGradient(ctx, req.Id)
	if err != nil {
		return nil, err
	}

	return &types.QueryGetStoredGradientResponse{StoredGradient: gradient}, nil
}

// GetDatasetProposal returns a specific dataset proposal by ID
func (q queryServer) GetDatasetProposal(ctx context.Context, req *types.QueryGetDatasetProposalRequest) (*types.QueryGetDatasetProposalResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	proposal, err := q.k.GetDatasetKeeper().GetDatasetProposal(ctx, req.ProposalId)
	if err != nil {
		return nil, err
	}

	return &types.QueryGetDatasetProposalResponse{Proposal: proposal}, nil
}

// ListDatasetProposals returns all dataset proposals
func (q queryServer) ListDatasetProposals(ctx context.Context, req *types.QueryListDatasetProposalsRequest) (*types.QueryListDatasetProposalsResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	proposals, err := q.k.GetDatasetKeeper().ListDatasetProposals(ctx)
	if err != nil {
		return nil, err
	}

	return &types.QueryListDatasetProposalsResponse{
		Proposals:  proposals,
		Pagination: nil,
	}, nil
}

// GetApprovedDataset returns an approved dataset by ID
func (q queryServer) GetApprovedDataset(ctx context.Context, req *types.QueryGetApprovedDatasetRequest) (*types.QueryGetApprovedDatasetResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	dataset, err := q.k.GetDatasetKeeper().GetApprovedDataset(ctx, req.DatasetId)
	if err != nil {
		return nil, err
	}

	return &types.QueryGetApprovedDatasetResponse{Dataset: dataset}, nil
}

// ListApprovedDatasets returns all approved datasets
func (q queryServer) ListApprovedDatasets(ctx context.Context, req *types.QueryListApprovedDatasetsRequest) (*types.QueryListApprovedDatasetsResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	datasets, err := q.k.GetDatasetKeeper().ListApprovedDatasets(ctx)
	if err != nil {
		return nil, err
	}

	return &types.QueryListApprovedDatasetsResponse{
		Datasets:   datasets,
		Pagination: nil,
	}, nil
}

// GetNodeRegistration returns a node registration by address
func (q queryServer) GetNodeRegistration(ctx context.Context, req *types.QueryGetNodeRegistrationRequest) (*types.QueryGetNodeRegistrationResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	node, err := q.k.GetNodeKeeper().GetNode(ctx, req.NodeAddress)
	if err != nil {
		return nil, err
	}

	return &types.QueryGetNodeRegistrationResponse{Registration: node}, nil
}

// ListNodeRegistrations returns all node registrations
func (q queryServer) ListNodeRegistrations(ctx context.Context, req *types.QueryListNodeRegistrationsRequest) (*types.QueryListNodeRegistrationsResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	nodes, err := q.k.GetNodeKeeper().ListNodes(ctx)
	if err != nil {
		return nil, err
	}

	return &types.QueryListNodeRegistrationsResponse{
		Registrations: nodes,
		Pagination:    nil,
	}, nil
}

// GetInferenceRequest returns an inference request by ID
func (q queryServer) GetInferenceRequest(ctx context.Context, req *types.QueryGetInferenceRequestRequest) (*types.QueryGetInferenceRequestResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	request, err := q.k.GetNodeKeeper().GetInferenceRequest(ctx, req.RequestId)
	if err != nil {
		return nil, err
	}

	return &types.QueryGetInferenceRequestResponse{Request: request}, nil
}

// GetServingNodeStatus returns serving node status
func (q queryServer) GetServingNodeStatus(ctx context.Context, req *types.QueryGetServingNodeStatusRequest) (*types.QueryGetServingNodeStatusResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	status, err := q.k.GetNodeKeeper().GetServingNodeStatus(ctx, req.NodeAddress)
	if err != nil {
		return nil, err
	}

	return &types.QueryGetServingNodeStatusResponse{Status: status}, nil
}

// ListServingNodes returns all serving nodes
func (q queryServer) ListServingNodes(ctx context.Context, req *types.QueryListServingNodesRequest) (*types.QueryListServingNodesResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	return &types.QueryListServingNodesResponse{
		ServingNodes: []types.ServingNodeStatus{},
		Pagination:   nil,
	}, nil
}

// GetRewardFormula returns transparent reward formulas
func (q queryServer) GetRewardFormula(ctx context.Context, req *types.QueryGetRewardFormulaRequest) (*types.QueryGetRewardFormulaResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	return &types.QueryGetRewardFormulaResponse{}, nil
}

// GetParticipantSyncState returns a participant's synchronization state
func (q queryServer) GetParticipantSyncState(ctx context.Context, req *types.QueryGetParticipantSyncStateRequest) (*types.QueryGetParticipantSyncStateResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	return &types.QueryGetParticipantSyncStateResponse{}, nil
}

// ListParticipantSyncStates returns all participant sync states with pagination
func (q queryServer) ListParticipantSyncStates(ctx context.Context, req *types.QueryListParticipantSyncStatesRequest) (*types.QueryListParticipantSyncStatesResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	return &types.QueryListParticipantSyncStatesResponse{
		SyncStates: []types.ParticipantSyncState{},
		Pagination: nil,
	}, nil
}

// GetGlobalModelState returns the current global model state
func (q queryServer) GetGlobalModelState(ctx context.Context, req *types.QueryGetGlobalModelStateRequest) (*types.QueryGetGlobalModelStateResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	state, err := q.k.GetModelKeeper().GetGlobalModelState(ctx)
	if err != nil {
		return nil, err
	}

	return &types.QueryGetGlobalModelStateResponse{State: state}, nil
}

// GetCatchUpInfo returns information needed for a participant to catch up after network partition
func (q queryServer) GetCatchUpInfo(ctx context.Context, req *types.QueryGetCatchUpInfoRequest) (*types.QueryGetCatchUpInfoResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	return &types.QueryGetCatchUpInfoResponse{}, nil
}

// GetGlobalSeed returns the global seed for a training round
func (q queryServer) GetGlobalSeed(ctx context.Context, req *types.QueryGetGlobalSeedRequest) (*types.QueryGetGlobalSeedResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	return &types.QueryGetGlobalSeedResponse{}, nil
}

// QueryMiners returns all miners with pagination (for dashboard API)
func (q queryServer) QueryMiners(ctx context.Context, req *types.QueryMinersRequest) (*types.QueryMinersResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	return &types.QueryMinersResponse{
		Miners:     []types.MinerInfo{},
		Pagination: nil,
	}, nil
}

// QueryStatistics returns network statistics (for dashboard API)
func (q queryServer) QueryStatistics(ctx context.Context, req *types.QueryStatisticsRequest) (*types.QueryStatisticsResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	return &types.QueryStatisticsResponse{}, nil
}

// QueryBlocks returns recent blocks with pagination (for dashboard API)
func (q queryServer) QueryBlocks(ctx context.Context, req *types.QueryBlocksRequest) (*types.QueryBlocksResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	return &types.QueryBlocksResponse{
		Blocks: []types.BlockInfo{},
		Total:  0,
	}, nil
}

// QueryBlock returns a specific block by height (for dashboard API)
func (q queryServer) QueryBlock(ctx context.Context, req *types.QueryBlockRequest) (*types.QueryBlockResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	return &types.QueryBlockResponse{}, nil
}

// QueryVaultStats returns genesis vault statistics (admin only)
func (q queryServer) QueryVaultStats(ctx context.Context, req *types.QueryVaultStatsRequest) (*types.QueryVaultStatsResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	return &types.QueryVaultStatsResponse{}, nil
}

// QueryMinerFraudScore returns miner's fraud score and trap statistics (admin only)
func (q queryServer) QueryMinerFraudScore(ctx context.Context, req *types.QueryMinerFraudScoreRequest) (*types.QueryMinerFraudScoreResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	return &types.QueryMinerFraudScoreResponse{}, nil
}

// QueryActivePool returns the currently active task pool ID
func (q queryServer) QueryActivePool(ctx context.Context, req *types.QueryActivePoolRequest) (*types.QueryActivePoolResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	return &types.QueryActivePoolResponse{}, nil
}

// QueryAvailableChunks returns available chunks from a task pool
func (q queryServer) QueryAvailableChunks(ctx context.Context, req *types.QueryAvailableChunksRequest) (*types.QueryAvailableChunksResponse, error) {
	if req == nil {
		return nil, types.ErrInvalidRequest
	}

	return &types.QueryAvailableChunksResponse{
		Chunks:         []types.TaskChunkResponse{},
		TotalAvailable: 0,
	}, nil
}
