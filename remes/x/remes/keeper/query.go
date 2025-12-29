package keeper

import (
	"context"

	"remes/x/remes/types"
)

var _ types.QueryServer = queryServer{}

// NewQueryServerImpl returns an implementation of the QueryServer interface
// for the provided Keeper.
func NewQueryServerImpl(k Keeper) types.QueryServer {
	return queryServer{k}
}

type queryServer struct {
	k Keeper
}

// GetDatasetProposal implements types.QueryServer
func (qs queryServer) GetDatasetProposal(ctx context.Context, req *types.QueryGetDatasetProposalRequest) (*types.QueryGetDatasetProposalResponse, error) {
	return qs.k.GetDatasetProposal(ctx, req)
}

// ListDatasetProposals implements types.QueryServer
func (qs queryServer) ListDatasetProposals(ctx context.Context, req *types.QueryListDatasetProposalsRequest) (*types.QueryListDatasetProposalsResponse, error) {
	return qs.k.ListDatasetProposals(ctx, req)
}

// GetApprovedDataset implements types.QueryServer
func (qs queryServer) GetApprovedDataset(ctx context.Context, req *types.QueryGetApprovedDatasetRequest) (*types.QueryGetApprovedDatasetResponse, error) {
	return qs.k.GetApprovedDataset(ctx, req)
}

// ListApprovedDatasets implements types.QueryServer
func (qs queryServer) ListApprovedDatasets(ctx context.Context, req *types.QueryListApprovedDatasetsRequest) (*types.QueryListApprovedDatasetsResponse, error) {
	return qs.k.ListApprovedDatasets(ctx, req)
}

// GetNodeRegistration implements types.QueryServer
func (qs queryServer) GetNodeRegistration(ctx context.Context, req *types.QueryGetNodeRegistrationRequest) (*types.QueryGetNodeRegistrationResponse, error) {
	return qs.k.GetNodeRegistration(ctx, req)
}

// ListNodeRegistrations implements types.QueryServer
func (qs queryServer) ListNodeRegistrations(ctx context.Context, req *types.QueryListNodeRegistrationsRequest) (*types.QueryListNodeRegistrationsResponse, error) {
	return qs.k.ListNodeRegistrations(ctx, req)
}

// GetInferenceRequest implements types.QueryServer
func (qs queryServer) GetInferenceRequest(ctx context.Context, req *types.QueryGetInferenceRequestRequest) (*types.QueryGetInferenceRequestResponse, error) {
	return qs.k.GetInferenceRequest(ctx, req)
}

// GetServingNodeStatus implements types.QueryServer
func (qs queryServer) GetServingNodeStatus(ctx context.Context, req *types.QueryGetServingNodeStatusRequest) (*types.QueryGetServingNodeStatusResponse, error) {
	return qs.k.GetServingNodeStatus(ctx, req)
}

// ListServingNodes implements types.QueryServer
func (qs queryServer) ListServingNodes(ctx context.Context, req *types.QueryListServingNodesRequest) (*types.QueryListServingNodesResponse, error) {
	return qs.k.ListServingNodes(ctx, req)
}

// GetRewardFormula implements types.QueryServer
func (qs queryServer) GetRewardFormula(ctx context.Context, req *types.QueryGetRewardFormulaRequest) (*types.QueryGetRewardFormulaResponse, error) {
	return qs.k.GetRewardFormula(ctx, req)
}

// GetParticipantSyncState implements types.QueryServer
func (qs queryServer) GetParticipantSyncState(ctx context.Context, req *types.QueryGetParticipantSyncStateRequest) (*types.QueryGetParticipantSyncStateResponse, error) {
	return qs.k.GetParticipantSyncState(ctx, req)
}

// ListParticipantSyncStates implements types.QueryServer
func (qs queryServer) ListParticipantSyncStates(ctx context.Context, req *types.QueryListParticipantSyncStatesRequest) (*types.QueryListParticipantSyncStatesResponse, error) {
	return qs.k.ListParticipantSyncStates(ctx, req)
}

// GetGlobalModelState implements types.QueryServer
func (qs queryServer) GetGlobalModelState(ctx context.Context, req *types.QueryGetGlobalModelStateRequest) (*types.QueryGetGlobalModelStateResponse, error) {
	return qs.k.GetGlobalModelState(ctx, req)
}

// GetCatchUpInfo implements types.QueryServer
func (qs queryServer) GetCatchUpInfo(ctx context.Context, req *types.QueryGetCatchUpInfoRequest) (*types.QueryGetCatchUpInfoResponse, error) {
	return qs.k.GetCatchUpInfo(ctx, req)
}

// GetGlobalSeed implements types.QueryServer
func (qs queryServer) GetGlobalSeed(ctx context.Context, req *types.QueryGetGlobalSeedRequest) (*types.QueryGetGlobalSeedResponse, error) {
	return qs.k.GetGlobalSeed(ctx, req)
}

// QueryMiners implements types.QueryServer (for dashboard API)
func (qs queryServer) QueryMiners(ctx context.Context, req *types.QueryMinersRequest) (*types.QueryMinersResponse, error) {
	return qs.k.QueryMiners(ctx, req)
}

// QueryStatistics implements types.QueryServer (for dashboard API)
func (qs queryServer) QueryStatistics(ctx context.Context, req *types.QueryStatisticsRequest) (*types.QueryStatisticsResponse, error) {
	return qs.k.QueryStatistics(ctx, req)
}

// QueryBlocks implements types.QueryServer (for dashboard API)
func (qs queryServer) QueryBlocks(ctx context.Context, req *types.QueryBlocksRequest) (*types.QueryBlocksResponse, error) {
	return qs.k.QueryBlocks(ctx, req)
}

// QueryBlock implements types.QueryServer (for dashboard API)
func (qs queryServer) QueryBlock(ctx context.Context, req *types.QueryBlockRequest) (*types.QueryBlockResponse, error) {
	return qs.k.QueryBlock(ctx, req)
}

// QueryVaultStats implements types.QueryServer (admin only)
func (qs queryServer) QueryVaultStats(ctx context.Context, req *types.QueryVaultStatsRequest) (*types.QueryVaultStatsResponse, error) {
	return qs.k.QueryVaultStats(ctx, req)
}

// QueryMinerFraudScore implements types.QueryServer (admin only)
func (qs queryServer) QueryMinerFraudScore(ctx context.Context, req *types.QueryMinerFraudScoreRequest) (*types.QueryMinerFraudScoreResponse, error) {
	return qs.k.QueryMinerFraudScore(ctx, req)
}

// QueryActivePool implements types.QueryServer
func (qs queryServer) QueryActivePool(ctx context.Context, req *types.QueryActivePoolRequest) (*types.QueryActivePoolResponse, error) {
	return qs.k.QueryActivePool(ctx, req)
}

// QueryAvailableChunks implements types.QueryServer
func (qs queryServer) QueryAvailableChunks(ctx context.Context, req *types.QueryAvailableChunksRequest) (*types.QueryAvailableChunksResponse, error) {
	return qs.k.QueryAvailableChunks(ctx, req)
}
