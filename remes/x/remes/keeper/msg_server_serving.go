package keeper

import (
	"context"
	"fmt"
	"strings"

	errorsmod "cosmossdk.io/errors"
	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// RequestInference handles MsgRequestInference
// Routes inference request to a serving node and collects fees
func (k msgServer) RequestInference(ctx context.Context, msg *types.MsgRequestInference) (*types.MsgRequestInferenceResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate requester address
	requesterAddr, err := k.addressCodec.StringToBytes(msg.Requester)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidMiner, "invalid requester address: %s", msg.Requester)
	}

	// 2. Validate serving node address
	_, err = k.addressCodec.StringToBytes(msg.ServingNode)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "invalid serving node address: %s", msg.ServingNode)
	}

	// 3. Verify serving node is registered and active
	registration, err := k.NodeRegistrations.Get(sdkCtx, msg.ServingNode)
	if err != nil {
		return nil, errorsmod.Wrap(err, "serving node not registered")
	}

	// Check if node has serving role
	hasServingRole := false
	for _, role := range registration.Roles {
		if role == types.NODE_TYPE_SERVING {
			hasServingRole = true
			break
		}
	}
	if !hasServingRole {
		return nil, errorsmod.Wrap(err, "node does not have serving role")
	}

	if registration.Status != types.NODE_STATUS_ACTIVE {
		return nil, errorsmod.Wrap(err, "serving node is not active")
	}

	// 4. Check serving node status
	servingStatus, err := k.ServingNodeStatuses.Get(ctx, msg.ServingNode)
	if err == nil {
		if !servingStatus.IsAvailable {
			return nil, errorsmod.Wrap(err, "serving node is not available")
		}
		// Verify model version matches if specified
		if msg.ModelVersion != "" && servingStatus.ModelVersion != msg.ModelVersion {
			return nil, errorsmod.Wrap(err, "serving node model version mismatch")
		}
	}

	// 5. Validate input data IPFS hash
	inputHash := strings.TrimSpace(msg.InputDataIpfsHash)
	if inputHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "input data IPFS hash cannot be empty")
	}
	if !strings.HasPrefix(inputHash, "Qm") && !strings.HasPrefix(inputHash, "bafy") && !strings.HasPrefix(inputHash, "bafk") {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "invalid IPFS hash format: %s", inputHash)
	}

	// 6. Validate fee
	fee, err := sdk.ParseCoinsNormalized(msg.Fee)
	if err != nil {
		return nil, errorsmod.Wrap(err, "invalid fee format")
	}
	if fee.IsZero() {
		return nil, errorsmod.Wrap(err, "fee cannot be zero")
	}

	// 7. Transfer fee from requester to module (will be distributed later)
	if err := k.bankKeeper.SendCoinsFromAccountToModule(ctx, requesterAddr, types.ModuleName, fee); err != nil {
		return nil, errorsmod.Wrap(err, "failed to transfer fee")
	}

	// 8. Generate unique request ID
	requestID := fmt.Sprintf("%s|%d|%x", msg.Requester, sdkCtx.BlockHeight(), sdkCtx.HeaderHash()[:8])

	// 9. Get current model version if not specified
	modelVersion := msg.ModelVersion
	if modelVersion == "" {
		modelState, err := k.GlobalModelState.Get(ctx)
		if err == nil {
			modelVersion = modelState.ModelVersion
		} else {
			modelVersion = "v1.0.0" // Default
		}
	}

	// 10. Create inference request
	inferenceRequest := types.InferenceRequest{
		RequestId:         requestID,
		Requester:         msg.Requester,
		ServingNode:       msg.ServingNode,
		ModelVersion:      modelVersion,
		InputDataIpfsHash: inputHash,
		Fee:               msg.Fee,
		RequestTime:       sdkCtx.BlockTime(),
		Status:            "pending",
	}

	// 11. Store inference request
	if err := k.InferenceRequests.Set(ctx, requestID, inferenceRequest); err != nil {
		return nil, errorsmod.Wrap(err, "failed to store inference request")
	}

	// 12. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeRequestInference,
			sdk.NewAttribute(types.AttributeKeyRequestID, requestID),
			sdk.NewAttribute(types.AttributeKeyRequester, msg.Requester),
			sdk.NewAttribute(types.AttributeKeyServingNode, msg.ServingNode),
			sdk.NewAttribute(types.AttributeKeyModelVersion, modelVersion),
			sdk.NewAttribute("fee", msg.Fee),
		),
	)

	return &types.MsgRequestInferenceResponse{
		RequestId: requestID,
	}, nil
}

// SubmitInferenceResult handles MsgSubmitInferenceResult
// Serving node submits inference result and receives fee
func (k msgServer) SubmitInferenceResult(ctx context.Context, msg *types.MsgSubmitInferenceResult) (*types.MsgSubmitInferenceResultResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate serving node address
	servingNodeAddr, err := k.addressCodec.StringToBytes(msg.ServingNode)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "invalid serving node address: %s", msg.ServingNode)
	}
	_ = servingNodeAddr

	// 2. Get inference request
	request, err := k.InferenceRequests.Get(ctx, msg.RequestId)
	if err != nil {
		return nil, errorsmod.Wrap(err, "inference request not found")
	}

	// 3. Verify serving node matches
	if request.ServingNode != msg.ServingNode {
		return nil, errorsmod.Wrap(err, "serving node does not match request")
	}

	// 4. Check if request is still pending
	if request.Status != "pending" {
		return nil, errorsmod.Wrap(err, "inference request is not pending")
	}

	// 5. Validate result IPFS hash
	resultHash := strings.TrimSpace(msg.ResultIpfsHash)
	if resultHash == "" {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "result IPFS hash cannot be empty")
	}
	if !strings.HasPrefix(resultHash, "Qm") && !strings.HasPrefix(resultHash, "bafy") && !strings.HasPrefix(resultHash, "bafk") {
		return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "invalid IPFS hash format: %s", resultHash)
	}

	// 6. Verify result exists in IPFS
	if err := k.VerifyIPFSContentExists(sdkCtx, resultHash); err != nil {
		return nil, errorsmod.Wrap(err, "result IPFS content does not exist")
	}

	// 7. Update inference request
	request.Status = "completed"
	request.ResultIpfsHash = resultHash
	request.LatencyMs = msg.LatencyMs
	responseTime := sdkCtx.BlockTime()
	request.ResponseTime = &responseTime
	if err := k.InferenceRequests.Set(ctx, msg.RequestId, request); err != nil {
		return nil, errorsmod.Wrap(err, "failed to update inference request")
	}

	// 8. Distribute serving reward (80% to serving node, 20% to miners)
	fee, err := sdk.ParseCoinsNormalized(request.Fee)
	if err == nil && !fee.IsZero() {
		// Calculate performance score based on latency (lower latency = higher score)
		// Target latency: 100ms, max acceptable: 1000ms
		targetLatency := uint64(100)
		maxLatency := uint64(1000)
		performanceScore := 1.0
		if msg.LatencyMs > targetLatency {
			if msg.LatencyMs > maxLatency {
				performanceScore = 0.1 // Very slow
			} else {
				// Linear interpolation between target and max
				performanceScore = 1.0 - (float64(msg.LatencyMs-targetLatency) / float64(maxLatency-targetLatency)) * 0.9
			}
		}
		
		// Fee distribution: 70% serving node, 20% treasury, 10% miners pool
		// Calculate shares
		servingFee70Percent := sdk.NewCoins()
		treasuryFee20Percent := sdk.NewCoins()
		minersFee10Percent := sdk.NewCoins()
		
		for _, coin := range fee {
			amount70Percent := coin.Amount.MulRaw(70).QuoRaw(100)
			amount20Percent := coin.Amount.MulRaw(20).QuoRaw(100)
			amount10Percent := coin.Amount.MulRaw(10).QuoRaw(100)
			
			if !amount70Percent.IsZero() {
				servingFee70Percent = servingFee70Percent.Add(sdk.NewCoin(coin.Denom, amount70Percent))
			}
			if !amount20Percent.IsZero() {
				treasuryFee20Percent = treasuryFee20Percent.Add(sdk.NewCoin(coin.Denom, amount20Percent))
			}
			if !amount10Percent.IsZero() {
				minersFee10Percent = minersFee10Percent.Add(sdk.NewCoin(coin.Denom, amount10Percent))
			}
		}
		
		// Send 70% to serving node
		if !servingFee70Percent.IsZero() {
			if err := k.bankKeeper.SendCoinsFromModuleToAccount(ctx, types.ModuleName, servingNodeAddr, servingFee70Percent); err != nil {
				sdkCtx.Logger().Error(fmt.Sprintf("Failed to send serving fee: %v", err))
			}
		}
		
		// Additional performance-based reward (from serving node's share)
		if err := k.DistributeServingReward(ctx, msg.ServingNode, performanceScore); err != nil {
			sdkCtx.Logger().Error(fmt.Sprintf("Failed to distribute serving reward: %v", err))
		}
		
		// Collect 20% for treasury (inference revenue)
		if !treasuryFee20Percent.IsZero() {
			if err := k.CollectInferenceRevenue(sdkCtx, treasuryFee20Percent); err != nil {
				sdkCtx.Logger().Error(fmt.Sprintf("Failed to collect treasury revenue: %v", err))
			}
		}
		
		// Distribute 10% to miners pool (proportional to mining contributions)
		if !minersFee10Percent.IsZero() {
			if err := k.DistributeMinersInferenceReward(sdkCtx, minersFee10Percent); err != nil {
				sdkCtx.Logger().Error(fmt.Sprintf("Failed to distribute miners inference reward: %v", err))
			}
		}
	}

	// 9. Update serving node status
	servingStatus, err := k.ServingNodeStatuses.Get(ctx, msg.ServingNode)
	if err != nil {
		// Create new status if doesn't exist
		servingStatus = types.ServingNodeStatus{
			NodeAddress:       msg.ServingNode,
			ModelVersion:      request.ModelVersion,
			IsAvailable:       true,
			TotalRequests:      0,
			SuccessfulRequests: 0,
			AverageLatencyMs:   0,
			LastHeartbeat:      sdkCtx.BlockTime(),
		}
	}
	servingStatus.TotalRequests++
	servingStatus.SuccessfulRequests++
	// Update average latency (simplified calculation)
	if servingStatus.AverageLatencyMs == 0 {
		servingStatus.AverageLatencyMs = msg.LatencyMs
	} else {
		servingStatus.AverageLatencyMs = (servingStatus.AverageLatencyMs + msg.LatencyMs) / 2
	}
	servingStatus.LastHeartbeat = sdkCtx.BlockTime()
	if err := k.ServingNodeStatuses.Set(ctx, msg.ServingNode, servingStatus); err != nil {
		return nil, errorsmod.Wrap(err, "failed to update serving node status")
	}

	// 10. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeSubmitInferenceResult,
			sdk.NewAttribute(types.AttributeKeyRequestID, msg.RequestId),
			sdk.NewAttribute(types.AttributeKeyServingNode, msg.ServingNode),
			sdk.NewAttribute(types.AttributeKeyResultIPFSHash, resultHash),
			sdk.NewAttribute(types.AttributeKeyLatencyMs, fmt.Sprintf("%d", msg.LatencyMs)),
		),
	)

	return &types.MsgSubmitInferenceResultResponse{
		Accepted: true,
	}, nil
}

// UpdateServingNodeStatus handles MsgUpdateServingNodeStatus
// Serving node updates its status and model version
func (k msgServer) UpdateServingNodeStatus(ctx context.Context, msg *types.MsgUpdateServingNodeStatus) (*types.MsgUpdateServingNodeStatusResponse, error) {
	sdkCtx := sdk.UnwrapSDKContext(ctx)

	// 1. Validate serving node address
	servingNodeAddr, err := k.addressCodec.StringToBytes(msg.ServingNode)
	if err != nil {
		return nil, errorsmod.Wrapf(types.ErrInvalidNodeAddress, "invalid serving node address: %s", msg.ServingNode)
	}
	_ = servingNodeAddr
	_ = servingNodeAddr

	// 2. Verify serving node is registered
	registration, err := k.NodeRegistrations.Get(sdkCtx, msg.ServingNode)
	if err != nil {
		return nil, errorsmod.Wrap(err, "serving node not registered")
	}

	// Check if node has serving role
	hasServingRole := false
	for _, role := range registration.Roles {
		if role == types.NODE_TYPE_SERVING {
			hasServingRole = true
			break
		}
	}
	if !hasServingRole {
		return nil, errorsmod.Wrap(err, "node does not have serving role")
	}

	// 3. Validate model IPFS hash if provided
	if msg.ModelIpfsHash != "" {
		modelHash := strings.TrimSpace(msg.ModelIpfsHash)
		if !strings.HasPrefix(modelHash, "Qm") && !strings.HasPrefix(modelHash, "bafy") && !strings.HasPrefix(modelHash, "bafk") {
			return nil, errorsmod.Wrapf(types.ErrInvalidIPFSHash, "invalid IPFS hash format: %s", modelHash)
		}
	}

	// 4. Get or create serving node status
	servingStatus, err := k.ServingNodeStatuses.Get(ctx, msg.ServingNode)
	if err != nil {
		// Create new status
		servingStatus = types.ServingNodeStatus{
			NodeAddress:        msg.ServingNode,
			ModelVersion:        msg.ModelVersion,
			ModelIpfsHash:       msg.ModelIpfsHash,
			IsAvailable:        msg.IsAvailable,
			TotalRequests:       0,
			SuccessfulRequests:  0,
			AverageLatencyMs:    0,
			LastHeartbeat:       sdkCtx.BlockTime(),
		}
	} else {
		// Update existing status
		if msg.ModelVersion != "" {
			servingStatus.ModelVersion = msg.ModelVersion
		}
		if msg.ModelIpfsHash != "" {
			servingStatus.ModelIpfsHash = msg.ModelIpfsHash
		}
		servingStatus.IsAvailable = msg.IsAvailable
		servingStatus.LastHeartbeat = sdkCtx.BlockTime()
	}

	// 5. Store serving node status
	if err := k.ServingNodeStatuses.Set(ctx, msg.ServingNode, servingStatus); err != nil {
		return nil, errorsmod.Wrap(err, "failed to update serving node status")
	}

	// 6. Emit event
	sdkCtx.EventManager().EmitEvent(
		sdk.NewEvent(
			types.EventTypeUpdateServingNodeStatus,
			sdk.NewAttribute(types.AttributeKeyServingNode, msg.ServingNode),
			sdk.NewAttribute(types.AttributeKeyModelVersion, msg.ModelVersion),
			sdk.NewAttribute(types.AttributeKeyIsAvailable, fmt.Sprintf("%t", msg.IsAvailable)),
		),
	)

	return &types.MsgUpdateServingNodeStatusResponse{
		Updated: true,
	}, nil
}

