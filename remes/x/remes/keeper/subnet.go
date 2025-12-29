package keeper

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"

	sdk "github.com/cosmos/cosmos-sdk/types"

	"remes/x/remes/types"
)

// AssignMinerToSubnet assigns a miner to a subnet based on deterministic algorithm
// Formula: (miner_address + block_hash) % total_subnets
// This ensures:
// - Same miner gets same subnet within a window (stable)
// - Different miners get different subnets (distributed)
// - Different windows can have different assignments (flexible)
func (k Keeper) AssignMinerToSubnet(
	ctx sdk.Context,
	minerAddress string,
	totalSubnets uint64,
	windowID uint64,
) (uint64, error) {
	if totalSubnets == 0 {
		return 0, fmt.Errorf("total_subnets cannot be zero")
	}

	// Get block hash (or block height as fallback)
	blockHash := ctx.HeaderHash()
	if len(blockHash) == 0 {
		// Fallback: use block height as deterministic seed
		blockHash = []byte(fmt.Sprintf("block_%d", ctx.BlockHeight()))
	}

	// Create deterministic input: miner_address + block_hash + window_id
	seed := append([]byte(minerAddress), blockHash...)
	seed = append(seed, []byte(fmt.Sprintf("%d", windowID))...)
	hash := sha256.Sum256(seed)

	// Convert first 8 bytes to uint64
	subnetID := binary.BigEndian.Uint64(hash[:8]) % totalSubnets

	return subnetID, nil
}

// CreateSubnet creates a new subnet configuration
func (k Keeper) CreateSubnet(
	ctx sdk.Context,
	subnetID uint64,
	layerRange types.LayerRange,
	nextSubnetID uint64,
	windowID uint64,
) error {
	// Validate layer range
	if layerRange.StartLayer >= layerRange.EndLayer {
		return fmt.Errorf("invalid layer range: start_layer (%d) must be less than end_layer (%d)", layerRange.StartLayer, layerRange.EndLayer)
	}

	// Check if subnet already exists
	_, err := k.SubnetConfigs.Get(ctx, subnetID)
	if err == nil {
		return fmt.Errorf("subnet %d already exists", subnetID)
	}

	// Create subnet config
	now := ctx.BlockTime()
	subnetConfig := types.SubnetConfig{
		SubnetId:      subnetID,
		LayerRange:    layerRange,
		AssignedMiners: []string{},
		ActivationHash: "",
		NextSubnetId:  nextSubnetID,
		Status:        "active",
		WindowId:      windowID,
		CreatedAt:     now,
		UpdatedAt:     now,
	}

	// Store subnet config
	if err := k.SubnetConfigs.Set(ctx, subnetID, subnetConfig); err != nil {
		return fmt.Errorf("failed to store subnet config: %w", err)
	}

	return nil
}

// GetSubnetConfig retrieves a subnet configuration
func (k Keeper) GetSubnetConfig(ctx sdk.Context, subnetID uint64) (types.SubnetConfig, error) {
	config, err := k.SubnetConfigs.Get(ctx, subnetID)
	if err != nil {
		return types.SubnetConfig{}, fmt.Errorf("subnet %d not found: %w", subnetID, err)
	}
	return config, nil
}

// AddMinerToSubnet adds a miner to a subnet's assigned miners list
func (k Keeper) AddMinerToSubnet(
	ctx sdk.Context,
	subnetID uint64,
	minerAddress string,
) error {
	// Get subnet config
	config, err := k.GetSubnetConfig(ctx, subnetID)
	if err != nil {
		return err
	}

	// Check if miner is already assigned
	for _, assignedMiner := range config.AssignedMiners {
		if assignedMiner == minerAddress {
			return fmt.Errorf("miner %s is already assigned to subnet %d", minerAddress, subnetID)
		}
	}

	// Add miner to list
	config.AssignedMiners = append(config.AssignedMiners, minerAddress)
	config.UpdatedAt = ctx.BlockTime()

	// Update subnet config
	if err := k.SubnetConfigs.Set(ctx, subnetID, config); err != nil {
		return fmt.Errorf("failed to update subnet config: %w", err)
	}

	return nil
}

// SubmitSubnetActivation submits activation data from a subnet
func (k Keeper) SubmitSubnetActivation(
	ctx sdk.Context,
	subnetID uint64,
	activationHash string,
	nextSubnetID uint64,
	signature []byte,
) (uint64, error) {
	// Get subnet config
	config, err := k.GetSubnetConfig(ctx, subnetID)
	if err != nil {
		return 0, err
	}

	// Validate subnet status
	if config.Status != "active" && config.Status != "aggregating" {
		return 0, fmt.Errorf("subnet %d is not in active or aggregating status (current: %s)", subnetID, config.Status)
	}

	// Generate transmission ID
	transmissionID, err := k.ActivationTransmissionID.Next(ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to generate transmission ID: %w", err)
	}

	// Create activation transmission
	transmission := types.ActivationTransmission{
		TransmissionId: transmissionID,
		FromSubnetId:  subnetID,
		ToSubnetId:    nextSubnetID,
		ActivationHash: activationHash,
		LayerOutput:   "",
		Timestamp:     ctx.BlockTime(),
		Signature:     signature,
		Verified:      false,
	}

	// Store transmission
	if err := k.ActivationTransmissions.Set(ctx, transmissionID, transmission); err != nil {
		return 0, fmt.Errorf("failed to store activation transmission: %w", err)
	}

	// Update subnet config
	config.ActivationHash = activationHash
	config.Status = "completed"
	config.UpdatedAt = ctx.BlockTime()

	if err := k.SubnetConfigs.Set(ctx, subnetID, config); err != nil {
		return 0, fmt.Errorf("failed to update subnet config: %w", err)
	}

	// Update next subnet status if it exists
	if nextSubnetID > 0 {
		nextConfig, err := k.GetSubnetConfig(ctx, nextSubnetID)
		if err == nil {
			nextConfig.Status = "active"
			nextConfig.UpdatedAt = ctx.BlockTime()
			if err := k.SubnetConfigs.Set(ctx, nextSubnetID, nextConfig); err != nil {
				return 0, fmt.Errorf("failed to update next subnet config: %w", err)
			}
		}
	}

	return transmissionID, nil
}

// CreateSubnetTrainingWorkflow creates a new subnet training workflow
func (k Keeper) CreateSubnetTrainingWorkflow(
	ctx sdk.Context,
	windowID uint64,
	subnets []types.SubnetConfig,
	globalSeed []byte,
) (uint64, error) {
	// Generate workflow ID
	workflowID, err := k.SubnetTrainingWorkflowID.Next(ctx)
	if err != nil {
		return 0, fmt.Errorf("failed to generate workflow ID: %w", err)
	}

	// Create workflow
	workflow := types.SubnetTrainingWorkflow{
		WorkflowId:     workflowID,
		WindowId:       windowID,
		Subnets:        subnets,
		ActivationChain: []string{},
		GlobalSeed:     globalSeed,
		Status:         "initializing",
		CreatedAt:     ctx.BlockTime(),
		CompletedAt:   nil,
	}

	// Store workflow
	if err := k.SubnetTrainingWorkflows.Set(ctx, workflowID, workflow); err != nil {
		return 0, fmt.Errorf("failed to store workflow: %w", err)
	}

	return workflowID, nil
}

// GetSubnetTrainingWorkflow retrieves a subnet training workflow
func (k Keeper) GetSubnetTrainingWorkflow(ctx sdk.Context, workflowID uint64) (types.SubnetTrainingWorkflow, error) {
	workflow, err := k.SubnetTrainingWorkflows.Get(ctx, workflowID)
	if err != nil {
		return types.SubnetTrainingWorkflow{}, fmt.Errorf("workflow %d not found: %w", workflowID, err)
	}
	return workflow, nil
}

// AddActivationToWorkflow adds an activation hash to the workflow's activation chain
func (k Keeper) AddActivationToWorkflow(
	ctx sdk.Context,
	workflowID uint64,
	activationHash string,
) error {
	// Get workflow
	workflow, err := k.GetSubnetTrainingWorkflow(ctx, workflowID)
	if err != nil {
		return err
	}

	// Add activation hash to chain
	workflow.ActivationChain = append(workflow.ActivationChain, activationHash)

	// Update workflow
	if err := k.SubnetTrainingWorkflows.Set(ctx, workflowID, workflow); err != nil {
		return fmt.Errorf("failed to update workflow: %w", err)
	}

	return nil
}

// CompleteSubnetTrainingWorkflow marks a workflow as completed
func (k Keeper) CompleteSubnetTrainingWorkflow(
	ctx sdk.Context,
	workflowID uint64,
) error {
	// Get workflow
	workflow, err := k.GetSubnetTrainingWorkflow(ctx, workflowID)
	if err != nil {
		return err
	}

	// Update workflow status
	workflow.Status = "completed"
	completedAt := ctx.BlockTime()
	workflow.CompletedAt = &completedAt

	// Update workflow
	if err := k.SubnetTrainingWorkflows.Set(ctx, workflowID, workflow); err != nil {
		return fmt.Errorf("failed to update workflow: %w", err)
	}

	return nil
}

