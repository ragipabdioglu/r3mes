package types

import (
	"encoding/json"
)

const (
	// IBCVersion defines the IBC version for R3MES module
	IBCVersion = "remes-1"

	// IBCPortID defines the default port ID for R3MES IBC module
	IBCPortID = "remes"
)

// IBCGradientPacketData represents gradient data sent via IBC
type IBCGradientPacketData struct {
	GradientID      uint64 `json:"gradient_id"`
	MinerAddress    string `json:"miner_address"`
	IPFSHash        string `json:"ipfs_hash"`
	ModelVersion    string `json:"model_version"`
	TrainingRoundID uint64 `json:"training_round_id"`
	ShardID         uint64 `json:"shard_id"`
	GradientHash    string `json:"gradient_hash"`
	GPUArchitecture string `json:"gpu_architecture"`
	SourceChain     string `json:"source_chain"`
}

// IBCGradientPacketAck represents acknowledgement for gradient packet
type IBCGradientPacketAck struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

// ValidateBasic performs basic validation of IBC gradient packet data
func (data IBCGradientPacketData) ValidateBasic() error {
	if data.GradientID == 0 {
		return ErrInvalidGradientID
	}

	if data.MinerAddress == "" {
		return ErrInvalidMinerAddress
	}

	if data.IPFSHash == "" {
		return ErrInvalidIPFSHash
	}

	if data.ModelVersion == "" {
		return ErrInvalidModelVersion
	}

	if data.GradientHash == "" {
		return ErrInvalidGradientHash
	}

	if data.SourceChain == "" {
		return ErrInvalidSourceChain
	}

	return nil
}

// GetBytes returns the byte representation of the packet data
func (data IBCGradientPacketData) GetBytes() []byte {
	// Use standard JSON marshaling instead of ModuleCdc
	bz, err := json.Marshal(&data)
	if err != nil {
		panic(err)
	}
	return bz
}
