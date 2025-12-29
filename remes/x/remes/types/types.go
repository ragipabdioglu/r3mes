package types

import (
	"fmt"
)

// Validate validates the GlobalModelState (proto-generated type)
func (gms *GlobalModelState) Validate() error {
	if gms.ModelIpfsHash == "" {
		return fmt.Errorf("model IPFS hash cannot be empty")
	}
	if gms.ModelVersion == "" {
		return fmt.Errorf("model version cannot be empty")
	}
	return nil
}

// Validate validates the AggregationRecord (proto-generated type)
func (ar *AggregationRecord) Validate() error {
	if ar.Proposer == "" {
		return fmt.Errorf("proposer address cannot be empty")
	}
	if ar.AggregatedGradientIpfsHash == "" {
		return fmt.Errorf("aggregated gradient IPFS hash cannot be empty")
	}
	if ar.MerkleRoot == "" {
		return fmt.Errorf("merkle root cannot be empty")
	}
	if len(ar.ParticipantGradientIds) == 0 {
		return fmt.Errorf("participant gradient IDs cannot be empty")
	}
	return nil
}

// Validate validates the MiningContribution (proto-generated type)
func (mc *MiningContribution) Validate() error {
	if mc.MinerAddress == "" {
		return fmt.Errorf("miner address cannot be empty")
	}
	// Trust score validation can be added if needed
	return nil
}
