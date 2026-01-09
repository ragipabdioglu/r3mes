package types

import (
	"fmt"
)

// DefaultGenesis returns the default genesis state
func DefaultGenesis() *GenesisState {
	return &GenesisState{
		Params:              DefaultParams(),
		ModelHash:           "", // Will be set during chain initialization
		ModelVersion:        "v1.0.0",
		InitialParticipants: []string{},
		StoredGradientList:  []StoredGradient{},
	}
}

// Validate performs basic genesis state validation returning an error upon any
// failure.
func (gs *GenesisState) Validate() error {
	if err := gs.Params.Validate(); err != nil {
		return err
	}
	
	// Validate model hash if provided
	if gs.ModelHash != "" {
		// Basic validation - IPFS hash should be non-empty
		if len(gs.ModelHash) < 10 {
			return fmt.Errorf("model hash appears to be invalid")
		}
	}
	
	// Validate model version
	if gs.ModelVersion == "" {
		return fmt.Errorf("model version cannot be empty")
	}
	
	// Validate stored gradients
	for i, sg := range gs.StoredGradientList {
		if sg.IpfsHash == "" {
			return fmt.Errorf("stored gradient at index %d has empty IPFS hash", i)
		}
	}
	
	return nil
}
