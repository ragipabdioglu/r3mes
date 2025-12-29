package types

import (
	"cosmossdk.io/math"
)

// NewParams creates a new Params instance.
func NewParams() Params {
	return Params{
		ScalabilityParams: DefaultScalabilityParams(),
	}
}

// DefaultParams returns a default set of parameters.
func DefaultParams() Params {
	return Params{
		ChallengePeriodBlocks: 100, // Default: 100 blocks (~8 minutes at 5s/block)
		ScalabilityParams:      DefaultScalabilityParams(),
	}
}

// Validate validates the set of params.
func (p Params) Validate() error {
	if err := p.ScalabilityParams.Validate(); err != nil {
		return err
	}
	return nil
}

// DefaultScalabilityParams returns default scalability parameters.
func DefaultScalabilityParams() ScalabilityParams {
	return ScalabilityParams{
		MaxParticipantsPerShard:     100,
		MaxGradientsPerBlock:        50,
		MaxAggregationsPerBlock:     10,
		MaxPendingGradients:         200,
		MaxPendingAggregations:      50,
		CompressionRatioThreshold:   "0.9",
		NetworkLoadThreshold:       "0.8",
		ResourceUtilizationThreshold: "0.8",
		EnableLoadBalancing:         true,
		LoadBalancingStrategy:       "round_robin",
		ShardReassignmentInterval:   100,
		EnableAdaptiveCompression:   true,
		EnableAdaptiveSharding:     true,
		EnableResourceOptimization: true,
	}
}

// Validate validates scalability parameters.
func (sp ScalabilityParams) Validate() error {
	if sp.MaxParticipantsPerShard == 0 {
		return ErrInvalidParameter
	}
	if sp.MaxGradientsPerBlock == 0 {
		return ErrInvalidParameter
	}
	if sp.MaxAggregationsPerBlock == 0 {
		return ErrInvalidParameter
	}
	// Validate decimal strings
	if _, err := math.LegacyNewDecFromStr(sp.CompressionRatioThreshold); err != nil {
		return ErrInvalidParameter
	}
	if _, err := math.LegacyNewDecFromStr(sp.NetworkLoadThreshold); err != nil {
		return ErrInvalidParameter
	}
	if _, err := math.LegacyNewDecFromStr(sp.ResourceUtilizationThreshold); err != nil {
		return ErrInvalidParameter
	}
	return nil
}
