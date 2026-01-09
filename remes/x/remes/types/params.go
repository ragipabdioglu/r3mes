package types

import (
	"fmt"

	sdkmath "cosmossdk.io/math"
	paramtypes "github.com/cosmos/cosmos-sdk/x/params/types"
)

// Parameter store keys
var (
	KeyNonceWindowSize       = []byte("NonceWindowSize")
	KeyMinStake              = []byte("MinStake")
	KeyStakeDenom            = []byte("StakeDenom")
	KeyMiningDifficulty      = []byte("MiningDifficulty")
	KeyRewardPerGradient     = []byte("RewardPerGradient")
	KeyMaxValidators         = []byte("MaxValidators")
	KeySlashingPenalty       = []byte("SlashingPenalty")
	KeyChallengePeriodBlocks = []byte("ChallengePeriodBlocks")
	KeyScalabilityParams     = []byte("ScalabilityParams")
)

// Default parameter values
const (
	DefaultNonceWindowSize       uint64 = 10000
	DefaultStakeDenom            string = "remes"
	DefaultMiningDifficulty      string = "1234.0"
	DefaultRewardPerGradient     string = "0.1"
	DefaultMaxValidators         uint64 = 100
	DefaultSlashingPenalty       string = "0.05" // 5%
	DefaultChallengePeriodBlocks int64  = 100    // ~8 minutes at 5s/block
)

var (
	DefaultMinStake = sdkmath.NewInt(1000) // 1000 tokens
)

// DefaultScalabilityParams returns default scalability parameters
func DefaultScalabilityParams() ScalabilityParams {
	return ScalabilityParams{
		MaxParticipantsPerShard:      1000,
		MaxGradientsPerBlock:         100,
		MaxAggregationsPerBlock:      50,
		MaxPendingGradients:          10000,
		MaxPendingAggregations:       1000,
		CompressionRatioThreshold:    "0.9",
		NetworkLoadThreshold:         "0.8",
		ResourceUtilizationThreshold: "0.8",
		EnableLoadBalancing:          true,
		LoadBalancingStrategy:        "round_robin",
		ShardReassignmentInterval:    100,
		EnableAdaptiveCompression:    true,
		EnableAdaptiveSharding:       true,
		EnableResourceOptimization:   true,
	}
}

// NewParams creates a new Params instance
func NewParams(
	nonceWindowSize uint64,
	minStake sdkmath.Int,
	stakeDenom string,
	miningDifficulty string,
	rewardPerGradient string,
	maxValidators uint64,
	slashingPenalty string,
	challengePeriodBlocks int64,
	scalabilityParams ScalabilityParams,
) Params {
	return Params{
		NonceWindowSize:       nonceWindowSize,
		MinStake:              minStake,
		StakeDenom:            stakeDenom,
		MiningDifficulty:      miningDifficulty,
		RewardPerGradient:     rewardPerGradient,
		MaxValidators:         maxValidators,
		SlashingPenalty:       slashingPenalty,
		ChallengePeriodBlocks: challengePeriodBlocks,
		ScalabilityParams:     scalabilityParams,
	}
}

// DefaultParams returns a default set of parameters
func DefaultParams() Params {
	return NewParams(
		DefaultNonceWindowSize,
		DefaultMinStake,
		DefaultStakeDenom,
		DefaultMiningDifficulty,
		DefaultRewardPerGradient,
		DefaultMaxValidators,
		DefaultSlashingPenalty,
		DefaultChallengePeriodBlocks,
		DefaultScalabilityParams(),
	)
}

// ParamSetPairs get the params.ParamSet
func (p *Params) ParamSetPairs() paramtypes.ParamSetPairs {
	return paramtypes.ParamSetPairs{
		paramtypes.NewParamSetPair(KeyNonceWindowSize, &p.NonceWindowSize, validateNonceWindowSize),
		paramtypes.NewParamSetPair(KeyMinStake, &p.MinStake, validateMinStake),
		paramtypes.NewParamSetPair(KeyStakeDenom, &p.StakeDenom, validateStakeDenom),
		paramtypes.NewParamSetPair(KeyMiningDifficulty, &p.MiningDifficulty, validateMiningDifficulty),
		paramtypes.NewParamSetPair(KeyRewardPerGradient, &p.RewardPerGradient, validateRewardPerGradient),
		paramtypes.NewParamSetPair(KeyMaxValidators, &p.MaxValidators, validateMaxValidators),
		paramtypes.NewParamSetPair(KeySlashingPenalty, &p.SlashingPenalty, validateSlashingPenalty),
		paramtypes.NewParamSetPair(KeyChallengePeriodBlocks, &p.ChallengePeriodBlocks, validateChallengePeriodBlocks),
		paramtypes.NewParamSetPair(KeyScalabilityParams, &p.ScalabilityParams, validateScalabilityParams),
	}
}

// Validate validates the set of params
func (p Params) Validate() error {
	if err := validateNonceWindowSize(p.NonceWindowSize); err != nil {
		return err
	}
	if err := validateMinStake(p.MinStake); err != nil {
		return err
	}
	if err := validateStakeDenom(p.StakeDenom); err != nil {
		return err
	}
	if err := validateMiningDifficulty(p.MiningDifficulty); err != nil {
		return err
	}
	if err := validateRewardPerGradient(p.RewardPerGradient); err != nil {
		return err
	}
	if err := validateMaxValidators(p.MaxValidators); err != nil {
		return err
	}
	if err := validateSlashingPenalty(p.SlashingPenalty); err != nil {
		return err
	}
	if err := validateChallengePeriodBlocks(p.ChallengePeriodBlocks); err != nil {
		return err
	}
	if err := validateScalabilityParams(p.ScalabilityParams); err != nil {
		return err
	}
	return nil
}

// Validation functions

func validateNonceWindowSize(i interface{}) error {
	v, ok := i.(uint64)
	if !ok {
		return fmt.Errorf("invalid parameter type: %T", i)
	}

	if v < 1000 {
		return fmt.Errorf("nonce window size must be at least 1000: %d", v)
	}

	if v > 1000000 {
		return fmt.Errorf("nonce window size must be at most 1,000,000: %d", v)
	}

	return nil
}

func validateMinStake(i interface{}) error {
	v, ok := i.(sdkmath.Int)
	if !ok {
		return fmt.Errorf("invalid parameter type: %T", i)
	}

	if v.IsNil() {
		return fmt.Errorf("min stake cannot be nil")
	}

	if v.IsNegative() {
		return fmt.Errorf("min stake cannot be negative: %s", v)
	}

	if v.GT(sdkmath.NewInt(1000000000)) { // 1 billion tokens max
		return fmt.Errorf("min stake too large: %s", v)
	}

	return nil
}

func validateStakeDenom(i interface{}) error {
	v, ok := i.(string)
	if !ok {
		return fmt.Errorf("invalid parameter type: %T", i)
	}

	if len(v) == 0 {
		return fmt.Errorf("stake denom cannot be empty")
	}

	if len(v) > 128 {
		return fmt.Errorf("stake denom too long: %s", v)
	}

	// Basic denom validation (alphanumeric + some special chars)
	for _, char := range v {
		if !((char >= 'a' && char <= 'z') ||
			(char >= 'A' && char <= 'Z') ||
			(char >= '0' && char <= '9') ||
			char == '-' || char == '_') {
			return fmt.Errorf("invalid character in stake denom: %c", char)
		}
	}

	return nil
}

func validateMiningDifficulty(i interface{}) error {
	v, ok := i.(string)
	if !ok {
		return fmt.Errorf("invalid parameter type: %T", i)
	}

	if len(v) == 0 {
		return fmt.Errorf("mining difficulty cannot be empty")
	}

	// Try to parse as float to validate format
	var difficulty float64
	if _, err := fmt.Sscanf(v, "%f", &difficulty); err != nil {
		return fmt.Errorf("invalid mining difficulty format: %s", v)
	}

	if difficulty <= 0 {
		return fmt.Errorf("mining difficulty must be positive: %f", difficulty)
	}

	return nil
}

func validateRewardPerGradient(i interface{}) error {
	v, ok := i.(string)
	if !ok {
		return fmt.Errorf("invalid parameter type: %T", i)
	}

	if len(v) == 0 {
		return fmt.Errorf("reward per gradient cannot be empty")
	}

	// Try to parse as float to validate format
	var reward float64
	if _, err := fmt.Sscanf(v, "%f", &reward); err != nil {
		return fmt.Errorf("invalid reward per gradient format: %s", v)
	}

	if reward < 0 {
		return fmt.Errorf("reward per gradient cannot be negative: %f", reward)
	}

	if reward > 1000 {
		return fmt.Errorf("reward per gradient too large: %f", reward)
	}

	return nil
}

func validateMaxValidators(i interface{}) error {
	v, ok := i.(uint64)
	if !ok {
		return fmt.Errorf("invalid parameter type: %T", i)
	}

	if v < 1 {
		return fmt.Errorf("max validators must be at least 1: %d", v)
	}

	if v > 1000 {
		return fmt.Errorf("max validators must be at most 1000: %d", v)
	}

	return nil
}

func validateSlashingPenalty(i interface{}) error {
	v, ok := i.(string)
	if !ok {
		return fmt.Errorf("invalid parameter type: %T", i)
	}

	if len(v) == 0 {
		return fmt.Errorf("slashing penalty cannot be empty")
	}

	// Try to parse as float to validate format
	var penalty float64
	if _, err := fmt.Sscanf(v, "%f", &penalty); err != nil {
		return fmt.Errorf("invalid slashing penalty format: %s", v)
	}

	if penalty < 0 {
		return fmt.Errorf("slashing penalty cannot be negative: %f", penalty)
	}

	if penalty > 1 {
		return fmt.Errorf("slashing penalty cannot exceed 100%%: %f", penalty)
	}

	return nil
}

func validateChallengePeriodBlocks(i interface{}) error {
	v, ok := i.(int64)
	if !ok {
		return fmt.Errorf("invalid parameter type: %T", i)
	}

	if v < 1 {
		return fmt.Errorf("challenge period blocks must be at least 1: %d", v)
	}

	if v > 10000 {
		return fmt.Errorf("challenge period blocks must be at most 10000: %d", v)
	}

	return nil
}

func validateScalabilityParams(i interface{}) error {
	v, ok := i.(ScalabilityParams)
	if !ok {
		return fmt.Errorf("invalid parameter type: %T", i)
	}

	// Validate thresholds
	if v.MaxParticipantsPerShard < 1 {
		return fmt.Errorf("max participants per shard must be at least 1: %d", v.MaxParticipantsPerShard)
	}

	if v.MaxGradientsPerBlock < 1 {
		return fmt.Errorf("max gradients per block must be at least 1: %d", v.MaxGradientsPerBlock)
	}

	if v.MaxAggregationsPerBlock < 1 {
		return fmt.Errorf("max aggregations per block must be at least 1: %d", v.MaxAggregationsPerBlock)
	}

	// Validate threshold strings
	thresholds := []string{
		v.CompressionRatioThreshold,
		v.NetworkLoadThreshold,
		v.ResourceUtilizationThreshold,
	}

	for _, threshold := range thresholds {
		var value float64
		if _, err := fmt.Sscanf(threshold, "%f", &value); err != nil {
			return fmt.Errorf("invalid threshold format: %s", threshold)
		}
		if value < 0 || value > 1 {
			return fmt.Errorf("threshold must be between 0 and 1: %f", value)
		}
	}

	// Validate load balancing strategy
	validStrategies := []string{"round_robin", "weighted", "least_connections"}
	validStrategy := false
	for _, strategy := range validStrategies {
		if v.LoadBalancingStrategy == strategy {
			validStrategy = true
			break
		}
	}
	if !validStrategy {
		return fmt.Errorf("invalid load balancing strategy: %s", v.LoadBalancingStrategy)
	}

	return nil
}
