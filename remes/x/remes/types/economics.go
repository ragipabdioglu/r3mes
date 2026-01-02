package types

// EconomicParams represents economic parameters for the system
type EconomicParams struct {
	RewardPerGradient string `json:"reward_per_gradient"`
	SlashingPenalty   string `json:"slashing_penalty"`
	MinStake          string `json:"min_stake"`
	StakeDenom        string `json:"stake_denom"`
}

// StakingInfo represents staking information for a participant
type StakingInfo struct {
	Address       string `json:"address"`
	StakedAmount  string `json:"staked_amount"`
	DelegatedTo   string `json:"delegated_to,omitempty"`
	RewardAddress string `json:"reward_address,omitempty"`
}

// Validate validates economic parameters
func (ep EconomicParams) Validate() error {
	if ep.RewardPerGradient == "" {
		return ErrInvalidParameter.Wrapf("reward_per_gradient cannot be empty")
	}
	if ep.SlashingPenalty == "" {
		return ErrInvalidParameter.Wrapf("slashing_penalty cannot be empty")
	}
	if ep.MinStake == "" {
		return ErrInvalidParameter.Wrapf("min_stake cannot be empty")
	}
	if ep.StakeDenom == "" {
		return ErrInvalidParameter.Wrapf("stake_denom cannot be empty")
	}
	return nil
}

// Validate validates staking information
func (si StakingInfo) Validate() error {
	if si.Address == "" {
		return ErrInvalidParameter.Wrapf("address cannot be empty")
	}
	if si.StakedAmount == "" {
		return ErrInvalidParameter.Wrapf("staked_amount cannot be empty")
	}
	return nil
}
