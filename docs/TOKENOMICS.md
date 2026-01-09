# R3MES Tokenomics

**Version**: 1.0  
**Last Updated**: 2025-01-01  
**Status**: Final

---

## Overview

R3MES (REMES) is the native token of the R3MES network, powering the Proof of Useful Work (PoUW) consensus mechanism for decentralized AI model training.

## Token Specifications

| Property | Value |
|----------|-------|
| **Token Name** | R3MES |
| **Token Symbol** | REMES |
| **Base Denomination** | uremes (micro-REMES) |
| **Decimals** | 6 |
| **Conversion** | 1 REMES = 1,000,000 uremes |
| **Total Supply** | 1,000,000,000 REMES (1 billion) |
| **Initial Circulating** | 300,000,000 REMES (30%) |

## Token Distribution

| Category | Allocation | Amount | Vesting |
|----------|------------|--------|---------|
| **Community Pool** | 20% | 200,000,000 REMES | Immediate (governance controlled) |
| **Mining Rewards** | 30% | 300,000,000 REMES | Released over 10 years |
| **Team & Foundation** | 15% | 150,000,000 REMES | 4-year vesting, 1-year cliff |
| **Ecosystem Development** | 15% | 150,000,000 REMES | 2-year vesting |
| **Public Sale** | 10% | 100,000,000 REMES | Immediate |
| **Genesis Validators** | 5% | 50,000,000 REMES | Immediate (staked) |
| **Reserve** | 5% | 50,000,000 REMES | Locked (governance unlock) |

### Distribution Chart

```
Mining Rewards (30%) ████████████████████████████████
Community Pool (20%) █████████████████████
Team & Foundation (15%) ████████████████
Ecosystem Dev (15%) ████████████████
Public Sale (10%) ███████████
Genesis Validators (5%) ██████
Reserve (5%) ██████
```

## Inflation & Emission

### Inflation Parameters

| Parameter | Value |
|-----------|-------|
| **Initial Inflation** | 10% annual |
| **Minimum Inflation** | 5% annual |
| **Maximum Inflation** | 20% annual |
| **Goal Bonded Ratio** | 67% |
| **Inflation Adjustment** | Dynamic based on staking ratio |

### Emission Schedule

| Year | Mining Rewards | Cumulative |
|------|----------------|------------|
| 1 | 60,000,000 REMES | 60,000,000 |
| 2 | 54,000,000 REMES | 114,000,000 |
| 3 | 48,600,000 REMES | 162,600,000 |
| 4 | 43,740,000 REMES | 206,340,000 |
| 5 | 39,366,000 REMES | 245,706,000 |
| 6-10 | ~54,294,000 REMES | 300,000,000 |

*Mining rewards decrease by 10% annually*

## Staking Economics

### Staking Parameters

| Parameter | Value |
|-----------|-------|
| **Unbonding Period** | 21 days |
| **Maximum Validators** | 100 |
| **Minimum Self-Delegation** | 1,000 REMES |
| **Redelegation Cooldown** | 7 days |

### Validator Commission

| Parameter | Value |
|-----------|-------|
| **Minimum Commission** | 5% |
| **Maximum Commission** | 100% |
| **Max Commission Change** | 1% per day |

### Staking Rewards

Staking rewards come from:
1. **Block Rewards**: New token inflation
2. **Transaction Fees**: Network usage fees
3. **Slashing Penalties**: Redistributed from slashed validators

Expected APY: 8-15% (varies with staking ratio)

## Mining Economics

### Gradient Rewards

Miners earn REMES by submitting valid gradients:

| Quality Tier | Reward Multiplier | Requirements |
|--------------|-------------------|--------------|
| **Platinum** | 2.0x | Loss improvement >5%, Trust Score >95 |
| **Gold** | 1.5x | Loss improvement >3%, Trust Score >85 |
| **Silver** | 1.2x | Loss improvement >1%, Trust Score >70 |
| **Bronze** | 1.0x | Valid gradient, Trust Score >50 |

### Base Reward Formula

```
Reward = BaseReward × QualityMultiplier × (1 + TrustBonus)

Where:
- BaseReward = 10 REMES per valid gradient
- QualityMultiplier = 1.0 - 2.0 based on tier
- TrustBonus = 0 - 0.2 based on reputation
```

### Mining Costs

| Resource | Estimated Cost |
|----------|----------------|
| GPU (RTX 4090) | ~$0.50/hour |
| Electricity | ~$0.10/hour |
| Network | ~$0.01/hour |
| **Total** | ~$0.61/hour |

Expected earnings: 2-5 REMES/hour (varies with network difficulty)

## Transaction Fees

### Fee Structure

| Transaction Type | Base Fee |
|------------------|----------|
| Transfer | 0.001 REMES |
| Delegate | 0.01 REMES |
| Undelegate | 0.01 REMES |
| Submit Gradient | 0.1 REMES |
| Governance Vote | 0.001 REMES |
| Create Proposal | 100 REMES (deposit) |

### Fee Distribution

| Recipient | Share |
|-----------|-------|
| Validators | 50% |
| Community Pool | 30% |
| Burn | 20% |

## Slashing

### Slashing Conditions

| Violation | Penalty | Jail Duration |
|-----------|---------|---------------|
| Double Signing | 5% of stake | Permanent |
| Downtime (>95% missed) | 0.01% of stake | 10 minutes |
| False Gradient Verification | 1% of stake | 24 hours |
| Lazy Validation | 0.1% of stake | 1 hour |

## Governance

### Proposal Types

| Type | Deposit | Voting Period |
|------|---------|---------------|
| Text Proposal | 100 REMES | 14 days |
| Parameter Change | 500 REMES | 14 days |
| Software Upgrade | 1,000 REMES | 21 days |
| Model Upgrade | 1,000 REMES | 21 days |
| Community Spend | 500 REMES | 14 days |

### Voting Thresholds

| Parameter | Value |
|-----------|-------|
| **Quorum** | 40% of staked tokens |
| **Pass Threshold** | 50% Yes votes |
| **Veto Threshold** | 33.4% NoWithVeto |

## Token Utility

### Primary Uses

1. **Staking**: Secure the network and earn rewards
2. **Mining**: Pay for gradient submission fees
3. **Governance**: Vote on proposals and upgrades
4. **Inference**: Pay for AI model inference services
5. **Data Marketplace**: Purchase training datasets

### Secondary Uses

1. **Validator Collateral**: Required for running validators
2. **Miner Reputation**: Stake for higher trust scores
3. **Priority Access**: Higher stakes = priority inference

## Economic Security

### Attack Cost Analysis

| Attack Type | Estimated Cost |
|-------------|----------------|
| 51% Attack | >$500M (at $1/REMES) |
| Gradient Spam | Rate limited + fees |
| Sybil Attack | Stake requirements |

### Security Measures

1. **Slashing**: Economic penalties for misbehavior
2. **Bonding**: 21-day unbonding prevents quick exits
3. **Governance**: Community can respond to attacks
4. **Rate Limiting**: Prevents spam attacks

## Future Considerations

### Potential Adjustments (via Governance)

- Inflation rate adjustments
- Fee structure changes
- Reward distribution modifications
- New token utility features

### Deflationary Mechanisms

- 20% fee burn
- Potential future burn proposals
- Unused community pool burns

---

## Summary

R3MES tokenomics are designed to:

1. **Incentivize Participation**: Rewards for miners, validators, and stakers
2. **Ensure Security**: Economic penalties for malicious behavior
3. **Enable Governance**: Token holders control network evolution
4. **Support Growth**: Sustainable emission for long-term development

For questions, join our [Discord](https://discord.gg/r3mes) or email tokenomics@r3mes.network.
