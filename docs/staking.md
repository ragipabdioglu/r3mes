# Staking Guide

Stake REMES tokens to earn passive rewards and help secure the network.

## Overview

Staking allows you to:

| Benefit | Description |
|---------|-------------|
| Earn Rewards | 8-15% APY on staked tokens |
| Secure Network | Your stake helps validate transactions |
| Governance | Vote on protocol proposals |
| Support Validators | Help decentralize the network |

## How Staking Works

| Step | Description |
|------|-------------|
| 1. Choose Validator | Select based on commission, uptime, reputation |
| 2. Delegate Tokens | Lock REMES with chosen validator |
| 3. Earn Rewards | Rewards accrue every block (~5 seconds) |
| 4. Claim or Compound | Withdraw rewards or restake for compound growth |

## Quick Start

**Via Web Dashboard:**

1. Go to [app.r3mes.network/staking](https://app.r3mes.network/staking)
2. Connect your wallet (Keplr, Leap, or Cosmostation)
3. Select a validator
4. Enter amount to stake
5. Confirm transaction

**Via CLI:**

```bash
# List validators
r3mes staking validators

# Delegate to a validator
r3mes staking delegate remes1validator... 1000uremes

# Check delegations
r3mes staking delegations

# Claim rewards
r3mes staking claim-rewards
```

## Choosing a Validator

### Key Metrics

| Metric | What to Look For | Why It Matters |
|--------|------------------|----------------|
| Commission | 5-10% | Lower = more rewards for you |
| Uptime | >99% | Higher = more consistent rewards |
| Voting Power | <10% | Avoid over-concentration |
| Self-Stake | >10,000 REMES | Shows validator commitment |
| Governance | Active voting | Engaged in protocol decisions |

### Validator Tiers

| Tier | Trust Score | Commission | Recommendation |
|------|-------------|------------|----------------|
| Diamond | ≥0.95 | 3-5% | Best for large stakes |
| Platinum | ≥0.90 | 5-7% | Excellent choice |
| Gold | ≥0.80 | 7-10% | Good balance |
| Silver | ≥0.70 | 10-15% | Acceptable |
| Bronze | <0.70 | 15%+ | Higher risk |

### Red Flags

Avoid validators with:
- Commission >20%
- Uptime <95%
- Recent slashing events
- No self-stake
- Inactive governance participation

## Staking Operations

### Delegate

```bash
r3mes staking delegate remes1validator... 1000000000uremes
```

### Undelegate

```bash
r3mes staking undelegate remes1validator... 500000000uremes
```

**Note:** Undelegation has a 21-day unbonding period. During this time, tokens are locked and earn no rewards.

### Redelegate

Move stake between validators without unbonding:

```bash
r3mes staking redelegate remes1validatorA... remes1validatorB... 500000000uremes
```

**Note:** You can only redelegate once per 21 days from the same source validator.

### Claim Rewards

```bash
# Claim from all validators
r3mes staking claim-rewards --all

# Auto-compound (claim and restake)
r3mes staking compound --all
```

## Reward Calculation

### APY Formula

```
Annual Rewards = Staked Amount × Network Inflation × (1 - Community Tax) × (1 - Validator Commission)
```

### Example

| Parameter | Value |
|-----------|-------|
| Staked Amount | 10,000 REMES |
| Network Inflation | 12% |
| Community Tax | 2% |
| Validator Commission | 5% |

**Result:** ~1,117 REMES/year (11.17% APY)

### Reward Distribution

| Timeframe | Approximate Rewards (10K stake, 11% APY) |
|-----------|------------------------------------------|
| Per Hour | ~0.13 REMES |
| Per Day | ~3.06 REMES |
| Per Week | ~21.4 REMES |
| Per Month | ~91.4 REMES |
| Per Year | ~1,100 REMES |

## Slashing Risks

| Offense | Slash Amount | Jail Duration |
|---------|--------------|---------------|
| Double Signing | 5% | Permanent |
| Downtime | 0.01% | 10 minutes |
| Lazy Validation | 1% | 24 hours |

### Protecting Your Stake

| Strategy | Description |
|----------|-------------|
| Diversify | Stake with multiple validators |
| Monitor | Check validator status regularly |
| Research | Choose reputable validators |
| Stay Informed | Follow validator announcements |

## Compound Strategies

| Strategy | 10K REMES @ 11% APY | After 1 Year | After 5 Years |
|----------|---------------------|--------------|---------------|
| No Compound | 10,000 | 11,100 | 15,500 |
| Monthly Compound | 10,000 | 11,157 | 17,024 |
| Daily Compound | 10,000 | 11,163 | 17,332 |

Enable auto-compound:

```bash
r3mes staking auto-compound --enable --interval 24h
```

## Multi-Validator Strategy

Recommended allocation:

| Stake Size | Number of Validators |
|------------|---------------------|
| <10K REMES | 3-5 validators |
| 10K-100K REMES | 5-10 validators |
| >100K REMES | 10+ validators |

## FAQ

**What's the minimum stake?**
1 REMES (1,000,000 uremes)

**When do I start earning rewards?**
Immediately after delegation is confirmed (next block)

**Can I stake with multiple validators?**
Yes, and it's recommended for risk diversification

**What happens if my validator gets slashed?**
Your stake is reduced proportionally to the slash amount

**Can I cancel an undelegation?**
No, once initiated, undelegation cannot be cancelled

**Do I need to run a node to stake?**
No, you can stake through the web dashboard or CLI
