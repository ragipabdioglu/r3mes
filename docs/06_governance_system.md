# Governance

R3MES implements a comprehensive governance system enabling community-driven decisions on protocol parameters, dataset approval, model versioning, and economic policies.

---

## Overview

The governance system allows token holders to participate in protocol evolution through proposals and voting.

| Governance Area | Description |
|-----------------|-------------|
| Dataset Governance | Training data approval and validation |
| Model Versioning | BitNet version upgrades (v1 → v2) |
| Economic Parameters | Slashing rates, reward formulas |
| Protocol Upgrades | Technical improvements and new features |

---

## Voting Mechanisms

| Mechanism | Use Case |
|-----------|----------|
| Token-weighted | Standard proposals |
| Quadratic | Critical decisions (prevents plutocracy) |
| Stake-weighted | Validator-specific decisions |
| Multi-tier | Different thresholds by proposal type |

---

## Dataset Proposals

### Creating a Proposal

**Via Web Dashboard:**
1. Navigate to Network page
2. Click "Create Dataset Proposal" in Governance panel
3. Select dataset file (JSONL, CSV, etc.)
4. Upload to IPFS (CID generated automatically)
5. Submit proposal (transaction hash displayed)

**Via CLI:**
```bash
remesd tx remes propose-dataset \
  --dataset-ipfs-hash QmXxxx... \
  --deposit 1000uremes \
  --from dataset-provider \
  --chain-id remes-1 \
  --yes
```

### Viewing Proposals

**REST API:**
```bash
curl http://localhost:1317/remes/remes/v1/dataset_proposals
```

**CLI:**
```bash
# List all proposals
remesd query remes list-dataset-proposals

# View specific proposal
remesd query remes dataset-proposal 1

# List approved datasets
remesd query remes list-approved-datasets
```

### Proposal Parameters

| Parameter | Value |
|-----------|-------|
| Voting Period | 7 days (10,080 blocks at 5s/block) |
| Approval Threshold | 67% |
| Veto Threshold | 33.4% |
| Minimum Deposit | 1,000 REMES |

---

## Voting

### Vote Options

| Option | Description |
|--------|-------------|
| Yes | Support the proposal |
| No | Oppose the proposal |
| Abstain | Neutral (counts toward quorum) |
| No with Veto | Strong rejection (>33.4% vetoes = rejected) |

### Vote Tracking

Each vote is tracked individually for auditability:

| Field | Description |
|-------|-------------|
| Vote ID | Unique identifier |
| Proposal ID | Proposal being voted on |
| Voter | Voter address |
| Option | yes/no/abstain/veto |
| Voting Power | Weight of vote |
| Quality Score | Dataset quality assessment (0-100) |
| Relevance Score | Dataset relevance assessment (0-100) |
| Block Height | When vote was cast |

### Vote Weight Calculation

- **Standard:** Token balance determines weight
- **Quadratic:** Square root of balance (for critical decisions)
- **Stake-weighted:** Validator stake determines weight

---

## Model Upgrade Proposals

Model upgrades require higher thresholds due to their impact on the network.

### Upgrade Parameters

| Parameter | Value |
|-----------|-------|
| Voting Period | 14 days (20,160 blocks) |
| Minimum Deposit | 100,000 REMES |
| Migration Window | Configurable (blocks) |

### Technical Specifications

| Specification | Description |
|---------------|-------------|
| Model Size | Size in bytes |
| Parameter Count | Number of parameters |
| LoRA Rank | Adapter rank |
| Required Memory | GPU memory requirement |
| Compute Requirement | Minimum compute capability |
| Backward Compatible | Compatibility with previous version |

### Upgrade Process

1. **Proposal Submission** - Technical specs and deposit
2. **Voting Period** - 14 days for community review
3. **Activation** - If approved, new version activated
4. **Migration Window** - Dual support during transition
5. **Deprecation** - Old version deprecated after window

---

## Economic Parameter Governance

### Configurable Parameters

| Category | Parameters |
|----------|------------|
| Slashing | Hash mismatch (5%), availability fault (2%), lazy mining (50%) |
| Rewards | Base mining reward, proposer reward, fraud bounty |
| Staking | Minimum requirement, unbonding time |
| Reputation | Spot check rate (10%), decay rate (5% per 1000 blocks) |
| Challenges | Response window (3 blocks), bond multiplier (10x) |

### Update Process

1. Governance proposal submitted
2. Parameter ranges validated
3. Community votes
4. If approved, parameters updated on-chain
5. Event emitted for transparency

---

## Emergency Governance

For critical security or economic issues, emergency proposals have expedited timelines.

### Emergency Types

| Type | Description |
|------|-------------|
| Security | Security vulnerabilities |
| Economic | Economic attacks or imbalances |
| Technical | Critical bugs or failures |

### Severity Levels

| Severity | Voting Period | Approval Threshold |
|----------|---------------|-------------------|
| Low | 7 days | 67% |
| Medium | 3 days | 70% |
| High | 1 day | 75% |
| Critical | 1 day | 75% |

### Emergency Actions

| Action | Description |
|--------|-------------|
| Pause | Temporarily halt affected functionality |
| Parameter Change | Immediate parameter adjustment |
| Upgrade | Emergency software upgrade |

---

## Anti-Spam Mechanisms

| Mechanism | Description |
|-----------|-------------|
| Deposit Requirement | Proposals require token deposit |
| Proposal Limit | Max 3 proposals per address per 7 days |
| Reputation Check | Minimum trust score required |
| Deposit Return | Returned on approval, burned on veto |

---

## Governance Analytics

Track governance health and participation:

| Metric | Description |
|--------|-------------|
| Total Proposals | All-time proposal count |
| Approval Rate | Percentage approved |
| Participation Rate | Voter turnout |
| Average Voting Power | Mean vote weight |
| Top Voters | Most active participants |

---

## Best Practices

### For Proposers

1. **Clear Justification** - Detailed rationale required
2. **Technical Specs** - Complete specifications for upgrades
3. **Impact Assessment** - Analysis of changes
4. **Community Discussion** - Pre-proposal discussion encouraged
5. **Appropriate Deposit** - Scaled to proposal impact

### For Voters

1. **Research** - Understand proposal implications
2. **Rationale** - Provide vote explanations (optional)
3. **Delegation** - Delegate if unable to participate
4. **Quorum** - Ensure minimum participation
5. **Veto Carefully** - Use veto for strong objections only

---

## Next Steps

- [Tokenomics →](tokenomics) - Economic model
- [Security →](security) - Verification system
- [Staking Guide →](staking-guide) - Participate in governance

---

**Need Help?** [Join Discord](https://discord.gg/r3mes) | [GitHub](https://github.com/r3mes-network/r3mes)
