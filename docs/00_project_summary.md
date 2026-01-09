# How R3MES Works

A technical overview of the R3MES protocol architecture and core mechanisms.

## Protocol Overview

R3MES is a decentralized protocol for machine learning computation. It provides a standardized way to execute AI training tasks across any GPU in the world, creating a global network for distributed model training.

The protocol is built on four foundational layers:

| Layer | Function | Technology |
|-------|----------|------------|
| **Execution** | Deterministic ML training | BitNet b1.58, LoRA adapters |
| **Verification** | Trustless result validation | Three-tier optimistic verification |
| **Coordination** | Consensus and rewards | Cosmos SDK, Tendermint BFT |
| **Storage** | Decentralized data | IPFS with content addressing |

## Proof of Useful Work (PoUW)

Unlike traditional Proof of Work that wastes energy on arbitrary puzzles, R3MES miners perform useful computation — training AI models.

### Mining Flow

| Step | Action | Output |
|------|--------|--------|
| 1 | Miner receives training task | Data chunk (2048 tokens) |
| 2 | Miner trains on local GPU | Gradient update |
| 3 | Miner uploads to IPFS | Content hash (CID) |
| 4 | Miner submits hash to chain | Transaction |
| 5 | Network verifies contribution | Reward distribution |

### Bandwidth Efficiency

R3MES uses LoRA (Low-Rank Adaptation) to achieve 99.6% bandwidth reduction:

| Approach | Initial Download | Per-Epoch Transfer |
|----------|------------------|-------------------|
| Traditional | 28GB | 28GB |
| R3MES LoRA | 28GB (one-time) | 10-100MB |

## Verification System

R3MES employs a three-tier verification system that balances speed with security:

### Layer 1: Optimistic Verification

| Property | Value |
|----------|-------|
| Speed | ~1 block |
| Method | Hash comparison |
| Success Rate | ~95% of cases |

Most submissions pass through Layer 1 with simple hash verification between miners on the same GPU architecture.

### Layer 2: Loss-Based Spot Check

| Property | Value |
|----------|-------|
| Speed | ~5 blocks |
| Method | Forward pass verification |
| Cost | ~100x cheaper than full training |

When disputes arise, validators download the miner's weights and run a forward pass on a random batch, comparing the calculated loss with the claimed loss.

### Layer 3: CPU Arbitration

| Property | Value |
|----------|-------|
| Speed | ~50 blocks |
| Method | Bit-exact CPU computation |
| Trigger | Layer 2 consensus supports challenge |

Final disputes are resolved by a 3-validator panel running deterministic CPU verification.

## Trap Job Security

The network uses cryptographically blinded trap jobs to detect lazy miners:

| Component | Description |
|-----------|-------------|
| Genesis Vault | Pre-computed "golden vectors" with known correct answers |
| Blind Delivery | 90% real tasks, 10% trap jobs (indistinguishable) |
| Tolerant Verification | Cosine similarity with hardware tolerance |
| Proof of Reuse | Verified real jobs become future trap jobs |

Miners cannot distinguish trap jobs from real work, creating a "Panopticon effect" that incentivizes honest behavior.

## Economic Model

### Token Distribution

| Category | Allocation |
|----------|------------|
| Mining Rewards | 30% |
| Community Pool | 20% |
| Team & Foundation | 15% |
| Ecosystem Development | 15% |
| Public Sale | 10% |
| Genesis Validators | 5% |
| Reserve | 5% |

### Reward Structure

| Recipient | Source |
|-----------|--------|
| Miners | Gradient submissions |
| Validators | Block production, verification |
| Stakers | Delegation rewards |
| Proposers | Aggregation coordination |

### Slashing Conditions

| Violation | Penalty |
|-----------|---------|
| Double signing | 5% of stake |
| Downtime | 0.01% of stake |
| Lazy mining (trap job failure) | 50% of stake |
| False verification | 20% of stake |

## Governance

Token holders participate in protocol governance:

| Proposal Type | Deposit | Voting Period |
|---------------|---------|---------------|
| Text Proposal | 100 REMES | 14 days |
| Parameter Change | 500 REMES | 14 days |
| Model Upgrade | 1,000 REMES | 21 days |
| Software Upgrade | 1,000 REMES | 21 days |

### Voting Thresholds

| Parameter | Value |
|-----------|-------|
| Quorum | 40% of staked tokens |
| Pass Threshold | 50% Yes votes |
| Veto Threshold | 33.4% NoWithVeto |

## Network Roles

| Role | Function | Requirements |
|------|----------|--------------|
| **Miner** | Train AI models | GPU, stake |
| **Validator** | Consensus participation | High uptime, stake |
| **Serving Node** | AI inference | Model hosting |
| **Proposer** | Gradient aggregation | Coordination |

## Technical Stack

| Component | Technology |
|-----------|------------|
| Blockchain | Cosmos SDK v0.50.x LTS |
| Consensus | CometBFT v0.38.27 |
| Smart Contracts | Go modules |
| Storage | IPFS |
| AI Framework | PyTorch, BitNet |
| Frontend | Next.js 14, TypeScript |

## Data Flow

| Stage | From | To | Data |
|-------|------|-----|------|
| Task Assignment | Blockchain | Miner | Data chunk hash |
| Training | Miner | Local GPU | Forward/backward pass |
| Upload | Miner | IPFS | Gradient tensor |
| Submission | Miner | Blockchain | IPFS hash + metadata |
| Verification | Validators | Blockchain | Verification result |
| Reward | Blockchain | Miner | REMES tokens |

## Security Guarantees

| Guarantee | Mechanism |
|-----------|-----------|
| Byzantine Fault Tolerance | Tendermint consensus (⅓ fault tolerance) |
| Economic Security | Slashing penalties make attacks unprofitable |
| Data Integrity | Cryptographic proofs throughout |
| Availability | 3-block response window with slashing |
| Trap Detection | >99.9% lazy mining detection rate |

## Learn More

| Topic | Link |
|-------|------|
| Mining | [Mining Guide](/docs/mining-guide) |
| Staking | [Staking Guide](/docs/staking-guide) |
| Tokenomics | [Tokenomics](/docs/tokenomics) |
| Security | [Security Model](/docs/security) |
| API | [API Reference](/docs/api-reference) |
