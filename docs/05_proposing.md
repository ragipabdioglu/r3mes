# Proposing in R3MES

Learn how to run a proposer node on R3MES and aggregate gradients from miners while earning rewards.

---

## What is Proposing?

Proposers aggregate gradients from miners and submit aggregated gradients to the blockchain. They play a crucial role in the distributed training process by combining contributions from multiple miners.

As a proposer, you:
- Aggregate gradients from multiple miners
- Verify gradient quality and consistency
- Submit aggregated gradients to blockchain
- Earn rewards for successful aggregations

---

## Why Become a Proposer?

### ✅ Network Critical Role

Proposers are essential for the training process. Without proposers, miner gradients cannot be aggregated and the model cannot be updated.

### ✅ Earn Rewards

Proposers earn tokens for successfully aggregating and submitting gradients. Active proposers with good quality control can earn steady rewards.

### ✅ Network Influence

Proposers can influence model training quality through gradient selection and aggregation strategies.

### ⚠️ Authorization Required

Proposer role requires authorization (not public). Contact network administrators to become a proposer.

---

## Requirements

### Hardware

- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 16GB+ (32GB+ for large aggregations)
- **Disk**: 100GB+ for gradient storage
- **Network**: High bandwidth connection (for receiving gradients from miners)

### Software

- **Python**: 3.10+
- **Blockchain Node**: Connection to R3MES blockchain
- **Proposer Engine**: R3MES proposer engine

### Network

- **Authorization**: Must be authorized before registration
- **Stake**: Minimum stake required
- **Registration**: Register as proposer on blockchain after authorization

---

## Quick Start

### Option 1: Desktop Launcher

1. **Download** Desktop Launcher
2. **Launch** and run setup wizard
3. **Request Authorization**: Contact network administrators for proposer authorization
4. **Select** "Proposer" role during setup (after authorization)
5. **Configure** proposer parameters:
   - Gradient aggregation settings
   - Quality thresholds
   - Submission parameters
6. **Register** on blockchain via Web Dashboard
7. **Start** proposer node from launcher

[Desktop Launcher Guide →](10_desktop_launcher.md)

### Option 2: CLI

1. **Install** the Python package:
   ```bash
   pip install r3mes
   ```

2. **Request Authorization**: Contact network administrators

3. **Configure** proposer:
   ```bash
   r3mes-proposer configure
   ```
   This will prompt for:
   - Gradient aggregation parameters
   - Quality verification settings
   - Blockchain RPC endpoint
   - Submission frequency

4. **Register** on blockchain:
   ```bash
   r3mes-proposer register --from my-key
   ```

5. **Start** proposer:
   ```bash
   r3mes-proposer start
   ```

---

## How Proposing Works

### Aggregation Process

1. **Receive Gradients**: Collect gradients from miners
2. **Verify Quality**: Check gradient validity and quality
3. **Aggregate**: Combine gradients using aggregation algorithm (e.g., weighted average)
4. **Submit**: Upload aggregated gradient to IPFS
5. **Broadcast**: Submit IPFS CID to blockchain
6. **Earn Rewards**: Receive tokens for successful submission

### Gradient Verification

Proposers verify gradients for:
- **Validity**: Gradient structure and format correctness
- **Quality**: Gradient quality metrics (magnitude, distribution, etc.)
- **Consistency**: Consistency with other gradients
- **Source Verification**: Verify gradient source (miner authentication)

### Aggregation Algorithms

Common aggregation methods:
- **Average**: Simple average of all gradients
- **Weighted Average**: Weight by miner reputation or contribution quality
- **Median**: Use median to filter outliers
- **Federated Averaging**: Standard federated learning aggregation

---

## Optimizing Your Proposer

### Quality Control

- **Strict Verification**: Enforce quality thresholds to ensure good aggregations
- **Reputation Weighting**: Weight gradients by miner reputation
- **Outlier Detection**: Detect and filter outlier gradients
- **Consistency Checks**: Verify gradient consistency across miners

### Efficiency

- **Batch Processing**: Aggregate multiple gradients in batches
- **Parallel Processing**: Use multi-threading for verification
- **Caching**: Cache verification results
- **Incremental Aggregation**: Incremental updates instead of full recomputation

### Network Optimization

- **Low Latency**: Minimize latency to miners for faster gradient collection
- **High Bandwidth**: Ensure sufficient bandwidth for gradient transfers
- **Reliable Connection**: Maintain stable connection for consistent operation

---

## Monitoring Your Proposer

### Key Metrics

Monitor these metrics:

- **Gradients Received**: Number of gradients collected
- **Aggregations Completed**: Successful aggregations
- **Submission Success Rate**: Percentage of successful blockchain submissions
- **Quality Metrics**: Average gradient quality
- **Earnings**: Tokens earned from aggregations

### Desktop Launcher

The Desktop Launcher shows:
- Proposer status
- Gradient aggregation statistics
- Submission history
- Earnings and rewards

### Web Dashboard

View proposer statistics at [dashboard.r3mes.network](https://dashboard.r3mes.network):
- Proposer details
- Aggregation history
- Earnings over time
- Quality metrics

### Command Line

```bash
# Check proposer status
r3mes-proposer status

# View recent aggregations
r3mes-proposer list-aggregations

# Check earnings
r3mes-proposer earnings
```

---

## Rewards and Economics

### Earning Model

Proposers earn tokens for:
- **Successful Aggregations**: Each successful aggregation submission
- **Quality Bonuses**: Additional rewards for high-quality aggregations
- **Consistency Rewards**: Rewards for consistent, reliable operation

### Reward Calculation

Rewards are based on:
- **Aggregation Frequency**: More aggregations = more rewards
- **Quality**: Higher quality aggregations may earn more
- **Network Contribution**: Contribution to overall network training progress
- **Reliability**: Consistent operation increases trust and rewards

### Claiming Rewards

```bash
# Claim proposer rewards
r3mes-proposer claim-rewards --from my-key
```

Or use the Web Dashboard to claim rewards.

---

## Security Considerations

### Gradient Verification

- **Cryptographic Verification**: Verify gradient signatures from miners
- **Source Authentication**: Ensure gradients come from registered miners
- **Integrity Checks**: Verify gradient integrity (no tampering)

### Attack Prevention

- **Sybil Resistance**: Prevent fake gradients from unauthorized sources
- **Quality Manipulation**: Detect and filter malicious gradients
- **Consensus Verification**: Ensure aggregated gradients meet network consensus

### Key Management

- **Secure Storage**: Store proposer keys securely
- **Access Control**: Limit access to proposer functionality
- **Backup**: Backup keys and configuration

---

## Troubleshooting

### Gradient Collection Issues

- Check miner connectivity
- Verify miner registrations
- Review network configuration
- Check firewall rules

### Aggregation Failures

- Verify gradient format compatibility
- Check aggregation algorithm parameters
- Review quality thresholds
- Check system resources (RAM, CPU)

### Submission Failures

- Verify blockchain connection
- Check transaction fees balance
- Review IPFS connectivity
- Check for network errors

### Low Earnings

- Improve aggregation frequency
- Enhance quality control
- Ensure consistent uptime
- Optimize aggregation efficiency

**More Help:** [Troubleshooting Guide →](TROUBLESHOOTING.md)

---

## Next Steps

- [Mining Guide →](02_mining.md) - Train AI models
- [Validating Guide →](03_validating.md) - Run a validator
- [Serving Guide →](04_serving.md) - Provide AI inference
- [Protocol Design →](06_protocol_design.md) - Technical deep-dive

---

**Need Help?** [Join Discord](https://discord.gg/r3mes) | [View Documentation](00_home.md)

