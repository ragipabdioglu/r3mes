# Validating in R3MES

Learn how to run a validator node on R3MES and help secure the network while earning rewards.

---

## What is Validating?

Validators are nodes that participate in blockchain consensus by validating transactions and proposing blocks. They are essential for network security and decentralization.

As a validator, you:
- Validate transactions and blocks
- Participate in consensus (Tendermint BFT)
- Earn rewards from transaction fees and block rewards
- Can accept delegations from token holders

---

## Why Become a Validator?

### ✅ Network Security

Validators **secure the R3MES network** by participating in consensus. Your node helps ensure transaction validity and network integrity.

### ✅ Earn Rewards

Validators earn rewards from:
- **Block rewards** — New tokens minted per block
- **Transaction fees** — Fees from transactions in blocks you propose
- **Delegator commissions** — Percentage of delegator rewards

### ✅ Influence

Validators participate in **governance proposals**, influencing protocol upgrades and network parameters.

### ⚠️ Responsibility

Validators have significant responsibility. **Slashing conditions** apply if you misbehave (double signing, downtime, etc.).

---

## Requirements

### Hardware

- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 16GB+ (32GB+ recommended)
- **Disk**: 200GB+ SSD (for blockchain state)
- **Network**: High bandwidth, low latency connection
- **Uptime**: 24/7 availability required

### Software

- **OS**: Linux (Ubuntu 22.04+ recommended), macOS, or Windows Server
- **Blockchain Node**: `remesd` node (Cosmos SDK)
- **Monitoring**: System monitoring tools recommended

### Network

- **Stake**: Minimum stake required (varies by network)
- **Authorization**: Validator role requires authorization (not public)
- **Slashing Risk**: Validators can be slashed for misbehavior

---

## Quick Start

### Option 1: Desktop Launcher (Easiest)

1. **Download** Desktop Launcher
2. **Launch** and run setup wizard
3. **Select** "Validator" role during setup
4. **Configure** validator parameters:
   - Commission rate (percentage of rewards to keep)
   - Minimum self-delegation
   - Validator name and description
5. **Request Authorization**: Validator role requires authorization before registration
6. **Register**: After authorization, register on blockchain via Web Dashboard
7. **Start** validator node from launcher

[Desktop Launcher Guide →](10_desktop_launcher.md)

### Option 2: Manual Setup

1. **Install** `remesd` (blockchain node):
   ```bash
   # Build from source or use pre-built binary
   git clone https://github.com/r3mes/remes
   cd remes
   make install
   ```

2. **Initialize** node:
   ```bash
   remesd init my-validator --chain-id remes-1
   ```

3. **Configure** validator:
   ```bash
   # Edit config.toml
   # Set validator details, commission rate, etc.
   ```

4. **Request Authorization**: Contact network administrators for validator authorization

5. **Register** validator:
   ```bash
   remesd tx remes register-node \
     --from my-key \
     --roles validator \
     --chain-id remes-1
   ```

6. **Start** validator:
   ```bash
   remesd start
   ```

---

## Validator Operations

### Commission Rate

Validators set a commission rate (percentage) on rewards earned by delegators. This is how validators earn income.

**Example**: If commission rate is 10%, and delegators earn 100 tokens, the validator keeps 10 tokens.

### Minimum Self-Delegation

Validators must stake a minimum amount of tokens themselves. This demonstrates commitment to the network.

### Accepting Delegations

Token holders can delegate tokens to your validator. You earn commission on their rewards.

### Governance Participation

Validators vote on governance proposals. Your vote carries weight based on total stake (self-delegated + delegated).

---

## Monitoring Your Validator

### Key Metrics

Monitor these metrics to ensure your validator is healthy:

- **Uptime**: Percentage of time online
- **Block Signing**: Percentage of blocks signed
- **Jail Status**: Whether validator is jailed (not signing blocks)
- **Commission**: Current commission rate
- **Delegations**: Total tokens delegated to you
- **Rewards**: Pending and earned rewards

### Desktop Launcher

The Desktop Launcher shows:
- Validator status
- Block signing rate
- Delegations and rewards
- System resource usage

### Web Dashboard

View validator statistics at [dashboard.r3mes.network](https://dashboard.r3mes.network):
- Validator details
- Delegation information
- Commission and rewards
- Governance voting history

### Command Line

```bash
# Check validator status
remesd query staking validator <validator-address>

# Check signing info
remesd query slashing signing-info <validator-consensus-address>

# Check delegations
remesd query staking delegations-to <validator-address>
```

---

## Slashing Conditions

Validators can be slashed (lose staked tokens) for:

### Double Signing

Signing two different blocks at the same height. **Penalty: 5% of stake**

### Downtime

Not signing blocks for extended periods. **Penalty: 0.01% of stake per infraction**

### Security Best Practices

To avoid slashing:
- Use sentry nodes to protect your validator key
- Implement monitoring and alerting
- Use hardware security modules (HSM) for key management
- Maintain high uptime (99%+ recommended)

---

## Security Considerations

### Key Management

**Critical**: Your validator key is extremely sensitive. Never share it or store it insecurely.

**Best Practices**:
- Use hardware wallets or HSMs for validator keys
- Use sentry nodes to protect validator from DDoS
- Keep validator key offline when possible
- Implement backup and recovery procedures

### Network Security

- Configure firewall rules properly
- Use VPN or private networking
- Monitor for suspicious activity
- Keep software up to date

### Monitoring

- Set up alerts for validator status
- Monitor system resources (CPU, RAM, disk)
- Track network connectivity
- Monitor blockchain sync status

---

## Rewards and Economics

### Earning Rewards

Validators earn rewards from:
1. **Block Rewards**: New tokens minted per block
2. **Transaction Fees**: Fees from transactions in blocks you propose
3. **Commission**: Percentage of delegator rewards

### Claiming Rewards

```bash
# Claim all rewards
remesd tx distribution withdraw-all-rewards --from my-key

# Claim validator commission
remesd tx distribution withdraw-rewards <validator-address> \
  --commission \
  --from my-key
```

Or use the Web Dashboard to claim rewards with a wallet interface.

### Commission Rate Updates

You can update your commission rate (subject to limits):
```bash
remesd tx staking edit-validator \
  --commission-rate 0.15 \
  --from my-key
```

---

## Troubleshooting

### Validator Not Signing Blocks

- Check if validator is jailed
- Verify validator key is correct
- Check network connectivity
- Review logs for errors

### High Resource Usage

- Optimize node configuration
- Use SSD for blockchain state
- Increase RAM if needed
- Monitor for memory leaks

### Getting Jailed

- Check slashing conditions
- Ensure high uptime
- Fix underlying issues
- Wait for jail period or submit unjail transaction

**More Help:** [Troubleshooting Guide →](TROUBLESHOOTING.md)

---

## Next Steps

- [Mining Guide →](02_mining.md) - Train AI models
- [Serving Guide →](04_serving.md) - Provide AI inference
- [Staking Guide →](07_staking.md) - Stake to validators
- [Governance Guide →](06_governance_system.md) - Participate in governance

---

**Need Help?** [Join Discord](https://discord.gg/r3mes) | [View Documentation](00_home.md)

