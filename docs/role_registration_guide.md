# Role Registration Guide

## Overview

This guide explains how to register your node for different roles in the R3MES network. Each role has specific requirements, minimum stake amounts, and access controls.

## Available Roles

### 1. Miner (Role ID: 1)
- **Access**: Public (Open Access)
- **Minimum Stake**: 1,000 REMES
- **Description**: AI model training node that contributes to decentralized AI training
- **Use Case**: Train AI models and submit gradients to earn rewards

### 2. Serving Node (Role ID: 2)
- **Access**: Public (Open Access)
- **Minimum Stake**: 1,000 REMES
- **Description**: Inference serving node that provides AI model inference services
- **Use Case**: Serve trained models and process inference requests for fees

### 3. Validator (Role ID: 3)
- **Access**: Restricted (Authorization Required)
- **Minimum Stake**: 100,000 REMES
- **Description**: Blockchain validator node that participates in consensus
- **Use Case**: Validate transactions and blocks, maintain network security
- **Authorization**: Requires whitelist approval or governance proposal

### 4. Proposer (Role ID: 4)
- **Access**: Restricted (Authorization Required)
- **Minimum Stake**: 50,000 REMES
- **Description**: Gradient aggregation proposer node
- **Use Case**: Aggregate gradients from miners and propose model updates
- **Authorization**: Requires validator role OR whitelist approval

## Registration Methods

### Method 1: Web Dashboard (Recommended)

The web dashboard provides the easiest way to register node roles with a user-friendly interface.

#### Steps:

1. **Connect Your Wallet**
   - Navigate to the web dashboard
   - Connect your Keplr wallet (or compatible Cosmos wallet)
   - Ensure your wallet has sufficient balance for stake and transaction fees

2. **Navigate to Roles Page**
   - Click on "Roles" in the navigation menu
   - Or visit `/roles` directly

3. **Select Roles**
   - Review available roles and their requirements
   - Select one or more roles you want to register for
   - Note: You can register for multiple roles simultaneously

4. **Set Stake Amount**
   - Enter your stake amount (in uremes, minimum 6 decimal places)
   - The interface will show the minimum stake required for your selected roles
   - Ensure you have sufficient balance in your wallet

5. **Review Authorization Requirements**
   - For public roles (Miner, Serving): No authorization needed
   - For restricted roles (Validator, Proposer): Review authorization requirements
   - See [Authorization Process](#authorization-process) section below

6. **Submit Registration**
   - Click "Register Node" button
   - Approve the transaction in your wallet
   - Wait for transaction confirmation
   - Your roles will be registered on-chain

#### Troubleshooting Web Dashboard Registration:

- **"Insufficient stake"**: Ensure your stake amount meets the minimum requirement for all selected roles
- **"Transaction failed"**: Check your wallet balance for transaction fees (gas)
- **"Authorization required"**: Validator/Proposer roles require approval (see Authorization section)
- **Wallet connection issues**: Ensure Keplr extension is installed and unlocked

### Method 2: Desktop Launcher

The desktop launcher Setup Wizard allows you to select roles during initial setup.

#### Steps:

1. **Run Setup Wizard**
   - Launch the R3MES Desktop Launcher
   - Complete hardware checks
   - Select roles in Step 2 (Role Selection)

2. **Complete Setup**
   - Review selected roles in Step 3
   - Complete the setup wizard

3. **Register Roles**
   - After setup completion, register your roles via web dashboard (`/roles`)
   - Or use blockchain CLI commands (see Method 3)

**Note**: The desktop launcher currently focuses on role selection during setup. Actual blockchain registration should be completed via web dashboard or CLI for better transaction control and error handling.

### Method 3: Blockchain CLI

Advanced users can register roles directly using blockchain CLI commands.

#### Using `remesd` CLI:

```bash
# Register as Miner
remesd tx remes register-node \
  --node-address $(remesd keys show mykey -a) \
  --roles 1 \
  --stake 1000000uremes \
  --from mykey \
  --chain-id remes-1 \
  --yes

# Register as Serving Node
remesd tx remes register-node \
  --node-address $(remesd keys show mykey -a) \
  --roles 2 \
  --stake 1000000uremes \
  --from mykey \
  --chain-id remes-1 \
  --yes

# Register with Multiple Roles (Miner + Serving)
remesd tx remes register-node \
  --node-address $(remesd keys show mykey -a) \
  --roles 1,2 \
  --stake 1000000uremes \
  --from mykey \
  --chain-id remes-1 \
  --yes
```

#### Using Python CLI (Future):

```bash
# Register via Python CLI (when implemented)
r3mes-miner register --roles 1,2 --stake 1000000uremes
```

## Authorization Process

### Public Roles (No Authorization Needed)

**Miner** and **Serving** roles are open to all users who meet the minimum stake requirement. No approval process is needed.

### Restricted Roles (Authorization Required)

**Validator** and **Proposer** roles require authorization before registration.

#### Option 1: Governance Proposal

1. **Submit Governance Proposal**
   - Create a governance proposal requesting validator/proposer authorization
   - Include your node address and justification
   - Submit via blockchain governance system

2. **Voting Period**
   - Community votes on your proposal
   - Proposal requires majority approval

3. **Approval and Registration**
   - Upon approval, your address is added to the whitelist
   - You can then register via web dashboard or CLI

#### Option 2: Admin Contact

1. **Contact Network Administrators**
   - Email: admin@r3mes.network (example)
   - Forum: [Link to governance forum]
   - Discord/Telegram: [Link to community channels]

2. **Provide Information**
   - Node address
   - Requested role (Validator/Proposer)
   - Justification and qualifications
   - Stake commitment

3. **Wait for Approval**
   - Admin reviews your request
   - Typical approval time: 1-2 weeks
   - You'll be notified via email or forum

4. **Register After Approval**
   - Once approved and whitelisted, register via web dashboard or CLI

### Authorization Requirements

**Validator Role**:
- Minimum stake: 100,000 REMES
- Stable network connection (24/7 uptime expected)
- Sufficient server resources
- Technical expertise in blockchain validation

**Proposer Role**:
- Minimum stake: 50,000 REMES
- Validator role OR separate authorization
- Computational resources for gradient aggregation
- Understanding of federated learning

## Minimum Stake Requirements

| Role | Minimum Stake | Notes |
|------|---------------|-------|
| Miner | 1,000 REMES | Public access |
| Serving | 1,000 REMES | Public access |
| Validator | 100,000 REMES | Authorization required |
| Proposer | 50,000 REMES | Authorization required |

**Stake Format**: All stake amounts are specified in `uremes` (micro-REMES) with 6 decimal places.
- 1 REMES = 1,000,000 uremes
- Example: 1,000 REMES = 1,000,000,000 uremes

## Resource Requirements

When registering, you may need to specify resource quotas:

```json
{
  "cpuCores": 4,
  "memoryGb": 8,
  "gpuCount": 1,
  "gpuMemoryGb": 12,
  "storageGb": 100,
  "networkBandwidthMbps": 1000
}
```

These are used for resource allocation and matching tasks to nodes.

## Updating Roles

You can update your registered roles at any time:

1. **Via Web Dashboard**
   - Navigate to `/roles` page
   - Select/deselect roles
   - Click "Update Roles"
   - Submit transaction

2. **Via CLI**
   ```bash
   remesd tx remes update-node-registration \
     --node-address $(remesd keys show mykey -a) \
     --roles 1,2,3 \
     --stake 100000000uremes \
     --from mykey \
     --chain-id remes-1 \
     --yes
   ```

**Note**: Updating roles does not require re-authorization for already-approved restricted roles.

## Checking Registration Status

### Via Web Dashboard

1. Navigate to `/roles` page
2. Your registered roles will be displayed under "Current Roles"
3. Status will show as "Active", "Inactive", etc.

### Via Blockchain Query

```bash
# Query your node registration
remesd query remes node $(remesd keys show mykey -a)

# List all registered nodes
remesd query remes nodes

# Get role statistics
remesd query remes role-statistics
```

### Via Backend API

```bash
# Get your node roles
curl http://localhost:8000/api/roles/node/{address}

# List all roles
curl http://localhost:8000/api/roles

# Get role statistics
curl http://localhost:8000/api/roles/statistics
```

## Troubleshooting

### Common Issues

**Issue**: "Insufficient stake"
- **Solution**: Ensure your stake amount meets the minimum requirement for all selected roles

**Issue**: "Authorization required"
- **Solution**: Validator/Proposer roles require whitelist approval. Complete authorization process first.

**Issue**: "Transaction failed: insufficient funds"
- **Solution**: Ensure your wallet has enough balance for both stake and transaction fees (gas)

**Issue**: "Node already registered"
- **Solution**: Use "Update Roles" instead of "Register Node" if you're already registered

**Issue**: "Invalid role ID"
- **Solution**: Ensure you're using valid role IDs (1=Miner, 2=Serving, 3=Validator, 4=Proposer)

### Getting Help

- **Documentation**: See [Role Access Control](../ROLE_ACCESS_CONTROL.md) for detailed access control information
- **Support**: Contact support via email, forum, or community channels
- **GitHub Issues**: Report bugs or request features on GitHub

## Best Practices

1. **Start with Public Roles**: If you're new, start with Miner or Serving roles (no authorization needed)
2. **Sufficient Stake**: Ensure you have enough stake and transaction fees before registering
3. **Role Selection**: Consider your hardware capabilities when selecting roles
4. **Multi-role Strategy**: You can register for multiple roles, but ensure you have resources for all
5. **Regular Updates**: Keep your node software updated for best performance
6. **Monitor Status**: Regularly check your node status and role performance

## Related Documentation

- [Role Access Control](../ROLE_ACCESS_CONTROL.md) - Detailed access control information
- [User Onboarding Guides](../09_user_onboarding_guides.md) - General onboarding information
- [Desktop Launcher](../10_desktop_launcher.md) - Desktop launcher usage
- [Web Dashboard](../08_web_dashboard_command_center.md) - Web dashboard guide

