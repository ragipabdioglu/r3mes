# Role Access Control Documentation

## Overview

R3MES implements role-based access control (RBAC) to ensure network security and prevent unauthorized access to critical roles.

## Role Categories

### Public Roles (Halka Açık)

These roles are open to all users who meet minimum stake requirements:

1. **Miner (Role ID: 1)**
   - **Access**: Public
   - **Minimum Stake**: 1,000 REMES
   - **Description**: AI model training node
   - **Use Case**: Decentralized AI training contribution

2. **Serving Node (Role ID: 2)**
   - **Access**: Public
   - **Minimum Stake**: 1,000 REMES
   - **Description**: Inference serving node
   - **Use Case**: Decentralized AI inference services

### Restricted Roles (Sınırlı Erişim)

These roles require special authorization:

3. **Validator (Role ID: 3)**
   - **Access**: Restricted (Whitelist/Governance Approval)
   - **Minimum Stake**: 100,000 REMES
   - **Description**: Blockchain validator node
   - **Authorization Required**: Yes
   - **Use Case**: Network consensus and security

4. **Proposer (Role ID: 4)**
   - **Access**: Restricted (Validator Role or Whitelist)
   - **Minimum Stake**: 50,000 REMES
   - **Description**: Gradient aggregation proposer node
   - **Authorization Required**: Yes (or must be Validator)
   - **Use Case**: Gradient aggregation and model updates

## Implementation Details

### Backend API (`backend/app/role_endpoints.py`)

Role access control is defined in `ROLE_ACCESS_CONTROL` dictionary:

```python
ROLE_ACCESS_CONTROL = {
    1: {  # Miner
        "public": True,
        "min_stake": "1000remes",
        "requires_approval": False,
    },
    2: {  # Serving
        "public": True,
        "min_stake": "1000remes",
        "requires_approval": False,
    },
    3: {  # Validator
        "public": False,
        "min_stake": "100000remes",
        "requires_approval": True,
        "whitelist_only": True,
    },
    4: {  # Proposer
        "public": False,
        "min_stake": "50000remes",
        "requires_approval": True,
        "requires_validator_role": True,
    },
}
```

### Blockchain Handler (`remes/x/remes/keeper/msg_server_register_node.go`)

The blockchain handler enforces access control:

1. **Validator Role Check**:
   - Checks `AuthorizedValidatorAddresses` whitelist
   - Falls back to module authority check (for genesis validators)
   - Validates minimum stake (100,000 REMES)

2. **Proposer Role Check**:
   - Checks `AuthorizedProposerAddresses` whitelist
   - OR checks if node has validator role
   - Validates minimum stake (50,000 REMES)

### Whitelist Storage

Authorized addresses are stored in blockchain state:

- `AuthorizedValidatorAddresses`: Map of authorized validator addresses
- `AuthorizedProposerAddresses`: Map of authorized proposer addresses

### Adding Authorized Addresses

To add authorized addresses (e.g., for genesis validators), use governance proposals or direct keeper methods:

```go
// Example: Add authorized validator (in genesis or via governance)
k.AuthorizedValidatorAddresses.Set(ctx, validatorAddress, true)

// Example: Add authorized proposer
k.AuthorizedProposerAddresses.Set(ctx, proposerAddress, true)
```

## Frontend UI (`web-dashboard/app/roles/page.tsx`)

The frontend displays access control information:

- **Public roles**: Green badge "Open Access"
- **Restricted roles**: Orange badge "Authorization Required"
- **Warning messages**: Displayed for restricted roles
- **Minimum stake**: Shown for each role

## Error Messages

When unauthorized registration is attempted:

- **Validator**: `"validator role requires authorization (whitelist or governance approval)"`
- **Proposer**: `"proposer role requires validator role or authorization (whitelist)"`

## Security Considerations

1. **Genesis Validators**: Module authority can always register as validator (for genesis setup)
2. **Whitelist Management**: Whitelist updates should be done via governance proposals
3. **Stake Requirements**: Minimum stake requirements prevent spam and ensure economic security
4. **Role Updates**: Existing validators/proposers can update their roles without re-authorization

## Authorization Request Process

For users who want to register as Validator or Proposer, authorization is required before registration.

### How to Request Authorization

#### Option 1: Governance Proposal

1. **Create a Governance Proposal**
   - Use the blockchain governance system to submit a proposal
   - Proposal type: Add to authorized validators/proposers
   - Include your node address and justification

2. **Proposal Details**
   - **Title**: "Request Validator/Proposer Authorization for [Your Address]"
   - **Description**: Include:
     - Your node address
     - Requested role (Validator/Proposer)
     - Technical qualifications
     - Stake commitment
     - Server specifications
     - Uptime guarantees

3. **Voting Period**
   - Community members vote on your proposal
   - Requires majority approval
   - Typical voting period: 1-2 weeks

4. **After Approval**
   - Upon approval, your address is automatically added to the whitelist
   - You can then register via web dashboard or CLI

**Governance Documentation**: See [Governance System](../06_governance_system.md) for detailed governance process.

#### Option 2: Admin Contact

1. **Contact Methods**
   - **Email**: admin@r3mes.network (or current admin email)
   - **Forum**: [Link to governance forum when available]
   - **Discord/Telegram**: [Link to community channels when available]

2. **Information to Provide**
   - Node address (bech32 format: `remes1...`)
   - Requested role (Validator or Proposer)
   - Justification: Why you want this role
   - Qualifications: Technical expertise, infrastructure
   - Stake commitment: Amount you're willing to stake
   - Server specifications: CPU, RAM, storage, network
   - Uptime commitment: Expected availability percentage

3. **Review Process**
   - Admin team reviews your request
   - May request additional information
   - Technical assessment of your capabilities
   - Typical review time: 1-2 weeks

4. **Approval Process**
   - If approved, your address is added to whitelist
   - You receive confirmation via email or forum
   - You can then proceed with registration

5. **If Rejected**
   - You'll receive feedback on why
   - You can reapply after addressing concerns
   - Or start with public roles (Miner/Serving) to build reputation

### Authorization Requirements Checklist

**For Validator Role**:
- [ ] Minimum stake commitment: 100,000 REMES
- [ ] Stable network connection (24/7 uptime capability)
- [ ] Sufficient server resources (CPU, RAM, storage)
- [ ] Technical expertise in blockchain validation
- [ ] Commitment to network security
- [ ] Ability to handle validator responsibilities

**For Proposer Role**:
- [ ] Minimum stake commitment: 50,000 REMES
- [ ] Validator role OR separate authorization
- [ ] Computational resources for gradient aggregation
- [ ] Understanding of federated learning
- [ ] Ability to handle aggregation tasks
- [ ] Stable network connection

### Typical Approval Timeline

- **Governance Proposal**: 2-4 weeks (includes proposal period + voting)
- **Admin Contact**: 1-2 weeks (review + whitelist update)

### After Authorization

Once authorized and whitelisted:

1. **Verify Whitelist Status**
   ```bash
   # Query whitelist (when query endpoint available)
   remesd query remes authorized-validators
   remesd query remes authorized-proposers
   ```

2. **Register Your Role**
   - Use web dashboard: Navigate to `/roles`, select role, register
   - Or use CLI: `remesd tx remes register-node --roles 3 --stake 100000000uremes ...`

3. **Monitor Registration**
   - Check registration status via web dashboard or CLI
   - Ensure your node is running and synced
   - Begin performing role responsibilities

## Future Enhancements

- Governance proposals for whitelist updates
- Automatic validator selection based on stake ranking
- Role expiration and renewal mechanisms
- Multi-sig whitelist management
- Self-service authorization application form

