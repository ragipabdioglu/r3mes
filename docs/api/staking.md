# Staking API Documentation

## Overview

The Staking API provides endpoints for delegating tokens, claiming rewards, and querying staking information. It integrates with the Cosmos SDK staking module.

**Important**: These are **Blockchain REST API endpoints** provided by the Cosmos SDK, not Backend FastAPI endpoints. They are accessed directly on the blockchain node's REST port (1317).

## Base URL

**Development**:
```
http://localhost:1317
```

**Production**:
```
https://rpc.r3mes.network
```

## Endpoints

### Get Staking Info

Get staking information for a delegator.

**Endpoint:** `GET /cosmos/staking/v1beta1/delegations/{address}`

**Response:**
```json
{
  "total_staked": "1000000000",
  "pending_rewards": "50000000",
  "unbonding": "0",
  "unbonding_end_time": "",
  "delegation_responses": [
    {
      "delegation": {
        "delegator_address": "remes1...",
        "validator_address": "remesvaloper1...",
        "shares": "1000000000.000000000000000000"
      },
      "balance": {
        "denom": "uremes",
        "amount": "1000000000"
      }
    }
  ]
}
```

### Get Validators

Get list of validators.

**Endpoint:** `GET /cosmos/staking/v1beta1/validators`

**Response:**
```json
{
  "validators": [
    {
      "operator_address": "remesvaloper1...",
      "description": {
        "moniker": "Validator Name",
        "identity": "",
        "website": "",
        "security_contact": "",
        "details": ""
      },
      "commission": {
        "commission_rates": {
          "rate": "0.10",
          "max_rate": "0.20",
          "max_change_rate": "0.01"
        },
        "update_time": "2024-01-01T00:00:00Z"
      },
      "tokens": "1000000000",
      "status": "BOND_STATUS_BONDED",
      "uptime": 99.5
    }
  ]
}
```

### Get Delegator Rewards

Get pending rewards for a delegator.

**Endpoint:** `GET /cosmos/distribution/v1beta1/delegators/{address}/rewards`

**Response:**
```json
{
  "rewards": [
    {
      "validator_address": "remesvaloper1...",
      "reward": [
        {
          "denom": "uremes",
          "amount": "50000000"
        }
      ]
    }
  ],
  "total": [
    {
      "denom": "uremes",
      "amount": "50000000"
    }
  ]
}
```

## Blockchain Transactions

All staking operations require blockchain transactions signed with Keplr or compatible wallet.

### Delegate

**Message Type:** `/cosmos.staking.v1beta1.MsgDelegate`

**Message:**
```json
{
  "typeUrl": "/cosmos.staking.v1beta1.MsgDelegate",
  "value": {
    "delegatorAddress": "remes1...",
    "validatorAddress": "remesvaloper1...",
    "amount": {
      "denom": "uremes",
      "amount": "1000000000"
    }
  }
}
```

### Undelegate

**Message Type:** `/cosmos.staking.v1beta1.MsgUndelegate`

**Message:**
```json
{
  "typeUrl": "/cosmos.staking.v1beta1.MsgUndelegate",
  "value": {
    "delegatorAddress": "remes1...",
    "validatorAddress": "remesvaloper1...",
    "amount": {
      "denom": "uremes",
      "amount": "1000000000"
    }
  }
}
```

### Redelegate

**Message Type:** `/cosmos.staking.v1beta1.MsgBeginRedelegate`

**Message:**
```json
{
  "typeUrl": "/cosmos.staking.v1beta1.MsgBeginRedelegate",
  "value": {
    "delegatorAddress": "remes1...",
    "validatorSrcAddress": "remesvaloper1...",
    "validatorDstAddress": "remesvaloper2...",
    "amount": {
      "denom": "uremes",
      "amount": "1000000000"
    }
  }
}
```

### Claim Rewards

**Message Type:** `/cosmos.distribution.v1beta1.MsgWithdrawDelegatorReward`

**Message:**
```json
{
  "typeUrl": "/cosmos.distribution.v1beta1.MsgWithdrawDelegatorReward",
  "value": {
    "delegatorAddress": "remes1...",
    "validatorAddress": "remesvaloper1..."
  }
}
```

**Note:** To claim rewards from all validators, batch multiple `MsgWithdrawDelegatorReward` messages in a single transaction.

## Examples

### Frontend (TypeScript/React)

```typescript
import { signAndBroadcastTransaction } from '@/lib/keplr';

// Delegate tokens
async function delegate(validatorAddress: string, amount: string) {
  const message = {
    typeUrl: "/cosmos.staking.v1beta1.MsgDelegate",
    value: {
      delegatorAddress: "", // Will be filled by Keplr
      validatorAddress: validatorAddress,
      amount: {
        denom: "uremes",
        amount: (parseFloat(amount) * 1e6).toString(), // Convert REMES to uremes
      },
    },
  };
  
  const txHash = await signAndBroadcastTransaction([message], `Delegate ${amount} REMES`);
  return txHash;
}

// Claim rewards
async function claimRewards(validators: string[]) {
  const messages = validators.map(validatorAddress => ({
    typeUrl: "/cosmos.distribution.v1beta1.MsgWithdrawDelegatorReward",
    value: {
      delegatorAddress: "", // Will be filled by Keplr
      validatorAddress: validatorAddress,
    },
  }));
  
  const txHash = await signAndBroadcastTransaction(messages, "Claim staking rewards");
  return txHash;
}
```

### Query Staking Info

```typescript
// Get staking info
const response = await fetch(
  `/cosmos/cosmos/staking/v1beta1/delegations/${walletAddress}`
);
const stakingInfo = await response.json();

console.log('Total staked:', stakingInfo.total_staked);
console.log('Pending rewards:', stakingInfo.pending_rewards);

// Get validators
const validatorsResponse = await fetch(
  '/cosmos/cosmos/staking/v1beta1/validators'
);
const validatorsData = await validatorsResponse.json();
const validators = validatorsData.validators;

// Get rewards
const rewardsResponse = await fetch(
  `/cosmos/cosmos/distribution/v1beta1/delegators/${walletAddress}/rewards`
);
const rewardsData = await rewardsResponse.json();
const totalRewards = rewardsData.total;
```

## Unbonding Period

- Default: 21 days (configurable via governance)
- Tokens cannot be transferred during unbonding period
- Unbonding can be cancelled before completion
- Check `unbonding_end_time` to see when tokens become available

## Commission Rates

- Validators charge a commission on staking rewards
- Commission rates vary by validator (typically 5-20%)
- Choose validators with lower commission for higher returns
- Commission rates are shown in validator list

## Slashing

- Validators can be slashed for downtime or double-signing
- Delegators share in slashing penalties
- Choose validators with high uptime to minimize risk
- Slashing events are tracked and displayed

## Error Handling

### Insufficient Balance

Ensure you have enough tokens to cover:
- Delegation amount
- Transaction fees (typically 1000-5000 uremes)

### Validator Not Found

- Verify validator address is correct
- Check if validator is active (not jailed)
- Query validators list to get current active validators

### Transaction Failed

- Check transaction fees
- Verify network connectivity
- Check transaction logs for detailed error messages
- Ensure validator is not jailed

## Best Practices

1. **Diversify Delegations**: Spread delegations across multiple validators
2. **Monitor Validator Performance**: Check uptime and commission rates
3. **Claim Rewards Regularly**: Claim rewards to compound returns
4. **Plan Unbonding**: Account for 21-day unbonding period
5. **Monitor Slashing**: Check validator slashing history

## Security Considerations

1. **Validator Selection**: Choose validators with high uptime and low commission
2. **Slashing Risk**: Understand that delegators share slashing penalties
3. **Private Keys**: Never share private keys or seed phrases
4. **Transaction Verification**: Always verify transaction details before signing

