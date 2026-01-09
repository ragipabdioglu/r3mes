# ADR-0005: Credit System for API Access

**Status**: Accepted  
**Date**: 2024-03-01  
**Deciders**: Product Team, Backend Team

## Context

We need a system to control API access and prevent abuse. Options include:
- Rate limiting only (doesn't account for cost)
- Subscription-based (complex billing)
- Pay-per-use (requires payment integration)
- Credit-based system (flexible, can be earned/spent)

We need a system that:
- Prevents abuse and controls costs
- Allows users to earn credits (mining, staking)
- Supports different credit costs per operation
- Tracks credit balance and history
- Integrates with blockchain for earning credits

## Decision

We will implement a **Credit-Based Access Control System** where:
1. Users have a credit balance (stored in database)
2. Each API operation costs credits (configurable per endpoint)
3. Credits can be earned through mining/staking (blockchain)
4. Credits are deducted before operation execution
5. Insufficient credits result in 402 Payment Required error

## Consequences

### Positive
- **Cost Control**: Prevents abuse and controls API costs
- **Flexible**: Can adjust credit costs per operation
- **Fair**: Users can earn credits through participation
- **Simple**: Easy to understand and implement
- **Blockchain Integration**: Credits can be earned on-chain

### Negative
- **Database Dependency**: Requires database for credit tracking
- **Race Conditions**: Need to handle concurrent credit deductions
- **Complexity**: Need to handle edge cases (negative balance, etc.)
- **User Experience**: Users need to understand credit system

### Neutral
- Credits are off-chain (database) but can be earned on-chain
- Credit costs can be adjusted without blockchain changes
- Can implement credit refunds for failed operations

## Implementation Details

- Store credit balance in `users` table
- Deduct credits atomically (database transaction)
- Check credits before operation (fail fast)
- Support credit refunds for failed operations
- Integrate with blockchain for credit earning
- Provide credit history and balance endpoints

## Alternatives Considered

1. **Rate Limiting Only**:
   - Pros: Simple, no database needed
   - Cons: Doesn't account for operation cost, can't earn credits

2. **Subscription Model**:
   - Pros: Predictable revenue, simple for users
   - Cons: Complex billing, doesn't scale with usage

3. **Pay-Per-Use**:
   - Pros: Direct payment, no credits needed
   - Cons: Requires payment integration, complex

---

**Related ADRs**: None

