# Transaction Hash Retrieval - Implementation Plan

## Problem

Currently, `submit_gradient` in `blockchain_client.py` always returns `"pending"` as the transaction hash because Cosmos SDK gRPC message handlers don't return transaction hashes directly.

## Current Architecture

```
Python Miner → gRPC → Go Node (MsgServer) → State Update
```

The gRPC message handler processes the message and updates state, but doesn't return a transaction hash because it's not a broadcast transaction.

## Solution Options

### Option 1: Transaction Broadcast Pattern (Recommended) ⭐

**Approach**: Sign and broadcast transaction via Tendermint RPC instead of using gRPC message handler.

**Architecture**:
```
Python Miner → Sign Transaction → Tendermint RPC Broadcast → Wait for Inclusion → Get TX Hash
```

**Implementation Steps**:

1. **Create Tendermint RPC Client** (`miner-engine/bridge/tendermint_client.py`):
   ```python
   class TendermintClient:
       def __init__(self, rpc_url: str = "http://localhost:26657"):
           self.rpc_url = rpc_url
       
       def broadcast_tx_sync(self, tx_bytes: bytes) -> Dict[str, Any]:
           """Broadcast transaction synchronously."""
           # POST to /broadcast_tx_sync
           pass
       
       def broadcast_tx_async(self, tx_bytes: bytes) -> Dict[str, Any]:
           """Broadcast transaction asynchronously."""
           # POST to /broadcast_tx_async
           pass
       
       def broadcast_tx_commit(self, tx_bytes: bytes) -> Dict[str, Any]:
           """Broadcast transaction and wait for commit."""
           # POST to /broadcast_tx_commit
           # Returns transaction hash
           pass
   ```

2. **Transaction Building** (`miner-engine/bridge/transaction_builder.py`):
   ```python
   class TransactionBuilder:
       def build_submit_gradient_tx(
           self,
           msg: MsgSubmitGradient,
           account: Account,
           chain_id: str,
           sequence: int,
           account_number: int,
       ) -> bytes:
           """Build and sign transaction."""
           # 1. Create transaction
           # 2. Sign with private key
           # 3. Encode to bytes
           pass
   ```

3. **Update BlockchainClient**:
   ```python
   def submit_gradient(self, ...) -> Dict[str, Any]:
       # 1. Build transaction
       tx_bytes = self.tx_builder.build_submit_gradient_tx(...)
       
       # 2. Broadcast via Tendermint RPC
       response = self.tendermint_client.broadcast_tx_commit(tx_bytes)
       
       # 3. Extract transaction hash
       tx_hash = response["result"]["hash"]
       
       return {
           "success": True,
           "tx_hash": tx_hash,
           ...
       }
   ```

**Dependencies**:
- `cosmpy` or `cosmos-sdk-python` for transaction building
- Tendermint RPC client library
- Account management (sequence, account_number)

**Pros**:
- ✅ Returns actual transaction hash
- ✅ Standard Cosmos SDK pattern
- ✅ Transaction tracking possible
- ✅ Can wait for inclusion

**Cons**:
- ❌ Requires significant refactoring
- ❌ Need to manage account sequence numbers
- ❌ More complex error handling
- ❌ Requires Tendermint RPC access

**Estimated Time**: 2-3 weeks

---

### Option 2: Response Message Enhancement

**Approach**: Add transaction hash field to `MsgSubmitGradientResponse` and calculate hash from SDK context.

**Implementation**:
1. Update `tx.proto` to include `tx_hash` in response
2. In Go message handler, calculate transaction hash from context
3. Return hash in response

**Pros**:
- ✅ Minimal changes to Python client
- ✅ No Tendermint RPC needed

**Cons**:
- ❌ SDK context doesn't have transaction hash in message handler
- ❌ Would require accessing transaction from context (not standard)
- ❌ May not be possible with current Cosmos SDK architecture

**Estimated Time**: 1 week (if feasible)

---

### Option 3: Hybrid Approach

**Approach**: Keep gRPC for message submission, query transaction hash separately.

**Implementation**:
1. Submit via gRPC (current method)
2. Query recent transactions from Tendermint RPC
3. Match by miner address, timestamp, or stored_gradient_id
4. Return matched transaction hash

**Pros**:
- ✅ Minimal changes to existing code
- ✅ Can be implemented incrementally

**Cons**:
- ❌ Not reliable (matching may fail)
- ❌ Race conditions possible
- ❌ Additional query overhead

**Estimated Time**: 1 week

---

## Recommended Implementation: Option 1

### Phase 1: Infrastructure (Week 1)

1. **Create Tendermint RPC Client**
   - File: `miner-engine/bridge/tendermint_client.py`
   - Functions: `broadcast_tx_sync`, `broadcast_tx_async`, `broadcast_tx_commit`
   - Error handling and retry logic

2. **Create Transaction Builder**
   - File: `miner-engine/bridge/transaction_builder.py`
   - Functions: `build_submit_gradient_tx`, `sign_transaction`
   - Account sequence management

3. **Update Dependencies**
   - Add `cosmpy` or similar library to `requirements.txt`
   - Add Tendermint RPC client library

### Phase 2: Integration (Week 2)

1. **Update BlockchainClient**
   - Add `TendermintClient` and `TransactionBuilder` instances
   - Refactor `submit_gradient` to use broadcast pattern
   - Handle account sequence numbers

2. **Account Management**
   - Query account info (sequence, account_number) from chain
   - Cache and increment sequence numbers
   - Handle sequence errors

3. **Error Handling**
   - Handle broadcast errors
   - Handle sequence mismatches
   - Retry logic for transient errors

### Phase 3: Testing & Polish (Week 3)

1. **Testing**
   - Unit tests for transaction building
   - Integration tests for broadcast
   - Error scenario testing

2. **Documentation**
   - Update API documentation
   - Add usage examples
   - Migration guide

---

## Migration Strategy

1. **Backward Compatibility**: Keep gRPC method as fallback
2. **Feature Flag**: Use environment variable to switch between methods
3. **Gradual Rollout**: Test with small subset of miners first

---

## Code Structure

```
miner-engine/
├── bridge/
│   ├── blockchain_client.py (updated)
│   ├── tendermint_client.py (new)
│   ├── transaction_builder.py (new)
│   └── account_manager.py (new)
└── requirements.txt (updated)
```

---

## Dependencies to Add

```txt
cosmpy>=0.8.0  # Cosmos SDK Python client
# OR
cosmos-sdk-python>=0.1.0  # Alternative library
```

---

## Testing Checklist

- [ ] Transaction building works correctly
- [ ] Transaction signing works correctly
- [ ] Broadcast succeeds
- [ ] Transaction hash is returned
- [ ] Sequence number management works
- [ ] Error handling works (network errors, sequence errors, etc.)
- [ ] Retry logic works
- [ ] Backward compatibility maintained

---

## Notes

- This is a **major refactoring** that affects core transaction flow
- Should be done carefully with thorough testing
- Consider keeping gRPC method as fallback during transition
- Account sequence management is critical for correctness

---

**Status**: Planned, not yet implemented  
**Priority**: High (but requires significant effort)  
**Estimated Completion**: 2-3 weeks

