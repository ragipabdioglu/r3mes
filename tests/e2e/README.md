# E2E Test Suite

End-to-end tests for R3MES protocol flows.

## Test Structure

- `test_protocol_flow.py` - Protocol flow tests (gradient submission, aggregation, challenges)
- `conftest.py` - Pytest fixtures and configuration

## Running Tests

### Run all E2E tests

```bash
cd tests/e2e
pytest -v
```

### Run specific test

```bash
pytest tests/e2e/test_protocol_flow.py::TestProtocolFlow::test_gradient_submission_flow -v
```

### Run with markers

```bash
# Run only E2E marked tests
pytest -m e2e -v

# Run async tests
pytest -m asyncio -v
```

## Test Scenarios

### Protocol Flows

1. **Gradient Submission Flow**
   - Miner generates gradient
   - Upload to IPFS
   - Submit to blockchain
   - Verify on blockchain
   - Check dashboard visibility

2. **Aggregation Flow**
   - Multiple gradients submitted
   - Aggregation triggered
   - Aggregation verified
   - Model updated

3. **Challenge-Response Flow**
   - Gradient submitted
   - Challenge issued
   - Response provided
   - Challenge resolved

4. **Pinning Flow**
   - Commit to pinning
   - Challenge issued
   - Response provided
   - Rewards distributed

5. **Nonce Replay Protection**
   - Submit gradient with nonce
   - Try to resubmit with same nonce
   - Verify rejection

### Integration Flows

1. **Backend-Blockchain Sync**
   - Verify backend database matches blockchain state

2. **Dashboard Real-time Updates**
   - Verify WebSocket events

3. **Cache Invalidation**
   - Verify cache updates on data changes

### E2E Scenarios

1. **Miner Lifecycle**
   - Registration → Submissions → Reputation → Rewards

2. **Validator Lifecycle**
   - Registration → Challenges → Verification → Rewards

3. **Network Growth**
   - Start with 1 node → Add miners → Add validators → Verify scalability

## Test Environment

Tests use:
- Mock blockchain client (can be configured to use testnet)
- Mock IPFS client (can be configured to use local IPFS)
- In-memory database (SQLite)
- Test Redis instance (DB 15)

## Configuration

Set environment variables for test configuration:

```bash
export TEST_BLOCKCHAIN_RPC_URL=http://localhost:26657
export TEST_BLOCKCHAIN_GRPC_URL=localhost:9090
export TEST_BACKEND_URL=http://localhost:8000
export TEST_REDIS_URL=redis://localhost:6379/15
```

## Continuous Integration

E2E tests run in CI pipeline:
- On every push/PR
- Before production deployment
- With test coverage reporting

