# Backend Integration Tests

This directory contains integration tests for the R3MES backend service.

## Test Structure

- `test_database.py` - Unit tests for database operations
- `test_api_integration.py` - Integration tests for API endpoints
- `test_blockchain_integration.py` - Integration tests for blockchain interactions
- `conftest.py` - Pytest fixtures and configuration

## Running Tests

### Run all tests
```bash
cd backend
python -m pytest tests/ -v
```

### Run specific test file
```bash
python -m pytest tests/test_api_integration.py -v
```

### Run with coverage
```bash
python -m pytest tests/ --cov=app --cov-report=html
```

## Test Environment

Tests use a temporary database and mock external services (blockchain, IPFS) to ensure:
- Tests are isolated and don't affect production data
- Tests run quickly without external dependencies
- Tests are deterministic and repeatable

## Environment Variables

Tests automatically set:
- `R3MES_ENV=test`
- `R3MES_TEST_MODE=true`
- Temporary database paths

## Writing New Tests

1. Use `unittest.TestCase` or `pytest` fixtures
2. Use `temp_db` fixture for database tests
3. Mock external services (blockchain, IPFS)
4. Clean up resources in `tearDown` or fixture cleanup

## Integration Test Coverage

### API Endpoints
- ✅ Health check
- ✅ Chat endpoint
- ✅ User info endpoint
- ✅ API key management
- ✅ Network stats
- ✅ Blocks endpoint

### Database Operations
- ✅ User credit lifecycle
- ✅ API key lifecycle
- ✅ Miner statistics
- ✅ Network statistics

### Blockchain Integration
- ✅ Query available chunks
- ✅ Get block height
- ✅ Get recent blocks
- ✅ Database-blockchain sync

