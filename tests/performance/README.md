# Performance and Load Testing

Load testing infrastructure for R3MES backend API.

## Requirements

```bash
pip install locust
```

## Running Load Tests

### Quick Start

```bash
cd tests/performance
locust -f locustfile.py --host=http://localhost:8000
```

Then open http://localhost:8089 in browser.

### Using Test Scenarios Script

```bash
# Normal load: 100 users
./load_test_scenarios.sh http://localhost:8000 normal

# High load: 1000 users
./load_test_scenarios.sh http://localhost:8000 high

# Stress test: 2000 users
./load_test_scenarios.sh http://localhost:8000 stress

# Run all scenarios
./load_test_scenarios.sh http://localhost:8000 all
```

## Test Scenarios

### Normal Load
- **Users**: 100 concurrent
- **Spawn Rate**: 10 users/second
- **Duration**: 5 minutes
- **Target**: < 200ms p95 latency

### High Load
- **Users**: 1000 concurrent
- **Spawn Rate**: 50 users/second
- **Duration**: 10 minutes
- **Target**: < 500ms p95 latency

### Stress Test
- **Users**: 2000+ concurrent
- **Spawn Rate**: 100 users/second
- **Duration**: 15 minutes
- **Target**: System stability, graceful degradation

## Metrics Collected

- **Request Rate**: Requests per second
- **Response Time**: p50, p95, p99 latencies
- **Error Rate**: Percentage of failed requests
- **Throughput**: Successful requests per second

## Reports

Test reports are generated in `reports/` directory:
- HTML reports: `reports/{scenario}_load.html`
- CSV data: `reports/{scenario}_load_*.csv`

## Performance Targets

| Metric | Normal Load | High Load | Stress Test |
|--------|------------|-----------|-------------|
| p95 Latency | < 200ms | < 500ms | < 2000ms |
| Error Rate | < 0.1% | < 1% | < 5% |
| Throughput | > 1000 req/s | > 5000 req/s | > 10000 req/s |

## Continuous Integration

Load tests run in CI:
- On release candidates
- Before production deployment
- Weekly performance regression tests

