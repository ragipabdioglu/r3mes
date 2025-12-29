# R3MES Test Suite

This directory contains comprehensive tests for the R3MES project.

## Structure

- `e2e/` - End-to-end tests (Playwright)
- `performance/` - Performance and load tests
- `security/` - Security and penetration tests

## Running Tests

### E2E Tests

```bash
cd web-dashboard
npm install
npx playwright install
npx playwright test
```

### Performance Tests

```bash
cd tests/performance
pip install -r requirements.txt
python load_test.py
python bandwidth_test.py
python memory_profiling.py
```

### Security Tests

```bash
cd tests/security
pip install -r ../performance/requirements.txt
python penetration_test.py
```

## Test Coverage Goals

- E2E: > 80% of critical user flows
- Performance: All endpoints < 200ms p95 latency
- Security: Zero critical vulnerabilities

