# R3MES Error Codes Reference

This document provides a comprehensive reference for all error codes used in the R3MES protocol.

## Blockchain Layer (Go/Cosmos SDK)

### Module: `remes`

All error codes are prefixed with module name and error code number.

#### Error Codes

| Code | Name | Description | HTTP Status |
|------|------|-------------|-------------|
| 1001 | `ErrInvalidMiner` | Invalid miner address format | 400 |
| 1002 | `ErrInvalidGradient` | Invalid gradient data or format | 400 |
| 1003 | `ErrInvalidIPFSHash` | Invalid IPFS hash format | 400 |
| 1004 | `ErrGradientNotFound` | Gradient not found in storage | 404 |
| 1005 | `ErrInvalidAggregation` | Invalid aggregation data | 400 |
| 1006 | `ErrAggregationNotFound` | Aggregation record not found | 404 |
| 1007 | `ErrInvalidChallenge` | Invalid challenge data or format | 400 |
| 1008 | `ErrChallengeNotFound` | Challenge record not found | 404 |
| 1009 | `ErrInvalidResponse` | Invalid challenge response | 400 |
| 1010 | `ErrInvalidSigner` | Invalid message signer | 401 |
| 1011 | `ErrNotImplemented` | Feature not yet implemented | 501 |
| 1012 | `ErrInvalidRequest` | Invalid request parameters | 400 |
| 1013 | `ErrInsufficientBalance` | Insufficient account balance | 400 |
| 1014 | `ErrInvalidNonce` | Invalid or reused nonce | 400 |
| 1015 | `ErrInvalidAmount` | Invalid amount format | 400 |
| 1016 | `ErrInvalidNonce` | Invalid nonce (replay attack prevention) | 400 |
| 1017 | `ErrInsufficientStake` | Insufficient stake amount | 400 |
| 1018 | `ErrDatasetNotFound` | Dataset not found | 404 |
| 1019 | `ErrInvalidParameter` | Invalid parameter value | 400 |

### Usage Examples

```go
// Return error with code
return nil, errorsmod.Wrapf(
    types.ErrInvalidMiner,
    "invalid miner address: %s", address,
)

// Check error type
if errors.Is(err, types.ErrInvalidNonce) {
    // Handle nonce error
}
```

## Backend API (Python/FastAPI)

### HTTP Status Codes

| Status | Description | Common Causes |
|--------|-------------|---------------|
| 200 | Success | Request processed successfully |
| 400 | Bad Request | Invalid parameters, missing required fields |
| 401 | Unauthorized | Missing or invalid API key |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource not found |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error |
| 503 | Service Unavailable | Service temporarily unavailable |

### Error Response Format

```json
{
  "detail": "Error message description",
  "error_code": "ERROR_CODE",
  "status_code": 400
}
```

### Common Error Codes

| Error Code | Description | Status |
|------------|-------------|--------|
| `INVALID_API_KEY` | Invalid or missing API key | 401 |
| `RATE_LIMIT_EXCEEDED` | Too many requests | 429 |
| `INSUFFICIENT_CREDITS` | User has insufficient credits | 400 |
| `INVALID_WALLET_ADDRESS` | Invalid wallet address format | 400 |
| `BLOCKCHAIN_CONNECTION_ERROR` | Cannot connect to blockchain | 503 |
| `DATABASE_ERROR` | Database operation failed | 500 |
| `MODEL_LOAD_ERROR` | Failed to load AI model | 500 |
| `INFERENCE_ERROR` | Inference processing failed | 500 |

## Rate Limiting

### Default Limits

| Endpoint Type | Limit | Window |
|---------------|-------|--------|
| Chat/Inference | 10 requests | 1 minute |
| GET requests | 30 requests | 1 minute |
| POST requests | 20 requests | 1 minute |
| WebSocket connections | 100 connections | Per IP |

### Rate Limit Headers

```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 5
X-RateLimit-Reset: 1640995200
```

### Rate Limit Exceeded Response

```json
{
  "detail": "Rate limit exceeded. Please try again later.",
  "error_code": "RATE_LIMIT_EXCEEDED",
  "retry_after": 30
}
```

## Error Handling Best Practices

1. **Always return descriptive error messages** - Help users understand what went wrong
2. **Include error codes** - Enable programmatic error handling
3. **Log errors with context** - Include request ID, user ID, timestamp
4. **Don't expose internal details** - Sanitize error messages in production
5. **Provide retry guidance** - Indicate if request can be retried

## Troubleshooting

### Common Issues

1. **401 Unauthorized**
   - Check API key is set correctly
   - Verify API key is not expired
   - Ensure API key has required permissions

2. **429 Too Many Requests**
   - Wait for rate limit window to reset
   - Implement exponential backoff
   - Consider upgrading rate limits

3. **400 Bad Request**
   - Validate request parameters
   - Check required fields are present
   - Verify data format matches API specification

4. **500 Internal Server Error**
   - Check service logs
   - Verify database connectivity
   - Ensure all dependencies are running

