# Faucet API Documentation

## Overview

The Faucet API allows users to claim free R3MES tokens. This is useful for new users who need initial tokens to start using the network.

## Endpoints

### Claim Tokens

Claim tokens from the faucet.

**Endpoint:** `POST /faucet/claim`

**Request Body:**
```json
{
  "address": "remes1abc...",
  "amount": "1000000uremes"  // Optional, defaults to FAUCET_AMOUNT
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Successfully sent 1000000uremes to remes1abc...",
  "tx_hash": "ABC123...",
  "amount": "1000000uremes",
  "next_claim_available_at": "2024-01-02T12:00:00Z"
}
```

**Response (Rate Limit Exceeded - 429):**
```json
{
  "error": "Rate limit exceeded",
  "message": "You can only claim from the faucet once per day",
  "next_claim_available_at": "2024-01-02T12:00:00Z"
}
```

**Response (Faucet Disabled - 503):**
```json
{
  "detail": "Faucet is currently disabled"
}
```

**Response (Invalid Amount - 400):**
```json
{
  "detail": "Requested amount exceeds daily limit of 5000000uremes"
}
```

**Response (Transaction Failed - 500):**
```json
{
  "detail": "Failed to send tokens. Please try again later or contact support."
}
```

### Get Faucet Status

Get faucet configuration and status.

**Endpoint:** `GET /faucet/status`

**Response:**
```json
{
  "enabled": true,
  "amount_per_claim": "1000000uremes",
  "daily_limit": "5000000uremes",
  "rate_limit": "1 request per day per IP and per address"
}
```

## Rate Limiting

- **Per IP Address**: 1 claim per 24 hours
- **Per Wallet Address**: 1 claim per 24 hours
- Both limits must be satisfied for a successful claim

## Configuration

### Environment Variables

- `FAUCET_ENABLED`: Enable/disable faucet (default: `true`)
- `FAUCET_AMOUNT`: Default amount per claim (default: `1000000uremes` = 1 REMES)
- `FAUCET_DAILY_LIMIT`: Maximum amount per day (default: `5000000uremes` = 5 REMES)

## Examples

### cURL

```bash
# Claim tokens
curl -X POST "https://api.r3mes.network/faucet/claim" \
  -H "Content-Type: application/json" \
  -d '{
    "address": "remes1abc...",
    "amount": "1000000uremes"
  }'

# Get faucet status
curl "https://api.r3mes.network/faucet/status"
```

### JavaScript/TypeScript

```typescript
import { claimFaucet, getFaucetStatus } from '@/lib/api';

// Claim tokens
try {
  const response = await claimFaucet({
    address: 'remes1abc...',
    amount: '1000000uremes'  // Optional
  });
  console.log('Success:', response.tx_hash);
} catch (error) {
  if (error.response?.status === 429) {
    console.log('Rate limit exceeded');
  }
}

// Get status
const status = await getFaucetStatus();
console.log('Faucet enabled:', status.enabled);
```

### Python

```python
import requests

# Claim tokens
response = requests.post(
    'https://api.r3mes.network/faucet/claim',
    json={
        'address': 'remes1abc...',
        'amount': '1000000uremes'  # Optional
    }
)

if response.status_code == 200:
    data = response.json()
    print(f"Success: {data['tx_hash']}")
elif response.status_code == 429:
    print("Rate limit exceeded")
else:
    print(f"Error: {response.text}")

# Get status
status = requests.get('https://api.r3mes.network/faucet/status').json()
print(f"Faucet enabled: {status['enabled']}")
```

## Error Handling

### Rate Limit Exceeded (429)

When rate limit is exceeded, the response includes `next_claim_available_at` timestamp. Calculate the time remaining:

```javascript
const nextClaim = new Date(response.data.next_claim_available_at);
const now = new Date();
const hoursUntil = Math.ceil((nextClaim - now) / (1000 * 60 * 60));
console.log(`Next claim available in ${hoursUntil} hours`);
```

### Invalid Address Format

Addresses must start with `remes` (R3MES network prefix). Invalid addresses return 400 error.

### Amount Validation

- Amount must be in `uremes` format (e.g., `1000000uremes`)
- Amount cannot exceed `FAUCET_DAILY_LIMIT`
- Amount must be a positive integer

## Security

- Rate limiting prevents abuse
- Address validation ensures only valid R3MES addresses can claim
- All transactions are recorded on-chain for audit
- IP-based rate limiting prevents multiple claims from same network

## Best Practices

1. **Check Status First**: Query `/faucet/status` before attempting to claim
2. **Handle Rate Limits**: Implement retry logic with exponential backoff
3. **Display Next Claim Time**: Show users when they can claim again
4. **Validate Addresses**: Validate wallet addresses before submitting
5. **Error Messages**: Display user-friendly error messages

