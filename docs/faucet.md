# Testnet Faucet

Get free testnet REMES tokens for development and testing.

## Quick Request

**Web Interface:**
Visit [faucet.r3mes.network](https://faucet.r3mes.network)

**CLI:**
```bash
r3mes faucet request
```

**cURL:**
```bash
curl -X POST https://faucet.r3mes.network/api/faucet/request \
  -H "Content-Type: application/json" \
  -d '{"address": "remes1your_address_here"}'
```

## API Reference

### Request Tokens

| Property | Value |
|----------|-------|
| Endpoint | `POST /api/faucet/request` |
| Content-Type | `application/json` |

**Request Body:**
```json
{
  "address": "remes1abc123def456..."
}
```

**Success Response:**
```json
{
  "success": true,
  "amount": "100000000",
  "denom": "uremes",
  "tx_hash": "ABC123...",
  "message": "100 REMES sent successfully"
}
```

**Rate Limited Response:**
```json
{
  "success": false,
  "error": "rate_limit_exceeded",
  "message": "Please wait 24 hours between requests",
  "retry_after": 86400
}
```

### Check Status

| Property | Value |
|----------|-------|
| Endpoint | `GET /api/faucet/status` |

**Response:**
```json
{
  "status": "operational",
  "balance": "1000000000000",
  "requests_today": 156
}
```

## Rate Limits

| Limit | Value |
|-------|-------|
| Requests per address | 1 per 24 hours |
| Requests per IP | 3 per 24 hours |
| Amount per request | 100 REMES |

## Faucet URLs

| Network | URL |
|---------|-----|
| Testnet | https://faucet.r3mes.network |
| Local | http://localhost:8080/api/faucet |

## Integration Examples

**Python:**
```python
import requests

def request_faucet(address: str) -> dict:
    response = requests.post(
        "https://faucet.r3mes.network/api/faucet/request",
        json={"address": address}
    )
    return response.json()

result = request_faucet("remes1abc123...")
print(f"Received: {result.get('amount', 0)} uremes")
```

**JavaScript:**
```javascript
async function requestFaucet(address) {
  const response = await fetch(
    'https://faucet.r3mes.network/api/faucet/request',
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ address })
    }
  );
  return response.json();
}
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Rate limit exceeded | Wait 24 hours or use a different address |
| Invalid address | Ensure address starts with `remes1` and is 39-45 characters |
| Faucet empty | Try again later or contact support |

## Notes

- Testnet tokens have no real value
- Faucet is for development purposes only
- Abuse may result in IP/address blocking
