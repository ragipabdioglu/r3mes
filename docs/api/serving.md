# Serving Node API Documentation

## Overview

The Serving Node API provides endpoints for managing serving nodes, viewing inference requests, and monitoring serving statistics.

## Base URL

**Development**:
```
http://localhost:8000/serving
```

**Production**:
```
https://api.r3mes.network/serving
```

## Endpoints

### List Serving Nodes

Get a list of all serving nodes.

**Endpoint**: `GET /serving/nodes`

**Query Parameters**:
- `limit` (optional, default: 100): Maximum number of nodes to return
- `offset` (optional, default: 0): Pagination offset

**Response**:
```json
{
  "nodes": [
    {
      "node_address": "remes1abc...",
      "model_version": "v1.0.0",
      "model_ipfs_hash": "Qm...",
      "is_available": true,
      "total_requests": 150,
      "successful_requests": 148,
      "average_latency_ms": 245,
      "last_heartbeat": "2025-01-15T10:30:00Z",
      "status": "ACTIVE"
    }
  ],
  "total": 1
}
```

**Example**:
```bash
curl http://localhost:8000/serving/nodes?limit=50&offset=0
```

### Get Serving Node Details

Get detailed information about a specific serving node.

**Endpoint**: `GET /serving/nodes/{address}`

**Path Parameters**:
- `address`: Serving node address

**Response**:
```json
{
  "node_address": "remes1abc...",
  "model_version": "v1.0.0",
  "model_ipfs_hash": "Qm...",
  "is_available": true,
  "total_requests": 150,
  "successful_requests": 148,
  "failed_requests": 2,
  "success_rate": 98.67,
  "average_latency_ms": 245,
  "last_heartbeat": "2025-01-15T10:30:00Z",
  "status": "ACTIVE",
  "resources": {
    "cpu_cores": 4,
    "memory_gb": 8,
    "gpu_count": 1
  },
  "stake": "1000000uremes"
}
```

**Example**:
```bash
curl http://localhost:8000/serving/nodes/remes1abc...
```

### Get Serving Node Requests

Get inference requests for a specific serving node.

**Endpoint**: `GET /serving/nodes/{address}/requests`

**Path Parameters**:
- `address`: Serving node address

**Query Parameters**:
- `limit` (optional, default: 50): Maximum number of requests to return
- `offset` (optional, default: 0): Pagination offset

**Response**:
```json
{
  "requests": [],
  "total": 0,
  "message": "Inference request history not yet implemented. Will query from blockchain events."
}
```

**Example**:
```bash
curl http://localhost:8000/serving/nodes/remes1abc.../requests?limit=20
```

### Get Serving Node Statistics

Get statistics for a specific serving node.

**Endpoint**: `GET /serving/nodes/{address}/stats`

**Path Parameters**:
- `address`: Serving node address

**Response**:
```json
{
  "node_address": "remes1abc...",
  "total_requests": 150,
  "successful_requests": 148,
  "failed_requests": 2,
  "success_rate": 98.67,
  "average_latency_ms": 245,
  "model_version": "v1.0.0",
  "is_available": true
}
```

**Example**:
```bash
curl http://localhost:8000/serving/nodes/remes1abc.../stats
```

### Update Serving Node Status

Update serving node status (informational endpoint).

**Endpoint**: `POST /serving/nodes/{address}/status`

**Path Parameters**:
- `address`: Serving node address

**Request Body**:
```json
{
  "is_available": true,
  "model_version": "v1.0.0",
  "model_ipfs_hash": "Qm..."
}
```

**Response**:
```json
{
  "message": "Status update submitted. Actual update requires blockchain transaction.",
  "note": "Use blockchain client to send MsgUpdateServingNodeStatus transaction."
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/serving/nodes/remes1abc.../status \
  -H "Content-Type: application/json" \
  -d '{
    "is_available": true,
    "model_version": "v2.0.0"
  }'
```

### Get Inference Request

Get details of a specific inference request.

**Endpoint**: `GET /serving/requests/{request_id}`

**Path Parameters**:
- `request_id`: Inference request ID

**Response**:
```json
{
  "request_id": "req_123",
  "requester": "remes1xyz...",
  "serving_node": "remes1abc...",
  "model_version": "v1.0.0",
  "input_data_ipfs_hash": "Qm...",
  "fee": "1000uremes",
  "status": "completed",
  "request_time": "2025-01-15T10:00:00Z",
  "result_ipfs_hash": "Qm...",
  "latency_ms": 245
}
```

**Example**:
```bash
curl http://localhost:8000/serving/requests/req_123
```

## Error Responses

All endpoints may return the following error responses:

**400 Bad Request**:
```json
{
  "detail": "Invalid request parameters"
}
```

**404 Not Found**:
```json
{
  "detail": "Serving node not found"
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Failed to fetch serving nodes"
}
```

## Rate Limiting

All endpoints are rate-limited. Default limits:
- GET requests: 100 requests/minute
- POST requests: 10 requests/minute

Rate limit headers are included in responses:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1642248000
```

## Authentication

Currently, endpoints do not require authentication. In production, API keys or JWT tokens may be required.

## Examples

### List All Serving Nodes

```bash
curl http://localhost:8000/serving/nodes
```

### Get Node Statistics

```bash
curl http://localhost:8000/serving/nodes/remes1abc.../stats
```

### Check Node Availability

```bash
curl http://localhost:8000/serving/nodes/remes1abc... | jq '.is_available'
```

