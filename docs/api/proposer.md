# Proposer API Documentation

## Overview

The Proposer API provides endpoints for managing proposer nodes, viewing aggregations, and monitoring gradient pools.

## Base URL

**Development**:
```
http://localhost:8000/proposer
```

**Production**:
```
https://api.r3mes.network/proposer
```

## Endpoints

### List Proposer Nodes

Get a list of all proposer nodes.

**Endpoint**: `GET /proposer/nodes`

**Query Parameters**:
- `limit` (optional, default: 100): Maximum number of nodes to return
- `offset` (optional, default: 0): Pagination offset

**Response**:
```json
{
  "nodes": [
    {
      "node_address": "remes1abc...",
      "status": "ACTIVE",
      "total_aggregations": 25,
      "total_rewards": "5000.5",
      "last_aggregation_height": 12345,
      "resources": {
        "cpu_cores": 8,
        "memory_gb": 16
      },
      "stake": "1000000uremes"
    }
  ],
  "total": 1
}
```

**Example**:
```bash
curl http://localhost:8000/proposer/nodes?limit=50&offset=0
```

### Get Proposer Node Details

Get detailed information about a specific proposer node.

**Endpoint**: `GET /proposer/nodes/{address}`

**Path Parameters**:
- `address`: Proposer node address

**Response**:
```json
{
  "node_address": "remes1abc...",
  "status": "ACTIVE",
  "total_aggregations": 25,
  "total_rewards": "5000.5",
  "last_aggregation_height": 12345,
  "resources": {
    "cpu_cores": 8,
    "memory_gb": 16,
    "storage_gb": 100
  },
  "stake": "1000000uremes"
}
```

**Example**:
```bash
curl http://localhost:8000/proposer/nodes/remes1abc...
```

### List Aggregations

Get a list of recent aggregations.

**Endpoint**: `GET /proposer/aggregations`

**Query Parameters**:
- `limit` (optional, default: 50): Maximum number of aggregations to return
- `offset` (optional, default: 0): Pagination offset
- `proposer` (optional): Filter by proposer address

**Response**:
```json
{
  "aggregations": [],
  "total": 0,
  "message": "Aggregation history not yet implemented. Will query from blockchain events."
}
```

**Example**:
```bash
curl http://localhost:8000/proposer/aggregations?limit=20&proposer=remes1abc...
```

### Get Aggregation Details

Get details of a specific aggregation.

**Endpoint**: `GET /proposer/aggregations/{aggregation_id}`

**Path Parameters**:
- `aggregation_id`: Aggregation ID

**Response**:
```json
{
  "aggregation_id": 123,
  "proposer": "remes1abc...",
  "aggregated_gradient_ipfs_hash": "Qm...",
  "merkle_root": "0x...",
  "participant_count": 10,
  "training_round_id": 5
}
```

**Example**:
```bash
curl http://localhost:8000/proposer/aggregations/123
```

### Get Gradient Pool

Get pending gradients available for aggregation.

**Endpoint**: `GET /proposer/pool`

**Query Parameters**:
- `limit` (optional, default: 100): Maximum number of gradients to return
- `offset` (optional, default: 0): Pagination offset
- `status` (optional, default: "pending"): Gradient status filter

**Response**:
```json
{
  "pending_gradients": [
    {
      "id": 1,
      "status": "pending",
      "ipfs_hash": "Qm...",
      "miner": "remes1xyz...",
      "training_round_id": 10
    }
  ],
  "total_count": 1
}
```

**Example**:
```bash
curl http://localhost:8000/proposer/pool?limit=50&status=pending
```

### List Pending Commits

Get pending aggregation commitments.

**Endpoint**: `GET /proposer/commits`

**Query Parameters**:
- `limit` (optional, default: 50): Maximum number of commits to return
- `offset` (optional, default: 0): Pagination offset

**Response**:
```json
{
  "commits": [],
  "total": 0,
  "message": "Commitment list not yet implemented. Will query from blockchain."
}
```

**Example**:
```bash
curl http://localhost:8000/proposer/commits?limit=20
```

### Submit Aggregation

Submit aggregation (informational endpoint).

**Endpoint**: `POST /proposer/aggregate`

**Request Body**:
```json
{
  "proposer": "remes1abc...",
  "gradient_ids": [1, 2, 3, 4, 5],
  "aggregated_hash": "Qm...",
  "merkle_root": "0x...",
  "training_round_id": 10
}
```

**Response**:
```json
{
  "message": "Aggregation submission initiated. Actual submission requires blockchain transaction.",
  "note": "Use blockchain client to send MsgSubmitAggregation transaction.",
  "proposer": "remes1abc...",
  "gradient_count": 5
}
```

**Example**:
```bash
curl -X POST http://localhost:8000/proposer/aggregate \
  -H "Content-Type: application/json" \
  -d '{
    "proposer": "remes1abc...",
    "gradient_ids": [1, 2, 3, 4, 5],
    "aggregated_hash": "Qm...",
    "merkle_root": "0x...",
    "training_round_id": 10
  }'
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
  "detail": "Proposer node not found"
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Failed to fetch proposer nodes"
}
```

## Rate Limiting

All endpoints are rate-limited. Default limits:
- GET requests: 100 requests/minute
- POST requests: 10 requests/minute

## Authentication

Currently, endpoints do not require authentication. In production, API keys or JWT tokens may be required.

## Examples

### List All Proposer Nodes

```bash
curl http://localhost:8000/proposer/nodes
```

### Get Gradient Pool

```bash
curl http://localhost:8000/proposer/pool
```

### Get Aggregation Details

```bash
curl http://localhost:8000/proposer/aggregations/123
```

