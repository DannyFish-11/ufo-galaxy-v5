# Node 04 - Router

Request Routing Node for UFO Galaxy v5.0

## Overview

The Router node provides intelligent request routing:
- Load balancing across nodes
- Service discovery
- Circuit breaker pattern
- Request forwarding and proxying

## Load Balancing Strategies

- `round_robin` - Distribute requests evenly
- `random` - Random selection
- `least_connections` - Select least loaded
- `weighted` - Weight-based selection
- `ip_hash` - IP-based sticky sessions

## API Endpoints

### Health Check
```
GET /health
```

### Services
```
POST /services
```
Register a new service endpoint.

**Request Body:**
```json
{
  "name": "my-service",
  "host": "localhost",
  "port": 8080,
  "path": "/",
  "protocol": "http",
  "weight": 1,
  "health_check_path": "/health"
}
```

```
GET /services
```
List registered services.

Query Parameters:
- `status`: Filter by status

```
DELETE /services/{endpoint_id}
```
Unregister a service endpoint.

### Routes
```
POST /routes
```
Create a new routing rule.

**Request Body:**
```json
{
  "name": "api-route",
  "path_prefix": "/api",
  "target_service": "my-service",
  "methods": ["GET", "POST"],
  "priority": 10,
  "strip_prefix": false
}
```

```
GET /routes
```
List all routing rules.

```
DELETE /routes/{rule_id}
```
Delete a routing rule.

### Circuit Breakers
```
GET /circuit-breakers
```
List all circuit breakers.

```
POST /circuit-breakers/{service_id}/reset
```
Reset a circuit breaker.

### Proxy
```
/proxy/{service_name}/{path}
```
Proxy a request to a service.

Supports all HTTP methods: GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS

### Statistics
```
GET /stats
```
Get routing statistics.

## Service Status

- `healthy` - Service is healthy
- `degraded` - Service is experiencing issues
- `unhealthy` - Service is unhealthy
- `offline` - Service is offline

## Circuit Breaker States

- `closed` - Normal operation
- `open` - Circuit is open (failing)
- `half_open` - Testing recovery

## Configuration

See `config.yaml` for configuration options.

## Usage Example

```python
import httpx

# Register a service
response = httpx.post("http://localhost:8004/services", json={
    "name": "my-api",
    "host": "localhost",
    "port": 8080,
    "health_check_path": "/health"
})
endpoint_id = response.json()["endpoint_id"]

# Create a route
response = httpx.post("http://localhost:8004/routes", json={
    "name": "api-route",
    "path_prefix": "/api",
    "target_service": "my-api",
    "methods": ["GET", "POST"]
})

# Proxy a request
response = httpx.get("http://localhost:8004/proxy/my-api/users")
print(response.json())
```

## Port

- HTTP API: `8004`

## Dependencies

- Node 00 - StateMachine (port 8000)
- Optional: Node 05 - Auth (port 8005)

## Author

UFO Galaxy Team
