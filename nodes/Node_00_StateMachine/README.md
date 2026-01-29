# Node 00 - StateMachine

Central State Management Node for UFO Galaxy v5.0

## Overview

The StateMachine node manages the global state of the UFO Galaxy system, providing:
- State transitions and state machine logic
- Global configuration management
- Event propagation
- Node registry and discovery

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the node.

### State Management
```
GET /state
```
Get the current global state.

```
POST /state/transition
```
Transition the system to a new state.

**Request Body:**
```json
{
  "from_state": "ready",
  "to_state": "running",
  "reason": "Starting system"
}
```

### Node Registration
```
POST /nodes/register
```
Register a node with the state machine.

```
POST /nodes/unregister/{node_id}
```
Unregister a node.

```
GET /nodes
```
List all registered nodes.

```
POST /nodes/heartbeat/{node_id}
```
Update node heartbeat.

### Configuration
```
GET /config
```
Get current global configuration.

```
POST /config/update
```
Update global configuration.

### State History
```
GET /state/history
```
Get state transition history.

## Configuration

See `config.yaml` for configuration options.

## System States

- `initializing` - System is starting up
- `ready` - System is ready for operation
- `running` - System is running normally
- `paused` - System is paused
- `error` - System encountered an error
- `shutdown` - System is shutting down

## Usage Example

```python
import httpx

# Register a node
response = httpx.post("http://localhost:8000/nodes/register", json={
    "node_id": "node_01",
    "state": "ready",
    "last_heartbeat": "2024-01-01T00:00:00Z",
    "metadata": {"version": "1.0.0"}
})

# Transition state
response = httpx.post("http://localhost:8000/state/transition", json={
    "from_state": "ready",
    "to_state": "running",
    "reason": "Starting services"
})
```

## Port

- HTTP API: `8000`

## Dependencies

- None (core node)

## Author

UFO Galaxy Team
