# Node 41 - MQTT

MQTT Broker Client Node for UFO Galaxy v5.0

## Overview

The MQTT node provides MQTT client functionality:
- Connect to MQTT brokers
- Publish/subscribe to topics
- Message persistence
- QoS management
- Last will and testament

## QoS Levels

- `0` - At most once (fire and forget)
- `1` - At least once (delivered at least once)
- `2` - Exactly once (delivered exactly once)

## API Endpoints

### Health Check
```
GET /health
```

### Brokers
```
POST /brokers
```
Add a new MQTT broker configuration.

**Request Body:**
```json
{
  "name": "My Broker",
  "host": "localhost",
  "port": 1883,
  "use_ssl": false,
  "username": "user",
  "password": "pass",
  "client_id": "my-client",
  "keepalive": 60
}
```

```
GET /brokers
```
List all configured brokers.

```
POST /brokers/{broker_id}/connect
```
Connect to a broker.

```
POST /brokers/{broker_id}/disconnect
```
Disconnect from a broker.

### Publishing
```
POST /publish
```
Publish a message to a topic.

**Request Body:**
```json
{
  "broker_id": "broker-id",
  "topic": "sensors/temperature",
  "payload": {"value": 25.5, "unit": "C"},
  "qos": 1,
  "retain": false
}
```

### Subscriptions
```
POST /subscribe
```
Subscribe to a topic.

**Request Body:**
```json
{
  "broker_id": "broker-id",
  "topic": "sensors/+",
  "qos": 1
}
```

```
POST /unsubscribe/{subscription_id}
```
Unsubscribe from a topic.

```
GET /subscriptions
```
List active subscriptions.

Query Parameters:
- `broker_id`: Filter by broker ID

### Messages
```
GET /messages
```
Get message history.

Query Parameters:
- `topic`: Filter by topic pattern
- `limit`: Maximum number of messages

### Statistics
```
GET /stats
```
Get MQTT statistics.

### WebSocket
```
/ws/{broker_id}/{topic}
```
WebSocket endpoint for real-time message streaming.

## Connection States

- `disconnected` - Not connected to broker
- `connecting` - Connection in progress
- `connected` - Successfully connected
- `reconnecting` - Attempting to reconnect
- `error` - Connection error

## Configuration

See `config.yaml` for configuration options.

## Usage Example

```python
import httpx
import asyncio
import websockets
import json

# Add broker
response = httpx.post("http://localhost:8441/brokers", json={
    "name": "Local MQTT",
    "host": "localhost",
    "port": 1883,
    "client_id": "ufo-galaxy-client"
})
broker_id = response.json()["broker_id"]

# Connect
response = httpx.post(f"http://localhost:8441/brokers/{broker_id}/connect")
print(response.json())

# Subscribe
response = httpx.post("http://localhost:8441/subscribe", json={
    "broker_id": broker_id,
    "topic": "test/topic",
    "qos": 1
})
sub_id = response.json()["subscription_id"]

# Publish
response = httpx.post("http://localhost:8441/publish", json={
    "broker_id": broker_id,
    "topic": "test/topic",
    "payload": {"message": "Hello MQTT!"},
    "qos": 1
})

# Get messages
response = httpx.get("http://localhost:8441/messages?topic=test/topic")
print(response.json())

# WebSocket for real-time messages
async def listen():
    uri = f"ws://localhost:8441/ws/{broker_id}/test/topic"
    async with websockets.connect(uri) as websocket:
        while True:
            message = await websocket.recv()
            print(json.loads(message))

asyncio.run(listen())
```

## Port

- HTTP API: `8441

## Dependencies

- Node 00 - StateMachine (port 8000)
- Optional: Node 02 - TaskEngine (port 8002)

## Author

UFO Galaxy Team
