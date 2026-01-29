# Node 58 - ModelRouter

AI Model Routing Node for UFO Galaxy v5.0

## Overview

The ModelRouter node provides intelligent AI model routing:
- Model selection based on task requirements
- Load balancing across model instances
- Model performance tracking
- Fallback and retry logic
- Cost optimization

## Model Capabilities

- `text_generation` - General text generation
- `chat_completion` - Chat/conversation models
- `embeddings` - Text embedding models
- `image_generation` - Image generation models
- `code_generation` - Code-specific models
- `function_calling` - Function calling models
- `vision` - Vision-capable models
- `multimodal` - Multimodal models

## Routing Strategies

- `round_robin` - Distribute requests evenly
- `random` - Random selection
- `least_latency` - Select fastest model
- `least_cost` - Select cheapest model
- `capability_match` - Best capability match (default)
- `priority` - Highest priority model

## API Endpoints

### Health Check
```
GET /health
```

### Model Management
```
POST /models
```
Register a new model instance.

**Request Body:**
```json
{
  "name": "GPT-4",
  "provider": "openai",
  "model_id": "gpt-4",
  "endpoint_url": "http://localhost:8001/v1/chat/completions",
  "capabilities": ["chat_completion", "text_generation"],
  "priority": 10,
  "cost_per_1k_tokens": 0.03,
  "max_tokens": 8192
}
```

```
GET /models
```
List registered model instances.

Query Parameters:
- `capability`: Filter by capability
- `provider`: Filter by provider
- `status`: Filter by status

```
GET /models/{instance_id}
```
Get model instance details.

```
POST /models/{instance_id}/status
```
Update model instance status.

Query Parameters:
- `status`: New status (`available`, `busy`, `unavailable`, `degraded`, `maintenance`)

```
DELETE /models/{instance_id}
```
Unregister a model instance.

### Routing
```
POST /route
```
Route a request to the best available model.

**Request Body:**
```json
{
  "task_type": "chat_completion",
  "prompt": "Hello, how are you?",
  "required_capabilities": ["chat_completion"],
  "preferred_model": "gpt-4",
  "max_cost": 0.05,
  "timeout_seconds": 60,
  "priority": 0
}
```

Query Parameters:
- `strategy`: Routing strategy (`capability_match`, `least_latency`, `least_cost`, etc.)

### Routing Rules
```
POST /rules
```
Create a routing rule.

**Request Body:**
```json
{
  "name": "Code Generation Rule",
  "required_capabilities": ["code_generation"],
  "preferred_providers": ["openai", "anthropic"],
  "fallback_enabled": true,
  "retry_count": 2
}
```

```
GET /rules
```
List all routing rules.

### Performance
```
GET /performance
```
Get performance metrics.

Query Parameters:
- `instance_id`: Filter by instance ID

### Statistics
```
GET /stats
```
Get routing statistics.

### Benchmark
```
POST /benchmark
```
Run benchmark on a model instance.

Query Parameters:
- `instance_id`: Model instance ID
- `iterations`: Number of iterations (default 10)
- `prompt`: Test prompt

## Model Status

- `available` - Model is available for requests
- `busy` - Model is currently busy
- `unavailable` - Model is unavailable
- `degraded` - Model is experiencing issues
- `maintenance` - Model is in maintenance mode

## Configuration

See `config.yaml` for configuration options.

## Usage Example

```python
import httpx

# Register a model
response = httpx.post("http://localhost:8558/models", json={
    "name": "GPT-4",
    "provider": "openai",
    "model_id": "gpt-4",
    "endpoint_url": "http://localhost:8001/v1/chat/completions",
    "capabilities": ["chat_completion", "text_generation"],
    "priority": 10,
    "cost_per_1k_tokens": 0.03
})
model_id = response.json()["instance_id"]

# Route a request
response = httpx.post("http://localhost:8558/route", json={
    "task_type": "chat_completion",
    "prompt": "What is the capital of France?",
    "required_capabilities": ["chat_completion"]
}, params={"strategy": "least_latency"})
print(response.json())

# Get performance metrics
response = httpx.get(f"http://localhost:8558/performance?instance_id={model_id}")
print(response.json())

# Run benchmark
response = httpx.post("http://localhost:8558/benchmark", params={
    "instance_id": model_id,
    "iterations": 5,
    "prompt": "Hello!"
})
print(response.json())

# Get statistics
response = httpx.get("http://localhost:8558/stats")
print(response.json())
```

## Port

- HTTP API: `8558`

## Dependencies

- Node 00 - StateMachine (port 8000)
- Node 01 - OneAPI (port 8001)
- Optional: Node 05 - Auth (port 8005)

## Author

UFO Galaxy Team
