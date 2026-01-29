# Node 02 - TaskEngine

Task Execution Engine Node for UFO Galaxy v5.0

## Overview

The TaskEngine node provides task execution and management:
- Task queue management
- Task scheduling and prioritization
- Parallel task execution
- Task status tracking and reporting

## Task Types

- `ai_inference` - AI model inference tasks
- `data_processing` - Data processing tasks
- `file_operation` - File operations
- `network_request` - HTTP/network requests
- `system_command` - System command execution
- `custom` - Custom task types

## API Endpoints

### Health Check
```
GET /health
```

### Tasks
```
POST /tasks
```
Create a new task.

**Request Body:**
```json
{
  "name": "My Task",
  "type": "ai_inference",
  "priority": 1,
  "config": {
    "timeout_seconds": 300,
    "max_retries": 3
  },
  "payload": {
    "model": "gpt-4",
    "prompt": "Hello!"
  }
}
```

```
POST /tasks/batch
```
Create multiple tasks in batch.

```
GET /tasks/{task_id}
```
Get task by ID.

```
GET /tasks
```
List tasks with optional filtering.

Query Parameters:
- `status`: Filter by status
- `task_type`: Filter by type
- `limit`: Maximum number of tasks

```
POST /tasks/{task_id}/cancel
```
Cancel a task.

### Workers
```
GET /workers
```
List all workers and their status.

### Statistics
```
GET /stats
```
Get task queue statistics.

## Task Priorities

- `0` - CRITICAL
- `1` - HIGH
- `2` - NORMAL
- `3` - LOW
- `4` - BACKGROUND

## Task Status

- `pending` - Task is pending
- `queued` - Task is in queue
- `running` - Task is running
- `completed` - Task completed successfully
- `failed` - Task failed
- `cancelled` - Task was cancelled
- `timeout` - Task timed out

## Configuration

See `config.yaml` for configuration options.

## Usage Example

```python
import httpx

# Create a task
response = httpx.post("http://localhost:8002/tasks", json={
    "name": "AI Inference",
    "type": "ai_inference",
    "priority": 1,
    "payload": {
        "model": "gpt-4",
        "prompt": "What is the capital of France?"
    }
})
task_id = response.json()["task_id"]

# Get task status
response = httpx.get(f"http://localhost:8002/tasks/{task_id}")
print(response.json())

# Get statistics
response = httpx.get("http://localhost:8002/stats")
print(response.json())
```

## Port

- HTTP API: `8002`

## Dependencies

- Node 00 - StateMachine (port 8000)
- Optional: Node 01 - OneAPI (port 8001)

## Author

UFO Galaxy Team
