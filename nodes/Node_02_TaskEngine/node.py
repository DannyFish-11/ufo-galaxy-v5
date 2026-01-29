"""
Node_02_TaskEngine - Task Execution Engine Node
UFO Galaxy v5.0 Core Node System

This node provides task execution and management:
- Task queue management
- Task scheduling and prioritization
- Parallel task execution
- Task status tracking and reporting
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal, Callable
from enum import Enum
import uvicorn
import asyncio
from datetime import datetime
from loguru import logger
import uuid
import json

# Configure logging
logger.add("task_engine.log", rotation="10 MB", retention="7 days")

app = FastAPI(
    title="Node 02 - TaskEngine",
    description="Task Execution Engine for UFO Galaxy v5.0",
    version="5.0.0"
)


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(int, Enum):
    """Task priority levels."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskType(str, Enum):
    """Task type enumeration."""
    AI_INFERENCE = "ai_inference"
    DATA_PROCESSING = "data_processing"
    FILE_OPERATION = "file_operation"
    NETWORK_REQUEST = "network_request"
    SYSTEM_COMMAND = "system_command"
    CUSTOM = "custom"


class TaskConfig(BaseModel):
    """Task configuration model."""
    timeout_seconds: int = 300
    max_retries: int = 3
    retry_delay_seconds: int = 5
    parallel_execution: bool = False
    dependencies: List[str] = Field(default_factory=list)


class Task(BaseModel):
    """Task model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: TaskType
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    config: TaskConfig = Field(default_factory=TaskConfig)
    payload: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    worker_id: Optional[str] = None


class TaskQueueStats(BaseModel):
    """Task queue statistics model."""
    total_tasks: int = 0
    pending: int = 0
    queued: int = 0
    running: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    avg_execution_time_ms: float = 0.0


class TaskBatchRequest(BaseModel):
    """Task batch request model."""
    tasks: List[Task]
    parallel: bool = False
    max_concurrency: int = 5


class WorkerInfo(BaseModel):
    """Worker information model."""
    worker_id: str
    status: str
    current_task: Optional[str] = None
    completed_tasks: int = 0
    failed_tasks: int = 0
    started_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)


# In-memory storage
_tasks: Dict[str, Task] = {}
_task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
_workers: Dict[str, WorkerInfo] = {}
_lock = asyncio.Lock()
_worker_semaphore = asyncio.Semaphore(10)  # Max concurrent workers


@app.on_event("startup")
async def startup_event():
    """Initialize the task engine."""
    logger.info("TaskEngine starting up...")
    
    # Start worker tasks
    for i in range(5):
        worker_id = f"worker_{i}"
        _workers[worker_id] = WorkerInfo(worker_id=worker_id, status="idle")
        asyncio.create_task(_worker_loop(worker_id))
    
    logger.info("TaskEngine ready with 5 workers")


async def _worker_loop(worker_id: str):
    """Worker loop for processing tasks."""
    while True:
        try:
            async with _worker_semaphore:
                # Get task from queue
                priority, task_id = await _task_queue.get()
                
                async with _lock:
                    if task_id not in _tasks:
                        continue
                    task = _tasks[task_id]
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.utcnow()
                    task.worker_id = worker_id
                    
                    _workers[worker_id].status = "busy"
                    _workers[worker_id].current_task = task_id
                    _workers[worker_id].last_activity = datetime.utcnow()
                
                logger.info(f"Worker {worker_id} processing task {task_id}")
                
                # Execute task
                try:
                    result = await _execute_task(task)
                    
                    async with _lock:
                        task.status = TaskStatus.COMPLETED
                        task.result = result
                        task.completed_at = datetime.utcnow()
                        _workers[worker_id].completed_tasks += 1
                        
                except Exception as e:
                    logger.error(f"Task {task_id} failed: {str(e)}")
                    async with _lock:
                        task.retry_count += 1
                        if task.retry_count < task.config.max_retries:
                            task.status = TaskStatus.QUEUED
                            await _task_queue.put((task.priority.value, task_id))
                        else:
                            task.status = TaskStatus.FAILED
                            task.error = str(e)
                            task.completed_at = datetime.utcnow()
                            _workers[worker_id].failed_tasks += 1
                
                finally:
                    async with _lock:
                        _workers[worker_id].status = "idle"
                        _workers[worker_id].current_task = None
                        _workers[worker_id].last_activity = datetime.utcnow()
                        
        except Exception as e:
            logger.error(f"Worker {worker_id} error: {str(e)}")
            await asyncio.sleep(1)


async def _execute_task(task: Task) -> Dict[str, Any]:
    """Execute a task based on its type."""
    if task.type == TaskType.AI_INFERENCE:
        return await _execute_ai_inference(task)
    elif task.type == TaskType.DATA_PROCESSING:
        return await _execute_data_processing(task)
    elif task.type == TaskType.FILE_OPERATION:
        return await _execute_file_operation(task)
    elif task.type == TaskType.NETWORK_REQUEST:
        return await _execute_network_request(task)
    elif task.type == TaskType.SYSTEM_COMMAND:
        return await _execute_system_command(task)
    else:
        return await _execute_custom_task(task)


async def _execute_ai_inference(task: Task) -> Dict[str, Any]:
    """Execute AI inference task."""
    # Simulate AI inference
    await asyncio.sleep(0.5)
    return {
        "inference_result": "success",
        "model": task.payload.get("model", "default"),
        "output": f"Processed with {task.payload.get('prompt', 'no prompt')}"
    }


async def _execute_data_processing(task: Task) -> Dict[str, Any]:
    """Execute data processing task."""
    # Simulate data processing
    await asyncio.sleep(0.3)
    return {
        "processed_items": task.payload.get("items", 0),
        "output_format": task.payload.get("format", "json")
    }


async def _execute_file_operation(task: Task) -> Dict[str, Any]:
    """Execute file operation task."""
    await asyncio.sleep(0.1)
    return {
        "operation": task.payload.get("operation", "read"),
        "path": task.payload.get("path", ""),
        "success": True
    }


async def _execute_network_request(task: Task) -> Dict[str, Any]:
    """Execute network request task."""
    import httpx
    
    url = task.payload.get("url")
    method = task.payload.get("method", "GET")
    headers = task.payload.get("headers", {})
    data = task.payload.get("data")
    
    async with httpx.AsyncClient(timeout=task.config.timeout_seconds) as client:
        if method.upper() == "GET":
            response = await client.get(url, headers=headers)
        elif method.upper() == "POST":
            response = await client.post(url, headers=headers, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        return {
            "status_code": response.status_code,
            "headers": dict(response.headers),
            "body": response.text[:1000]  # Limit response size
        }


async def _execute_system_command(task: Task) -> Dict[str, Any]:
    """Execute system command task."""
    import subprocess
    
    command = task.payload.get("command")
    if not command:
        raise ValueError("No command specified")
    
    result = subprocess.run(
        command,
        shell=True,
        capture_output=True,
        text=True,
        timeout=task.config.timeout_seconds
    )
    
    return {
        "returncode": result.returncode,
        "stdout": result.stdout[:1000],
        "stderr": result.stderr[:1000]
    }


async def _execute_custom_task(task: Task) -> Dict[str, Any]:
    """Execute custom task."""
    # Custom task execution logic
    await asyncio.sleep(0.2)
    return {
        "custom_result": "success",
        "payload_received": task.payload
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "node": "02",
        "name": "TaskEngine",
        "workers": len(_workers),
        "queue_size": _task_queue.qsize(),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/tasks")
async def create_task(task: Task) -> Dict[str, Any]:
    """
    Create a new task.
    
    Args:
        task: Task to create
        
    Returns:
        Created task information
    """
    async with _lock:
        _tasks[task.id] = task
        task.status = TaskStatus.QUEUED
        await _task_queue.put((task.priority.value, task.id))
    
    logger.info(f"Task created: {task.id} ({task.name})")
    
    return {
        "success": True,
        "task_id": task.id,
        "status": task.status.value,
        "created_at": task.created_at.isoformat()
    }


@app.post("/tasks/batch")
async def create_batch(request: TaskBatchRequest) -> Dict[str, Any]:
    """
    Create multiple tasks in batch.
    
    Args:
        request: Batch request with tasks
        
    Returns:
        Batch creation result
    """
    created_tasks = []
    
    async with _lock:
        for task in request.tasks:
            _tasks[task.id] = task
            task.status = TaskStatus.QUEUED
            await _task_queue.put((task.priority.value, task.id))
            created_tasks.append(task.id)
    
    logger.info(f"Batch created: {len(created_tasks)} tasks")
    
    return {
        "success": True,
        "task_ids": created_tasks,
        "count": len(created_tasks)
    }


@app.get("/tasks/{task_id}")
async def get_task(task_id: str) -> Task:
    """
    Get task by ID.
    
    Args:
        task_id: Task ID
        
    Returns:
        Task information
    """
    async with _lock:
        if task_id not in _tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return _tasks[task_id]


@app.get("/tasks")
async def list_tasks(
    status: Optional[TaskStatus] = None,
    task_type: Optional[TaskType] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    List tasks with optional filtering.
    
    Args:
        status: Filter by status
        task_type: Filter by type
        limit: Maximum number of tasks to return
        
    Returns:
        List of tasks
    """
    async with _lock:
        tasks = list(_tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        if task_type:
            tasks = [t for t in tasks if t.type == task_type]
        
        tasks = sorted(tasks, key=lambda t: t.created_at, reverse=True)[:limit]
        
        return {
            "tasks": [t.dict() for t in tasks],
            "total": len(tasks),
            "filters": {
                "status": status.value if status else None,
                "type": task_type.value if task_type else None
            }
        }


@app.post("/tasks/{task_id}/cancel")
async def cancel_task(task_id: str) -> Dict[str, Any]:
    """
    Cancel a task.
    
    Args:
        task_id: Task ID to cancel
        
    Returns:
        Cancellation result
    """
    async with _lock:
        if task_id not in _tasks:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
        task = _tasks[task_id]
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            raise HTTPException(status_code=400, detail=f"Cannot cancel {task.status} task")
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()
    
    logger.info(f"Task cancelled: {task_id}")
    
    return {
        "success": True,
        "task_id": task_id,
        "status": TaskStatus.CANCELLED.value
    }


@app.get("/stats")
async def get_stats() -> TaskQueueStats:
    """Get task queue statistics."""
    async with _lock:
        stats = TaskQueueStats()
        stats.total_tasks = len(_tasks)
        
        for task in _tasks.values():
            if task.status == TaskStatus.PENDING:
                stats.pending += 1
            elif task.status == TaskStatus.QUEUED:
                stats.queued += 1
            elif task.status == TaskStatus.RUNNING:
                stats.running += 1
            elif task.status == TaskStatus.COMPLETED:
                stats.completed += 1
            elif task.status == TaskStatus.FAILED:
                stats.failed += 1
            elif task.status == TaskStatus.CANCELLED:
                stats.cancelled += 1
        
        # Calculate average execution time
        completed_tasks = [t for t in _tasks.values() if t.status == TaskStatus.COMPLETED and t.completed_at and t.started_at]
        if completed_tasks:
            total_time = sum(
                (t.completed_at - t.started_at).total_seconds() * 1000
                for t in completed_tasks
            )
            stats.avg_execution_time_ms = total_time / len(completed_tasks)
        
        return stats


@app.get("/workers")
async def list_workers() -> Dict[str, Any]:
    """List all workers and their status."""
    async with _lock:
        return {
            "workers": [w.dict() for w in _workers.values()],
            "total": len(_workers),
            "busy": sum(1 for w in _workers.values() if w.status == "busy")
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
