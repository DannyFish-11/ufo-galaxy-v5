"""
Node_58_ModelRouter - AI Model Routing Node
UFO Galaxy v5.0 Core Node System

This node provides intelligent AI model routing:
- Model selection based on task requirements
- Load balancing across model instances
- Model performance tracking
- Fallback and retry logic
- Cost optimization
"""

from fastapi import FastAPI, HTTPException, Header, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import uvicorn
import asyncio
import httpx
from datetime import datetime
from loguru import logger
import uuid
import random

# Configure logging
logger.add("model_router.log", rotation="10 MB", retention="7 days")

app = FastAPI(
    title="Node 58 - ModelRouter",
    description="AI Model Routing for UFO Galaxy v5.0",
    version="5.0.0"
)


class ModelCapability(str, Enum):
    """Model capability enumeration."""
    TEXT_GENERATION = "text_generation"
    CHAT_COMPLETION = "chat_completion"
    EMBEDDINGS = "embeddings"
    IMAGE_GENERATION = "image_generation"
    CODE_GENERATION = "code_generation"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    MULTIMODAL = "multimodal"


class ModelStatus(str, Enum):
    """Model status enumeration."""
    AVAILABLE = "available"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"


class RoutingStrategy(str, Enum):
    """Routing strategy enumeration."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_LATENCY = "least_latency"
    LEAST_COST = "least_cost"
    CAPABILITY_MATCH = "capability_match"
    PRIORITY = "priority"


class ModelInstance(BaseModel):
    """Model instance configuration."""
    instance_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    provider: str
    model_id: str
    endpoint_url: str
    api_key: Optional[str] = None
    capabilities: List[ModelCapability] = Field(default_factory=list)
    status: ModelStatus = ModelStatus.AVAILABLE
    priority: int = 0
    cost_per_1k_tokens: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: int = 60
    weight: int = 1
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0


class RoutingRule(BaseModel):
    """Routing rule model."""
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    required_capabilities: List[ModelCapability] = Field(default_factory=list)
    preferred_providers: List[str] = Field(default_factory=list)
    excluded_models: List[str] = Field(default_factory=list)
    min_priority: int = 0
    max_cost_per_1k: Optional[float] = None
    timeout_seconds: int = 60
    fallback_enabled: bool = True
    retry_count: int = 2
    enabled: bool = True


class RouteRequest(BaseModel):
    """Route request model."""
    task_type: ModelCapability
    prompt: str
    required_capabilities: List[ModelCapability] = Field(default_factory=list)
    preferred_model: Optional[str] = None
    max_cost: Optional[float] = None
    timeout_seconds: int = 60
    priority: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RouteResponse(BaseModel):
    """Route response model."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    instance_id: str
    model_name: str
    provider: str
    status: str
    response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    tokens_used: Optional[int] = None
    cost: Optional[float] = None
    routed_at: datetime = Field(default_factory=datetime.utcnow)


class ModelPerformance(BaseModel):
    """Model performance metrics."""
    instance_id: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_latency_ms: float = 0.0
    avg_tokens_per_request: float = 0.0
    total_cost: float = 0.0
    availability_percent: float = 100.0
    last_updated: datetime = Field(default_factory=datetime.utcnow)


# In-memory storage
_model_instances: Dict[str, ModelInstance] = {}
_routing_rules: Dict[str, RoutingRule] = {}
_performance_metrics: Dict[str, ModelPerformance] = {}
_round_robin_index: int = 0
_lock = asyncio.Lock()
_request_history: List[Dict[str, Any]] = []


@app.on_event("startup")
async def startup_event():
    """Initialize the model router."""
    logger.info("ModelRouter starting up...")
    
    # Register default models
    await _register_default_models()
    
    logger.info("ModelRouter ready")


async def _register_default_models():
    """Register default model instances."""
    default_models = [
        ModelInstance(
            name="GPT-4",
            provider="openai",
            model_id="gpt-4",
            endpoint_url="http://localhost:8001/v1/chat/completions",
            capabilities=[
                ModelCapability.CHAT_COMPLETION,
                ModelCapability.TEXT_GENERATION,
                ModelCapability.FUNCTION_CALLING
            ],
            priority=10,
            cost_per_1k_tokens=0.03,
            max_tokens=8192
        ),
        ModelInstance(
            name="GPT-3.5-Turbo",
            provider="openai",
            model_id="gpt-3.5-turbo",
            endpoint_url="http://localhost:8001/v1/chat/completions",
            capabilities=[
                ModelCapability.CHAT_COMPLETION,
                ModelCapability.TEXT_GENERATION
            ],
            priority=5,
            cost_per_1k_tokens=0.0015,
            max_tokens=4096
        ),
        ModelInstance(
            name="Claude-3-Opus",
            provider="anthropic",
            model_id="claude-3-opus-20240229",
            endpoint_url="http://localhost:8001/v1/chat/completions",
            capabilities=[
                ModelCapability.CHAT_COMPLETION,
                ModelCapability.TEXT_GENERATION,
                ModelCapability.VISION,
                ModelCapability.MULTIMODAL
            ],
            priority=9,
            cost_per_1k_tokens=0.015,
            max_tokens=200000
        ),
        ModelInstance(
            name="Text-Embedding-3",
            provider="openai",
            model_id="text-embedding-3-small",
            endpoint_url="http://localhost:8001/v1/embeddings",
            capabilities=[ModelCapability.EMBEDDINGS],
            priority=5,
            cost_per_1k_tokens=0.00002,
            max_tokens=8191
        ),
    ]
    
    for model in default_models:
        async with _lock:
            _model_instances[model.instance_id] = model
            _performance_metrics[model.instance_id] = ModelPerformance(
                instance_id=model.instance_id
            )
        
        logger.info(f"Default model registered: {model.name}")


def _select_model(
    request: RouteRequest,
    strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_MATCH
) -> Optional[ModelInstance]:
    """Select a model instance based on routing strategy."""
    
    # Filter available models with required capabilities
    required_caps = set(request.required_capabilities or [request.task_type])
    
    available_models = [
        m for m in _model_instances.values()
        if m.status == ModelStatus.AVAILABLE
        and required_caps.issubset(set(m.capabilities))
        and m.priority >= request.priority
        and (request.max_cost is None or m.cost_per_1k_tokens <= request.max_cost)
    ]
    
    if not available_models:
        return None
    
    if strategy == RoutingStrategy.RANDOM:
        return random.choice(available_models)
    
    elif strategy == RoutingStrategy.LEAST_LATENCY:
        return min(available_models, key=lambda m: m.avg_latency_ms or float('inf'))
    
    elif strategy == RoutingStrategy.LEAST_COST:
        return min(available_models, key=lambda m: m.cost_per_1k_tokens)
    
    elif strategy == RoutingStrategy.PRIORITY:
        return max(available_models, key=lambda m: m.priority)
    
    elif strategy == RoutingStrategy.ROUND_ROBIN:
        global _round_robin_index
        model = available_models[_round_robin_index % len(available_models)]
        _round_robin_index = (_round_robin_index + 1) % len(available_models)
        return model
    
    else:  # CAPABILITY_MATCH
        # Select model with best capability match and priority
        def score_model(m: ModelInstance) -> float:
            cap_score = len(set(m.capabilities) & required_caps) / len(required_caps)
            priority_score = m.priority / 10.0
            latency_score = 1.0 / (1.0 + (m.avg_latency_ms or 1000) / 1000)
            return cap_score * 0.5 + priority_score * 0.3 + latency_score * 0.2
        
        return max(available_models, key=score_model)


async def _call_model(instance: ModelInstance, request: RouteRequest) -> Dict[str, Any]:
    """Call a model instance."""
    start_time = datetime.utcnow()
    
    headers = {"Content-Type": "application/json"}
    if instance.api_key:
        headers["Authorization"] = f"Bearer {instance.api_key}"
    
    payload = {
        "model": instance.model_id,
        "messages": [{"role": "user", "content": request.prompt}],
        "max_tokens": min(instance.max_tokens, 4096)
    }
    
    try:
        async with httpx.AsyncClient(timeout=instance.timeout_seconds) as client:
            response = await client.post(
                instance.endpoint_url,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                "success": True,
                "response": response.json(),
                "latency_ms": latency_ms
            }
    
    except Exception as e:
        latency_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        return {
            "success": False,
            "error": str(e),
            "latency_ms": latency_ms
        }


def _update_metrics(instance_id: str, success: bool, latency_ms: float, tokens: int = 0):
    """Update performance metrics for a model instance."""
    if instance_id not in _performance_metrics:
        return
    
    metrics = _performance_metrics[instance_id]
    metrics.total_requests += 1
    
    if success:
        metrics.successful_requests += 1
    else:
        metrics.failed_requests += 1
    
    # Update average latency
    metrics.avg_latency_ms = (
        (metrics.avg_latency_ms * (metrics.total_requests - 1) + latency_ms)
        / metrics.total_requests
    )
    
    # Update average tokens
    if tokens > 0:
        metrics.avg_tokens_per_request = (
            (metrics.avg_tokens_per_request * (metrics.total_requests - 1) + tokens)
            / metrics.total_requests
        )
    
    # Update availability
    metrics.availability_percent = (
        metrics.successful_requests / metrics.total_requests * 100
    )
    
    metrics.last_updated = datetime.utcnow()


@app.get("/health")
async def health():
    """Health check endpoint."""
    available = sum(1 for m in _model_instances.values() if m.status == ModelStatus.AVAILABLE)
    return {
        "status": "healthy",
        "node": "58",
        "name": "ModelRouter",
        "models": len(_model_instances),
        "available": available,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/models")
async def register_model(instance: ModelInstance) -> Dict[str, Any]:
    """
    Register a new model instance.
    
    Args:
        instance: Model instance configuration
        
    Returns:
        Registration result
    """
    async with _lock:
        _model_instances[instance.instance_id] = instance
        _performance_metrics[instance.instance_id] = ModelPerformance(
            instance_id=instance.instance_id
        )
    
    logger.info(f"Model registered: {instance.name} ({instance.instance_id})")
    
    return {
        "success": True,
        "instance_id": instance.instance_id,
        "name": instance.name,
        "provider": instance.provider,
        "capabilities": [c.value for c in instance.capabilities],
        "registered_at": instance.created_at.isoformat()
    }


@app.get("/models")
async def list_models(
    capability: Optional[ModelCapability] = None,
    provider: Optional[str] = None,
    status: Optional[ModelStatus] = None
) -> Dict[str, Any]:
    """
    List registered model instances.
    
    Args:
        capability: Filter by capability
        provider: Filter by provider
        status: Filter by status
        
    Returns:
        List of model instances
    """
    async with _lock:
        models = list(_model_instances.values())
        
        if capability:
            models = [m for m in models if capability in m.capabilities]
        if provider:
            models = [m for m in models if m.provider == provider]
        if status:
            models = [m for m in models if m.status == status]
        
        return {
            "models": [
                {
                    "instance_id": m.instance_id,
                    "name": m.name,
                    "provider": m.provider,
                    "model_id": m.model_id,
                    "capabilities": [c.value for c in m.capabilities],
                    "status": m.status.value,
                    "priority": m.priority,
                    "cost_per_1k_tokens": m.cost_per_1k_tokens,
                    "max_tokens": m.max_tokens,
                    "avg_latency_ms": m.avg_latency_ms,
                    "total_requests": m.total_requests
                }
                for m in models
            ],
            "total": len(models)
        }


@app.get("/models/{instance_id}")
async def get_model(instance_id: str) -> Dict[str, Any]:
    """
    Get model instance details.
    
    Args:
        instance_id: Model instance ID
        
    Returns:
        Model instance details
    """
    async with _lock:
        if instance_id not in _model_instances:
            raise HTTPException(status_code=404, detail=f"Model instance {instance_id} not found")
        
        model = _model_instances[instance_id]
        metrics = _performance_metrics.get(instance_id)
        
        return {
            "instance_id": model.instance_id,
            "name": model.name,
            "provider": model.provider,
            "model_id": model.model_id,
            "capabilities": [c.value for c in model.capabilities],
            "status": model.status.value,
            "priority": model.priority,
            "cost_per_1k_tokens": model.cost_per_1k_tokens,
            "max_tokens": model.max_tokens,
            "performance": metrics.dict() if metrics else None
        }


@app.post("/models/{instance_id}/status")
async def update_model_status(
    instance_id: str,
    status: ModelStatus
) -> Dict[str, Any]:
    """
    Update model instance status.
    
    Args:
        instance_id: Model instance ID
        status: New status
        
    Returns:
        Update result
    """
    async with _lock:
        if instance_id not in _model_instances:
            raise HTTPException(status_code=404, detail=f"Model instance {instance_id} not found")
        
        _model_instances[instance_id].status = status
    
    logger.info(f"Model {instance_id} status updated to {status.value}")
    
    return {
        "success": True,
        "instance_id": instance_id,
        "status": status.value,
        "updated_at": datetime.utcnow().isoformat()
    }


@app.delete("/models/{instance_id}")
async def unregister_model(instance_id: str) -> Dict[str, Any]:
    """
    Unregister a model instance.
    
    Args:
        instance_id: Model instance ID
        
    Returns:
        Unregistration result
    """
    async with _lock:
        if instance_id not in _model_instances:
            raise HTTPException(status_code=404, detail=f"Model instance {instance_id} not found")
        
        model = _model_instances.pop(instance_id)
        _performance_metrics.pop(instance_id, None)
    
    logger.info(f"Model unregistered: {model.name}")
    
    return {
        "success": True,
        "instance_id": instance_id,
        "name": model.name,
        "unregistered_at": datetime.utcnow().isoformat()
    }


@app.post("/route")
async def route_request(
    request: RouteRequest,
    strategy: RoutingStrategy = RoutingStrategy.CAPABILITY_MATCH
) -> RouteResponse:
    """
    Route a request to the best available model.
    
    Args:
        request: Route request
        strategy: Routing strategy
        
    Returns:
        Route response
    """
    # Select model
    instance = _select_model(request, strategy)
    
    if not instance:
        return RouteResponse(
            instance_id="",
            model_name="",
            provider="",
            status="error",
            error="No suitable model found for the request"
        )
    
    # Call model
    result = await _call_model(instance, request)
    
    # Update instance metrics
    async with _lock:
        instance.last_used = datetime.utcnow()
        instance.total_requests += 1
        if result["success"]:
            instance.successful_requests += 1
        else:
            instance.failed_requests += 1
        
        # Update latency
        if instance.avg_latency_ms == 0:
            instance.avg_latency_ms = result["latency_ms"]
        else:
            instance.avg_latency_ms = (
                instance.avg_latency_ms * 0.9 + result["latency_ms"] * 0.1
            )
    
    # Update performance metrics
    _update_metrics(
        instance.instance_id,
        result["success"],
        result["latency_ms"]
    )
    
    # Record request
    _request_history.append({
        "request_id": str(uuid.uuid4()),
        "instance_id": instance.instance_id,
        "task_type": request.task_type.value,
        "success": result["success"],
        "latency_ms": result["latency_ms"],
        "timestamp": datetime.utcnow().isoformat()
    })
    
    return RouteResponse(
        instance_id=instance.instance_id,
        model_name=instance.name,
        provider=instance.provider,
        status="success" if result["success"] else "error",
        response=result.get("response"),
        error=result.get("error"),
        latency_ms=result["latency_ms"]
    )


@app.post("/rules")
async def create_routing_rule(rule: RoutingRule) -> Dict[str, Any]:
    """
    Create a routing rule.
    
    Args:
        rule: Routing rule configuration
        
    Returns:
        Creation result
    """
    async with _lock:
        _routing_rules[rule.rule_id] = rule
    
    logger.info(f"Routing rule created: {rule.name}")
    
    return {
        "success": True,
        "rule_id": rule.rule_id,
        "name": rule.name,
        "created_at": datetime.utcnow().isoformat()
    }


@app.get("/rules")
async def list_routing_rules() -> Dict[str, Any]:
    """List all routing rules."""
    async with _lock:
        return {
            "rules": [
                {
                    "rule_id": r.rule_id,
                    "name": r.name,
                    "required_capabilities": [c.value for c in r.required_capabilities],
                    "preferred_providers": r.preferred_providers,
                    "fallback_enabled": r.fallback_enabled,
                    "retry_count": r.retry_count,
                    "enabled": r.enabled
                }
                for r in _routing_rules.values()
            ],
            "total": len(_routing_rules)
        }


@app.get("/performance")
async def get_performance_metrics(instance_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get performance metrics.
    
    Args:
        instance_id: Filter by instance ID
        
    Returns:
        Performance metrics
    """
    async with _lock:
        if instance_id:
            if instance_id not in _performance_metrics:
                raise HTTPException(status_code=404, detail=f"Instance {instance_id} not found")
            return {"metrics": _performance_metrics[instance_id].dict()}
        
        return {
            "metrics": [
                m.dict()
                for m in _performance_metrics.values()
            ]
        }


@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Get routing statistics."""
    async with _lock:
        total_requests = len(_request_history)
        successful = sum(1 for r in _request_history if r.get("success"))
        
        avg_latency = 0.0
        if _request_history:
            avg_latency = sum(r.get("latency_ms", 0) for r in _request_history) / len(_request_history)
        
        return {
            "total_models": len(_model_instances),
            "available_models": sum(1 for m in _model_instances.values() if m.status == ModelStatus.AVAILABLE),
            "total_requests": total_requests,
            "successful_requests": successful,
            "failed_requests": total_requests - successful,
            "success_rate": (successful / total_requests * 100) if total_requests > 0 else 0,
            "avg_latency_ms": avg_latency,
            "routing_rules": len(_routing_rules)
        }


@app.post("/benchmark")
async def run_benchmark(
    instance_id: str,
    iterations: int = 10,
    prompt: str = "Hello, this is a benchmark test."
) -> Dict[str, Any]:
    """
    Run benchmark on a model instance.
    
    Args:
        instance_id: Model instance ID
        iterations: Number of iterations
        prompt: Test prompt
        
    Returns:
        Benchmark results
    """
    async with _lock:
        if instance_id not in _model_instances:
            raise HTTPException(status_code=404, detail=f"Model instance {instance_id} not found")
        
        instance = _model_instances[instance_id]
    
    latencies = []
    successes = 0
    
    request = RouteRequest(
        task_type=ModelCapability.CHAT_COMPLETION,
        prompt=prompt
    )
    
    for _ in range(iterations):
        result = await _call_model(instance, request)
        latencies.append(result["latency_ms"])
        if result["success"]:
            successes += 1
    
    return {
        "instance_id": instance_id,
        "iterations": iterations,
        "successes": successes,
        "failures": iterations - successes,
        "avg_latency_ms": sum(latencies) / len(latencies),
        "min_latency_ms": min(latencies),
        "max_latency_ms": max(latencies),
        "benchmarked_at": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8558)
