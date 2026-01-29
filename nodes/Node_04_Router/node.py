"""
Node_04_Router - Request Routing Node
UFO Galaxy v5.0 Core Node System

This node provides intelligent request routing:
- Load balancing across nodes
- Service discovery
- Circuit breaker pattern
- Request forwarding and proxying
"""

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import uvicorn
import asyncio
import httpx
from datetime import datetime, timedelta
from loguru import logger
import uuid
import random

# Configure logging
logger.add("router.log", rotation="10 MB", retention="7 days")

app = FastAPI(
    title="Node 04 - Router",
    description="Request Routing for UFO Galaxy v5.0",
    version="5.0.0"
)


class ServiceStatus(str, Enum):
    """Service status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


class LoadBalanceStrategy(str, Enum):
    """Load balancing strategy."""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    IP_HASH = "ip_hash"


class ServiceEndpoint(BaseModel):
    """Service endpoint model."""
    endpoint_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    host: str
    port: int
    path: str = "/"
    protocol: Literal["http", "https"] = "http"
    weight: int = 1
    status: ServiceStatus = ServiceStatus.HEALTHY
    health_check_path: str = "/health"
    health_check_interval: int = 30
    max_failures: int = 3
    failure_count: int = 0
    last_health_check: Optional[datetime] = None
    response_time_ms: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class RouteRule(BaseModel):
    """Route rule model."""
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    path_prefix: str
    target_service: str
    methods: List[str] = Field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    priority: int = 0
    enabled: bool = True
    rewrite_path: Optional[str] = None
    add_headers: Dict[str, str] = Field(default_factory=dict)
    strip_prefix: bool = False


class CircuitBreakerState(str, Enum):
    """Circuit breaker state."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker(BaseModel):
    """Circuit breaker model."""
    service_id: str
    state: CircuitBreakerState = CircuitBreakerState.CLOSED
    failure_threshold: int = 5
    recovery_timeout: int = 30
    half_open_max_calls: int = 3
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    total_calls: int = 0
    total_failures: int = 0


class RoutingStats(BaseModel):
    """Routing statistics model."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    active_connections: int = 0


# In-memory storage
_services: Dict[str, ServiceEndpoint] = {}
_routes: Dict[str, RouteRule] = {}
_circuit_breakers: Dict[str, CircuitBreaker] = {}
_stats = RoutingStats()
_lock = asyncio.Lock()
_round_robin_index: Dict[str, int] = {}
_request_history: List[Dict[str, Any]] = []


@app.on_event("startup")
async def startup_event():
    """Initialize the router."""
    logger.info("Router starting up...")
    
    # Start health check task
    asyncio.create_task(_health_check_loop())
    
    logger.info("Router ready")


async def _health_check_loop():
    """Background health check loop."""
    while True:
        await asyncio.sleep(30)
        await _check_all_services()


async def _check_all_services():
    """Check health of all registered services."""
    async with _lock:
        for service in _services.values():
            try:
                url = f"{service.protocol}://{service.host}:{service.port}{service.health_check_path}"
                async with httpx.AsyncClient(timeout=10.0) as client:
                    start = datetime.utcnow()
                    response = await client.get(url)
                    elapsed = (datetime.utcnow() - start).total_seconds() * 1000
                    
                    service.response_time_ms = elapsed
                    service.last_health_check = datetime.utcnow()
                    
                    if response.status_code == 200:
                        if service.failure_count > 0:
                            service.failure_count = 0
                        service.status = ServiceStatus.HEALTHY
                    else:
                        service.failure_count += 1
                        if service.failure_count >= service.max_failures:
                            service.status = ServiceStatus.UNHEALTHY
                        else:
                            service.status = ServiceStatus.DEGRADED
                            
            except Exception as e:
                service.failure_count += 1
                service.last_health_check = datetime.utcnow()
                if service.failure_count >= service.max_failures:
                    service.status = ServiceStatus.OFFLINE
                else:
                    service.status = ServiceStatus.DEGRADED
                
                logger.warning(f"Health check failed for {service.name}: {str(e)}")


def _get_circuit_breaker(service_id: str) -> CircuitBreaker:
    """Get or create circuit breaker for a service."""
    if service_id not in _circuit_breakers:
        _circuit_breakers[service_id] = CircuitBreaker(service_id=service_id)
    return _circuit_breakers[service_id]


def _can_call(circuit: CircuitBreaker) -> bool:
    """Check if call is allowed through circuit breaker."""
    if circuit.state == CircuitBreakerState.CLOSED:
        return True
    elif circuit.state == CircuitBreakerState.OPEN:
        if circuit.last_failure_time:
            elapsed = (datetime.utcnow() - circuit.last_failure_time).total_seconds()
            if elapsed >= circuit.recovery_timeout:
                circuit.state = CircuitBreakerState.HALF_OPEN
                circuit.failure_count = 0
                circuit.success_count = 0
                return True
        return False
    elif circuit.state == CircuitBreakerState.HALF_OPEN:
        return circuit.success_count < circuit.half_open_max_calls
    return True


def _record_success(circuit: CircuitBreaker):
    """Record successful call."""
    circuit.total_calls += 1
    circuit.success_count += 1
    
    if circuit.state == CircuitBreakerState.HALF_OPEN:
        if circuit.success_count >= circuit.half_open_max_calls:
            circuit.state = CircuitBreakerState.CLOSED
            circuit.failure_count = 0
            logger.info(f"Circuit breaker for {circuit.service_id} closed")


def _record_failure(circuit: CircuitBreaker):
    """Record failed call."""
    circuit.total_calls += 1
    circuit.total_failures += 1
    circuit.failure_count += 1
    circuit.last_failure_time = datetime.utcnow()
    
    if circuit.state == CircuitBreakerState.HALF_OPEN:
        circuit.state = CircuitBreakerState.OPEN
        logger.warning(f"Circuit breaker for {circuit.service_id} opened")
    elif circuit.failure_count >= circuit.failure_threshold:
        circuit.state = CircuitBreakerState.OPEN
        logger.warning(f"Circuit breaker for {circuit.service_id} opened")


def _select_endpoint(service_name: str, strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN) -> Optional[ServiceEndpoint]:
    """Select an endpoint using the specified strategy."""
    healthy_services = [
        s for s in _services.values()
        if s.name == service_name and s.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
    ]
    
    if not healthy_services:
        return None
    
    if strategy == LoadBalanceStrategy.RANDOM:
        return random.choice(healthy_services)
    
    elif strategy == LoadBalanceStrategy.ROUND_ROBIN:
        if service_name not in _round_robin_index:
            _round_robin_index[service_name] = 0
        index = _round_robin_index[service_name] % len(healthy_services)
        _round_robin_index[service_name] = (index + 1) % len(healthy_services)
        return healthy_services[index]
    
    elif strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
        return min(healthy_services, key=lambda s: s.failure_count)
    
    elif strategy == LoadBalanceStrategy.WEIGHTED:
        total_weight = sum(s.weight for s in healthy_services)
        r = random.uniform(0, total_weight)
        cumulative = 0
        for service in healthy_services:
            cumulative += service.weight
            if r <= cumulative:
                return service
        return healthy_services[-1]
    
    return healthy_services[0]


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "node": "04",
        "name": "Router",
        "services": len(_services),
        "routes": len(_routes),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/services")
async def register_service(service: ServiceEndpoint) -> Dict[str, Any]:
    """
    Register a new service endpoint.
    
    Args:
        service: Service endpoint configuration
        
    Returns:
        Registration result
    """
    async with _lock:
        _services[service.endpoint_id] = service
        
        logger.info(f"Service registered: {service.name} at {service.host}:{service.port}")
        
        return {
            "success": True,
            "endpoint_id": service.endpoint_id,
            "name": service.name,
            "url": f"{service.protocol}://{service.host}:{service.port}",
            "registered_at": service.created_at.isoformat()
        }


@app.get("/services")
async def list_services(
    status: Optional[ServiceStatus] = None
) -> Dict[str, Any]:
    """
    List registered services.
    
    Args:
        status: Filter by status
        
    Returns:
        List of services
    """
    async with _lock:
        services = list(_services.values())
        
        if status:
            services = [s for s in services if s.status == status]
        
        return {
            "services": [
                {
                    "endpoint_id": s.endpoint_id,
                    "name": s.name,
                    "url": f"{s.protocol}://{s.host}:{s.port}{s.path}",
                    "status": s.status.value,
                    "weight": s.weight,
                    "response_time_ms": s.response_time_ms,
                    "last_health_check": s.last_health_check.isoformat() if s.last_health_check else None
                }
                for s in services
            ],
            "total": len(services)
        }


@app.delete("/services/{endpoint_id}")
async def unregister_service(endpoint_id: str) -> Dict[str, Any]:
    """
    Unregister a service endpoint.
    
    Args:
        endpoint_id: Service endpoint ID
        
    Returns:
        Unregistration result
    """
    async with _lock:
        if endpoint_id not in _services:
            raise HTTPException(status_code=404, detail=f"Service {endpoint_id} not found")
        
        service = _services.pop(endpoint_id)
        
        logger.info(f"Service unregistered: {service.name}")
        
        return {
            "success": True,
            "endpoint_id": endpoint_id,
            "name": service.name,
            "unregistered_at": datetime.utcnow().isoformat()
        }


@app.post("/routes")
async def create_route(rule: RouteRule) -> Dict[str, Any]:
    """
    Create a new routing rule.
    
    Args:
        rule: Route rule configuration
        
    Returns:
        Creation result
    """
    async with _lock:
        _routes[rule.rule_id] = rule
        
        logger.info(f"Route created: {rule.name} -> {rule.target_service}")
        
        return {
            "success": True,
            "rule_id": rule.rule_id,
            "name": rule.name,
            "path_prefix": rule.path_prefix,
            "target_service": rule.target_service,
            "created_at": datetime.utcnow().isoformat()
        }


@app.get("/routes")
async def list_routes() -> Dict[str, Any]:
    """List all routing rules."""
    async with _lock:
        return {
            "routes": [
                {
                    "rule_id": r.rule_id,
                    "name": r.name,
                    "path_prefix": r.path_prefix,
                    "target_service": r.target_service,
                    "methods": r.methods,
                    "priority": r.priority,
                    "enabled": r.enabled
                }
                for r in sorted(_routes.values(), key=lambda x: x.priority, reverse=True)
            ],
            "total": len(_routes)
        }


@app.delete("/routes/{rule_id}")
async def delete_route(rule_id: str) -> Dict[str, Any]:
    """
    Delete a routing rule.
    
    Args:
        rule_id: Rule ID to delete
        
    Returns:
        Deletion result
    """
    async with _lock:
        if rule_id not in _routes:
            raise HTTPException(status_code=404, detail=f"Route {rule_id} not found")
        
        rule = _routes.pop(rule_id)
        
        logger.info(f"Route deleted: {rule.name}")
        
        return {
            "success": True,
            "rule_id": rule_id,
            "deleted_at": datetime.utcnow().isoformat()
        }


@app.get("/circuit-breakers")
async def list_circuit_breakers() -> Dict[str, Any]:
    """List all circuit breakers."""
    return {
        "circuit_breakers": [
            {
                "service_id": cb.service_id,
                "state": cb.state.value,
                "failure_count": cb.failure_count,
                "success_count": cb.success_count,
                "total_calls": cb.total_calls,
                "total_failures": cb.total_failures
            }
            for cb in _circuit_breakers.values()
        ]
    }


@app.post("/circuit-breakers/{service_id}/reset")
async def reset_circuit_breaker(service_id: str) -> Dict[str, Any]:
    """
    Reset a circuit breaker.
    
    Args:
        service_id: Service ID
        
    Returns:
        Reset result
    """
    if service_id not in _circuit_breakers:
        raise HTTPException(status_code=404, detail=f"Circuit breaker for {service_id} not found")
    
    cb = _circuit_breakers[service_id]
    cb.state = CircuitBreakerState.CLOSED
    cb.failure_count = 0
    cb.success_count = 0
    
    logger.info(f"Circuit breaker reset for {service_id}")
    
    return {
        "success": True,
        "service_id": service_id,
        "new_state": cb.state.value,
        "reset_at": datetime.utcnow().isoformat()
    }


@app.api_route("/proxy/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy_request(
    service_name: str,
    path: str,
    request: Request
):
    """
    Proxy a request to a service.
    
    Args:
        service_name: Target service name
        path: Request path
        request: Incoming request
        
    Returns:
        Proxied response
    """
    async with _lock:
        _stats.total_requests += 1
    
    # Check circuit breaker
    circuit = _get_circuit_breaker(service_name)
    if not _can_call(circuit):
        raise HTTPException(status_code=503, detail=f"Circuit breaker open for {service_name}")
    
    # Select endpoint
    endpoint = _select_endpoint(service_name)
    if not endpoint:
        _record_failure(circuit)
        raise HTTPException(status_code=503, detail=f"No healthy endpoints for {service_name}")
    
    # Build target URL
    target_url = f"{endpoint.protocol}://{endpoint.host}:{endpoint.port}/{path}"
    if request.query_params:
        target_url += f"?{request.query_params}"
    
    try:
        # Forward request
        async with httpx.AsyncClient(timeout=60.0) as client:
            body = await request.body()
            
            headers = dict(request.headers)
            headers.pop("host", None)
            
            response = await client.request(
                method=request.method,
                url=target_url,
                headers=headers,
                content=body,
                follow_redirects=True
            )
            
            _record_success(circuit)
            
            async with _lock:
                _stats.successful_requests += 1
            
            return response.json() if response.headers.get("content-type", "").startswith("application/json") else {"content": response.text}
            
    except Exception as e:
        _record_failure(circuit)
        async with _lock:
            _stats.failed_requests += 1
        logger.error(f"Proxy error for {service_name}: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Proxy error: {str(e)}")


@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Get routing statistics."""
    return {
        "stats": {
            "total_requests": _stats.total_requests,
            "successful_requests": _stats.successful_requests,
            "failed_requests": _stats.failed_requests,
            "success_rate": (_stats.successful_requests / _stats.total_requests * 100) if _stats.total_requests > 0 else 0,
            "services": len(_services),
            "routes": len(_routes)
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
