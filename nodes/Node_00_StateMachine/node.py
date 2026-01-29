"""
Node_00_StateMachine - Central State Management Node
UFO Galaxy v5.0 Core Node System

This node manages the global state of the UFO Galaxy system, including:
- State transitions and state machine logic
- Global configuration management
- Event propagation
- Node registry and discovery
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import uvicorn
import asyncio
from datetime import datetime
from loguru import logger
import yaml
import json

# Configure logging
logger.add("state_machine.log", rotation="10 MB", retention="7 days")

app = FastAPI(
    title="Node 00 - StateMachine",
    description="Central State Management for UFO Galaxy v5.0",
    version="5.0.0"
)


class SystemState(str, Enum):
    """System-wide state enumeration."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class NodeStatus(BaseModel):
    """Node status model."""
    node_id: str
    state: str
    last_heartbeat: datetime
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StateTransition(BaseModel):
    """State transition request model."""
    from_state: SystemState
    to_state: SystemState
    reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GlobalState(BaseModel):
    """Global state model."""
    current_state: SystemState = SystemState.INITIALIZING
    previous_state: Optional[SystemState] = None
    state_history: List[Dict[str, Any]] = Field(default_factory=list)
    nodes: Dict[str, NodeStatus] = Field(default_factory=dict)
    config: Dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Global state instance
_global_state = GlobalState()
_state_lock = asyncio.Lock()


@app.on_event("startup")
async def startup_event():
    """Initialize the state machine on startup."""
    logger.info("StateMachine node starting up...")
    async with _state_lock:
        _global_state.current_state = SystemState.READY
        _global_state.timestamp = datetime.utcnow()
    logger.info("StateMachine node ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("StateMachine node shutting down...")
    async with _state_lock:
        _global_state.current_state = SystemState.SHUTDOWN
    logger.info("StateMachine node shutdown complete")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "node": "00",
        "name": "StateMachine",
        "state": _global_state.current_state.value,
        "timestamp": datetime.utcnow().isoformat(),
        "registered_nodes": len(_global_state.nodes)
    }


@app.get("/state")
async def get_state() -> GlobalState:
    """Get the current global state."""
    async with _state_lock:
        return _global_state


@app.post("/state/transition")
async def transition_state(transition: StateTransition) -> Dict[str, Any]:
    """
    Transition the system to a new state.
    
    Args:
        transition: State transition request
        
    Returns:
        Result of the state transition
    """
    async with _state_lock:
        if _global_state.current_state != transition.from_state:
            raise HTTPException(
                status_code=400,
                detail=f"Current state {_global_state.current_state} does not match from_state {transition.from_state}"
            )
        
        # Record transition
        transition_record = {
            "from": transition.from_state.value,
            "to": transition.to_state.value,
            "reason": transition.reason,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": transition.metadata
        }
        
        _global_state.previous_state = _global_state.current_state
        _global_state.current_state = transition.to_state
        _global_state.state_history.append(transition_record)
        _global_state.timestamp = datetime.utcnow()
        
        logger.info(f"State transition: {transition.from_state} -> {transition.to_state}")
        
        return {
            "success": True,
            "transition": transition_record,
            "current_state": _global_state.current_state.value
        }


@app.post("/nodes/register")
async def register_node(node_status: NodeStatus) -> Dict[str, Any]:
    """
    Register a node with the state machine.
    
    Args:
        node_status: Node status information
        
    Returns:
        Registration result
    """
    async with _state_lock:
        _global_state.nodes[node_status.node_id] = node_status
        logger.info(f"Node registered: {node_status.node_id}")
        
        return {
            "success": True,
            "node_id": node_status.node_id,
            "registered_at": datetime.utcnow().isoformat(),
            "total_nodes": len(_global_state.nodes)
        }


@app.post("/nodes/unregister/{node_id}")
async def unregister_node(node_id: str) -> Dict[str, Any]:
    """
    Unregister a node from the state machine.
    
    Args:
        node_id: ID of the node to unregister
        
    Returns:
        Unregistration result
    """
    async with _state_lock:
        if node_id not in _global_state.nodes:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        
        del _global_state.nodes[node_id]
        logger.info(f"Node unregistered: {node_id}")
        
        return {
            "success": True,
            "node_id": node_id,
            "unregistered_at": datetime.utcnow().isoformat(),
            "total_nodes": len(_global_state.nodes)
        }


@app.get("/nodes")
async def list_nodes() -> Dict[str, Any]:
    """List all registered nodes."""
    async with _state_lock:
        return {
            "nodes": {
                node_id: node.dict()
                for node_id, node in _global_state.nodes.items()
            },
            "total": len(_global_state.nodes)
        }


@app.post("/nodes/heartbeat/{node_id}")
async def node_heartbeat(node_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Update node heartbeat.
    
    Args:
        node_id: ID of the node
        metadata: Optional metadata to update
        
    Returns:
        Heartbeat acknowledgment
    """
    async with _state_lock:
        if node_id not in _global_state.nodes:
            raise HTTPException(status_code=404, detail=f"Node {node_id} not found")
        
        _global_state.nodes[node_id].last_heartbeat = datetime.utcnow()
        if metadata:
            _global_state.nodes[node_id].metadata.update(metadata)
        
        return {
            "success": True,
            "node_id": node_id,
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/state/history")
async def get_state_history(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get state transition history.
    
    Args:
        limit: Maximum number of history entries to return
        
    Returns:
        List of state transitions
    """
    async with _state_lock:
        return _global_state.state_history[-limit:]


@app.post("/config/update")
async def update_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update global configuration.
    
    Args:
        config: Configuration dictionary to merge
        
    Returns:
        Updated configuration
    """
    async with _state_lock:
        _global_state.config.update(config)
        _global_state.timestamp = datetime.utcnow()
        
        logger.info("Global configuration updated")
        
        return {
            "success": True,
            "config": _global_state.config,
            "updated_at": datetime.utcnow().isoformat()
        }


@app.get("/config")
async def get_config() -> Dict[str, Any]:
    """Get current global configuration."""
    async with _state_lock:
        return {
            "config": _global_state.config,
            "timestamp": _global_state.timestamp.isoformat()
        }


@app.post("/broadcast")
async def broadcast_event(
    event_type: str,
    payload: Dict[str, Any],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Broadcast an event to all registered nodes.
    
    Args:
        event_type: Type of event
        payload: Event payload
        background_tasks: Background task manager
        
    Returns:
        Broadcast result
    """
    event = {
        "type": event_type,
        "payload": payload,
        "timestamp": datetime.utcnow().isoformat(),
        "source": "state_machine"
    }
    
    # In a real implementation, this would broadcast to all nodes
    # For now, we just log the event
    logger.info(f"Broadcasting event: {event_type}")
    
    return {
        "success": True,
        "event": event,
        "target_nodes": len(_global_state.nodes)
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
