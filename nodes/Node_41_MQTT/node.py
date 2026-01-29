"""
Node_41_MQTT - MQTT Broker Client Node
UFO Galaxy v5.0 Core Node System

This node provides MQTT client functionality:
- Connect to MQTT brokers
- Publish/subscribe to topics
- Message persistence
- QoS management
- Last will and testament
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
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
logger.add("mqtt.log", rotation="10 MB", retention="7 days")

# Try to import paho-mqtt
try:
    import paho.mqtt.client as mqtt
    PAHO_AVAILABLE = True
except ImportError:
    PAHO_AVAILABLE = False
    logger.warning("paho-mqtt not installed. Using mock implementation.")

app = FastAPI(
    title="Node 41 - MQTT",
    description="MQTT Broker Client for UFO Galaxy v5.0",
    version="5.0.0"
)


class QoSLevel(int, Enum):
    """MQTT QoS levels."""
    AT_MOST_ONCE = 0
    AT_LEAST_ONCE = 1
    EXACTLY_ONCE = 2


class ConnectionState(str, Enum):
    """Connection state."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"


class BrokerConfig(BaseModel):
    """MQTT broker configuration."""
    broker_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    host: str
    port: int = 1883
    use_ssl: bool = False
    username: Optional[str] = None
    password: Optional[str] = None
    client_id: Optional[str] = None
    keepalive: int = 60
    clean_session: bool = True
    reconnect_on_failure: bool = True
    max_reconnect_attempts: int = 10
    reconnect_delay: int = 5


class LastWill(BaseModel):
    """Last will and testament."""
    topic: str
    payload: str
    qos: QoSLevel = QoSLevel.AT_MOST_ONCE
    retain: bool = False


class MqttMessage(BaseModel):
    """MQTT message model."""
    topic: str
    payload: Any
    qos: QoSLevel = QoSLevel.AT_MOST_ONCE
    retain: bool = False
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class Subscription(BaseModel):
    """Subscription model."""
    subscription_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    broker_id: str
    topic: str
    qos: QoSLevel = QoSLevel.AT_MOST_ONCE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    message_count: int = 0
    is_active: bool = True


class BrokerStatus(BaseModel):
    """Broker connection status."""
    broker_id: str
    state: ConnectionState
    connected_at: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    messages_sent: int = 0
    messages_received: int = 0
    reconnect_attempts: int = 0
    error_message: Optional[str] = None


class PublishRequest(BaseModel):
    """Publish request model."""
    broker_id: str
    topic: str
    payload: Any
    qos: QoSLevel = QoSLevel.AT_MOST_ONCE
    retain: bool = False


class SubscribeRequest(BaseModel):
    """Subscribe request model."""
    broker_id: str
    topic: str
    qos: QoSLevel = QoSLevel.AT_MOST_ONCE


# In-memory storage
_brokers: Dict[str, BrokerConfig] = {}
_broker_status: Dict[str, BrokerStatus] = {}
_subscriptions: Dict[str, Subscription] = {}
_message_history: List[MqttMessage] = []
_clients: Dict[str, Any] = {}  # broker_id -> mqtt.Client
_message_handlers: Dict[str, List[Callable]] = {}  # topic -> handlers
_lock = asyncio.Lock()


class MockMQTTClient:
    """Mock MQTT client for when paho-mqtt is not available."""
    
    def __init__(self, client_id: str = None):
        self.client_id = client_id or str(uuid.uuid4())
        self._connected = False
        self._callbacks = {}
        self._subscriptions = {}
    
    def on_connect(self, callback):
        self._callbacks['on_connect'] = callback
    
    def on_disconnect(self, callback):
        self._callbacks['on_disconnect'] = callback
    
    def on_message(self, callback):
        self._callbacks['on_message'] = callback
    
    def connect(self, host: str, port: int = 1883, keepalive: int = 60):
        self._connected = True
        if 'on_connect' in self._callbacks:
            self._callbacks['on_connect'](self, None, None, 0)
        logger.info(f"Mock MQTT client connected to {host}:{port}")
    
    def disconnect(self):
        self._connected = False
        if 'on_disconnect' in self._callbacks:
            self._callbacks['on_disconnect'](self, None, 0)
    
    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False):
        logger.info(f"Mock publish to {topic}: {payload}")
        return type('Result', (), {'rc': 0, 'mid': 1})()
    
    def subscribe(self, topic: str, qos: int = 0):
        self._subscriptions[topic] = qos
        logger.info(f"Mock subscribe to {topic}")
        return type('Result', (), {'rc': 0, 'mid': 1})()
    
    def unsubscribe(self, topic: str):
        if topic in self._subscriptions:
            del self._subscriptions[topic]
        return type('Result', (), {'rc': 0, 'mid': 1})()
    
    def loop_start(self):
        pass
    
    def loop_stop(self):
        pass
    
    def username_pw_set(self, username: str, password: str = None):
        pass
    
    def will_set(self, topic: str, payload: str = None, qos: int = 0, retain: bool = False):
        pass


def create_mqtt_client(client_id: str = None) -> Any:
    """Create MQTT client (real or mock)."""
    if PAHO_AVAILABLE:
        return mqtt.Client(client_id=client_id)
    else:
        return MockMQTTClient(client_id=client_id)


def on_connect_callback(broker_id: str):
    """Create on_connect callback for a broker."""
    def callback(client, userdata, flags, rc):
        asyncio.create_task(_handle_connect(broker_id, rc))
    return callback


def on_disconnect_callback(broker_id: str):
    """Create on_disconnect callback for a broker."""
    def callback(client, userdata, rc):
        asyncio.create_task(_handle_disconnect(broker_id, rc))
    return callback


def on_message_callback(broker_id: str):
    """Create on_message callback for a broker."""
    def callback(client, userdata, msg):
        asyncio.create_task(_handle_message(broker_id, msg))
    return callback


async def _handle_connect(broker_id: str, rc: int):
    """Handle connection event."""
    async with _lock:
        if broker_id in _broker_status:
            if rc == 0:
                _broker_status[broker_id].state = ConnectionState.CONNECTED
                _broker_status[broker_id].connected_at = datetime.utcnow()
                _broker_status[broker_id].error_message = None
                logger.info(f"Connected to broker: {broker_id}")
            else:
                _broker_status[broker_id].state = ConnectionState.ERROR
                _broker_status[broker_id].error_message = f"Connection failed with code {rc}"


async def _handle_disconnect(broker_id: str, rc: int):
    """Handle disconnection event."""
    async with _lock:
        if broker_id in _broker_status:
            if rc == 0:
                _broker_status[broker_id].state = ConnectionState.DISCONNECTED
                logger.info(f"Disconnected from broker: {broker_id}")
            else:
                _broker_status[broker_id].state = ConnectionState.ERROR
                _broker_status[broker_id].error_message = f"Unexpected disconnection with code {rc}"
                
                # Attempt reconnection
                broker = _brokers.get(broker_id)
                if broker and broker.reconnect_on_failure:
                    _broker_status[broker_id].state = ConnectionState.RECONNECTING
                    asyncio.create_task(_reconnect_broker(broker_id))


async def _handle_message(broker_id: str, msg):
    """Handle incoming message."""
    message = MqttMessage(
        topic=msg.topic,
        payload=msg.payload.decode() if isinstance(msg.payload, bytes) else msg.payload,
        qos=msg.qos,
        retain=msg.retain
    )
    
    async with _lock:
        _message_history.append(message)
        if broker_id in _broker_status:
            _broker_status[broker_id].messages_received += 1
            _broker_status[broker_id].last_activity = datetime.utcnow()
        
        # Update subscription message count
        for sub in _subscriptions.values():
            if sub.broker_id == broker_id and _topic_matches(sub.topic, msg.topic):
                sub.message_count += 1
    
    # Call registered handlers
    for topic_pattern, handlers in _message_handlers.items():
        if _topic_matches(topic_pattern, msg.topic):
            for handler in handlers:
                try:
                    await handler(message)
                except Exception as e:
                    logger.error(f"Message handler error: {e}")


def _topic_matches(pattern: str, topic: str) -> bool:
    """Check if topic matches pattern (with wildcards)."""
    pattern_parts = pattern.split("/")
    topic_parts = topic.split("/")
    
    for i, part in enumerate(pattern_parts):
        if part == "#":
            return True
        if part == "+":
            continue
        if i >= len(topic_parts) or part != topic_parts[i]:
            return False
    
    return len(pattern_parts) == len(topic_parts)


async def _reconnect_broker(broker_id: str):
    """Attempt to reconnect to a broker."""
    broker = _brokers.get(broker_id)
    if not broker:
        return
    
    status = _broker_status.get(broker_id)
    if not status:
        return
    
    for attempt in range(broker.max_reconnect_attempts):
        logger.info(f"Reconnection attempt {attempt + 1} for broker {broker_id}")
        
        try:
            await connect_broker(broker_id)
            if _broker_status[broker_id].state == ConnectionState.CONNECTED:
                logger.info(f"Reconnected to broker: {broker_id}")
                return
        except Exception as e:
            logger.error(f"Reconnection attempt {attempt + 1} failed: {e}")
        
        await asyncio.sleep(broker.reconnect_delay * (attempt + 1))
    
    logger.error(f"Failed to reconnect to broker {broker_id} after {broker.max_reconnect_attempts} attempts")


async def connect_broker(broker_id: str):
    """Connect to a broker."""
    broker = _brokers.get(broker_id)
    if not broker:
        raise ValueError(f"Broker {broker_id} not found")
    
    # Create client
    client = create_mqtt_client(broker.client_id)
    
    # Set callbacks
    client.on_connect = on_connect_callback(broker_id)
    client.on_disconnect = on_disconnect_callback(broker_id)
    client.on_message = on_message_callback(broker_id)
    
    # Set credentials
    if broker.username:
        client.username_pw_set(broker.username, broker.password)
    
    # Connect
    try:
        client.connect(broker.host, broker.port, broker.keepalive)
        client.loop_start()
        
        async with _lock:
            _clients[broker_id] = client
            _broker_status[broker_id].state = ConnectionState.CONNECTING
    except Exception as e:
        async with _lock:
            _broker_status[broker_id].state = ConnectionState.ERROR
            _broker_status[broker_id].error_message = str(e)
        raise


@app.on_event("startup")
async def startup_event():
    """Initialize the MQTT node."""
    logger.info("MQTT node starting up...")
    logger.info(f"paho-mqtt available: {PAHO_AVAILABLE}")
    logger.info("MQTT node ready")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("MQTT node shutting down...")
    
    for broker_id, client in _clients.items():
        try:
            client.loop_stop()
            client.disconnect()
        except Exception as e:
            logger.error(f"Error disconnecting from {broker_id}: {e}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    connected = sum(1 for s in _broker_status.values() if s.state == ConnectionState.CONNECTED)
    return {
        "status": "healthy",
        "node": "41",
        "name": "MQTT",
        "brokers": len(_brokers),
        "connected": connected,
        "paho_available": PAHO_AVAILABLE,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/brokers")
async def add_broker(config: BrokerConfig) -> Dict[str, Any]:
    """
    Add a new MQTT broker configuration.
    
    Args:
        config: Broker configuration
        
    Returns:
        Broker information
    """
    async with _lock:
        _brokers[config.broker_id] = config
        _broker_status[config.broker_id] = BrokerStatus(
            broker_id=config.broker_id,
            state=ConnectionState.DISCONNECTED
        )
    
    logger.info(f"Broker added: {config.name} ({config.host}:{config.port})")
    
    return {
        "success": True,
        "broker_id": config.broker_id,
        "name": config.name,
        "host": config.host,
        "port": config.port,
        "added_at": datetime.utcnow().isoformat()
    }


@app.get("/brokers")
async def list_brokers() -> Dict[str, Any]:
    """List all configured brokers."""
    async with _lock:
        return {
            "brokers": [
                {
                    "broker_id": b.broker_id,
                    "name": b.name,
                    "host": b.host,
                    "port": b.port,
                    "use_ssl": b.use_ssl,
                    "state": _broker_status.get(b.broker_id, BrokerStatus(broker_id=b.broker_id, state=ConnectionState.DISCONNECTED)).state.value
                }
                for b in _brokers.values()
            ],
            "total": len(_brokers)
        }


@app.post("/brokers/{broker_id}/connect")
async def connect_broker_endpoint(broker_id: str) -> Dict[str, Any]:
    """
    Connect to a broker.
    
    Args:
        broker_id: Broker ID
        
    Returns:
        Connection result
    """
    async with _lock:
        if broker_id not in _brokers:
            raise HTTPException(status_code=404, detail=f"Broker {broker_id} not found")
    
    try:
        await connect_broker(broker_id)
        
        return {
            "success": True,
            "broker_id": broker_id,
            "state": _broker_status[broker_id].state.value,
            "connected_at": _broker_status[broker_id].connected_at.isoformat() if _broker_status[broker_id].connected_at else None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Connection failed: {str(e)}")


@app.post("/brokers/{broker_id}/disconnect")
async def disconnect_broker(broker_id: str) -> Dict[str, Any]:
    """
    Disconnect from a broker.
    
    Args:
        broker_id: Broker ID
        
    Returns:
        Disconnection result
    """
    async with _lock:
        if broker_id not in _brokers:
            raise HTTPException(status_code=404, detail=f"Broker {broker_id} not found")
        
        if broker_id in _clients:
            client = _clients.pop(broker_id)
            client.loop_stop()
            client.disconnect()
        
        _broker_status[broker_id].state = ConnectionState.DISCONNECTED
    
    return {
        "success": True,
        "broker_id": broker_id,
        "disconnected_at": datetime.utcnow().isoformat()
    }


@app.post("/publish")
async def publish_message(request: PublishRequest) -> Dict[str, Any]:
    """
    Publish a message to a topic.
    
    Args:
        request: Publish request
        
    Returns:
        Publish result
    """
    async with _lock:
        if request.broker_id not in _brokers:
            raise HTTPException(status_code=404, detail=f"Broker {request.broker_id} not found")
        
        if request.broker_id not in _clients:
            raise HTTPException(status_code=400, detail=f"Not connected to broker {request.broker_id}")
        
        client = _clients[request.broker_id]
        
        payload = request.payload
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload)
        
        result = client.publish(
            request.topic,
            payload,
            qos=request.qos.value,
            retain=request.retain
        )
        
        if result.rc == 0:
            _broker_status[request.broker_id].messages_sent += 1
            _broker_status[request.broker_id].last_activity = datetime.utcnow()
        
        return {
            "success": result.rc == 0,
            "broker_id": request.broker_id,
            "topic": request.topic,
            "qos": request.qos.value,
            "retain": request.retain,
            "message_id": result.mid
        }


@app.post("/subscribe")
async def subscribe(request: SubscribeRequest) -> Dict[str, Any]:
    """
    Subscribe to a topic.
    
    Args:
        request: Subscribe request
        
    Returns:
        Subscription result
    """
    async with _lock:
        if request.broker_id not in _brokers:
            raise HTTPException(status_code=404, detail=f"Broker {request.broker_id} not found")
        
        if request.broker_id not in _clients:
            raise HTTPException(status_code=400, detail=f"Not connected to broker {request.broker_id}")
        
        client = _clients[request.broker_id]
        result = client.subscribe(request.topic, qos=request.qos.value)
        
        if result.rc == 0:
            subscription = Subscription(
                broker_id=request.broker_id,
                topic=request.topic,
                qos=request.qos
            )
            _subscriptions[subscription.subscription_id] = subscription
            
            return {
                "success": True,
                "subscription_id": subscription.subscription_id,
                "broker_id": request.broker_id,
                "topic": request.topic,
                "qos": request.qos.value
            }
        else:
            raise HTTPException(status_code=500, detail=f"Subscribe failed with code {result.rc}")


@app.post("/unsubscribe/{subscription_id}")
async def unsubscribe(subscription_id: str) -> Dict[str, Any]:
    """
    Unsubscribe from a topic.
    
    Args:
        subscription_id: Subscription ID
        
    Returns:
        Unsubscription result
    """
    async with _lock:
        if subscription_id not in _subscriptions:
            raise HTTPException(status_code=404, detail=f"Subscription {subscription_id} not found")
        
        subscription = _subscriptions[subscription_id]
        
        if subscription.broker_id in _clients:
            client = _clients[subscription.broker_id]
            client.unsubscribe(subscription.topic)
        
        subscription.is_active = False
        del _subscriptions[subscription_id]
    
    return {
        "success": True,
        "subscription_id": subscription_id,
        "unsubscribed_at": datetime.utcnow().isoformat()
    }


@app.get("/subscriptions")
async def list_subscriptions(broker_id: Optional[str] = None) -> Dict[str, Any]:
    """
    List active subscriptions.
    
    Args:
        broker_id: Filter by broker ID
        
    Returns:
        List of subscriptions
    """
    async with _lock:
        subs = list(_subscriptions.values())
        
        if broker_id:
            subs = [s for s in subs if s.broker_id == broker_id]
        
        return {
            "subscriptions": [
                {
                    "subscription_id": s.subscription_id,
                    "broker_id": s.broker_id,
                    "topic": s.topic,
                    "qos": s.qos.value,
                    "message_count": s.message_count,
                    "created_at": s.created_at.isoformat(),
                    "is_active": s.is_active
                }
                for s in subs
            ],
            "total": len(subs)
        }


@app.get("/messages")
async def get_messages(
    topic: Optional[str] = None,
    limit: int = 100
) -> Dict[str, Any]:
    """
    Get message history.
    
    Args:
        topic: Filter by topic pattern
        limit: Maximum number of messages
        
    Returns:
        Message history
    """
    async with _lock:
        messages = _message_history
        
        if topic:
            messages = [m for m in messages if _topic_matches(topic, m.topic)]
        
        messages = sorted(messages, key=lambda m: m.timestamp, reverse=True)[:limit]
        
        return {
            "messages": [
                {
                    "topic": m.topic,
                    "payload": m.payload,
                    "qos": m.qos.value,
                    "retain": m.retain,
                    "timestamp": m.timestamp.isoformat()
                }
                for m in messages
            ],
            "total": len(messages)
        }


@app.get("/stats")
async def get_stats() -> Dict[str, Any]:
    """Get MQTT statistics."""
    async with _lock:
        total_sent = sum(s.messages_sent for s in _broker_status.values())
        total_received = sum(s.messages_received for s in _broker_status.values())
        
        return {
            "brokers": len(_brokers),
            "connected": sum(1 for s in _broker_status.values() if s.state == ConnectionState.CONNECTED),
            "subscriptions": len(_subscriptions),
            "messages_sent": total_sent,
            "messages_received": total_received,
            "message_history": len(_message_history)
        }


@app.websocket("/ws/{broker_id}/{topic}")
async def websocket_endpoint(websocket: WebSocket, broker_id: str, topic: str):
    """
    WebSocket endpoint for real-time message streaming.
    
    Args:
        websocket: WebSocket connection
        broker_id: Broker ID
        topic: Topic to subscribe to
    """
    await websocket.accept()
    
    # Register message handler
    async def message_handler(message: MqttMessage):
        await websocket.send_json({
            "topic": message.topic,
            "payload": message.payload,
            "qos": message.qos.value,
            "timestamp": message.timestamp.isoformat()
        })
    
    if topic not in _message_handlers:
        _message_handlers[topic] = []
    _message_handlers[topic].append(message_handler)
    
    try:
        while True:
            # Keep connection alive
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        # Unregister handler
        if topic in _message_handlers:
            _message_handlers[topic].remove(message_handler)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8441)
