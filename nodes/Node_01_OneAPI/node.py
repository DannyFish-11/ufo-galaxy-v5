"""
Node_01_OneAPI - AI API Gateway Node
UFO Galaxy v5.0 Core Node System

This node provides a unified gateway for AI API services:
- Multi-provider AI API routing (OpenAI, Anthropic, Google, etc.)
- Request/response transformation
- Rate limiting and quota management
- API key rotation and management
"""

from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal, AsyncGenerator
from enum import Enum
import uvicorn
import asyncio
import httpx
from datetime import datetime
from loguru import logger

# Configure logging
logger.add("oneapi_gateway.log", rotation="10 MB", retention="7 days")

app = FastAPI(
    title="Node 01 - OneAPI",
    description="AI API Gateway for UFO Galaxy v5.0",
    version="5.0.0"
)


class AIProvider(str, Enum):
    """Supported AI providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    COHERE = "cohere"
    MISTRAL = "mistral"
    AZURE = "azure"


class ModelConfig(BaseModel):
    """Model configuration."""
    provider: AIProvider
    model_name: str
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    timeout: int = 60


class ChatMessage(BaseModel):
    """Chat message model."""
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    """Chat completion request model."""
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    user: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Chat completion response model."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


class APIKeyConfig(BaseModel):
    """API key configuration."""
    provider: AIProvider
    key: str
    base_url: Optional[str] = None
    quota_limit: int = 10000
    quota_used: int = 0
    is_active: bool = True


class ProviderStatus(BaseModel):
    """Provider status model."""
    provider: AIProvider
    status: str
    latency_ms: float
    available_models: List[str]
    quota_remaining: int


# In-memory storage (replace with persistent storage in production)
_api_keys: Dict[str, APIKeyConfig] = {}
_request_counts: Dict[str, int] = {}
_provider_status: Dict[AIProvider, ProviderStatus] = {}
_lock = asyncio.Lock()


# Provider base URLs
PROVIDER_URLS = {
    AIProvider.OPENAI: "https://api.openai.com/v1",
    AIProvider.ANTHROPIC: "https://api.anthropic.com/v1",
    AIProvider.GOOGLE: "https://generativelanguage.googleapis.com/v1",
    AIProvider.COHERE: "https://api.cohere.com/v1",
    AIProvider.MISTRAL: "https://api.mistral.ai/v1",
}


@app.on_event("startup")
async def startup_event():
    """Initialize the OneAPI gateway."""
    logger.info("OneAPI gateway starting up...")
    
    # Initialize provider status
    for provider in AIProvider:
        _provider_status[provider] = ProviderStatus(
            provider=provider,
            status="unknown",
            latency_ms=0.0,
            available_models=[],
            quota_remaining=0
        )
    
    logger.info("OneAPI gateway ready")


async def get_api_key(provider: AIProvider) -> Optional[APIKeyConfig]:
    """Get active API key for a provider."""
    async with _lock:
        for key_id, key_config in _api_keys.items():
            if key_config.provider == provider and key_config.is_active:
                if key_config.quota_used < key_config.quota_limit:
                    return key_config
        return None


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "node": "01",
        "name": "OneAPI",
        "providers": len(_provider_status),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/providers")
async def list_providers() -> Dict[str, Any]:
    """List all supported AI providers."""
    return {
        "providers": [
            {
                "id": provider.value,
                "name": provider.name,
                "base_url": PROVIDER_URLS.get(provider)
            }
            for provider in AIProvider
        ]
    }


@app.get("/providers/status")
async def get_providers_status() -> Dict[str, Any]:
    """Get status of all providers."""
    async with _lock:
        return {
            "providers": [
                status.dict()
                for status in _provider_status.values()
            ],
            "timestamp": datetime.utcnow().isoformat()
        }


@app.post("/keys/register")
async def register_api_key(config: APIKeyConfig) -> Dict[str, Any]:
    """
    Register a new API key.
    
    Args:
        config: API key configuration
        
    Returns:
        Registration result
    """
    async with _lock:
        key_id = f"{config.provider.value}_{len(_api_keys)}"
        _api_keys[key_id] = config
        
        logger.info(f"API key registered for provider: {config.provider.value}")
        
        return {
            "success": True,
            "key_id": key_id,
            "provider": config.provider.value,
            "registered_at": datetime.utcnow().isoformat()
        }


@app.get("/keys")
async def list_api_keys() -> Dict[str, Any]:
    """List all registered API keys (masked)."""
    async with _lock:
        return {
            "keys": [
                {
                    "key_id": key_id,
                    "provider": config.provider.value,
                    "quota_limit": config.quota_limit,
                    "quota_used": config.quota_used,
                    "is_active": config.is_active,
                    "key_preview": f"{config.key[:8]}..." if len(config.key) > 8 else "***"
                }
                for key_id, config in _api_keys.items()
            ]
        }


@app.post("/chat/completions")
async def chat_completion(
    request: ChatCompletionRequest,
    x_provider: Optional[str] = Header(None, description="Target AI provider"),
    x_api_key: Optional[str] = Header(None, description="Custom API key")
) -> Dict[str, Any]:
    """
    Create a chat completion.
    
    Args:
        request: Chat completion request
        x_provider: Target AI provider (optional)
        x_api_key: Custom API key (optional)
        
    Returns:
        Chat completion response
    """
    # Determine provider
    provider = AIProvider.OPENAI  # Default provider
    if x_provider:
        try:
            provider = AIProvider(x_provider.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown provider: {x_provider}")
    
    # Get API key
    api_key = x_api_key
    if not api_key:
        key_config = await get_api_key(provider)
        if not key_config:
            raise HTTPException(status_code=503, detail=f"No available API key for {provider.value}")
        api_key = key_config.key
    
    # Build request based on provider
    try:
        if provider == AIProvider.OPENAI:
            return await _call_openai(request, api_key)
        elif provider == AIProvider.ANTHROPIC:
            return await _call_anthropic(request, api_key)
        else:
            # Generic implementation for other providers
            return await _call_generic(provider, request, api_key)
    except Exception as e:
        logger.error(f"Error calling {provider.value}: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Provider error: {str(e)}")


async def _call_openai(request: ChatCompletionRequest, api_key: str) -> Dict[str, Any]:
    """Call OpenAI API."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": request.model,
        "messages": [msg.dict() for msg in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "stream": request.stream
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


async def _call_anthropic(request: ChatCompletionRequest, api_key: str) -> Dict[str, Any]:
    """Call Anthropic API."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    # Convert messages to Anthropic format
    system_msg = ""
    messages = []
    for msg in request.messages:
        if msg.role == "system":
            system_msg = msg.content
        else:
            messages.append({"role": msg.role, "content": msg.content})
    
    payload = {
        "model": request.model,
        "messages": messages,
        "max_tokens": request.max_tokens or 4096,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "system": system_msg
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Transform to OpenAI-compatible format
        return {
            "id": data.get("id", ""),
            "object": "chat.completion",
            "created": int(datetime.utcnow().timestamp()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": data.get("content", [{}])[0].get("text", "")
                },
                "finish_reason": data.get("stop_reason", "stop")
            }],
            "usage": data.get("usage", {})
        }


async def _call_generic(
    provider: AIProvider,
    request: ChatCompletionRequest,
    api_key: str
) -> Dict[str, Any]:
    """Generic API call for other providers."""
    base_url = PROVIDER_URLS.get(provider)
    if not base_url:
        raise HTTPException(status_code=400, detail=f"Unknown provider: {provider.value}")
    
    url = f"{base_url}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": request.model,
        "messages": [msg.dict() for msg in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": request.stream
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


@app.get("/models")
async def list_models(
    provider: Optional[AIProvider] = None
) -> Dict[str, Any]:
    """
    List available models.
    
    Args:
        provider: Filter by provider (optional)
        
    Returns:
        List of available models
    """
    models = {
        AIProvider.OPENAI: [
            "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini",
            "gpt-3.5-turbo", "gpt-3.5-turbo-16k"
        ],
        AIProvider.ANTHROPIC: [
            "claude-3-opus-20240229", "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307", "claude-2.1", "claude-2.0"
        ],
        AIProvider.GOOGLE: [
            "gemini-pro", "gemini-pro-vision", "gemini-ultra"
        ],
        AIProvider.COHERE: [
            "command", "command-light", "command-nightly"
        ],
        AIProvider.MISTRAL: [
            "mistral-tiny", "mistral-small", "mistral-medium", "mistral-large"
        ],
    }
    
    if provider:
        return {
            "provider": provider.value,
            "models": [{"id": m, "object": "model"} for m in models.get(provider, [])]
        }
    
    all_models = []
    for prov, model_list in models.items():
        for m in model_list:
            all_models.append({
                "id": m,
                "object": "model",
                "owned_by": prov.value
            })
    
    return {"object": "list", "data": all_models}


@app.post("/embeddings")
async def create_embeddings(
    model: str,
    input: List[str],
    x_provider: Optional[str] = Header(None)
) -> Dict[str, Any]:
    """
    Create embeddings for input text.
    
    Args:
        model: Embedding model
        input: List of input texts
        x_provider: Target provider
        
    Returns:
        Embeddings response
    """
    provider = AIProvider.OPENAI
    if x_provider:
        provider = AIProvider(x_provider.lower())
    
    key_config = await get_api_key(provider)
    if not key_config:
        raise HTTPException(status_code=503, detail=f"No API key available")
    
    base_url = PROVIDER_URLS.get(provider)
    url = f"{base_url}/embeddings"
    headers = {
        "Authorization": f"Bearer {key_config.key}",
        "Content-Type": "application/json"
    }
    payload = {"model": model, "input": input}
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
