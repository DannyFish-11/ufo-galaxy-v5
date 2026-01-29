"""
Node_05_Auth - Authentication/Authorization Node
UFO Galaxy v5.0 Core Node System

This node provides authentication and authorization services:
- User authentication (password, token, API key)
- JWT token generation and validation
- Role-based access control (RBAC)
- Permission management
"""

from fastapi import FastAPI, HTTPException, Header, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import uvicorn
import asyncio
from datetime import datetime, timedelta
from loguru import logger
import uuid
import jwt
import hashlib
import secrets

# Configure logging
logger.add("auth.log", rotation="10 MB", retention="7 days")

app = FastAPI(
    title="Node 05 - Auth",
    description="Authentication/Authorization for UFO Galaxy v5.0",
    version="5.0.0"
)

# Security scheme
security = HTTPBearer()


class UserStatus(str, Enum):
    """User status enumeration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class TokenType(str, Enum):
    """Token type enumeration."""
    ACCESS = "access"
    REFRESH = "refresh"
    API_KEY = "api_key"


class Permission(str, Enum):
    """Permission enumeration."""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXECUTE = "execute"


class Role(str, Enum):
    """Role enumeration."""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    SERVICE = "service"
    READONLY = "readonly"


# Role permissions mapping
ROLE_PERMISSIONS = {
    Role.ADMIN: [Permission.READ, Permission.WRITE, Permission.DELETE, Permission.ADMIN, Permission.EXECUTE],
    Role.USER: [Permission.READ, Permission.WRITE, Permission.EXECUTE],
    Role.GUEST: [Permission.READ],
    Role.SERVICE: [Permission.READ, Permission.WRITE, Permission.EXECUTE],
    Role.READONLY: [Permission.READ],
}


class User(BaseModel):
    """User model."""
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    username: str
    email: str
    password_hash: str
    salt: str
    roles: List[Role] = Field(default_factory=lambda: [Role.USER])
    status: UserStatus = UserStatus.PENDING
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None


class TokenPayload(BaseModel):
    """Token payload model."""
    sub: str  # user_id
    username: str
    roles: List[str]
    permissions: List[str]
    token_type: TokenType
    exp: datetime
    iat: datetime
    jti: str  # token ID


class LoginRequest(BaseModel):
    """Login request model."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    username: str
    roles: List[str]


class RegisterRequest(BaseModel):
    """Registration request model."""
    username: str
    email: str
    password: str
    roles: List[Role] = Field(default_factory=lambda: [Role.USER])


class APIKeyRequest(BaseModel):
    """API key request model."""
    name: str
    permissions: List[Permission] = Field(default_factory=list)
    expires_in_days: int = 30


class APIKey(BaseModel):
    """API key model."""
    key_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    key: str
    user_id: str
    permissions: List[Permission]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    expires_at: datetime
    last_used: Optional[datetime] = None
    is_active: bool = True


class PermissionCheck(BaseModel):
    """Permission check model."""
    user_id: str
    permission: Permission
    resource: Optional[str] = None


# In-memory storage (use database in production)
_users: Dict[str, User] = {}  # user_id -> User
_usernames: Dict[str, str] = {}  # username -> user_id
_tokens: Dict[str, TokenPayload] = {}  # jti -> TokenPayload
_api_keys: Dict[str, APIKey] = {}  # key -> APIKey
_refresh_tokens: Dict[str, str] = {}  # refresh_token -> user_id
_lock = asyncio.Lock()

# JWT configuration
JWT_SECRET = secrets.token_urlsafe(32)
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 30


def _hash_password(password: str, salt: str) -> str:
    """Hash password with salt."""
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100000).hex()


def _generate_salt() -> str:
    """Generate random salt."""
    return secrets.token_hex(16)


def _generate_token(user: User, token_type: TokenType, expires_delta: timedelta) -> str:
    """Generate JWT token."""
    now = datetime.utcnow()
    jti = str(uuid.uuid4())
    
    # Get permissions from roles
    permissions = set()
    for role in user.roles:
        permissions.update(ROLE_PERMISSIONS.get(role, []))
    
    payload = {
        "sub": user.user_id,
        "username": user.username,
        "roles": [r.value for r in user.roles],
        "permissions": [p.value for p in permissions],
        "token_type": token_type.value,
        "exp": now + expires_delta,
        "iat": now,
        "jti": jti
    }
    
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    # Store token payload
    _tokens[jti] = TokenPayload(**payload)
    
    return token


def _verify_token(token: str) -> Optional[TokenPayload]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        jti = payload.get("jti")
        
        if jti not in _tokens:
            return None
        
        return _tokens[jti]
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)) -> User:
    """Get current user from token."""
    token = credentials.credentials
    payload = _verify_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    async with _lock:
        if payload.sub not in _users:
            raise HTTPException(status_code=401, detail="User not found")
        
        user = _users[payload.sub]
        if user.status != UserStatus.ACTIVE:
            raise HTTPException(status_code=403, detail=f"User account is {user.status.value}")
        
        return user


@app.on_event("startup")
async def startup_event():
    """Initialize the auth service."""
    logger.info("Auth service starting up...")
    
    # Create default admin user
    await _create_default_admin()
    
    logger.info("Auth service ready")


async def _create_default_admin():
    """Create default admin user."""
    async with _lock:
        if "admin" not in _usernames:
            salt = _generate_salt()
            password_hash = _hash_password("admin", salt)
            
            admin = User(
                username="admin",
                email="admin@ufo-galaxy.local",
                password_hash=password_hash,
                salt=salt,
                roles=[Role.ADMIN],
                status=UserStatus.ACTIVE
            )
            
            _users[admin.user_id] = admin
            _usernames[admin.username] = admin.user_id
            
            logger.info(f"Default admin user created: {admin.user_id}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "node": "05",
        "name": "Auth",
        "users": len(_users),
        "active_tokens": len(_tokens),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/register")
async def register(request: RegisterRequest) -> Dict[str, Any]:
    """
    Register a new user.
    
    Args:
        request: Registration request
        
    Returns:
        Registration result
    """
    async with _lock:
        if request.username in _usernames:
            raise HTTPException(status_code=409, detail="Username already exists")
        
        salt = _generate_salt()
        password_hash = _hash_password(request.password, salt)
        
        user = User(
            username=request.username,
            email=request.email,
            password_hash=password_hash,
            salt=salt,
            roles=request.roles,
            status=UserStatus.PENDING
        )
        
        _users[user.user_id] = user
        _usernames[user.username] = user.user_id
        
        logger.info(f"User registered: {user.username} ({user.user_id})")
        
        return {
            "success": True,
            "user_id": user.user_id,
            "username": user.username,
            "status": user.status.value,
            "created_at": user.created_at.isoformat()
        }


@app.post("/login")
async def login(request: LoginRequest) -> TokenResponse:
    """
    Authenticate user and issue tokens.
    
    Args:
        request: Login request
        
    Returns:
        Token response
    """
    async with _lock:
        if request.username not in _usernames:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        user_id = _usernames[request.username]
        user = _users[user_id]
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.utcnow():
            raise HTTPException(status_code=403, detail="Account is locked. Try again later.")
        
        # Verify password
        password_hash = _hash_password(request.password, user.salt)
        if password_hash != user.password_hash:
            user.login_attempts += 1
            
            if user.login_attempts >= MAX_LOGIN_ATTEMPTS:
                user.locked_until = datetime.utcnow() + timedelta(minutes=LOCKOUT_DURATION_MINUTES)
                logger.warning(f"User {request.username} locked due to failed login attempts")
            
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        # Reset login attempts
        user.login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.utcnow()
        
        # Generate tokens
        access_token = _generate_token(
            user,
            TokenType.ACCESS,
            timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        refresh_token = _generate_token(
            user,
            TokenType.REFRESH,
            timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        )
        
        _refresh_tokens[refresh_token] = user.user_id
        
        logger.info(f"User logged in: {user.username}")
        
        return TokenResponse(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user.user_id,
            username=user.username,
            roles=[r.value for r in user.roles]
        )


@app.post("/refresh")
async def refresh_token(refresh_token: str) -> TokenResponse:
    """
    Refresh access token.
    
    Args:
        refresh_token: Refresh token
        
    Returns:
        New token response
    """
    payload = _verify_token(refresh_token)
    
    if not payload or payload.token_type != TokenType.REFRESH:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    
    async with _lock:
        if refresh_token not in _refresh_tokens:
            raise HTTPException(status_code=401, detail="Refresh token revoked")
        
        user_id = _refresh_tokens[refresh_token]
        if user_id not in _users:
            raise HTTPException(status_code=401, detail="User not found")
        
        user = _users[user_id]
        
        # Generate new tokens
        new_access_token = _generate_token(
            user,
            TokenType.ACCESS,
            timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        new_refresh_token = _generate_token(
            user,
            TokenType.REFRESH,
            timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
        )
        
        # Revoke old refresh token
        del _refresh_tokens[refresh_token]
        _refresh_tokens[new_refresh_token] = user_id
        
        logger.info(f"Token refreshed for user: {user.username}")
        
        return TokenResponse(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            user_id=user.user_id,
            username=user.username,
            roles=[r.value for r in user.roles]
        )


@app.post("/logout")
async def logout(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """
    Logout user and revoke token.
    
    Args:
        credentials: Authorization credentials
        
    Returns:
        Logout result
    """
    token = credentials.credentials
    payload = _verify_token(token)
    
    if payload and payload.jti in _tokens:
        del _tokens[payload.jti]
    
    logger.info(f"User logged out: {payload.username if payload else 'unknown'}")
    
    return {
        "success": True,
        "message": "Logged out successfully"
    }


@app.get("/me")
async def get_me(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Get current user information.
    
    Args:
        user: Current user
        
    Returns:
        User information
    """
    permissions = set()
    for role in user.roles:
        permissions.update(ROLE_PERMISSIONS.get(role, []))
    
    return {
        "user_id": user.user_id,
        "username": user.username,
        "email": user.email,
        "roles": [r.value for r in user.roles],
        "permissions": [p.value for p in permissions],
        "status": user.status.value,
        "created_at": user.created_at.isoformat(),
        "last_login": user.last_login.isoformat() if user.last_login else None
    }


@app.post("/api-keys")
async def create_api_key(
    request: APIKeyRequest,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Create API key for user.
    
    Args:
        request: API key request
        user: Current user
        
    Returns:
        API key information
    """
    key_value = f"ufog_{secrets.token_urlsafe(32)}"
    
    api_key = APIKey(
        name=request.name,
        key=key_value,
        user_id=user.user_id,
        permissions=request.permissions,
        expires_at=datetime.utcnow() + timedelta(days=request.expires_in_days)
    )
    
    async with _lock:
        _api_keys[key_value] = api_key
    
    logger.info(f"API key created for user: {user.username}")
    
    return {
        "success": True,
        "key_id": api_key.key_id,
        "key": key_value,  # Only shown once
        "name": api_key.name,
        "expires_at": api_key.expires_at.isoformat()
    }


@app.get("/api-keys")
async def list_api_keys(user: User = Depends(get_current_user)) -> Dict[str, Any]:
    """
    List user's API keys.
    
    Args:
        user: Current user
        
    Returns:
        List of API keys
    """
    async with _lock:
        keys = [
            {
                "key_id": k.key_id,
                "name": k.name,
                "permissions": [p.value for p in k.permissions],
                "created_at": k.created_at.isoformat(),
                "expires_at": k.expires_at.isoformat(),
                "last_used": k.last_used.isoformat() if k.last_used else None,
                "is_active": k.is_active
            }
            for k in _api_keys.values()
            if k.user_id == user.user_id
        ]
    
    return {
        "api_keys": keys,
        "total": len(keys)
    }


@app.post("/check-permission")
async def check_permission(
    check: PermissionCheck,
    user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Check if user has a specific permission.
    
    Args:
        check: Permission check request
        user: Current user
        
    Returns:
        Permission check result
    """
    user_permissions = set()
    for role in user.roles:
        user_permissions.update(ROLE_PERMISSIONS.get(role, []))
    
    has_permission = check.permission in user_permissions
    
    return {
        "user_id": user.user_id,
        "permission": check.permission.value,
        "granted": has_permission,
        "resource": check.resource
    }


@app.get("/verify")
async def verify_token_endpoint(
    credentials: HTTPAuthorizationCredentials = Security(security)
) -> Dict[str, Any]:
    """
    Verify a token.
    
    Args:
        credentials: Authorization credentials
        
    Returns:
        Token verification result
    """
    token = credentials.credentials
    payload = _verify_token(token)
    
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    
    return {
        "valid": True,
        "user_id": payload.sub,
        "username": payload.username,
        "roles": payload.roles,
        "permissions": payload.permissions,
        "expires": payload.exp.isoformat()
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
