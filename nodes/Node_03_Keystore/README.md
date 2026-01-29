# Node 03 - Keystore

Secure Key Management Node for UFO Galaxy v5.0

## Overview

The Keystore node provides secure key management:
- API key storage and retrieval
- Encryption/decryption services
- Key rotation and expiration
- Access control and auditing

## Key Types

- `api_key` - API keys for external services
- `encryption_key` - Symmetric encryption keys
- `signing_key` - Digital signing keys
- `jwt_secret` - JWT signing secrets
- `password` - Encrypted passwords
- `certificate` - TLS certificates

## API Endpoints

### Health Check
```
GET /health
```

### Keys
```
POST /keys
```
Store a new key securely.

**Request Body:**
```json
{
  "name": "OpenAI API Key",
  "key_type": "api_key",
  "value": "sk-...",
  "expires_in_days": 365,
  "tags": ["openai", "production"]
}
```

```
GET /keys/{key_id}
```
Retrieve key information.

Query Parameters:
- `include_value`: Whether to include the decrypted value

```
GET /keys
```
List stored keys with optional filtering.

Query Parameters:
- `key_type`: Filter by key type
- `status`: Filter by status
- `tag`: Filter by tag

```
POST /keys/{key_id}/rotate
```
Rotate a key to a new value.

```
POST /keys/{key_id}/revoke
```
Revoke a key.

```
POST /keys/{key_id}/delete
```
Permanently delete a key.

### Encryption
```
POST /encrypt
```
Encrypt data using a stored key or ephemeral key.

**Request Body:**
```json
{
  "plaintext": "Hello, World!",
  "key_id": "optional-key-id",
  "encoding": "base64"
}
```

```
POST /decrypt
```
Decrypt data using a stored key or master key.

### Access Logs
```
GET /logs
```
Get access logs.

Query Parameters:
- `key_id`: Filter by key ID
- `limit`: Maximum number of logs

### Statistics
```
GET /stats
```
Get keystore statistics.

## Key Status

- `active` - Key is active and can be used
- `expired` - Key has expired
- `revoked` - Key has been revoked
- `pending` - Key is pending activation

## Configuration

See `config.yaml` for configuration options.

## Usage Example

```python
import httpx

# Store a key
response = httpx.post("http://localhost:8003/keys", json={
    "name": "OpenAI API Key",
    "key_type": "api_key",
    "value": "sk-...",
    "expires_in_days": 365
})
key_id = response.json()["key_id"]

# Retrieve key (without value)
response = httpx.get(f"http://localhost:8003/keys/{key_id}")
print(response.json())

# Retrieve key with value
response = httpx.get(f"http://localhost:8003/keys/{key_id}?include_value=true")
print(response.json())

# Encrypt data
response = httpx.post("http://localhost:8003/encrypt", json={
    "plaintext": "Secret message",
    "encoding": "base64"
})
ciphertext = response.json()["ciphertext"]

# Decrypt data
response = httpx.post("http://localhost:8003/decrypt", json={
    "ciphertext": ciphertext,
    "encoding": "base64"
})
print(response.json()["plaintext"])
```

## Port

- HTTP API: `8003`

## Dependencies

- Node 00 - StateMachine (port 8000)

## Author

UFO Galaxy Team
