# Node 33 - ADB

Android Debug Bridge Node for UFO Galaxy v5.0

## Overview

The ADB node provides Android Debug Bridge functionality:
- Device discovery and connection
- Shell command execution
- File transfer (push/pull)
- Screenshot capture
- Logcat streaming

## Requirements

- Android SDK Platform Tools (adb)
- USB debugging enabled on devices
- Network debugging enabled (for WiFi connections)

## API Endpoints

### Health Check
```
GET /health
```

### Devices
```
GET /devices
```
List all known devices.

```
GET /devices/{device_id}
```
Get detailed device information.

```
POST /devices/connect
```
Connect to a device over network.

Query Parameters:
- `ip_address`: Device IP address
- `port`: Device port (default 5555)

```
POST /devices/disconnect
```
Disconnect from a network device.

Query Parameters:
- `device_id`: Device identifier

### Shell Commands
```
POST /shell
```
Execute a shell command on a device.

**Request Body:**
```json
{
  "device_id": "device-id",
  "command": "ls -la",
  "timeout": 30,
  "as_root": false
}
```

### File Transfer
```
POST /files/pull
```
Pull a file from device.

**Request Body:**
```json
{
  "device_id": "device-id",
  "source_path": "/sdcard/file.txt",
  "destination_path": "/local/path/file.txt"
}
```

```
POST /files/push
```
Push a file to device.

Form Data:
- `device_id`: Device identifier
- `destination_path`: Destination path on device
- `file`: File to upload

### Screenshot
```
POST /screenshot
```
Capture a screenshot from device.

Query Parameters:
- `device_id`: Device identifier

Returns base64-encoded PNG image.

### Logcat
```
POST /logcat/start
```
Start logcat streaming.

**Request Body:**
```json
{
  "device_id": "device-id",
  "buffer": "main",
  "priority": "V",
  "tag_filter": "MyApp",
  "max_lines": 1000
}
```

```
POST /logcat/stop
```
Stop logcat streaming.

Query Parameters:
- `device_id`: Device identifier

### Applications
```
GET /apps
```
List installed applications.

Query Parameters:
- `device_id`: Device identifier
- `include_system`: Include system apps (default false)

```
POST /apps/install
```
Install an APK on device.

Form Data:
- `device_id`: Device identifier
- `file`: APK file to install

### Device Control
```
POST /reboot
```
Reboot a device.

Query Parameters:
- `device_id`: Device identifier
- `mode`: Reboot mode (`system`, `recovery`, `bootloader`)

## Device Status

- `connected` - Device is connected and ready
- `disconnected` - Device is disconnected
- `unauthorized` - Device requires authorization
- `offline` - Device is offline
- `recovery` - Device is in recovery mode
- `bootloader` - Device is in bootloader mode

## Configuration

See `config.yaml` for configuration options.

## Usage Example

```python
import httpx

# List devices
response = httpx.get("http://localhost:8433/devices")
devices = response.json()["devices"]

if devices:
    device_id = devices[0]["device_id"]
    
    # Execute shell command
    response = httpx.post("http://localhost:8433/shell", json={
        "device_id": device_id,
        "command": "getprop ro.product.model"
    })
    print(response.json()["stdout"])
    
    # Take screenshot
    response = httpx.post(f"http://localhost:8433/screenshot?device_id={device_id}")
    screenshot_data = response.json()["screenshot"]
    
    # Install APK
    with open("app.apk", "rb") as f:
        response = httpx.post(
            f"http://localhost:8433/apps/install?device_id={device_id}",
            files={"file": f}
        )
        print(response.json())
```

## Port

- HTTP API: `8433`

## Dependencies

- Node 00 - StateMachine (port 8000)
- Optional: Node 02 - TaskEngine (port 8002)

## Author

UFO Galaxy Team
