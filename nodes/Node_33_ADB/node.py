"""
Node_33_ADB - Android Debug Bridge Node
UFO Galaxy v5.0 Core Node System

This node provides Android Debug Bridge functionality:
- Device discovery and connection
- Shell command execution
- File transfer (push/pull)
- Screenshot capture
- Logcat streaming
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import uvicorn
import asyncio
import subprocess
import re
from datetime import datetime
from loguru import logger
import uuid
import os
import tempfile
import json

# Configure logging
logger.add("adb.log", rotation="10 MB", retention="7 days")

app = FastAPI(
    title="Node 33 - ADB",
    description="Android Debug Bridge for UFO Galaxy v5.0",
    version="5.0.0"
)


class DeviceStatus(str, Enum):
    """Device connection status."""
    CONNECTED = "connected"
    DISCONNECTED = "disconnected"
    UNAUTHORIZED = "unauthorized"
    OFFLINE = "offline"
    RECOVERY = "recovery"
    BOOTLOADER = "bootloader"


class Device(BaseModel):
    """Android device model."""
    device_id: str
    status: DeviceStatus
    model: Optional[str] = None
    manufacturer: Optional[str] = None
    android_version: Optional[str] = None
    sdk_version: Optional[str] = None
    product: Optional[str] = None
    ip_address: Optional[str] = None
    port: int = 5555
    last_seen: datetime = Field(default_factory=datetime.utcnow)
    properties: Dict[str, str] = Field(default_factory=dict)


class ShellCommand(BaseModel):
    """Shell command model."""
    device_id: str
    command: str
    timeout: int = 30
    as_root: bool = False


class FileTransfer(BaseModel):
    """File transfer model."""
    device_id: str
    source_path: str
    destination_path: str


class AppInfo(BaseModel):
    """Application info model."""
    package_name: str
    version_name: Optional[str] = None
    version_code: Optional[str] = None
    is_system_app: bool = False
    is_enabled: bool = True


class LogcatConfig(BaseModel):
    """Logcat configuration model."""
    device_id: str
    buffer: Literal["main", "system", "crash", "events", "radio", "all"] = "main"
    priority: Literal["V", "D", "I", "W", "E", "F", "S"] = "V"
    tag_filter: Optional[str] = None
    pid_filter: Optional[int] = None
    max_lines: int = 1000


# In-memory storage
_devices: Dict[str, Device] = {}
_logcat_processes: Dict[str, subprocess.Popen] = {}
_lock = asyncio.Lock()


def _run_adb_command(args: List[str], timeout: int = 30) -> tuple:
    """Run ADB command and return output."""
    cmd = ["adb"] + args
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"
    except FileNotFoundError:
        return -1, "", "ADB not found. Please install Android SDK."


async def _refresh_devices():
    """Refresh device list from ADB."""
    returncode, stdout, stderr = _run_adb_command(["devices", "-l"])
    
    if returncode != 0:
        logger.error(f"Failed to list devices: {stderr}")
        return
    
    # Parse device list
    lines = stdout.strip().split("\n")[1:]  # Skip header
    
    found_devices = set()
    for line in lines:
        if not line.strip():
            continue
        
        parts = line.split()
        if len(parts) >= 2:
            device_id = parts[0]
            status_str = parts[1]
            
            found_devices.add(device_id)
            
            # Parse additional info
            props = {}
            for part in parts[2:]:
                if ":" in part:
                    key, value = part.split(":", 1)
                    props[key] = value
            
            # Map status
            status_map = {
                "device": DeviceStatus.CONNECTED,
                "unauthorized": DeviceStatus.UNAUTHORIZED,
                "offline": DeviceStatus.OFFLINE,
                "recovery": DeviceStatus.RECOVERY,
                "bootloader": DeviceStatus.BOOTLOADER,
            }
            status = status_map.get(status_str, DeviceStatus.DISCONNECTED)
            
            async with _lock:
                if device_id in _devices:
                    _devices[device_id].status = status
                    _devices[device_id].last_seen = datetime.utcnow()
                    _devices[device_id].properties.update(props)
                else:
                    _devices[device_id] = Device(
                        device_id=device_id,
                        status=status,
                        properties=props
                    )
                
                # Extract known properties
                device = _devices[device_id]
                device.model = props.get("model")
                device.manufacturer = props.get("device")
                device.product = props.get("product")
    
    # Mark missing devices as disconnected
    async with _lock:
        for device_id in list(_devices.keys()):
            if device_id not in found_devices:
                _devices[device_id].status = DeviceStatus.DISCONNECTED


async def _get_device_properties(device_id: str):
    """Get detailed device properties."""
    # Get Android version
    returncode, stdout, _ = _run_adb_command(
        ["-s", device_id, "shell", "getprop", "ro.build.version.release"]
    )
    if returncode == 0:
        async with _lock:
            if device_id in _devices:
                _devices[device_id].android_version = stdout.strip()
    
    # Get SDK version
    returncode, stdout, _ = _run_adb_command(
        ["-s", device_id, "shell", "getprop", "ro.build.version.sdk"]
    )
    if returncode == 0:
        async with _lock:
            if device_id in _devices:
                _devices[device_id].sdk_version = stdout.strip()
    
    # Get IP address
    returncode, stdout, _ = _run_adb_command(
        ["-s", device_id, "shell", "ip", "addr", "show", "wlan0"]
    )
    if returncode == 0:
        match = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", stdout)
        if match:
            async with _lock:
                if device_id in _devices:
                    _devices[device_id].ip_address = match.group(1)


@app.on_event("startup")
async def startup_event():
    """Initialize the ADB node."""
    logger.info("ADB node starting up...")
    
    # Start server
    _run_adb_command(["start-server"])
    
    # Initial device scan
    await _refresh_devices()
    
    # Start background device monitoring
    asyncio.create_task(_device_monitor_loop())
    
    logger.info("ADB node ready")


async def _device_monitor_loop():
    """Background device monitoring loop."""
    while True:
        await asyncio.sleep(10)
        await _refresh_devices()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("ADB node shutting down...")
    
    # Kill logcat processes
    for process in _logcat_processes.values():
        process.terminate()
    
    _run_adb_command(["kill-server"])


@app.get("/health")
async def health():
    """Health check endpoint."""
    connected = sum(1 for d in _devices.values() if d.status == DeviceStatus.CONNECTED)
    return {
        "status": "healthy",
        "node": "33",
        "name": "ADB",
        "devices": len(_devices),
        "connected": connected,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/devices")
async def list_devices() -> Dict[str, Any]:
    """List all known devices."""
    await _refresh_devices()
    
    async with _lock:
        return {
            "devices": [
                {
                    "device_id": d.device_id,
                    "status": d.status.value,
                    "model": d.model,
                    "manufacturer": d.manufacturer,
                    "android_version": d.android_version,
                    "sdk_version": d.sdk_version,
                    "ip_address": d.ip_address,
                    "last_seen": d.last_seen.isoformat()
                }
                for d in _devices.values()
            ],
            "total": len(_devices)
        }


@app.get("/devices/{device_id}")
async def get_device(device_id: str) -> Dict[str, Any]:
    """
    Get detailed device information.
    
    Args:
        device_id: Device identifier
        
    Returns:
        Device details
    """
    async with _lock:
        if device_id not in _devices:
            raise HTTPException(status_code=404, detail=f"Device {device_id} not found")
        
        device = _devices[device_id]
        
        return {
            "device_id": device.device_id,
            "status": device.status.value,
            "model": device.model,
            "manufacturer": device.manufacturer,
            "android_version": device.android_version,
            "sdk_version": device.sdk_version,
            "product": device.product,
            "ip_address": device.ip_address,
            "properties": device.properties,
            "last_seen": device.last_seen.isoformat()
        }


@app.post("/devices/connect")
async def connect_device(ip_address: str, port: int = 5555) -> Dict[str, Any]:
    """
    Connect to a device over network.
    
    Args:
        ip_address: Device IP address
        port: Device port (default 5555)
        
    Returns:
        Connection result
    """
    returncode, stdout, stderr = _run_adb_command(
        ["connect", f"{ip_address}:{port}"],
        timeout=10
    )
    
    if returncode != 0:
        raise HTTPException(status_code=500, detail=f"Connection failed: {stderr}")
    
    # Refresh device list
    await _refresh_devices()
    
    return {
        "success": "connected" in stdout.lower() or "already" in stdout.lower(),
        "message": stdout.strip(),
        "address": f"{ip_address}:{port}"
    }


@app.post("/devices/disconnect")
async def disconnect_device(device_id: str) -> Dict[str, Any]:
    """
    Disconnect from a network device.
    
    Args:
        device_id: Device identifier
        
    Returns:
        Disconnection result
    """
    returncode, stdout, stderr = _run_adb_command(
        ["disconnect", device_id],
        timeout=10
    )
    
    if returncode != 0:
        raise HTTPException(status_code=500, detail=f"Disconnection failed: {stderr}")
    
    # Refresh device list
    await _refresh_devices()
    
    return {
        "success": True,
        "message": stdout.strip(),
        "device_id": device_id
    }


@app.post("/shell")
async def execute_shell(cmd: ShellCommand) -> Dict[str, Any]:
    """
    Execute a shell command on a device.
    
    Args:
        cmd: Shell command configuration
        
    Returns:
        Command output
    """
    async with _lock:
        if cmd.device_id not in _devices:
            raise HTTPException(status_code=404, detail=f"Device {cmd.device_id} not found")
        
        if _devices[cmd.device_id].status != DeviceStatus.CONNECTED:
            raise HTTPException(status_code=400, detail=f"Device is not connected")
    
    # Build command
    shell_cmd = cmd.command
    if cmd.as_root:
        shell_cmd = f"su -c '{cmd.command}'"
    
    returncode, stdout, stderr = _run_adb_command(
        ["-s", cmd.device_id, "shell", shell_cmd],
        timeout=cmd.timeout
    )
    
    return {
        "device_id": cmd.device_id,
        "command": cmd.command,
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "executed_at": datetime.utcnow().isoformat()
    }


@app.post("/files/pull")
async def pull_file(transfer: FileTransfer) -> Dict[str, Any]:
    """
    Pull a file from device.
    
    Args:
        transfer: File transfer configuration
        
    Returns:
        Transfer result
    """
    async with _lock:
        if transfer.device_id not in _devices:
            raise HTTPException(status_code=404, detail=f"Device {transfer.device_id} not found")
    
    returncode, stdout, stderr = _run_adb_command(
        ["-s", transfer.device_id, "pull", transfer.source_path, transfer.destination_path],
        timeout=60
    )
    
    if returncode != 0:
        raise HTTPException(status_code=500, detail=f"Pull failed: {stderr}")
    
    return {
        "success": True,
        "device_id": transfer.device_id,
        "source": transfer.source_path,
        "destination": transfer.destination_path,
        "message": stdout.strip()
    }


@app.post("/files/push")
async def push_file(
    device_id: str,
    destination_path: str,
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Push a file to device.
    
    Args:
        device_id: Device identifier
        destination_path: Destination path on device
        file: File to upload
        
    Returns:
        Transfer result
    """
    async with _lock:
        if device_id not in _devices:
            raise HTTPException(status_code=404, detail=f"Device {device_id} not found")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        returncode, stdout, stderr = _run_adb_command(
            ["-s", device_id, "push", tmp_path, destination_path],
            timeout=60
        )
        
        if returncode != 0:
            raise HTTPException(status_code=500, detail=f"Push failed: {stderr}")
        
        return {
            "success": True,
            "device_id": device_id,
            "filename": file.filename,
            "destination": destination_path,
            "size": len(content),
            "message": stdout.strip()
        }
    finally:
        os.unlink(tmp_path)


@app.post("/screenshot")
async def capture_screenshot(device_id: str) -> Dict[str, Any]:
    """
    Capture a screenshot from device.
    
    Args:
        device_id: Device identifier
        
    Returns:
        Screenshot information
    """
    async with _lock:
        if device_id not in _devices:
            raise HTTPException(status_code=404, detail=f"Device {device_id} not found")
    
    # Create temporary file for screenshot
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        # Capture screenshot to device
        returncode, _, stderr = _run_adb_command(
            ["-s", device_id, "shell", "screencap", "-p", "/data/local/tmp/screenshot.png"],
            timeout=10
        )
        
        if returncode != 0:
            raise HTTPException(status_code=500, detail=f"Screenshot capture failed: {stderr}")
        
        # Pull screenshot
        returncode, _, stderr = _run_adb_command(
            ["-s", device_id, "pull", "/data/local/tmp/screenshot.png", tmp_path],
            timeout=10
        )
        
        if returncode != 0:
            raise HTTPException(status_code=500, detail=f"Screenshot pull failed: {stderr}")
        
        # Clean up device
        _run_adb_command(
            ["-s", device_id, "shell", "rm", "/data/local/tmp/screenshot.png"],
            timeout=5
        )
        
        # Read and encode screenshot
        with open(tmp_path, "rb") as f:
            import base64
            screenshot_data = base64.b64encode(f.read()).decode()
        
        return {
            "success": True,
            "device_id": device_id,
            "screenshot": screenshot_data,
            "format": "png",
            "captured_at": datetime.utcnow().isoformat()
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@app.post("/logcat/start")
async def start_logcat(config: LogcatConfig) -> Dict[str, Any]:
    """
    Start logcat streaming.
    
    Args:
        config: Logcat configuration
        
    Returns:
        Stream information
    """
    async with _lock:
        if config.device_id not in _devices:
            raise HTTPException(status_code=404, detail=f"Device {config.device_id} not found")
        
        # Stop existing logcat if any
        if config.device_id in _logcat_processes:
            _logcat_processes[config.device_id].terminate()
            del _logcat_processes[config.device_id]
    
    # Build logcat command
    cmd = ["adb", "-s", config.device_id, "logcat"]
    
    if config.buffer != "all":
        cmd.extend(["-b", config.buffer])
    
    if config.tag_filter:
        cmd.extend(["-s", f"{config.tag_filter}:{config.priority}"])
    
    if config.pid_filter:
        cmd.extend(["--pid", str(config.pid_filter)])
    
    # Start logcat process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    _logcat_processes[config.device_id] = process
    
    return {
        "success": True,
        "device_id": config.device_id,
        "buffer": config.buffer,
        "priority": config.priority,
        "pid": process.pid
    }


@app.post("/logcat/stop")
async def stop_logcat(device_id: str) -> Dict[str, Any]:
    """
    Stop logcat streaming.
    
    Args:
        device_id: Device identifier
        
    Returns:
        Stop result
    """
    async with _lock:
        if device_id not in _logcat_processes:
            raise HTTPException(status_code=404, detail=f"No active logcat for device {device_id}")
        
        process = _logcat_processes.pop(device_id)
        process.terminate()
    
    return {
        "success": True,
        "device_id": device_id,
        "message": "Logcat stopped"
    }


@app.get("/apps")
async def list_apps(device_id: str, include_system: bool = False) -> Dict[str, Any]:
    """
    List installed applications.
    
    Args:
        device_id: Device identifier
        include_system: Include system apps
        
    Returns:
        List of applications
    """
    async with _lock:
        if device_id not in _devices:
            raise HTTPException(status_code=404, detail=f"Device {device_id} not found")
    
    # Get package list
    flag = "-s" if not include_system else "-e"
    returncode, stdout, stderr = _run_adb_command(
        ["-s", device_id, "shell", "pm", "list", "packages", flag],
        timeout=30
    )
    
    if returncode != 0:
        raise HTTPException(status_code=500, detail=f"Failed to list apps: {stderr}")
    
    apps = []
    for line in stdout.strip().split("\n"):
        if line.startswith("package:"):
            package_name = line.replace("package:", "").strip()
            apps.append({
                "package_name": package_name,
                "is_system_app": not include_system
            })
    
    return {
        "device_id": device_id,
        "apps": apps,
        "total": len(apps)
    }


@app.post("/apps/install")
async def install_app(device_id: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Install an APK on device.
    
    Args:
        device_id: Device identifier
        file: APK file to install
        
    Returns:
        Installation result
    """
    async with _lock:
        if device_id not in _devices:
            raise HTTPException(status_code=404, detail=f"Device {device_id} not found")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(suffix=".apk", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    try:
        returncode, stdout, stderr = _run_adb_command(
            ["-s", device_id, "install", "-r", tmp_path],
            timeout=120
        )
        
        success = returncode == 0 and "Success" in stdout
        
        return {
            "success": success,
            "device_id": device_id,
            "filename": file.filename,
            "message": stdout.strip() if success else stderr.strip()
        }
    finally:
        os.unlink(tmp_path)


@app.post("/reboot")
async def reboot_device(device_id: str, mode: Literal["system", "recovery", "bootloader"] = "system") -> Dict[str, Any]:
    """
    Reboot a device.
    
    Args:
        device_id: Device identifier
        mode: Reboot mode
        
    Returns:
        Reboot result
    """
    async with _lock:
        if device_id not in _devices:
            raise HTTPException(status_code=404, detail=f"Device {device_id} not found")
    
    cmd = ["-s", device_id, "reboot"]
    if mode != "system":
        cmd.append(mode)
    
    returncode, stdout, stderr = _run_adb_command(cmd, timeout=10)
    
    if returncode != 0:
        raise HTTPException(status_code=500, detail=f"Reboot failed: {stderr}")
    
    return {
        "success": True,
        "device_id": device_id,
        "mode": mode,
        "message": "Reboot initiated"
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8433)
