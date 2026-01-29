#!/usr/bin/env python3
"""
UFO Galaxy v5.0 - System Manager
统一系统管理工具
"""

import asyncio
import click
import json
import subprocess
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Node configuration
NODES = {
    "core": [
        ("Node_00_StateMachine", 8000),
        ("Node_01_OneAPI", 8001),
        ("Node_02_TaskEngine", 8002),
        ("Node_03_Keystore", 8003),
        ("Node_04_Router", 8004),
        ("Node_05_Auth", 8005),
    ],
    "adb": [("Node_33_ADB", 8433)],
    "mqtt": [("Node_41_MQTT", 8441)],
    "model": [("Node_58_ModelRouter", 8558)],
    "learning": [("enhancements/learning/learning_node", 8070)],
    "multidevice": [
        ("enhancements/multidevice/device_coordinator", 8055),
        ("enhancements/multidevice/device_manager", 8056),
    ],
}


@click.group()
def cli():
    """🛸 UFO Galaxy v5.0 System Manager"""
    pass


@cli.command()
@click.argument("service", default="all")
def start(service):
    """Start services (all, core, learning, multidevice, adb, mqtt, model)"""
    services = _get_services(service)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for name, port in services:
            task = progress.add_task(f"Starting {name}...", total=None)
            _start_node(name, port)
            progress.update(task, completed=True)
    
    console.print(Panel.fit("✅ Services started successfully!", style="green"))


@cli.command()
@click.argument("service", default="all")
def stop(service):
    """Stop services"""
    services = _get_services(service)
    
    for name, port in services:
        _stop_node(name)
    
    console.print(Panel.fit("🛑 Services stopped!", style="yellow"))


@cli.command()
def status():
    """Show system status"""
    table = Table(title="🛸 UFO Galaxy v5.0 - System Status")
    table.add_column("Node", style="cyan")
    table.add_column("Port", style="magenta")
    table.add_column("Status", style="green")
    table.add_column("Health", style="blue")
    
    all_services = []
    for category, services in NODES.items():
        all_services.extend(services)
    
    for name, port in all_services:
        status_icon = "🟢" if _is_port_open(port) else "🔴"
        health = _check_health(port) if _is_port_open(port) else "N/A"
        table.add_row(name, str(port), status_icon, health)
    
    console.print(table)


@cli.command()
def dashboard():
    """Launch Dashboard"""
    console.print("🚀 Starting Dashboard...")
    dashboard_path = Path(__file__).parent / "dashboard"
    subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=dashboard_path,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    console.print(Panel.fit("📊 Dashboard started at http://localhost:5173", style="blue"))


@cli.command()
def deploy():
    """Deploy with Docker Compose"""
    console.print("🐳 Deploying with Docker Compose...")
    deploy_path = Path(__file__).parent / "deploy"
    subprocess.run(["make", "deploy"], cwd=deploy_path)


@cli.command()
def logs():
    """View system logs"""
    logs_path = Path(__file__).parent / "logs"
    if logs_path.exists():
        subprocess.run(["tail", "-f"] + list(logs_path.glob("*.log")))
    else:
        console.print("❌ No logs found", style="red")


@cli.command()
def test():
    """Run tests"""
    console.print("🧪 Running tests...")
    subprocess.run(["pytest", "tests/", "-v"])


def _get_services(service):
    """Get list of services to manage"""
    if service == "all":
        all_services = []
        for category, services in NODES.items():
            all_services.extend(services)
        return all_services
    elif service in NODES:
        return NODES[service]
    else:
        console.print(f"❌ Unknown service: {service}", style="red")
        sys.exit(1)


def _start_node(name, port):
    """Start a single node"""
    node_path = Path(__file__).parent / name
    if not node_path.exists():
        node_path = Path(__file__).parent / "nodes" / name
    
    if node_path.exists():
        subprocess.Popen(
            [sys.executable, "node.py"],
            cwd=node_path,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def _stop_node(name):
    """Stop a single node"""
    subprocess.run(["pkill", "-f", f"{name}/node.py"], capture_output=True)


def _is_port_open(port):
    """Check if port is open"""
    import socket
    try:
        with socket.create_connection(("localhost", port), timeout=1):
            return True
    except:
        return False


def _check_health(port):
    """Check node health"""
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as resp:
            data = json.loads(resp.read())
            return data.get("status", "unknown")
    except:
        return "error"


if __name__ == "__main__":
    cli()
