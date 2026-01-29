# UFO Galaxy v5.0 - Main Makefile
.PHONY: help install start stop restart build clean test lint format logs health deploy

# Default target
help:
	@echo "🛸 UFO Galaxy v5.0 - Available Commands"
	@echo "========================================"
	@echo ""
	@echo "Setup:"
	@echo "  make install      - Install all dependencies"
	@echo "  make build        - Build all services"
	@echo ""
	@echo "Runtime:"
	@echo "  make start        - Start all services"
	@echo "  make stop         - Stop all services"
	@echo "  make restart      - Restart all services"
	@echo "  make logs         - View all logs"
	@echo ""
	@echo "Development:"
	@echo "  make test         - Run all tests"
	@echo "  make lint         - Run linters"
	@echo "  make format       - Format code"
	@echo ""
	@echo "Deployment:"
	@echo "  make deploy       - Deploy with Docker Compose"
	@echo "  make deploy-podman - Deploy with Podman"
	@echo "  make health       - Check health status"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Clean up containers and volumes"
	@echo "  make backup       - Backup data"
	@echo "  make update       - Update to latest version"

# Installation
install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "📦 Installing Dashboard dependencies..."
	cd dashboard && npm install
	@echo "✅ Installation complete!"

# Build
build:
	@echo "🔨 Building services..."
	cd deploy && make build

# Start all services
start:
	@echo "🚀 Starting UFO Galaxy v5.0..."
	@echo "Starting core nodes..."
	@cd nodes/Node_00_StateMachine && python node.py &
	@cd nodes/Node_01_OneAPI && python node.py &
	@cd nodes/Node_02_TaskEngine && python node.py &
	@cd nodes/Node_03_Keystore && python node.py &
	@cd nodes/Node_04_Router && python node.py &
	@cd nodes/Node_05_Auth && python node.py &
	@echo "Starting enhancements..."
	@cd enhancements/learning && python learning_node.py &
	@cd enhancements/multidevice && python device_coordinator.py &
	@cd enhancements/multidevice && python device_manager.py &
	@echo "✅ Services started!"
	@echo "Dashboard: http://localhost:3000"
	@echo "Gateway: http://localhost:8080"

# Stop all services
stop:
	@echo "🛑 Stopping UFO Galaxy v5.0..."
	@pkill -f "python node.py" || true
	@pkill -f "python learning_node.py" || true
	@pkill -f "python device_coordinator.py" || true
	@pkill -f "python device_manager.py" || true
	@echo "✅ Services stopped!"

# Restart
restart: stop start

# View logs
logs:
	@echo "📋 Viewing logs..."
	@tail -f logs/*.log 2>/dev/null || echo "No logs found"

# Run tests
test:
	@echo "🧪 Running tests..."
	pytest tests/ -v --cov=. --cov-report=html

# Lint code
lint:
	@echo "🔍 Running linters..."
	@flake8 nodes/ enhancements/ --max-line-length=120 || true
	@cd dashboard && npm run lint || true

# Format code
format:
	@echo "✨ Formatting code..."
	@black nodes/ enhancements/ --line-length=120 || true
	@cd dashboard && npm run format || true

# Deploy with Docker
deploy:
	@echo "🐳 Deploying with Docker Compose..."
	cd deploy && make deploy

# Deploy with Podman
deploy-podman:
	@echo "🦭 Deploying with Podman..."
	cd deploy && make deploy-podman

# Health check
health:
	@echo "🏥 Checking health status..."
	@curl -s http://localhost:8000/health || echo "❌ StateMachine not responding"
	@curl -s http://localhost:8070/health || echo "❌ LearningNode not responding"
	@curl -s http://localhost:8055/health || echo "❌ DeviceCoordinator not responding"

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	cd deploy && make clean

# Backup
backup:
	@echo "💾 Creating backup..."
	@mkdir -p backups
	@tar -czf backups/ufo-galaxy-backup-$(shell date +%Y%m%d-%H%M%S).tar.gz \
		data/ logs/ config/ 2>/dev/null || echo "Nothing to backup"
	@echo "✅ Backup created!"

# Update
update:
	@echo "🔄 Updating UFO Galaxy..."
	@git pull origin main
	@pip install -r requirements.txt --upgrade
	@cd dashboard && npm update
	@echo "✅ Update complete! Run 'make restart' to apply changes."
