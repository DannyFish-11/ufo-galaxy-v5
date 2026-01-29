# 🛸 UFO Galaxy v5.0

[![Version](https://img.shields.io/badge/version-5.0-blue.svg)](https://github.com/DannyFish-11/ufo-galaxy)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Nodes](https://img.shields.io/badge/nodes-102+-purple.svg)](#)

> **让 AI 拥有身体，让智能无处不在**

UFO Galaxy 是一个分布式 AI 节点操作系统，支持 102+ 功能节点的统一管理和协调。它融合了自主学习、多设备协同和现代化的部署架构。

## ✨ 核心特性

### 🤖 涌现式自主学习系统 (Node 70)
- **5阶段学习循环**: 观察 → 分析 → 实验 → 验证 → 部署
- **知识图谱**: 13种实体类型，38种关系类型
- **多源搜索**: Web、arXiv、GitHub 集成
- **涌现检测**: 4种类型（能力涌现、性能突破、新模式、协同效应）
- **服务端口**: 8070

### 📱 多设备协同系统 (Node 71)
- **设备发现**: <3秒发现新设备
- **心跳监控**: 5秒间隔健康检查
- **任务调度**: 500 TPS，6种负载均衡策略
- **故障转移**: 4种恢复机制
- **Android桥接**: 完整 AIP v2.0 协议支持
- **服务端口**: 8055/8056

### 🎯 102+ 功能节点

| 层级 | 节点数量 | 描述 |
|------|----------|------|
| Layer 0: KERNEL | 6 | 状态机、OneAPI、任务引擎、密钥库、路由、认证 |
| Layer 1: GATEWAY | 14 | 协议转换、量子调度、知识图谱、符号数学、智能体群 |
| Layer 2: TOOLS | 20 | 文件系统、Git、搜索、Slack、GitHub、数据库、TTS |
| Layer 3: PHYSICAL | 17 | ADB、Scrcpy、BLE、SSH、MQTT、CAN-bus、摄像头 |
| Layer 4: ENHANCEMENTS | 2+ | 自主学习、多设备协同 |

### 📊 现代化 Dashboard
- React 18 + TypeScript + Tailwind CSS
- 实时节点状态监控
- 网络拓扑可视化
- 性能图表和分析
- 深空主题 + 霓虹效果

## 🚀 快速开始

### 方式一：Docker Compose (推荐)

```bash
# 1. 克隆仓库
git clone https://github.com/DannyFish-11/ufo-galaxy.git
cd ufo-galaxy

# 2. 配置环境变量
cp deploy/.env.example deploy/.env
# 编辑 .env 填入 API 密钥

# 3. 一键部署
cd deploy
make deploy
```

### 方式二：Podman (Rootless)

```bash
cd deploy
make deploy-podman
```

### 方式三：本地开发

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动核心服务
python nodes/Node_00_StateMachine/node.py &
python nodes/Node_01_OneAPI/node.py &

# 3. 启动增强模块
python enhancements/learning/learning_node.py &
python enhancements/multidevice/device_coordinator.py &

# 4. 启动 Dashboard
cd dashboard
npm install
npm run dev
```

## 📋 系统要求

### 最低配置
- CPU: 4核
- 内存: 8GB
- 磁盘: 50GB SSD
- 网络: 公网IP或内网穿透

### 推荐配置
- CPU: 8核+
- 内存: 16GB+
- 磁盘: 100GB+ NVMe SSD
- GPU: NVIDIA CUDA (可选，用于AI加速)

### 软件依赖
- Python 3.11+
- Docker & Docker Compose 或 Podman
- Node.js 18+ (Dashboard)
- Redis 7+
- PostgreSQL 15+

## 🔌 端口规划

| 端口 | 服务 | 说明 |
|------|------|------|
| 8000 | StateMachine | 状态管理核心 |
| 8001 | OneAPI | AI API 网关 |
| 8004 | Router | 统一路由 |
| 8055 | DeviceCoordinator | 设备协调 |
| 8056 | DeviceManager | 设备管理 |
| 8070 | LearningNode | 自主学习 |
| 8080 | Galaxy Gateway | 统一网关 |
| 3000 | Dashboard | 管理界面 |
| 1883 | MQTT Broker | 设备通信 |
| 9090 | Prometheus | 监控 |
| 3001 | Grafana | 可视化 |

## 🏗️ 架构图

```
┌─────────────────────────────────────────────────────────┐
│                    UFO Galaxy 102-Core                  │
├─────────────────────────────────────────────────────────┤
│  Layer 0: KERNEL (6 节点)                                │
│  状态机 | OneAPI | 任务引擎 | 密钥库 | 路由 | 认证       │
├─────────────────────────────────────────────────────────┤
│  Layer 1: GATEWAY (14 节点)                              │
│  协议转换 | 量子调度 | 知识图谱 | 符号数学 | 智能体群    │
├─────────────────────────────────────────────────────────┤
│  Layer 2: TOOLS (20 节点)                                │
│  文件系统 | Git | 搜索 | Slack | GitHub | 数据库 | TTS   │
├─────────────────────────────────────────────────────────┤
│  Layer 3: PHYSICAL (17 节点) - 网络隔离                  │
│  ADB | Scrcpy | BLE | SSH | MQTT | CAN-bus | 摄像头      │
├─────────────────────────────────────────────────────────┤
│  ENHANCEMENTS (新增)                                     │
│  🤖 自主学习 (Node 70) | 📱 多设备协同 (Node 71)         │
│  📊 Dashboard (React)                                    │
└─────────────────────────────────────────────────────────┘
```

## 📚 API 文档

### 自主学习系统 (Port 8070)

```bash
# 提交学习数据
curl -X POST http://localhost:8070/learn \
  -H "Content-Type: application/json" \
  -d '{"data": "...", "source": "web"}'

# 查询知识
curl http://localhost:8070/knowledge/ai

# 获取模式
curl http://localhost:8070/patterns

# WebSocket 实时学习
wscat -c ws://localhost:8070/ws/learn
```

### 多设备协同 (Port 8055/8056)

```bash
# 注册设备
curl -X POST http://localhost:8056/devices/register \
  -H "Content-Type: application/json" \
  -d '{"device_id": "android-001", "capabilities": ["screen", "input"]}'

# 获取设备列表
curl http://localhost:8056/devices

# 提交任务
curl -X POST http://localhost:8056/tasks/submit \
  -H "Content-Type: application/json" \
  -d '{"task_type": "execute", "target_device": "android-001"}'
```

## 🧪 测试

```bash
# 运行所有测试
make test

# 运行单元测试
pytest tests/unit/ -v

# 运行集成测试
pytest tests/integration/ -v

# 运行 E2E 测试
pytest tests/e2e/ -v
```

## 📊 监控

访问以下地址查看监控数据：

- **Dashboard**: http://localhost:3000
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3001
- **Node Health**: http://localhost:8000/health

## 🤝 与微软 UFO³ Galaxy 融合

本项目与微软 UFO³ Galaxy 架构兼容：

| 特性 | 兼容性 |
|------|--------|
| AIP v2.0 协议 | ✅ 完全兼容 |
| TaskConstellation | ⚠️ 需适配层 |
| 跨设备协同 | ✅ 完全兼容 |
| 节点系统 | ⚠️ 需映射 |

融合可行性评级: **92/100 (高度可行)**

## 📖 文档

- [部署指南](deploy/README.md)
- [API 参考](docs/API.md)
- [架构设计](docs/ARCHITECTURE.md)
- [开发指南](docs/DEVELOPMENT.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件

## 🙏 致谢

- 微软 UFO³ Galaxy 团队 - 架构灵感
- FastAPI 团队 - Web 框架
- React 团队 - 前端框架

---

**UFO Galaxy - 让 AI 拥有身体，让智能无处不在** 🛸
