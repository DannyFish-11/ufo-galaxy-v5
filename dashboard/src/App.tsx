import { useState, useCallback } from 'react';
import { BrowserRouter as Router, Routes, Route, NavLink } from 'react-router-dom';
import { 
  Cpu, 
  Network, 
  BarChart3, 
  ListTodo, 
  ScrollText, 
  Settings, 
  Zap,
  Activity,
  Globe,
  Server,
  Menu,
  X
} from 'lucide-react';
import type { Node, Task, LogEntry, PerformanceMetrics, NetworkTopology } from './types';
import { useSimulatedWebSocket } from './hooks/useWebSocket';

// Components
import NodeStatus from './components/NodeStatus';
import NetworkTopology from './components/NetworkTopology';
import PerformanceCharts from './components/PerformanceCharts';
import TaskQueue from './components/TaskQueue';
import LogViewer from './components/LogViewer';

// Navigation Item Component
interface NavItemProps {
  to: string;
  icon: React.ReactNode;
  label: string;
  badge?: number;
}

function NavItem({ to, icon, label, badge }: NavItemProps) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) => `
        flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-300
        ${isActive 
          ? 'bg-gradient-to-r from-blue-500/20 to-purple-500/20 border-l-2 border-blue-500 text-blue-400' 
          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
        }
      `}
    >
      {icon}
      <span className="font-medium">{label}</span>
      {badge !== undefined && badge > 0 && (
        <span className="ml-auto bg-blue-500 text-white text-xs px-2 py-0.5 rounded-full">
          {badge}
        </span>
      )}
    </NavLink>
  );
}

// Stats Card Component
interface StatsCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  color: 'blue' | 'purple' | 'green' | 'orange';
  trend?: number;
}

function StatsCard({ title, value, subtitle, icon, color, trend }: StatsCardProps) {
  const colorClasses = {
    blue: 'from-blue-500/20 to-blue-600/10 border-blue-500/30',
    purple: 'from-purple-500/20 to-purple-600/10 border-purple-500/30',
    green: 'from-green-500/20 to-green-600/10 border-green-500/30',
    orange: 'from-orange-500/20 to-orange-600/10 border-orange-500/30',
  };

  const iconColors = {
    blue: 'text-blue-400',
    purple: 'text-purple-400',
    green: 'text-green-400',
    orange: 'text-orange-400',
  };

  return (
    <div className={`glass-card p-5 bg-gradient-to-br ${colorClasses[color]} border`}>
      <div className="flex items-start justify-between">
        <div>
          <p className="text-slate-400 text-sm font-medium mb-1">{title}</p>
          <h3 className="text-2xl font-orbitron font-bold text-white">{value}</h3>
          {subtitle && <p className="text-slate-500 text-xs mt-1">{subtitle}</p>}
        </div>
        <div className={`p-3 rounded-lg bg-slate-800/50 ${iconColors[color]}`}>
          {icon}
        </div>
      </div>
      {trend !== undefined && (
        <div className="mt-3 flex items-center gap-2">
          <span className={`text-xs ${trend >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {trend >= 0 ? '+' : ''}{trend.toFixed(1)}%
          </span>
          <span className="text-slate-500 text-xs">vs last hour</span>
        </div>
      )}
    </div>
  );
}

// Connection Status Component
function ConnectionStatus({ isConnected }: { isConnected: boolean }) {
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-800/50 border border-slate-700">
      <div className={`w-2 h-2 rounded-full ${isConnected ? 'status-online pulse-glow' : 'status-offline'}`} />
      <span className={`text-xs font-medium ${isConnected ? 'text-green-400' : 'text-red-400'}`}>
        {isConnected ? 'LIVE' : 'OFFLINE'}
      </span>
    </div>
  );
}

// Header Component
function Header({ 
  isConnected, 
  onMenuToggle, 
  isMobileMenuOpen 
}: { 
  isConnected: boolean; 
  onMenuToggle: () => void;
  isMobileMenuOpen: boolean;
}) {
  return (
    <header className="h-16 glass-card border-b border-slate-800 flex items-center justify-between px-4 lg:px-6 sticky top-0 z-50">
      <div className="flex items-center gap-4">
        <button 
          onClick={onMenuToggle}
          className="lg:hidden p-2 rounded-lg hover:bg-slate-800 text-slate-400"
        >
          {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <Zap className="text-white" size={20} />
          </div>
          <div>
            <h1 className="font-orbitron font-bold text-lg text-white">UFO GALAXY</h1>
            <p className="text-xs text-slate-500">v5.0 Dashboard</p>
          </div>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <ConnectionStatus isConnected={isConnected} />
        <div className="hidden md:flex items-center gap-4 text-sm text-slate-400">
          <span className="flex items-center gap-1.5">
            <Globe size={14} className="text-blue-400" />
            Global Network
          </span>
          <span className="flex items-center gap-1.5">
            <Activity size={14} className="text-green-400" />
            Operational
          </span>
        </div>
        <button className="p-2 rounded-lg hover:bg-slate-800 text-slate-400 transition-colors">
          <Settings size={18} />
        </button>
      </div>
    </header>
  );
}

// Sidebar Component
function Sidebar({ 
  isOpen, 
  onClose, 
  pendingTasks 
}: { 
  isOpen: boolean; 
  onClose: () => void;
  pendingTasks: number;
}) {
  return (
    <>
      {/* Mobile Overlay */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}
      
      {/* Sidebar */}
      <aside className={`
        fixed lg:static inset-y-0 left-0 z-50 w-64 glass-card border-r border-slate-800
        transform transition-transform duration-300 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        <div className="p-4 space-y-2">
          <div className="px-4 py-2 mb-4">
            <p className="text-xs font-orbitron text-slate-500 uppercase tracking-wider">Main Menu</p>
          </div>
          
          <NavItem 
            to="/" 
            icon={<Cpu size={18} />} 
            label="Node Status" 
          />
          <NavItem 
            to="/network" 
            icon={<Network size={18} />} 
            label="Network Topology" 
          />
          <NavItem 
            to="/performance" 
            icon={<BarChart3 size={18} />} 
            label="Performance" 
          />
          <NavItem 
            to="/tasks" 
            icon={<ListTodo size={18} />} 
            label="Task Queue" 
            badge={pendingTasks}
          />
          <NavItem 
            to="/logs" 
            icon={<ScrollText size={18} />} 
            label="System Logs" 
          />
        </div>

        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-slate-800">
          <div className="glass-card p-3 rounded-lg">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                <Server size={14} className="text-white" />
              </div>
              <div>
                <p className="text-sm font-medium text-white">System Status</p>
                <p className="text-xs text-green-400">All Systems Normal</p>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}

// Dashboard Overview Component
function DashboardOverview({ metrics }: { metrics: PerformanceMetrics[] }) {
  const latestMetrics = metrics[metrics.length - 1];
  
  if (!latestMetrics) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="glass-card p-5 animate-pulse">
            <div className="h-4 bg-slate-700 rounded w-1/2 mb-2"></div>
            <div className="h-8 bg-slate-700 rounded w-3/4"></div>
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      <StatsCard
        title="Active Nodes"
        value={`${latestMetrics.onlineNodes}/${latestMetrics.totalNodes}`}
        subtitle={`${((latestMetrics.onlineNodes / latestMetrics.totalNodes) * 100).toFixed(1)}% online`}
        icon={<Server size={20} />}
        color="blue"
        trend={2.5}
      />
      <StatsCard
        title="Task Throughput"
        value={`${latestMetrics.taskThroughput.toFixed(0)}/s`}
        subtitle="Tasks per second"
        icon={<Zap size={20} />}
        color="purple"
        trend={5.2}
      />
      <StatsCard
        title="Running Tasks"
        value={latestMetrics.runningTasks}
        subtitle={`${latestMetrics.pendingTasks} pending`}
        icon={<Activity size={20} />}
        color="green"
        trend={-1.3}
      />
      <StatsCard
        title="Network Traffic"
        value={`${(latestMetrics.totalNetworkIn / 1024 / 1024).toFixed(1)} MB/s`}
        subtitle="Total throughput"
        icon={<Globe size={20} />}
        color="orange"
        trend={8.7}
      />
    </div>
  );
}

// Main App Component
function App() {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [nodes, setNodes] = useState<Node[]>([]);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [metrics, setMetrics] = useState<PerformanceMetrics[]>([]);
  const [topology, setTopology] = useState<NetworkTopology>({ nodes: [], connections: [] });

  // WebSocket handlers
  const handleNodeUpdate = useCallback((node: Node) => {
    setNodes(prev => {
      const index = prev.findIndex(n => n.id === node.id);
      if (index >= 0) {
        const updated = [...prev];
        updated[index] = node;
        return updated;
      }
      return [...prev, node];
    });
  }, []);

  const handleTaskUpdate = useCallback((task: Task) => {
    setTasks(prev => {
      const index = prev.findIndex(t => t.id === task.id);
      if (index >= 0) {
        const updated = [...prev];
        updated[index] = task;
        return updated;
      }
      return [task, ...prev].slice(0, 100);
    });
  }, []);

  const handleMetricsUpdate = useCallback((newMetrics: PerformanceMetrics) => {
    setMetrics(prev => [...prev.slice(-50), newMetrics]);
  }, []);

  const handleLogEntry = useCallback((log: LogEntry) => {
    setLogs(prev => [log, ...prev].slice(0, 500));
  }, []);

  const handleTopologyUpdate = useCallback((newTopology: NetworkTopology) => {
    setTopology(newTopology);
  }, []);

  // Use simulated WebSocket for demo
  const { isConnected } = useSimulatedWebSocket({
    onNodeUpdate: handleNodeUpdate,
    onTaskUpdate: handleTaskUpdate,
    onMetricsUpdate: handleMetricsUpdate,
    onLogEntry: handleLogEntry,
    onTopologyUpdate: handleTopologyUpdate,
  });

  // Generate initial mock nodes
  useState(() => {
    const mockNodes: Node[] = Array.from({ length: 102 }, (_, i) => ({
      id: `node-${i + 1}`,
      name: `Node ${i + 1}`,
      status: Math.random() > 0.15 ? 'online' : (Math.random() > 0.5 ? 'busy' : 'idle'),
      cpuUsage: Math.random() * 100,
      memoryUsage: Math.random() * 100,
      networkIn: Math.random() * 1000,
      networkOut: Math.random() * 1000,
      tasksCompleted: Math.floor(Math.random() * 10000),
      uptime: Math.random() * 86400,
      lastSeen: new Date(),
      location: ['US-East', 'US-West', 'EU-Central', 'Asia-Pacific'][Math.floor(Math.random() * 4)],
      deviceType: ['desktop', 'mobile', 'server', 'embedded'][Math.floor(Math.random() * 4)] as Node['deviceType'],
      capabilities: ['cpu', 'gpu', 'tpu'].slice(0, Math.floor(Math.random() * 3) + 1),
    }));
    setNodes(mockNodes);
  });

  const pendingTasksCount = tasks.filter(t => t.status === 'pending').length;

  return (
    <Router>
      <div className="min-h-screen bg-[#0A0E17]">
        {/* Background Effects */}
        <div className="space-bg" />
        <div className="stars" />
        
        <div className="flex">
          <Sidebar 
            isOpen={mobileMenuOpen} 
            onClose={() => setMobileMenuOpen(false)}
            pendingTasks={pendingTasksCount}
          />
          
          <div className="flex-1 flex flex-col min-h-screen">
            <Header 
              isConnected={isConnected} 
              onMenuToggle={() => setMobileMenuOpen(!mobileMenuOpen)}
              isMobileMenuOpen={mobileMenuOpen}
            />
            
            <main className="flex-1 p-4 lg:p-6 overflow-auto">
              <Routes>
                <Route 
                  path="/" 
                  element={
                    <div className="space-y-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <h2 className="text-2xl font-orbitron font-bold text-white">Dashboard Overview</h2>
                          <p className="text-slate-400">Real-time system monitoring and analytics</p>
                        </div>
                      </div>
                      <DashboardOverview metrics={metrics} />
                      <NodeStatus nodes={nodes} />
                    </div>
                  } 
                />
                <Route 
                  path="/network" 
                  element={<NetworkTopology nodes={nodes} topology={topology} />} 
                />
                <Route 
                  path="/performance" 
                  element={<PerformanceCharts metrics={metrics} />} 
                />
                <Route 
                  path="/tasks" 
                  element={<TaskQueue tasks={tasks} />} 
                />
                <Route 
                  path="/logs" 
                  element={<LogViewer logs={logs} />} 
                />
              </Routes>
            </main>
            
            <footer className="h-12 glass-card border-t border-slate-800 flex items-center justify-between px-4 lg:px-6">
              <p className="text-xs text-slate-500">
                UFO Galaxy v5.0 - Distributed AI Computing Network
              </p>
              <p className="text-xs text-slate-500">
                {isConnected ? 'Connected to WebSocket' : 'Disconnected'}
              </p>
            </footer>
          </div>
        </div>
      </div>
    </Router>
  );
}

export default App;
