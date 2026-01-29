import { useState, useMemo } from 'react';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  BarChart3,
  Cpu,
  HardDrive,
  Wifi,
  Zap,
  TrendingUp,
  TrendingDown,
  Calendar,
  Download,
  RefreshCw
} from 'lucide-react';
import type { PerformanceMetrics } from '../types';

interface PerformanceChartsProps {
  metrics: PerformanceMetrics[];
}

// Custom Tooltip Component
const CustomTooltip = ({ active, payload, label }: any) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card p-3 border border-slate-700">
        <p className="text-slate-400 text-xs mb-2">{label}</p>
        {payload.map((entry: any, index: number) => (
          <p key={index} className="text-sm" style={{ color: entry.color }}>
            {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

// Metric Card Component
interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: React.ReactNode;
  color: string;
  trend?: number;
}

function MetricCard({ title, value, subtitle, icon, color, trend }: MetricCardProps) {
  return (
    <div className="glass-card p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-slate-400 text-sm mb-1">{title}</p>
          <h3 className="text-2xl font-orbitron font-bold text-white">{value}</h3>
          {subtitle && <p className="text-slate-500 text-xs mt-1">{subtitle}</p>}
        </div>
        <div className="p-2 rounded-lg" style={{ backgroundColor: `${color}20`, color }}>
          {icon}
        </div>
      </div>
      {trend !== undefined && (
        <div className="mt-3 flex items-center gap-2">
          {trend >= 0 ? (
            <TrendingUp size={14} className="text-green-400" />
          ) : (
            <TrendingDown size={14} className="text-red-400" />
          )}
          <span className={`text-xs ${trend >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {trend >= 0 ? '+' : ''}{trend.toFixed(1)}%
          </span>
          <span className="text-slate-500 text-xs">vs last period</span>
        </div>
      )}
    </div>
  );
}

// Main Performance Charts Component
export default function PerformanceCharts({ metrics }: PerformanceChartsProps) {
  const [timeRange, setTimeRange] = useState<'1h' | '6h' | '24h' | '7d'>('1h');
  const [selectedMetrics, setSelectedMetrics] = useState({
    cpu: true,
    memory: true,
    network: true,
    tasks: true,
  });

  // Process data for charts
  const chartData = useMemo(() => {
    return metrics.map((m, index) => ({
      time: new Date(m.timestamp).toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit',
        second: '2-digit'
      }),
      cpu: m.avgCpuUsage,
      memory: m.avgMemoryUsage,
      networkIn: m.totalNetworkIn / 1024 / 1024,
      networkOut: m.totalNetworkOut / 1024 / 1024,
      throughput: m.taskThroughput,
      pending: m.pendingTasks,
      running: m.runningTasks,
      completed: m.completedTasks,
      failed: m.failedMetrics,
    }));
  }, [metrics]);

  // Calculate summary statistics
  const stats = useMemo(() => {
    if (metrics.length === 0) return null;

    const latest = metrics[metrics.length - 1];
    const avgCpu = metrics.reduce((sum, m) => sum + m.avgCpuUsage, 0) / metrics.length;
    const avgMemory = metrics.reduce((sum, m) => sum + m.avgMemoryUsage, 0) / metrics.length;
    const avgThroughput = metrics.reduce((sum, m) => sum + m.taskThroughput, 0) / metrics.length;
    const peakCpu = Math.max(...metrics.map(m => m.avgCpuUsage));
    const peakMemory = Math.max(...metrics.map(m => m.avgMemoryUsage));

    return {
      latest,
      avgCpu,
      avgMemory,
      avgThroughput,
      peakCpu,
      peakMemory,
    };
  }, [metrics]);

  // Task distribution data for pie chart
  const taskDistribution = useMemo(() => {
    if (!stats) return [];
    return [
      { name: 'Running', value: stats.latest.runningTasks, color: '#3B82F6' },
      { name: 'Pending', value: stats.latest.pendingTasks, color: '#F59E0B' },
      { name: 'Completed', value: stats.latest.completedTasks % 100, color: '#10B981' },
      { name: 'Failed', value: stats.latest.failedTasks, color: '#EF4444' },
    ];
  }, [stats]);

  // Node status distribution
  const nodeDistribution = useMemo(() => {
    if (!stats) return [];
    return [
      { name: 'Online', value: stats.latest.onlineNodes, color: '#10B981' },
      { name: 'Offline', value: stats.latest.totalNodes - stats.latest.onlineNodes, color: '#EF4444' },
    ];
  }, [stats]);

  if (!stats) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="loading-spinner" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-orbitron font-bold text-white flex items-center gap-2">
            <BarChart3 className="text-blue-400" />
            Performance Metrics
          </h2>
          <p className="text-slate-400">Real-time system performance analytics</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value as any)}
            className="input-field"
          >
            <option value="1h">Last 1 Hour</option>
            <option value="6h">Last 6 Hours</option>
            <option value="24h">Last 24 Hours</option>
            <option value="7d">Last 7 Days</option>
          </select>
          <button className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 transition-colors">
            <RefreshCw size={18} />
          </button>
          <button className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 transition-colors">
            <Download size={18} />
          </button>
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard
          title="Avg CPU Usage"
          value={`${stats.avgCpu.toFixed(1)}%`}
          subtitle={`Peak: ${stats.peakCpu.toFixed(1)}%`}
          icon={<Cpu size={20} />}
          color="#3B82F6"
          trend={2.5}
        />
        <MetricCard
          title="Avg Memory Usage"
          value={`${stats.avgMemory.toFixed(1)}%`}
          subtitle={`Peak: ${stats.peakMemory.toFixed(1)}%`}
          icon={<HardDrive size={20} />}
          color="#8B5CF6"
          trend={-1.2}
        />
        <MetricCard
          title="Task Throughput"
          value={`${stats.avgThroughput.toFixed(0)}/s`}
          subtitle="Tasks per second"
          icon={<Zap size={20} />}
          color="#10B981"
          trend={5.8}
        />
        <MetricCard
          title="Network I/O"
          value={`${((stats.latest.totalNetworkIn + stats.latest.totalNetworkOut) / 1024 / 1024).toFixed(1)} MB/s`}
          subtitle="Total throughput"
          icon={<Wifi size={20} />}
          color="#06B6D4"
          trend={3.4}
        />
      </div>

      {/* Metric Toggles */}
      <div className="glass-card p-4">
        <p className="text-sm text-slate-400 mb-3">Display Metrics</p>
        <div className="flex flex-wrap gap-2">
          {[
            { key: 'cpu', label: 'CPU Usage', color: '#3B82F6' },
            { key: 'memory', label: 'Memory Usage', color: '#8B5CF6' },
            { key: 'network', label: 'Network I/O', color: '#06B6D4' },
            { key: 'tasks', label: 'Task Metrics', color: '#10B981' },
          ].map((metric) => (
            <button
              key={metric.key}
              onClick={() => setSelectedMetrics(prev => ({ ...prev, [metric.key]: !prev[metric.key as keyof typeof prev] }))}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors flex items-center gap-2 ${
                selectedMetrics[metric.key as keyof typeof selectedMetrics]
                  ? 'bg-slate-700 text-white'
                  : 'bg-slate-800 text-slate-500'
              }`}
            >
              <div 
                className="w-2 h-2 rounded-full" 
                style={{ backgroundColor: metric.color }}
              />
              {metric.label}
            </button>
          ))}
        </div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* CPU & Memory Usage Chart */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-orbitron font-semibold text-white mb-4 flex items-center gap-2">
            <Cpu size={18} className="text-blue-400" />
            CPU & Memory Usage
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="cpuGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                </linearGradient>
                <linearGradient id="memoryGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8B5CF6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#8B5CF6" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis 
                dataKey="time" 
                stroke="#64748B" 
                fontSize={10}
                tickLine={false}
              />
              <YAxis 
                stroke="#64748B" 
                fontSize={10}
                tickLine={false}
                domain={[0, 100]}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              {selectedMetrics.cpu && (
                <Area
                  type="monotone"
                  dataKey="cpu"
                  name="CPU Usage %"
                  stroke="#3B82F6"
                  fillOpacity={1}
                  fill="url(#cpuGradient)"
                  strokeWidth={2}
                />
              )}
              {selectedMetrics.memory && (
                <Area
                  type="monotone"
                  dataKey="memory"
                  name="Memory Usage %"
                  stroke="#8B5CF6"
                  fillOpacity={1}
                  fill="url(#memoryGradient)"
                  strokeWidth={2}
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Network Traffic Chart */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-orbitron font-semibold text-white mb-4 flex items-center gap-2">
            <Wifi size={18} className="text-cyan-400" />
            Network Traffic
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis 
                dataKey="time" 
                stroke="#64748B" 
                fontSize={10}
                tickLine={false}
              />
              <YAxis 
                stroke="#64748B" 
                fontSize={10}
                tickLine={false}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              {selectedMetrics.network && (
                <>
                  <Line
                    type="monotone"
                    dataKey="networkIn"
                    name="Inbound (MB/s)"
                    stroke="#06B6D4"
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="networkOut"
                    name="Outbound (MB/s)"
                    stroke="#EC4899"
                    strokeWidth={2}
                    dot={false}
                  />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>

        {/* Task Throughput Chart */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-orbitron font-semibold text-white mb-4 flex items-center gap-2">
            <Zap size={18} className="text-green-400" />
            Task Throughput
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={chartData}>
              <defs>
                <linearGradient id="throughputGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10B981" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#10B981" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis 
                dataKey="time" 
                stroke="#64748B" 
                fontSize={10}
                tickLine={false}
              />
              <YAxis 
                stroke="#64748B" 
                fontSize={10}
                tickLine={false}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              {selectedMetrics.tasks && (
                <Area
                  type="monotone"
                  dataKey="throughput"
                  name="Tasks/sec"
                  stroke="#10B981"
                  fillOpacity={1}
                  fill="url(#throughputGradient)"
                  strokeWidth={2}
                />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Task Queue Status */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-orbitron font-semibold text-white mb-4 flex items-center gap-2">
            <BarChart3 size={18} className="text-orange-400" />
            Task Queue Status
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1E293B" />
              <XAxis 
                dataKey="time" 
                stroke="#64748B" 
                fontSize={10}
                tickLine={false}
              />
              <YAxis 
                stroke="#64748B" 
                fontSize={10}
                tickLine={false}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              {selectedMetrics.tasks && (
                <>
                  <Bar dataKey="pending" name="Pending" fill="#F59E0B" />
                  <Bar dataKey="running" name="Running" fill="#3B82F6" />
                  <Bar dataKey="failed" name="Failed" fill="#EF4444" />
                </>
              )}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Distribution Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Task Distribution */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-orbitron font-semibold text-white mb-4">Task Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={taskDistribution}
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
              >
                {taskDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Node Status Distribution */}
        <div className="glass-card p-5">
          <h3 className="text-lg font-orbitron font-semibold text-white mb-4">Node Status</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={nodeDistribution}
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={80}
                paddingAngle={5}
                dataKey="value"
              >
                {nodeDistribution.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip content={<CustomTooltip />} />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
