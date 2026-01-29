import { useState, useMemo } from 'react';
import {
  ListTodo,
  Play,
  CheckCircle,
  XCircle,
  Clock,
  AlertCircle,
  Search,
  Filter,
  ChevronDown,
  ChevronUp,
  MoreHorizontal,
  Pause,
  RotateCcw,
  Trash2,
  Cpu,
  Brain,
  Database,
  RefreshCw,
  X
} from 'lucide-react';
import type { Task } from '../types';

interface TaskQueueProps {
  tasks: Task[];
}

// Task Type Icons
const taskTypeIcons = {
  inference: <Brain size={16} />,
  training: <Cpu size={16} />,
  data_processing: <Database size={16} />,
  model_sync: <RefreshCw size={16} />,
};

const taskTypeLabels = {
  inference: 'Inference',
  training: 'Training',
  data_processing: 'Data Processing',
  model_sync: 'Model Sync',
};

// Priority Colors
const priorityColors = {
  low: 'bg-slate-500/20 text-slate-400',
  medium: 'bg-blue-500/20 text-blue-400',
  high: 'bg-orange-500/20 text-orange-400',
  critical: 'bg-red-500/20 text-red-400',
};

// Status Colors
const statusColors = {
  pending: 'bg-yellow-500/20 text-yellow-400',
  running: 'bg-blue-500/20 text-blue-400',
  completed: 'bg-green-500/20 text-green-400',
  failed: 'bg-red-500/20 text-red-400',
};

// Status Icons
const statusIcons = {
  pending: <Clock size={14} />,
  running: <Play size={14} />,
  completed: <CheckCircle size={14} />,
  failed: <XCircle size={14} />,
};

// Task Detail Modal
interface TaskDetailModalProps {
  task: Task | null;
  onClose: () => void;
}

function TaskDetailModal({ task, onClose }: TaskDetailModalProps) {
  if (!task) return null;

  const formatDuration = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours}h ${minutes}m ${secs}s`;
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
    return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="glass-card w-full max-w-2xl max-h-[90vh] overflow-auto">
        <div className="p-6 border-b border-slate-700 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${statusColors[task.status]}`}>
              {statusIcons[task.status]}
            </div>
            <div>
              <h2 className="text-xl font-orbitron font-bold text-white">Task Details</h2>
              <p className="text-sm text-slate-500">{task.id}</p>
            </div>
          </div>
          <button 
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-slate-700 text-slate-400 transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        <div className="p-6 space-y-6">
          {/* Basic Info */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <p className="text-xs text-slate-500 mb-1">Type</p>
              <div className="flex items-center gap-2">
                {taskTypeIcons[task.type]}
                <span className="text-white font-medium">{taskTypeLabels[task.type]}</span>
              </div>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <p className="text-xs text-slate-500 mb-1">Status</p>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusColors[task.status]}`}>
                {task.status.charAt(0).toUpperCase() + task.status.slice(1)}
              </span>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <p className="text-xs text-slate-500 mb-1">Priority</p>
              <span className={`px-2 py-1 rounded-full text-xs font-medium ${priorityColors[task.priority]}`}>
                {task.priority.charAt(0).toUpperCase() + task.priority.slice(1)}
              </span>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <p className="text-xs text-slate-500 mb-1">Progress</p>
              <div className="flex items-center gap-2">
                <div className="flex-1 progress-bar h-2">
                  <div 
                    className="progress-fill progress-fill-blue h-2"
                    style={{ width: `${task.progress}%` }}
                  />
                </div>
                <span className="text-white text-sm">{task.progress.toFixed(0)}%</span>
              </div>
            </div>
          </div>

          {/* Node Assignment */}
          {task.nodeId && (
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <p className="text-xs text-slate-500 mb-2">Assigned Node</p>
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                  <Cpu size={18} className="text-blue-400" />
                </div>
                <div>
                  <p className="text-white font-medium">{task.nodeId}</p>
                  <p className="text-xs text-slate-500">Processing Node</p>
                </div>
              </div>
            </div>
          )}

          {/* Timing */}
          <div>
            <h3 className="text-sm font-orbitron text-slate-400 mb-3 uppercase tracking-wider">Timing Information</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-slate-800/50 p-4 rounded-lg">
                <p className="text-xs text-slate-500 mb-1">Created</p>
                <p className="text-white font-medium">{task.createdAt.toLocaleTimeString()}</p>
              </div>
              {task.startedAt && (
                <div className="bg-slate-800/50 p-4 rounded-lg">
                  <p className="text-xs text-slate-500 mb-1">Started</p>
                  <p className="text-white font-medium">{task.startedAt.toLocaleTimeString()}</p>
                </div>
              )}
              {task.completedAt && (
                <div className="bg-slate-800/50 p-4 rounded-lg">
                  <p className="text-xs text-slate-500 mb-1">Completed</p>
                  <p className="text-white font-medium">{task.completedAt.toLocaleTimeString()}</p>
                </div>
              )}
              <div className="bg-slate-800/50 p-4 rounded-lg">
                <p className="text-xs text-slate-500 mb-1">Estimated Duration</p>
                <p className="text-white font-medium">{formatDuration(task.estimatedDuration)}</p>
              </div>
              {task.actualDuration && (
                <div className="bg-slate-800/50 p-4 rounded-lg">
                  <p className="text-xs text-slate-500 mb-1">Actual Duration</p>
                  <p className="text-white font-medium">{formatDuration(task.actualDuration)}</p>
                </div>
              )}
            </div>
          </div>

          {/* Data Sizes */}
          <div>
            <h3 className="text-sm font-orbitron text-slate-400 mb-3 uppercase tracking-wider">Data Information</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-slate-800/50 p-4 rounded-lg flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-blue-500/20 flex items-center justify-center">
                  <Database size={18} className="text-blue-400" />
                </div>
                <div>
                  <p className="text-xs text-slate-500">Input Size</p>
                  <p className="text-white font-medium">{formatSize(task.inputSize)}</p>
                </div>
              </div>
              {task.outputSize && (
                <div className="bg-slate-800/50 p-4 rounded-lg flex items-center gap-3">
                  <div className="w-10 h-10 rounded-lg bg-green-500/20 flex items-center justify-center">
                    <CheckCircle size={18} className="text-green-400" />
                  </div>
                  <div>
                    <p className="text-xs text-slate-500">Output Size</p>
                    <p className="text-white font-medium">{formatSize(task.outputSize)}</p>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Error Message */}
          {task.errorMessage && (
            <div className="bg-red-500/10 border border-red-500/30 p-4 rounded-lg">
              <div className="flex items-center gap-2 mb-2">
                <AlertCircle size={16} className="text-red-400" />
                <p className="text-sm font-medium text-red-400">Error</p>
              </div>
              <p className="text-sm text-red-300">{task.errorMessage}</p>
            </div>
          )}

          {/* Actions */}
          <div className="flex gap-2">
            {task.status === 'pending' && (
              <button className="btn-primary flex items-center gap-2">
                <Play size={16} />
                Start Task
              </button>
            )}
            {task.status === 'running' && (
              <button className="btn-secondary flex items-center gap-2 text-orange-400 border-orange-500/30">
                <Pause size={16} />
                Pause Task
              </button>
            )}
            {(task.status === 'failed' || task.status === 'completed') && (
              <button className="btn-secondary flex items-center gap-2">
                <RotateCcw size={16} />
                Retry Task
              </button>
            )}
            <button className="btn-secondary flex items-center gap-2 text-red-400 border-red-500/30 ml-auto">
              <Trash2 size={16} />
              Delete Task
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

// Main TaskQueue Component
export default function TaskQueue({ tasks }: TaskQueueProps) {
  const [selectedTask, setSelectedTask] = useState<Task | null>(null);
  const [activeTab, setActiveTab] = useState<'all' | 'pending' | 'running' | 'completed' | 'failed'>('all');
  const [filterType, setFilterType] = useState<'all' | Task['type']>('all');
  const [filterPriority, setFilterPriority] = useState<'all' | Task['priority']>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState<'created' | 'priority' | 'progress'>('created');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Filter and sort tasks
  const filteredTasks = useMemo(() => {
    let result = [...tasks];

    // Filter by tab/status
    if (activeTab !== 'all') {
      result = result.filter(t => t.status === activeTab);
    }

    // Filter by type
    if (filterType !== 'all') {
      result = result.filter(t => t.type === filterType);
    }

    // Filter by priority
    if (filterPriority !== 'all') {
      result = result.filter(t => t.priority === filterPriority);
    }

    // Filter by search
    if (searchQuery) {
      result = result.filter(t => 
        t.id.toLowerCase().includes(searchQuery.toLowerCase())
      );
    }

    // Sort
    result.sort((a, b) => {
      let comparison = 0;
      switch (sortBy) {
        case 'created':
          comparison = a.createdAt.getTime() - b.createdAt.getTime();
          break;
        case 'priority':
          const priorityOrder = { critical: 4, high: 3, medium: 2, low: 1 };
          comparison = priorityOrder[a.priority] - priorityOrder[b.priority];
          break;
        case 'progress':
          comparison = a.progress - b.progress;
          break;
      }
      return sortOrder === 'asc' ? comparison : -comparison;
    });

    return result;
  }, [tasks, activeTab, filterType, filterPriority, searchQuery, sortBy, sortOrder]);

  // Calculate statistics
  const stats = useMemo(() => {
    const total = tasks.length;
    const pending = tasks.filter(t => t.status === 'pending').length;
    const running = tasks.filter(t => t.status === 'running').length;
    const completed = tasks.filter(t => t.status === 'completed').length;
    const failed = tasks.filter(t => t.status === 'failed').length;
    return { total, pending, running, completed, failed };
  }, [tasks]);

  const tabs = [
    { key: 'all', label: 'All Tasks', count: stats.total },
    { key: 'pending', label: 'Pending', count: stats.pending },
    { key: 'running', label: 'Running', count: stats.running },
    { key: 'completed', label: 'Completed', count: stats.completed },
    { key: 'failed', label: 'Failed', count: stats.failed },
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-orbitron font-bold text-white flex items-center gap-2">
            <ListTodo className="text-blue-400" />
            Task Queue
          </h2>
          <p className="text-slate-400">Manage and monitor distributed tasks</p>
        </div>
        <div className="flex items-center gap-2">
          <button className="btn-primary flex items-center gap-2">
            <Play size={16} />
            New Task
          </button>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key as any)}
            className={`glass-card p-4 text-left transition-all ${
              activeTab === tab.key 
                ? 'border-blue-500/50 bg-blue-500/10' 
                : 'hover:border-slate-600'
            }`}
          >
            <p className="text-2xl font-orbitron font-bold text-white">{tab.count}</p>
            <p className="text-xs text-slate-500">{tab.label}</p>
          </button>
        ))}
      </div>

      {/* Filters */}
      <div className="glass-card p-4">
        <div className="flex flex-col md:flex-row gap-4">
          {/* Search */}
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
            <input
              type="text"
              placeholder="Search tasks..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="input-field w-full pl-10"
            />
          </div>

          {/* Type Filter */}
          <div className="flex items-center gap-2">
            <Filter size={18} className="text-slate-500" />
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value as any)}
              className="input-field"
            >
              <option value="all">All Types</option>
              <option value="inference">Inference</option>
              <option value="training">Training</option>
              <option value="data_processing">Data Processing</option>
              <option value="model_sync">Model Sync</option>
            </select>
          </div>

          {/* Priority Filter */}
          <select
            value={filterPriority}
            onChange={(e) => setFilterPriority(e.target.value as any)}
            className="input-field"
          >
            <option value="all">All Priorities</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>

          {/* Sort */}
          <div className="flex items-center gap-2">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="input-field"
            >
              <option value="created">Sort by Created</option>
              <option value="priority">Sort by Priority</option>
              <option value="progress">Sort by Progress</option>
            </select>
            <button
              onClick={() => setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc')}
              className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-400 transition-colors"
            >
              {sortOrder === 'asc' ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
            </button>
          </div>
        </div>
      </div>

      {/* Task List */}
      <div className="glass-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="data-table">
            <thead>
              <tr>
                <th>Task ID</th>
                <th>Type</th>
                <th>Status</th>
                <th>Priority</th>
                <th>Progress</th>
                <th>Node</th>
                <th>Created</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {filteredTasks.map((task) => (
                <tr 
                  key={task.id} 
                  className="cursor-pointer hover:bg-slate-800/50"
                  onClick={() => setSelectedTask(task)}
                >
                  <td>
                    <span className="font-mono text-sm">{task.id}</span>
                  </td>
                  <td>
                    <div className="flex items-center gap-2">
                      <span className="text-slate-400">{taskTypeIcons[task.type]}</span>
                      <span className="text-sm">{taskTypeLabels[task.type]}</span>
                    </div>
                  </td>
                  <td>
                    <span className={`badge ${statusColors[task.status]}`}>
                      {statusIcons[task.status]}
                      <span className="ml-1">{task.status.charAt(0).toUpperCase() + task.status.slice(1)}</span>
                    </span>
                  </td>
                  <td>
                    <span className={`badge ${priorityColors[task.priority]}`}>
                      {task.priority.charAt(0).toUpperCase() + task.priority.slice(1)}
                    </span>
                  </td>
                  <td>
                    <div className="flex items-center gap-2">
                      <div className="w-20 progress-bar">
                        <div 
                          className="progress-fill progress-fill-blue"
                          style={{ width: `${task.progress}%` }}
                        />
                      </div>
                      <span className="text-xs text-slate-400">{task.progress.toFixed(0)}%</span>
                    </div>
                  </td>
                  <td>
                    <span className="text-sm text-slate-400">{task.nodeId || '-'}</span>
                  </td>
                  <td>
                    <span className="text-sm text-slate-400">
                      {task.createdAt.toLocaleTimeString()}
                    </span>
                  </td>
                  <td>
                    <button 
                      className="p-2 rounded-lg hover:bg-slate-700 text-slate-400 transition-colors"
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedTask(task);
                      }}
                    >
                      <MoreHorizontal size={16} />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Empty State */}
        {filteredTasks.length === 0 && (
          <div className="p-12 text-center">
            <ListTodo size={48} className="mx-auto text-slate-600 mb-4" />
            <h3 className="text-lg font-orbitron text-white mb-2">No Tasks Found</h3>
            <p className="text-slate-500">Try adjusting your filters or search query</p>
          </div>
        )}
      </div>

      {/* Task Detail Modal */}
      {selectedTask && (
        <TaskDetailModal 
          task={selectedTask} 
          onClose={() => setSelectedTask(null)} 
        />
      )}
    </div>
  );
}
