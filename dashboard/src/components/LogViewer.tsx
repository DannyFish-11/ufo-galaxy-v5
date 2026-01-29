import { useState, useMemo, useRef, useEffect } from 'react';
import {
  ScrollText,
  Search,
  Filter,
  Download,
  Trash2,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  AlertCircle,
  Info,
  AlertTriangle,
  XCircle,
  Bug,
  CheckCircle,
  Copy,
  X
} from 'lucide-react';
import type { LogEntry } from '../types';

interface LogViewerProps {
  logs: LogEntry[];
}

// Log Level Configuration
const logLevels = {
  debug: { 
    color: 'text-slate-400', 
    bgColor: 'bg-slate-500/10',
    borderColor: 'border-slate-500/30',
    icon: <Bug size={14} />,
    label: 'DEBUG'
  },
  info: { 
    color: 'text-blue-400', 
    bgColor: 'bg-blue-500/10',
    borderColor: 'border-blue-500/30',
    icon: <Info size={14} />,
    label: 'INFO'
  },
  warn: { 
    color: 'text-orange-400', 
    bgColor: 'bg-orange-500/10',
    borderColor: 'border-orange-500/30',
    icon: <AlertTriangle size={14} />,
    label: 'WARN'
  },
  error: { 
    color: 'text-red-400', 
    bgColor: 'bg-red-500/10',
    borderColor: 'border-red-500/30',
    icon: <XCircle size={14} />,
    label: 'ERROR'
  },
  fatal: { 
    color: 'text-purple-400', 
    bgColor: 'bg-purple-500/10',
    borderColor: 'border-purple-500/30',
    icon: <AlertCircle size={14} />,
    label: 'FATAL'
  },
};

// Log Detail Modal
interface LogDetailModalProps {
  log: LogEntry | null;
  onClose: () => void;
}

function LogDetailModal({ log, onClose }: LogDetailModalProps) {
  const [copied, setCopied] = useState(false);

  if (!log) return null;

  const levelConfig = logLevels[log.level];

  const handleCopy = () => {
    const logText = `[${log.timestamp.toISOString()}] [${log.level.toUpperCase()}] [${log.source}] ${log.message}`;
    navigator.clipboard.writeText(logText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
      <div className="glass-card w-full max-w-3xl max-h-[90vh] overflow-auto">
        <div className="p-6 border-b border-slate-700 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className={`p-2 rounded-lg ${levelConfig.bgColor} ${levelConfig.color}`}>
              {levelConfig.icon}
            </div>
            <div>
              <h2 className="text-xl font-orbitron font-bold text-white">Log Entry</h2>
              <p className="text-sm text-slate-500">{log.id}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button 
              onClick={handleCopy}
              className="p-2 rounded-lg hover:bg-slate-700 text-slate-400 transition-colors"
              title="Copy to clipboard"
            >
              {copied ? <CheckCircle size={18} className="text-green-400" /> : <Copy size={18} />}
            </button>
            <button 
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-slate-700 text-slate-400 transition-colors"
            >
              <X size={20} />
            </button>
          </div>
        </div>

        <div className="p-6 space-y-6">
          {/* Basic Info */}
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <p className="text-xs text-slate-500 mb-1">Timestamp</p>
              <p className="text-white font-mono text-sm">{log.timestamp.toLocaleString()}</p>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <p className="text-xs text-slate-500 mb-1">Level</p>
              <span className={`inline-flex items-center gap-1 px-2 py-1 rounded text-xs font-medium ${levelConfig.bgColor} ${levelConfig.color}`}>
                {levelConfig.icon}
                {levelConfig.label}
              </span>
            </div>
            <div className="bg-slate-800/50 p-4 rounded-lg">
              <p className="text-xs text-slate-500 mb-1">Source</p>
              <p className="text-white font-medium">{log.source}</p>
            </div>
          </div>

          {/* Message */}
          <div>
            <h3 className="text-sm font-orbitron text-slate-400 mb-3 uppercase tracking-wider">Message</h3>
            <div className={`bg-slate-800/50 p-4 rounded-lg border ${levelConfig.borderColor}`}>
              <p className="text-white font-mono text-sm whitespace-pre-wrap">{log.message}</p>
            </div>
          </div>

          {/* Metadata */}
          {log.metadata && Object.keys(log.metadata).length > 0 && (
            <div>
              <h3 className="text-sm font-orbitron text-slate-400 mb-3 uppercase tracking-wider">Metadata</h3>
              <div className="bg-slate-800/50 p-4 rounded-lg">
                <pre className="text-sm font-mono text-slate-300 overflow-x-auto">
                  {JSON.stringify(log.metadata, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {/* Raw Log */}
          <div>
            <h3 className="text-sm font-orbitron text-slate-400 mb-3 uppercase tracking-wider">Raw Log Entry</h3>
            <div className="bg-slate-900 p-4 rounded-lg font-mono text-xs text-slate-400 overflow-x-auto">
              {`[${log.timestamp.toISOString()}] [${log.level.toUpperCase()}] [${log.source}] ${log.message}`}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// Main LogViewer Component
export default function LogViewer({ logs }: LogViewerProps) {
  const [selectedLog, setSelectedLog] = useState<LogEntry | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedLevels, setSelectedLevels] = useState<Set<LogEntry['level']>>(new Set(['debug', 'info', 'warn', 'error', 'fatal']));
  const [selectedSources, setSelectedSources] = useState<Set<string>>(new Set());
  const [autoScroll, setAutoScroll] = useState(true);
  const [expandedLogs, setExpandedLogs] = useState<Set<string>>(new Set());
  const logsEndRef = useRef<HTMLDivElement>(null);
  const logsContainerRef = useRef<HTMLDivElement>(null);

  // Get unique sources
  const sources = useMemo(() => {
    const sourceSet = new Set(logs.map(log => log.source));
    return Array.from(sourceSet).sort();
  }, [logs]);

  // Initialize all sources as selected
  useEffect(() => {
    if (sources.length > 0 && selectedSources.size === 0) {
      setSelectedSources(new Set(sources));
    }
  }, [sources]);

  // Filter logs
  const filteredLogs = useMemo(() => {
    return logs.filter(log => {
      // Filter by level
      if (!selectedLevels.has(log.level)) return false;
      
      // Filter by source
      if (!selectedSources.has(log.source)) return false;
      
      // Filter by search query
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        return (
          log.message.toLowerCase().includes(query) ||
          log.source.toLowerCase().includes(query) ||
          log.id.toLowerCase().includes(query)
        );
      }
      
      return true;
    });
  }, [logs, selectedLevels, selectedSources, searchQuery]);

  // Auto-scroll to bottom
  useEffect(() => {
    if (autoScroll && logsEndRef.current) {
      logsEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [filteredLogs, autoScroll]);

  // Calculate statistics
  const stats = useMemo(() => {
    const total = logs.length;
    const debug = logs.filter(l => l.level === 'debug').length;
    const info = logs.filter(l => l.level === 'info').length;
    const warn = logs.filter(l => l.level === 'warn').length;
    const error = logs.filter(l => l.level === 'error').length;
    const fatal = logs.filter(l => l.level === 'fatal').length;
    return { total, debug, info, warn, error, fatal };
  }, [logs]);

  // Toggle level selection
  const toggleLevel = (level: LogEntry['level']) => {
    setSelectedLevels(prev => {
      const next = new Set(prev);
      if (next.has(level)) {
        next.delete(level);
      } else {
        next.add(level);
      }
      return next;
    });
  };

  // Toggle source selection
  const toggleSource = (source: string) => {
    setSelectedSources(prev => {
      const next = new Set(prev);
      if (next.has(source)) {
        next.delete(source);
      } else {
        next.add(source);
      }
      return next;
    });
  };

  // Toggle log expansion
  const toggleExpand = (logId: string) => {
    setExpandedLogs(prev => {
      const next = new Set(prev);
      if (next.has(logId)) {
        next.delete(logId);
      } else {
        next.add(logId);
      }
      return next;
    }));
  };

  // Export logs
  const exportLogs = () => {
    const logText = filteredLogs.map(log => 
      `[${log.timestamp.toISOString()}] [${log.level.toUpperCase()}] [${log.source}] ${log.message}`
    ).join('\n');
    
    const blob = new Blob([logText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `ufo-galaxy-logs-${new Date().toISOString()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Clear logs
  const clearLogs = () => {
    // In a real app, this would call an API to clear logs
    console.log('Clear logs');
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-orbitron font-bold text-white flex items-center gap-2">
            <ScrollText className="text-blue-400" />
            System Logs
          </h2>
          <p className="text-slate-400">Real-time log stream and analysis</p>
        </div>
        <div className="flex items-center gap-2">
          <button 
            onClick={exportLogs}
            className="btn-secondary flex items-center gap-2"
          >
            <Download size={16} />
            Export
          </button>
          <button 
            onClick={clearLogs}
            className="btn-secondary flex items-center gap-2 text-red-400 border-red-500/30"
          >
            <Trash2 size={16} />
            Clear
          </button>
        </div>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
        <div className="glass-card p-3 text-center">
          <p className="text-xl font-orbitron font-bold text-white">{stats.total}</p>
          <p className="text-xs text-slate-500">Total</p>
        </div>
        <div className="glass-card p-3 text-center">
          <p className="text-xl font-orbitron font-bold text-slate-400">{stats.debug}</p>
          <p className="text-xs text-slate-500">Debug</p>
        </div>
        <div className="glass-card p-3 text-center">
          <p className="text-xl font-orbitron font-bold text-blue-400">{stats.info}</p>
          <p className="text-xs text-slate-500">Info</p>
        </div>
        <div className="glass-card p-3 text-center">
          <p className="text-xl font-orbitron font-bold text-orange-400">{stats.warn}</p>
          <p className="text-xs text-slate-500">Warn</p>
        </div>
        <div className="glass-card p-3 text-center">
          <p className="text-xl font-orbitron font-bold text-red-400">{stats.error}</p>
          <p className="text-xs text-slate-500">Error</p>
        </div>
        <div className="glass-card p-3 text-center">
          <p className="text-xl font-orbitron font-bold text-purple-400">{stats.fatal}</p>
          <p className="text-xs text-slate-500">Fatal</p>
        </div>
      </div>

      {/* Filters */}
      <div className="glass-card p-4 space-y-4">
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" size={18} />
          <input
            type="text"
            placeholder="Search logs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="input-field w-full pl-10"
          />
        </div>

        {/* Level Filters */}
        <div>
          <p className="text-sm text-slate-400 mb-2">Log Levels</p>
          <div className="flex flex-wrap gap-2">
            {(Object.keys(logLevels) as LogEntry['level'][]).map((level) => (
              <button
                key={level}
                onClick={() => toggleLevel(level)}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                  selectedLevels.has(level)
                    ? `${logLevels[level].bgColor} ${logLevels[level].color}`
                    : 'bg-slate-800 text-slate-500'
                }`}
              >
                {logLevels[level].icon}
                {logLevels[level].label}
              </button>
            ))}
          </div>
        </div>

        {/* Source Filters */}
        {sources.length > 0 && (
          <div>
            <p className="text-sm text-slate-400 mb-2">Sources</p>
            <div className="flex flex-wrap gap-2">
              {sources.map((source) => (
                <button
                  key={source}
                  onClick={() => toggleSource(source)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
                    selectedSources.has(source)
                      ? 'bg-blue-500/20 text-blue-400'
                      : 'bg-slate-800 text-slate-500'
                  }`}
                >
                  {source}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Log Stream */}
      <div className="glass-card overflow-hidden">
        <div className="p-3 border-b border-slate-700 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <span className="text-sm text-slate-400">
              Showing {filteredLogs.length} of {logs.length} logs
            </span>
          </div>
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-2 text-sm text-slate-400 cursor-pointer">
              <input
                type="checkbox"
                checked={autoScroll}
                onChange={(e) => setAutoScroll(e.target.checked)}
                className="rounded bg-slate-800 border-slate-600 text-blue-500 focus:ring-blue-500"
              />
              Auto-scroll
            </label>
          </div>
        </div>

        <div 
          ref={logsContainerRef}
          className="max-h-[500px] overflow-y-auto font-mono text-sm"
        >
          {filteredLogs.length === 0 ? (
            <div className="p-12 text-center">
              <ScrollText size={48} className="mx-auto text-slate-600 mb-4" />
              <h3 className="text-lg font-orbitron text-white mb-2">No Logs Found</h3>
              <p className="text-slate-500">Try adjusting your filters or search query</p>
            </div>
          ) : (
            <div className="divide-y divide-slate-800">
              {filteredLogs.map((log) => {
                const levelConfig = logLevels[log.level];
                const isExpanded = expandedLogs.has(log.id);
                
                return (
                  <div 
                    key={log.id}
                    className={`p-3 hover:bg-slate-800/50 cursor-pointer transition-colors ${levelConfig.bgColor}`}
                    onClick={() => setSelectedLog(log)}
                  >
                    <div className="flex items-start gap-3">
                      <div className={`mt-0.5 ${levelConfig.color}`}>
                        {levelConfig.icon}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 flex-wrap">
                          <span className="text-slate-500 text-xs">
                            {log.timestamp.toLocaleTimeString()}
                          </span>
                          <span className={`px-1.5 py-0.5 rounded text-xs font-medium ${levelConfig.bgColor} ${levelConfig.color}`}>
                            {levelConfig.label}
                          </span>
                          <span className="text-slate-400 text-xs">[{log.source}]</span>
                        </div>
                        <p className={`mt-1 ${levelConfig.color} ${isExpanded ? '' : 'truncate'}`}>
                          {log.message}
                        </p>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleExpand(log.id);
                        }}
                        className="p-1 rounded hover:bg-slate-700 text-slate-500"
                      >
                        {isExpanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                      </button>
                    </div>
                  </div>
                );
              })}
            </div>
          )}
          <div ref={logsEndRef} />
        </div>
      </div>

      {/* Log Detail Modal */}
      {selectedLog && (
        <LogDetailModal 
          log={selectedLog} 
          onClose={() => setSelectedLog(null)} 
        />
      )}
    </div>
  );
}
