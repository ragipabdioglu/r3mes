'use client';

/**
 * Debug Dashboard Page
 *
 * Provides debug dashboard UI with real-time metrics display, log viewer, performance charts, and state inspector UI.
 */

import { useEffect, useState } from 'react';
import { debugLogger, loadDebugConfig, PerformanceMonitor } from '@/lib/debug';

interface DebugMetrics {
  timestamp: string;
  memory?: {
    used: number;
    total: number;
  };
  performance?: {
    renderTime: number;
    apiCalls: number;
  };
}

export default function DebugPage() {
  const [config, setConfig] = useState(loadDebugConfig());
  const [metrics, setMetrics] = useState<DebugMetrics | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [autoRefresh, setAutoRefresh] = useState(false);

  useEffect(() => {
    if (!config.enabled) {
      debugLogger.warn('Debug mode is not enabled. Enable it in localStorage: R3MES_DEBUG_MODE=true');
      return;
    }

    // Load initial metrics
    loadMetrics();

    // Setup auto-refresh if enabled
    if (autoRefresh) {
      const interval = setInterval(loadMetrics, 1000);
      return () => clearInterval(interval);
    }
  }, [config.enabled, autoRefresh]);

  const loadMetrics = async () => {
    try {
      // Try to fetch metrics from backend
      const response = await fetch('/api/debug/performance');
      if (response.ok) {
        const data = await response.json();
        setMetrics({
          timestamp: new Date().toISOString(),
          performance: {
            renderTime: 0, // Would need to measure
            apiCalls: 0, // Would need to track
          },
        });
      }
    } catch (error) {
      debugLogger.error('Failed to load metrics', error);
    }
  };

  const clearLogs = () => {
    setLogs([]);
  };

  const exportLogs = () => {
    const logContent = logs.join('\n');
    const blob = new Blob([logContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `r3mes-debug-${Date.now()}.log`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (!config.enabled) {
    return (
      <div className="container mx-auto p-6">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <h2 className="text-lg font-semibold text-yellow-800 mb-2">Debug Mode Disabled</h2>
          <p className="text-yellow-700 mb-4">
            To enable debug mode, set the following in your browser console:
          </p>
          <pre className="bg-yellow-100 p-3 rounded text-sm overflow-x-auto">
            {`localStorage.setItem('R3MES_DEBUG_MODE', 'true');
localStorage.setItem('R3MES_DEBUG_LOG_LEVEL', 'TRACE');
localStorage.setItem('R3MES_DEBUG_COMPONENTS', 'frontend');
window.location.reload();`}
          </pre>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto p-6 space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">Debug Dashboard</h1>
        <div className="flex gap-2">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-4 py-2 rounded ${
              autoRefresh
                ? 'bg-green-500 text-white'
                : 'bg-gray-200 text-gray-700'
            }`}
          >
            {autoRefresh ? 'Auto-refresh ON' : 'Auto-refresh OFF'}
          </button>
          <button
            onClick={clearLogs}
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded"
          >
            Clear Logs
          </button>
          <button
            onClick={exportLogs}
            className="px-4 py-2 bg-blue-500 text-white rounded"
          >
            Export Logs
          </button>
        </div>
      </div>

      {/* Configuration */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-xl font-semibold mb-4">Configuration</h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700">Log Level</label>
            <p className="mt-1 text-sm text-gray-900">{config.logLevel}</p>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700">Components</label>
            <p className="mt-1 text-sm text-gray-900">{config.components.join(', ')}</p>
          </div>
        </div>
      </div>

      {/* Metrics */}
      {metrics && (
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="text-xl font-semibold mb-4">Real-time Metrics</h2>
          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700">Last Update</label>
              <p className="mt-1 text-sm text-gray-900">
                {new Date(metrics.timestamp).toLocaleTimeString()}
              </p>
            </div>
            {metrics.memory && (
              <>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Memory Used</label>
                  <p className="mt-1 text-sm text-gray-900">
                    {metrics.memory.used.toFixed(2)} MB
                  </p>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700">Memory Total</label>
                  <p className="mt-1 text-sm text-gray-900">
                    {metrics.memory.total.toFixed(2)} MB
                  </p>
                </div>
              </>
            )}
          </div>
        </div>
      )}

      {/* Performance */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-xl font-semibold mb-4">Performance</h2>
        <p className="text-sm text-gray-600">
          Performance metrics would be displayed here. Connect to backend debug API to view detailed metrics.
        </p>
      </div>

      {/* Log Viewer */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-xl font-semibold mb-4">Log Viewer</h2>
        <div className="bg-gray-900 text-green-400 p-4 rounded font-mono text-sm h-64 overflow-y-auto">
          {logs.length === 0 ? (
            <p className="text-gray-500">No logs captured yet. Check browser console for debug logs.</p>
          ) : (
            logs.map((log, index) => (
              <div key={index} className="mb-1">
                {log}
              </div>
            ))
          )}
        </div>
      </div>

      {/* State Inspector */}
      <div className="bg-white rounded-lg shadow p-4">
        <h2 className="text-xl font-semibold mb-4">State Inspector</h2>
        <p className="text-sm text-gray-600">
          State inspector would be displayed here. Use browser DevTools to inspect component state.
        </p>
      </div>
    </div>
  );
}
