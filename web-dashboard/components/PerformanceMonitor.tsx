"use client";

import React, { useEffect, useState } from 'react';
import { performanceMonitor, PerformanceMetrics, MemoryManager } from '@/lib/performance';
import { logger } from '@/lib/logger';

interface PerformanceMonitorProps {
  showInProduction?: boolean;
  position?: 'top-left' | 'top-right' | 'bottom-left' | 'bottom-right';
  onMetricsUpdate?: (metrics: Partial<PerformanceMetrics>) => void;
}

export default function PerformanceMonitor({
  showInProduction = false,
  position = 'bottom-right',
  onMetricsUpdate,
}: PerformanceMonitorProps) {
  const [metrics, setMetrics] = useState<Partial<PerformanceMetrics>>({});
  const [memoryUsage, setMemoryUsage] = useState<ReturnType<typeof MemoryManager.getMemoryUsage>>(null);
  const [isVisible, setIsVisible] = useState(false);
  const [budget, setBudget] = useState<{ passed: boolean; violations: string[] }>({ passed: true, violations: [] });

  // Only show in development or when explicitly enabled
  const shouldShow = process.env.NODE_ENV === 'development' || showInProduction;

  useEffect(() => {
    if (!shouldShow) return;

    const updateMetrics = () => {
      const currentMetrics = performanceMonitor.getMetrics();
      const currentMemory = MemoryManager.getMemoryUsage();
      const currentBudget = performanceMonitor.checkPerformanceBudget();

      setMetrics(currentMetrics);
      setMemoryUsage(currentMemory);
      setBudget(currentBudget);
      
      onMetricsUpdate?.(currentMetrics);
    };

    // Initial update
    updateMetrics();

    // Update every 2 seconds
    const interval = setInterval(updateMetrics, 2000);

    return () => clearInterval(interval);
  }, [shouldShow, onMetricsUpdate]);

  if (!shouldShow) return null;

  const positionClasses = {
    'top-left': 'top-4 left-4',
    'top-right': 'top-4 right-4',
    'bottom-left': 'bottom-4 left-4',
    'bottom-right': 'bottom-4 right-4',
  };

  const formatMetric = (value: number | undefined, unit: string = 'ms') => {
    if (value === undefined) return 'N/A';
    return `${Math.round(value)}${unit}`;
  };

  const getMetricColor = (metric: keyof PerformanceMetrics, value: number | undefined) => {
    if (value === undefined) return 'text-gray-400';
    
    const thresholds = {
      lcp: { good: 2500, poor: 4000 },
      fid: { good: 100, poor: 300 },
      cls: { good: 0.1, poor: 0.25 },
      fcp: { good: 1800, poor: 3000 },
      ttfb: { good: 600, poor: 1000 },
    };

    const threshold = thresholds[metric];
    if (!threshold) return 'text-gray-400';

    if (value <= threshold.good) return 'text-green-400';
    if (value <= threshold.poor) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className={`fixed ${positionClasses[position]} z-50`}>
      {/* Toggle Button */}
      <button
        onClick={() => setIsVisible(!isVisible)}
        className="mb-2 px-3 py-1 bg-black/80 text-white text-xs rounded-lg hover:bg-black/90 transition-colors"
        title="Toggle Performance Monitor"
      >
        {isVisible ? '📊 Hide' : '📊 Perf'}
      </button>

      {/* Performance Panel */}
      {isVisible && (
        <div className="bg-black/90 text-white p-4 rounded-lg shadow-lg min-w-[280px] text-xs font-mono">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold text-sm">Performance Monitor</h3>
            <div className={`w-2 h-2 rounded-full ${budget.passed ? 'bg-green-400' : 'bg-red-400'}`} />
          </div>

          {/* Core Web Vitals */}
          <div className="space-y-2 mb-4">
            <h4 className="text-yellow-400 font-semibold">Core Web Vitals</h4>
            
            <div className="flex justify-between">
              <span>LCP:</span>
              <span className={getMetricColor('lcp', metrics.lcp)}>
                {formatMetric(metrics.lcp)}
              </span>
            </div>
            
            <div className="flex justify-between">
              <span>FID:</span>
              <span className={getMetricColor('fid', metrics.fid)}>
                {formatMetric(metrics.fid)}
              </span>
            </div>
            
            <div className="flex justify-between">
              <span>CLS:</span>
              <span className={getMetricColor('cls', metrics.cls)}>
                {formatMetric(metrics.cls, '')}
              </span>
            </div>
          </div>

          {/* Other Metrics */}
          <div className="space-y-2 mb-4">
            <h4 className="text-blue-400 font-semibold">Other Metrics</h4>
            
            <div className="flex justify-between">
              <span>FCP:</span>
              <span className={getMetricColor('fcp', metrics.fcp)}>
                {formatMetric(metrics.fcp)}
              </span>
            </div>
            
            <div className="flex justify-between">
              <span>TTFB:</span>
              <span className={getMetricColor('ttfb', metrics.ttfb)}>
                {formatMetric(metrics.ttfb)}
              </span>
            </div>
          </div>

          {/* Memory Usage */}
          {memoryUsage && (
            <div className="space-y-2 mb-4">
              <h4 className="text-purple-400 font-semibold">Memory</h4>
              
              <div className="flex justify-between">
                <span>Used:</span>
                <span className="text-gray-300">
                  {Math.round(memoryUsage.used / 1024 / 1024)}MB
                </span>
              </div>
              
              <div className="flex justify-between">
                <span>Total:</span>
                <span className="text-gray-300">
                  {Math.round(memoryUsage.total / 1024 / 1024)}MB
                </span>
              </div>
              
              <div className="flex justify-between">
                <span>Usage:</span>
                <span className={memoryUsage.percentage > 80 ? 'text-red-400' : 'text-green-400'}>
                  {Math.round(memoryUsage.percentage)}%
                </span>
              </div>
            </div>
          )}

          {/* Budget Violations */}
          {budget.violations.length > 0 && (
            <div className="space-y-1">
              <h4 className="text-red-400 font-semibold">Issues</h4>
              {budget.violations.map((violation, index) => (
                <div key={index} className="text-red-300 text-xs">
                  • {violation}
                </div>
              ))}
            </div>
          )}

          {/* Actions */}
          <div className="mt-4 pt-3 border-t border-gray-600 flex gap-2">
            <button
              onClick={() => performanceMonitor.sendMetrics()}
              className="px-2 py-1 bg-blue-600 hover:bg-blue-700 rounded text-xs transition-colors"
              title="Send metrics to analytics"
            >
              Send
            </button>
            
            <button
              onClick={() => {
                logger.debug('Performance Metrics:', metrics);
                logger.debug('Memory Usage:', memoryUsage);
                logger.debug('Budget Check:', budget);
              }}
              className="px-2 py-1 bg-gray-600 hover:bg-gray-700 rounded text-xs transition-colors"
              title="Log metrics to console"
            >
              Log
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
