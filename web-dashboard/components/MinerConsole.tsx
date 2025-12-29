"use client";

import { useEffect, useState } from "react";
import { useWebSocket } from "@/hooks/useWebSocket";
import TrainingGraph from "@/components/TrainingGraph";
import HardwareMonitor from "@/components/HardwareMonitor";
import LogStream from "@/components/LogStream";

interface MinerStats {
  gpu_temp: number;
  fan_speed: number;
  vram_usage: number;
  power_draw: number;
  hash_rate: number;
  uptime: number;
  timestamp: number;
}

interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy: number;
  gradient_norm: number;
  timestamp: number;
}

export default function MinerConsole() {
  const [minerStats, setMinerStats] = useState<MinerStats | null>(null);
  const [trainingMetrics, setTrainingMetrics] = useState<TrainingMetrics | null>(null);

  // Get WebSocket URL from environment variable
  const getWebSocketUrl = (topic: string): string => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL;
    if (wsUrl) {
      return `${wsUrl}/ws?topic=${topic}`;
    }
    
    // Only allow localhost fallback in development
    if (process.env.NODE_ENV === 'development') {
      return `ws://localhost:1317/ws?topic=${topic}`;
    }
    
    // Production: fail if not configured
    throw new Error('NEXT_PUBLIC_WS_URL environment variable must be set in production');
  };

  // WebSocket connections
  const {
    data: statsData,
    isConnected: statsConnected,
    error: statsError,
  } = useWebSocket<MinerStats>(getWebSocketUrl('miner_stats'));
  const {
    data: metricsData,
    isConnected: metricsConnected,
    error: metricsError,
  } = useWebSocket<TrainingMetrics>(getWebSocketUrl('training_metrics'));

  useEffect(() => {
    if (statsData) {
      setMinerStats(statsData);
    }
  }, [statsData]);

  useEffect(() => {
    if (metricsData) {
      setTrainingMetrics(metricsData);
    }
  }, [metricsData]);

  return (
    <div className="space-y-6">
      <div className="glass-panel p-6">
        <h2 className="text-2xl font-semibold text-slate-50 mb-2">
          Miner Console
        </h2>
        <p className="text-slate-400 text-sm">
          Real-time monitoring of your mining operations. Zero GPU usage interface.
        </p>
        <div className="mt-3 flex flex-wrap gap-3 text-xs">
          <span
            className={`inline-flex items-center rounded-full px-3 py-1 border ${
              statsConnected
                ? "border-emerald-400/60 bg-emerald-500/10 text-emerald-200"
                : "border-slate-600/70 bg-slate-900/80 text-slate-300"
            }`}
          >
            <span
              className={`mr-2 h-1.5 w-1.5 rounded-full ${
                statsConnected ? "bg-emerald-400" : "bg-slate-500"
              }`}
            />
            Stats stream {statsConnected ? "online" : "offline"}
          </span>
          <span
            className={`inline-flex items-center rounded-full px-3 py-1 border ${
              metricsConnected
                ? "border-cyan-400/60 bg-cyan-500/10 text-cyan-200"
                : "border-slate-600/70 bg-slate-900/80 text-slate-300"
            }`}
          >
            <span
              className={`mr-2 h-1.5 w-1.5 rounded-full ${
                metricsConnected ? "bg-cyan-400" : "bg-slate-500"
              }`}
            />
            Training metrics {metricsConnected ? "online" : "offline"}
          </span>
          {(statsError || metricsError) && (
            <span className="inline-flex items-center rounded-full px-3 py-1 border border-amber-400/60 bg-amber-500/10 text-amber-100">
              Miner WebSocket stream not reachable. Ensure node REST/WebSocket
              endpoint at{" "}
              <span className="font-mono ml-1">Blockchain API</span> is running.
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-panel p-6">
          <h3 className="text-lg font-semibold text-slate-100 mb-4">
            Training Progress
          </h3>
          <TrainingGraph metrics={trainingMetrics} />
        </div>

        <div className="glass-panel p-6">
          <h3 className="text-lg font-semibold text-slate-100 mb-4">
            Hardware Monitor
          </h3>
          <HardwareMonitor stats={minerStats} />
        </div>
      </div>

      <div className="glass-panel p-6">
        <h3 className="text-lg font-semibold text-slate-100 mb-4">
          Log Stream
        </h3>
        <LogStream />
      </div>
    </div>
  );
}

