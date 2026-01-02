import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/tauri";
import "./MiningDashboard.css";

interface MiningStats {
  hashrate: number; // gradients/hour
  loss: number;
  loss_trend: "decreasing" | "increasing" | "stable";
  estimated_earnings_per_day: number; // REMES
  current_balance: number; // REMES
  gpu_temp: number; // Celsius
  gpu_temp_status: "normal" | "high" | "critical";
  vram_usage_mb: number;
  vram_total_mb: number;
  training_epoch: number;
  gradient_norm: number;
  uptime_seconds: number;
}

export default function MiningDashboard() {
  const [stats, setStats] = useState<MiningStats | null>(null);
  const [, setLoading] = useState(false);

  useEffect(() => {
    const interval = setInterval(async () => {
      try {
        const miningStats = await invoke<MiningStats>("get_mining_stats");
        setStats(miningStats);
      } catch (error) {
        console.error("Failed to fetch mining stats:", error);
      } finally {
        setLoading(false);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(interval);
  }, []);

  const formatUptime = (seconds: number) => {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    
    if (days > 0) return `${days}d ${hours}h ${minutes}m`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  const getTempColor = (status: string) => {
    switch (status) {
      case "normal":
        return "#22c55e";
      case "high":
        return "#f59e0b";
      case "critical":
        return "#ef4444";
      default:
        return "#94a3b8";
    }
  };

  if (!stats) {
    return (
      <div className="mining-dashboard">
        <h3 className="dashboard-title">Mining Dashboard</h3>
        <div className="loading">Loading mining stats...</div>
      </div>
    );
  }

  return (
    <div className="mining-dashboard">
      <h3 className="dashboard-title">Mining Dashboard</h3>
      
      <div className="stats-grid">
        {/* Hashrate/Loss Widget */}
        <div className="stat-widget">
          <div className="widget-header">
            <span className="widget-icon">🔥</span>
            <span className="widget-label">Hashrate</span>
          </div>
          <div className="widget-value">{stats.hashrate.toLocaleString()}</div>
          <div className="widget-unit">gradients/hour</div>
          <div className="widget-footer">
            <span className="loss-indicator">
              📉 Loss: {stats.loss.toFixed(4)}
              {stats.loss_trend === "decreasing" && (
                <span className="trend-down"> ↓</span>
              )}
              {stats.loss_trend === "increasing" && (
                <span className="trend-up"> ↑</span>
              )}
            </span>
          </div>
        </div>

        {/* Earnings Widget */}
        <div className="stat-widget">
          <div className="widget-header">
            <span className="widget-icon">💰</span>
            <span className="widget-label">Estimated Earnings</span>
          </div>
          <div className="widget-value">{stats.estimated_earnings_per_day.toFixed(2)}</div>
          <div className="widget-unit">REMES/day</div>
          <div className="widget-footer">
            <span className="balance">Balance: {stats.current_balance.toFixed(2)} REMES</span>
          </div>
        </div>

        {/* GPU Temperature Widget */}
        <div className="stat-widget">
          <div className="widget-header">
            <span className="widget-icon">🌡️</span>
            <span className="widget-label">GPU Temperature</span>
          </div>
          <div
            className="widget-value"
            style={{ color: getTempColor(stats.gpu_temp_status) }}
          >
            {stats.gpu_temp}°C
          </div>
          <div className="widget-unit">
            {stats.gpu_temp_status === "normal" && "Normal"}
            {stats.gpu_temp_status === "high" && "⚠️ High"}
            {stats.gpu_temp_status === "critical" && "🔴 Critical"}
          </div>
          <div className="widget-footer">
            {stats.gpu_temp_status === "critical" && (
              <span className="warning-text">
                Mining should be stopped!
              </span>
            )}
          </div>
        </div>

        {/* VRAM Usage Widget */}
        <div className="stat-widget">
          <div className="widget-header">
            <span className="widget-icon">💾</span>
            <span className="widget-label">VRAM Usage</span>
          </div>
          <div className="widget-value">
            {(stats.vram_usage_mb / 1024).toFixed(1)} / {(stats.vram_total_mb / 1024).toFixed(1)}
          </div>
          <div className="widget-unit">GB</div>
          <div className="widget-footer">
            <div className="progress-bar">
              <div
                className="progress-fill"
                style={{
                  width: `${(stats.vram_usage_mb / stats.vram_total_mb) * 100}%`,
                }}
              />
            </div>
            <span className="progress-text">
              {((stats.vram_usage_mb / stats.vram_total_mb) * 100).toFixed(0)}%
            </span>
          </div>
        </div>

        {/* Training Epoch Widget */}
        <div className="stat-widget">
          <div className="widget-header">
            <span className="widget-icon">📊</span>
            <span className="widget-label">Training Epoch</span>
          </div>
          <div className="widget-value">{stats.training_epoch}</div>
          <div className="widget-unit">epoch</div>
        </div>

        {/* Gradient Norm Widget */}
        <div className="stat-widget">
          <div className="widget-header">
            <span className="widget-icon">📐</span>
            <span className="widget-label">Gradient Norm</span>
          </div>
          <div className="widget-value">{stats.gradient_norm.toFixed(6)}</div>
          <div className="widget-unit">norm</div>
        </div>

        {/* Uptime Widget */}
        <div className="stat-widget">
          <div className="widget-header">
            <span className="widget-icon">⏱️</span>
            <span className="widget-label">Uptime</span>
          </div>
          <div className="widget-value">{formatUptime(stats.uptime_seconds)}</div>
          <div className="widget-unit">running</div>
        </div>
      </div>
    </div>
  );
}

