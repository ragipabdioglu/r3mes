"use client";

import { useQuery } from "@tanstack/react-query";
import "./NetworkStats.css";

interface NetworkStats {
  total_stake: string;
  inflation_rate: string;
  model_version: string;
  active_validators: number;
  total_miners: number;
  network_hashrate: number;
  bonded_tokens: string;
  unbonded_tokens: string;
}

export default function NetworkStats() {
  const { data: stats, isLoading, error } = useQuery<NetworkStats>({
    queryKey: ["explorer", "network-stats"],
    queryFn: async () => {
      const response = await fetch("/api/blockchain/dashboard/statistics");
      if (!response.ok) {
        throw new Error("Failed to fetch network stats");
      }
      return response.json();
    },
    refetchInterval: 30000,
  });

  const formatStake = (stake: string) => {
    const num = parseFloat(stake) / 1e6; // Convert from uremes to REMES
    if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B REMES`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M REMES`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K REMES`;
    return `${num.toFixed(2)} REMES`;
  };

  const formatHashrate = (hashrate: number) => {
    if (hashrate >= 1e9) return `${(hashrate / 1e9).toFixed(2)}B gradients/h`;
    if (hashrate >= 1e6) return `${(hashrate / 1e6).toFixed(2)}M gradients/h`;
    if (hashrate >= 1e3) return `${(hashrate / 1e3).toFixed(2)}K gradients/h`;
    return `${hashrate.toFixed(2)} gradients/h`;
  };

  if (isLoading) {
    return (
      <div className="network-stats-container">
        <div className="loading">Loading network stats...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="network-stats-container">
        <div className="error">Failed to load network stats. Please try again.</div>
      </div>
    );
  }

  return (
    <div className="network-stats-container">
      <h3 className="stats-title">Network Statistics</h3>
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-label">Total Stake</div>
          <div className="stat-value">
            {stats ? formatStake(stats.total_stake) : "0 REMES"}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Inflation Rate</div>
          <div className="stat-value">
            {stats ? `${(parseFloat(stats.inflation_rate) * 100).toFixed(2)}%` : "0%"}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Model Version</div>
          <div className="stat-value">{stats?.model_version || "N/A"}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Active Validators</div>
          <div className="stat-value">{stats?.active_validators || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Total Miners</div>
          <div className="stat-value">{stats?.total_miners || 0}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Network Hashrate</div>
          <div className="stat-value">
            {stats ? formatHashrate(stats.network_hashrate) : "0 gradients/h"}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Bonded Tokens</div>
          <div className="stat-value">
            {stats ? formatStake(stats.bonded_tokens) : "0 REMES"}
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Unbonded Tokens</div>
          <div className="stat-value">
            {stats ? formatStake(stats.unbonded_tokens) : "0 REMES"}
          </div>
        </div>
      </div>
    </div>
  );
}

