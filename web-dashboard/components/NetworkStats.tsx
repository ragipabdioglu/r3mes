"use client";

import { useQuery } from "@tanstack/react-query";

// API base URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://api.r3mes.network';

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
      const response = await fetch(`${API_BASE_URL}/api/blockchain/dashboard/statistics`);
      if (!response.ok) {
        throw new Error("Failed to fetch network stats");
      }
      return response.json();
    },
    refetchInterval: 30000,
    retry: 3,
    retryDelay: 1000,
  });

  const formatStake = (stake: string) => {
    const num = parseFloat(stake) / 1e6;
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
      <div className="mb-8">
        <div className="text-center py-10 text-slate-400">Loading network stats...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mb-8">
        <div className="text-center py-10 text-red-500">Failed to load network stats. Please try again.</div>
      </div>
    );
  }

  return (
    <div className="mb-8">
      <h3 className="text-xl font-semibold text-slate-100 mb-5">Network Statistics</h3>
      <div className="grid grid-cols-[repeat(auto-fit,minmax(200px,1fr))] gap-4">
        <StatCard label="Total Stake" value={stats ? formatStake(stats.total_stake) : "0 REMES"} />
        <StatCard label="Inflation Rate" value={stats ? `${(parseFloat(stats.inflation_rate) * 100).toFixed(2)}%` : "0%"} />
        <StatCard label="Model Version" value={stats?.model_version || "N/A"} />
        <StatCard label="Active Validators" value={String(stats?.active_validators || 0)} />
        <StatCard label="Total Miners" value={String(stats?.total_miners || 0)} />
        <StatCard label="Network Hashrate" value={stats ? formatHashrate(stats.network_hashrate) : "0 gradients/h"} />
        <StatCard label="Bonded Tokens" value={stats ? formatStake(stats.bonded_tokens) : "0 REMES"} />
        <StatCard label="Unbonded Tokens" value={stats ? formatStake(stats.unbonded_tokens) : "0 REMES"} />
      </div>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-slate-800 border border-slate-700 rounded-lg p-5 transition-all hover:border-slate-600 hover:shadow-lg hover:shadow-black/30">
      <div className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">{label}</div>
      <div className="text-lg font-semibold text-slate-100">{value}</div>
    </div>
  );
}
