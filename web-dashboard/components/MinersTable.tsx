"use client";

import { useQuery } from "@tanstack/react-query";

// API base URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://api.r3mes.network';

interface Miner {
  address: string;
  moniker: string;
  reputation_score: number;
  total_staked: string;
  active_tasks: number;
  completed_tasks: number;
  gpu_count: number;
  status: "active" | "inactive" | "jailed";
}

type MinerTier = "diamond" | "platinum" | "gold" | "silver" | "bronze";

interface TierInfo {
  name: string;
  color: string;
  bgColor: string;
  icon: string;
  minScore: number;
  rewardMultiplier: number;
}

const TIER_CONFIG: Record<MinerTier, TierInfo> = {
  diamond: { name: "Diamond", color: "#b9f2ff", bgColor: "rgba(185, 242, 255, 0.15)", icon: "💎", minScore: 95, rewardMultiplier: 2.0 },
  platinum: { name: "Platinum", color: "#e5e4e2", bgColor: "rgba(229, 228, 226, 0.15)", icon: "🏆", minScore: 85, rewardMultiplier: 1.5 },
  gold: { name: "Gold", color: "#ffd700", bgColor: "rgba(255, 215, 0, 0.15)", icon: "🥇", minScore: 70, rewardMultiplier: 1.25 },
  silver: { name: "Silver", color: "#c0c0c0", bgColor: "rgba(192, 192, 192, 0.15)", icon: "🥈", minScore: 50, rewardMultiplier: 1.1 },
  bronze: { name: "Bronze", color: "#cd7f32", bgColor: "rgba(205, 127, 50, 0.15)", icon: "🥉", minScore: 0, rewardMultiplier: 1.0 },
};

function getTierFromScore(score: number): MinerTier {
  if (score >= 95) return "diamond";
  if (score >= 85) return "platinum";
  if (score >= 70) return "gold";
  if (score >= 50) return "silver";
  return "bronze";
}

function TierBadge({ score }: { score: number }) {
  const tier = getTierFromScore(score);
  const tierInfo = TIER_CONFIG[tier];

  return (
    <div 
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md border text-xs font-semibold transition-transform hover:scale-105 hover:shadow-lg"
      style={{ backgroundColor: tierInfo.bgColor, borderColor: tierInfo.color }}
      title={`${tierInfo.name} Tier - ${tierInfo.rewardMultiplier}x reward multiplier`}
    >
      <span className="text-sm">{tierInfo.icon}</span>
      <span className="uppercase tracking-wide" style={{ color: tierInfo.color }}>{tierInfo.name}</span>
    </div>
  );
}

export default function MinersTable() {
  const { data: miners, isLoading, error } = useQuery<Miner[]>({
    queryKey: ["explorer", "miners"],
    queryFn: async () => {
      const response = await fetch(`${API_BASE_URL}/api/blockchain/dashboard/miners`);
      if (!response.ok) throw new Error("Failed to fetch miners");
      const data = await response.json();
      return data.miners || [];
    },
    refetchInterval: 30000,
    retry: 3,
    retryDelay: 1000,
  });

  const formatStake = (stake: string) => {
    const num = parseFloat(stake) / 1e6;
    if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M REMES`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K REMES`;
    return `${num.toFixed(2)} REMES`;
  };

  const getReputationColor = (score: number) => {
    if (score >= 80) return "#22c55e";
    if (score >= 60) return "#3b82f6";
    if (score >= 40) return "#f59e0b";
    return "#ef4444";
  };

  const getRankBadgeClass = (index: number) => {
    const base = "inline-flex items-center justify-center w-8 h-8 rounded-full text-xs font-semibold";
    if (index === 0) return `${base} bg-gradient-to-br from-yellow-400 to-yellow-600 text-slate-900 shadow-[0_0_12px_rgba(255,215,0,0.4)]`;
    if (index === 1) return `${base} bg-gradient-to-br from-slate-300 to-slate-400 text-slate-900 shadow-[0_0_12px_rgba(192,192,192,0.4)]`;
    if (index === 2) return `${base} bg-gradient-to-br from-amber-600 to-amber-700 text-slate-900 shadow-[0_0_12px_rgba(205,127,50,0.4)]`;
    return `${base} bg-slate-700 text-slate-100`;
  };

  const getStatusClass = (status: string) => {
    const base = "px-3 py-1 rounded-md text-xs font-semibold uppercase";
    if (status === "active") return `${base} bg-green-500/10 text-green-500`;
    if (status === "jailed") return `${base} bg-red-500/10 text-red-500`;
    return `${base} bg-slate-400/10 text-slate-400`;
  };

  if (isLoading) return <div className="mb-8"><div className="text-center py-10 text-slate-400">Loading miners...</div></div>;
  if (error) return <div className="mb-8"><div className="text-center py-10 text-red-500">Failed to load miners. Please try again.</div></div>;

  return (
    <div className="mb-8">
      <h3 className="text-xl font-semibold text-slate-100 mb-5">Active Miners</h3>
      <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
        {/* Header */}
        <div className="grid grid-cols-[60px_2fr_100px_150px_1fr_100px_80px_100px] gap-4 px-5 py-4 bg-slate-900 border-b border-slate-700 text-xs font-semibold text-slate-400 uppercase tracking-wide">
          <div>Rank</div>
          <div>Miner</div>
          <div>Tier</div>
          <div>Reputation</div>
          <div>Total Staked</div>
          <div>Tasks</div>
          <div>GPUs</div>
          <div>Status</div>
        </div>
        {/* Body */}
        <div className="flex flex-col">
          {miners && miners.length > 0 ? (
            miners
              .sort((a, b) => b.reputation_score - a.reputation_score)
              .map((miner, index) => (
                <div key={miner.address} className="grid grid-cols-[60px_2fr_100px_150px_1fr_100px_80px_100px] gap-4 px-5 py-4 border-b border-slate-700 last:border-b-0 transition-colors hover:bg-slate-900">
                  <div className="flex items-center">
                    <span className={getRankBadgeClass(index)}>#{index + 1}</span>
                  </div>
                  <div className="flex flex-col gap-1">
                    <div className="text-sm font-semibold text-slate-100">{miner.moniker || "Unknown"}</div>
                    <div className="text-xs font-mono text-slate-500">{miner.address.slice(0, 12)}...{miner.address.slice(-8)}</div>
                  </div>
                  <div className="flex items-center">
                    <TierBadge score={miner.reputation_score} />
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 h-2 bg-slate-700 rounded overflow-hidden">
                      <div className="h-full transition-all duration-300" style={{ width: `${miner.reputation_score}%`, backgroundColor: getReputationColor(miner.reputation_score) }} />
                    </div>
                    <span className="text-xs font-semibold text-slate-100 min-w-[30px]">{miner.reputation_score}</span>
                  </div>
                  <div className="flex items-center text-sm text-slate-100">{formatStake(miner.total_staked)}</div>
                  <div className="flex items-center gap-1 text-sm">
                    <span className="text-blue-500 font-semibold">{miner.active_tasks}</span>
                    <span className="text-slate-500">/</span>
                    <span className="text-green-500">{miner.completed_tasks}</span>
                  </div>
                  <div className="flex items-center text-sm text-slate-100">{miner.gpu_count}</div>
                  <div className="flex items-center">
                    <span className={getStatusClass(miner.status)}>{miner.status}</span>
                  </div>
                </div>
              ))
          ) : (
            <div className="py-10 text-center text-slate-400">No miners found</div>
          )}
        </div>
      </div>
    </div>
  );
}
