"use client";

import { useQuery } from "@tanstack/react-query";
import "./MinersTable.css";

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
  diamond: {
    name: "Diamond",
    color: "#b9f2ff",
    bgColor: "rgba(185, 242, 255, 0.15)",
    icon: "💎",
    minScore: 95,
    rewardMultiplier: 2.0,
  },
  platinum: {
    name: "Platinum",
    color: "#e5e4e2",
    bgColor: "rgba(229, 228, 226, 0.15)",
    icon: "🏆",
    minScore: 85,
    rewardMultiplier: 1.5,
  },
  gold: {
    name: "Gold",
    color: "#ffd700",
    bgColor: "rgba(255, 215, 0, 0.15)",
    icon: "🥇",
    minScore: 70,
    rewardMultiplier: 1.25,
  },
  silver: {
    name: "Silver",
    color: "#c0c0c0",
    bgColor: "rgba(192, 192, 192, 0.15)",
    icon: "🥈",
    minScore: 50,
    rewardMultiplier: 1.1,
  },
  bronze: {
    name: "Bronze",
    color: "#cd7f32",
    bgColor: "rgba(205, 127, 50, 0.15)",
    icon: "🥉",
    minScore: 0,
    rewardMultiplier: 1.0,
  },
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
      className="tier-badge"
      style={{ 
        backgroundColor: tierInfo.bgColor,
        borderColor: tierInfo.color,
      }}
      title={`${tierInfo.name} Tier - ${tierInfo.rewardMultiplier}x reward multiplier`}
    >
      <span className="tier-icon">{tierInfo.icon}</span>
      <span className="tier-name" style={{ color: tierInfo.color }}>
        {tierInfo.name}
      </span>
    </div>
  );
}

export default function MinersTable() {
  const { data: miners, isLoading, error } = useQuery<Miner[]>({
    queryKey: ["explorer", "miners"],
    queryFn: async () => {
      const response = await fetch("/api/blockchain/dashboard/miners");
      if (!response.ok) {
        throw new Error("Failed to fetch miners");
      }
      const data = await response.json();
      return data.miners || [];
    },
    refetchInterval: 30000,
  });

  const formatStake = (stake: string) => {
    const num = parseFloat(stake) / 1e6; // Convert from uremes to REMES
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

  if (isLoading) {
    return (
      <div className="miners-table-container">
        <div className="loading">Loading miners...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="miners-table-container">
        <div className="error">Failed to load miners. Please try again.</div>
      </div>
    );
  }

  return (
    <div className="miners-table-container">
      <h3 className="table-title">Active Miners</h3>
      <div className="miners-table">
        <div className="table-header">
          <div className="col-rank">Rank</div>
          <div className="col-miner">Miner</div>
          <div className="col-tier">Tier</div>
          <div className="col-reputation">Reputation</div>
          <div className="col-stake">Total Staked</div>
          <div className="col-tasks">Tasks</div>
          <div className="col-gpu">GPUs</div>
          <div className="col-status">Status</div>
        </div>
        <div className="table-body">
          {miners && miners.length > 0 ? (
            miners
              .sort((a, b) => b.reputation_score - a.reputation_score)
              .map((miner, index) => (
                <div key={miner.address} className="table-row">
                  <div className="col-rank">
                    <span className={`rank-badge rank-${index < 3 ? index + 1 : 'default'}`}>
                      #{index + 1}
                    </span>
                  </div>
                  <div className="col-miner">
                    <div className="miner-name">{miner.moniker || "Unknown"}</div>
                    <div className="miner-address">
                      {miner.address.slice(0, 12)}...{miner.address.slice(-8)}
                    </div>
                  </div>
                  <div className="col-tier">
                    <TierBadge score={miner.reputation_score} />
                  </div>
                  <div className="col-reputation">
                    <div className="reputation-bar">
                      <div
                        className="reputation-fill"
                        style={{
                          width: `${miner.reputation_score}%`,
                          backgroundColor: getReputationColor(miner.reputation_score),
                        }}
                      />
                    </div>
                    <span className="reputation-score">{miner.reputation_score}</span>
                  </div>
                  <div className="col-stake">{formatStake(miner.total_staked)}</div>
                  <div className="col-tasks">
                    <span className="task-active">{miner.active_tasks}</span>
                    <span className="task-separator">/</span>
                    <span className="task-completed">{miner.completed_tasks}</span>
                  </div>
                  <div className="col-gpu">{miner.gpu_count}</div>
                  <div className="col-status">
                    <span className={`status-badge status-${miner.status}`}>
                      {miner.status}
                    </span>
                  </div>
                </div>
              ))
          ) : (
            <div className="table-empty">No miners found</div>
          )}
        </div>
      </div>
    </div>
  );
}

