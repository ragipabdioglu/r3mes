"use client";

import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { Trophy, Medal, Award, TrendingUp } from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { getLeaderboard } from "@/lib/api";
import { useAnnouncer } from "@/hooks/useAccessibility";
import { formatAddress, formatNumber, formatPercentage } from "@/utils/formatters";

export default function LeaderboardPage() {
  const [activeTab, setActiveTab] = useState<"miners" | "validators">("miners");
  const { announce, announceLoading, announceError } = useAnnouncer();

  const { data: minersData, isLoading: minersLoading, error: minersError } = useQuery({
    queryKey: ["leaderboard", "miners"],
    queryFn: () => getLeaderboard("miners"),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const { data: validatorsData, isLoading: validatorsLoading, error: validatorsError } = useQuery({
    queryKey: ["leaderboard", "validators"],
    queryFn: () => getLeaderboard("validators"),
    refetchInterval: 30000,
  });

  // Announce loading and error states
  useEffect(() => {
    if (activeTab === "miners") {
      if (minersLoading) {
        announceLoading("miners leaderboard", true);
      } else if (minersData?.miners?.length) {
        announce(`Loaded ${minersData.miners.length} miners`);
      }
      if (minersError) {
        announceError("Failed to load miners leaderboard");
      }
    } else {
      if (validatorsLoading) {
        announceLoading("validators leaderboard", true);
      } else if (validatorsData?.validators?.length) {
        announce(`Loaded ${validatorsData.validators.length} validators`);
      }
      if (validatorsError) {
        announceError("Failed to load validators leaderboard");
      }
    }
  }, [activeTab, minersLoading, validatorsLoading, minersData, validatorsData, minersError, validatorsError, announce, announceLoading, announceError]);

  const handleTabChange = (tab: "miners" | "validators") => {
    setActiveTab(tab);
    announce(`Switched to ${tab} leaderboard`);
  };

  const getTierBadge = (tier: string) => {
    const tiers: Record<string, { icon: JSX.Element; color: string; label: string }> = {
      bronze: {
        icon: <Medal className="w-5 h-5" />,
        color: "text-amber-600",
        label: "B",
      },
      silver: {
        icon: <Medal className="w-5 h-5" />,
        color: "text-gray-400",
        label: "S",
      },
      gold: {
        icon: <Award className="w-5 h-5" />,
        color: "text-yellow-400",
        label: "G",
      },
      platinum: {
        icon: <Trophy className="w-5 h-5" />,
        color: "text-cyan-400",
        label: "P",
      },
      diamond: {
        icon: <Trophy className="w-5 h-5" />,
        color: "text-purple-400",
        label: "D",
      },
    };

    return tiers[tier.toLowerCase()] || tiers.bronze;
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <div className="container mx-auto px-4 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-12"
        >
          <h1 className="text-5xl md:text-6xl font-bold mb-6 bg-gradient-to-r from-green-400 to-cyan-400 bg-clip-text text-transparent">
            Leaderboard
          </h1>
          <p className="text-xl text-slate-400">
            Top performers in the R3MES network
          </p>
        </motion.div>

        {/* Tabs */}
        <div className="flex gap-4 justify-center mb-8" role="tablist" aria-label="Leaderboard categories">
          <button
            onClick={() => handleTabChange("miners")}
            role="tab"
            aria-selected={activeTab === "miners"}
            aria-controls="miners-panel"
            className={`px-6 py-3 rounded-lg font-semibold transition-colors ${
              activeTab === "miners"
                ? "bg-green-500 text-white"
                : "bg-slate-800 text-slate-300 hover:bg-slate-700"
            }`}
          >
            Top Miners
          </button>
          <button
            onClick={() => handleTabChange("validators")}
            role="tab"
            aria-selected={activeTab === "validators"}
            aria-controls="validators-panel"
            className={`px-6 py-3 rounded-lg font-semibold transition-colors ${
              activeTab === "validators"
                ? "bg-green-500 text-white"
                : "bg-slate-800 text-slate-300 hover:bg-slate-700"
            }`}
          >
            Most Trusted Validators
          </button>
        </div>

        {/* Leaderboard Table */}
        <div className="card overflow-hidden">
          {activeTab === "miners" ? (
            <div id="miners-panel" role="tabpanel" aria-labelledby="miners-tab">
              <MinersTable data={minersData} loading={minersLoading} getTierBadge={getTierBadge} />
            </div>
          ) : (
            <div id="validators-panel" role="tabpanel" aria-labelledby="validators-tab">
              <ValidatorsTable data={validatorsData} loading={validatorsLoading} getTierBadge={getTierBadge} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MinersTable({
  data,
  loading,
  getTierBadge,
}: {
  data: any;
  loading: boolean;
  getTierBadge: (tier: string) => any;
}) {
  if (loading) {
    return (
      <div className="p-8 text-center">
        <div className="animate-pulse text-slate-400">Loading leaderboard...</div>
      </div>
    );
  }

  if (!data || !data.miners || data.miners.length === 0) {
    return (
      <div className="p-8 text-center text-slate-400">
        No miners found in the leaderboard.
      </div>
    );
  }

  return (
    <table className="w-full">
      <thead className="bg-slate-800">
        <tr>
          <th className="px-6 py-4 text-left">Rank</th>
          <th className="px-6 py-4 text-left">Address</th>
          <th className="px-6 py-4 text-left">Tier</th>
          <th className="px-6 py-4 text-right">Total Submissions</th>
          <th className="px-6 py-4 text-right">Reputation</th>
          <th className="px-6 py-4 text-right">Trend</th>
        </tr>
      </thead>
      <tbody>
        {data.miners.map((miner: any, index: number) => {
          const tier = getTierBadge(miner.tier || "bronze");
          return (
            <tr
              key={miner.address}
              className="border-t border-slate-700 hover:bg-slate-800/50 transition-colors"
            >
              <td className="px-6 py-4">
                <div className="flex items-center gap-2">
                  {index < 3 && <Trophy className={`w-5 h-5 ${index === 0 ? "text-yellow-400" : index === 1 ? "text-gray-400" : "text-amber-600"}`} />}
                  <span className="font-bold">{index + 1}</span>
                </div>
              </td>
              <td className="px-6 py-4 font-mono text-sm">
                {formatAddress(miner.address, 8, 6)}
              </td>
              <td className="px-6 py-4">
                <div className={`flex items-center gap-2 ${tier.color}`}>
                  {tier.icon}
                  <span className="font-semibold">{tier.label}</span>
                </div>
              </td>
              <td className="px-6 py-4 text-right">{formatNumber(miner.total_submissions || 0)}</td>
              <td className="px-6 py-4 text-right">{miner.reputation?.toFixed(2) || "0.00"}</td>
              <td className="px-6 py-4 text-right">
                {miner.trend && miner.trend > 0 && (
                  <div className="flex items-center justify-end gap-1 text-green-400">
                    <TrendingUp className="w-4 h-4" />
                    <span>+{miner.trend}</span>
                  </div>
                )}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function ValidatorsTable({
  data,
  loading,
  getTierBadge,
}: {
  data: any;
  loading: boolean;
  getTierBadge: (tier: string) => any;
}) {
  if (loading) {
    return (
      <div className="p-8 text-center">
        <div className="animate-pulse text-slate-400">Loading leaderboard...</div>
      </div>
    );
  }

  if (!data || !data.validators || data.validators.length === 0) {
    return (
      <div className="p-8 text-center text-slate-400">
        No validators found in the leaderboard.
      </div>
    );
  }

  return (
    <table className="w-full">
      <thead className="bg-slate-800">
        <tr>
          <th className="px-6 py-4 text-left">Rank</th>
          <th className="px-6 py-4 text-left">Address</th>
          <th className="px-6 py-4 text-left">Tier</th>
          <th className="px-6 py-4 text-right">Trust Score</th>
          <th className="px-6 py-4 text-right">Uptime</th>
          <th className="px-6 py-4 text-right">Voting Power</th>
        </tr>
      </thead>
      <tbody>
        {data.validators.map((validator: any, index: number) => {
          const tier = getTierBadge(validator.tier || "bronze");
          return (
            <tr
              key={validator.address}
              className="border-t border-slate-700 hover:bg-slate-800/50 transition-colors"
            >
              <td className="px-6 py-4">
                <div className="flex items-center gap-2">
                  {index < 3 && <Trophy className={`w-5 h-5 ${index === 0 ? "text-yellow-400" : index === 1 ? "text-gray-400" : "text-amber-600"}`} />}
                  <span className="font-bold">{index + 1}</span>
                </div>
              </td>
              <td className="px-6 py-4 font-mono text-sm">
                {formatAddress(validator.address, 8, 6)}
              </td>
              <td className="px-6 py-4">
                <div className={`flex items-center gap-2 ${tier.color}`}>
                  {tier.icon}
                  <span className="font-semibold">{tier.label}</span>
                </div>
              </td>
              <td className="px-6 py-4 text-right">{validator.trust_score?.toFixed(2) || "0.00"}</td>
              <td className="px-6 py-4 text-right">{formatPercentage(validator.uptime || 0)}</td>
              <td className="px-6 py-4 text-right">{formatNumber(validator.voting_power || 0)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

