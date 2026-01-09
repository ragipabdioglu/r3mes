"use client";

import { useEffect, useState } from "react";
import { Download, TrendingUp, Cpu, Thermometer, Clock, Activity } from "lucide-react";
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from "recharts";
import { logger } from "@/lib/logger";
import { useUserInfo, useMinerStats, useEarningsHistory, useHashrateHistory } from "@/hooks/useMinerData";
import { useRecentBlocks } from "@/hooks/useNetworkStats";
import StatCard from "@/components/StatCard";
import StatusBadge from "@/components/StatusBadge";
import WalletGuard from "@/components/WalletGuard";
import { SkeletonStatCard, SkeletonChart } from "@/components/SkeletonLoader";
import { formatTimeAgo, formatAddress } from "@/utils/formatters";

// Constants
const DEFAULT_BLOCK_REWARD = 10.5; // REMES per block

function MinePageContent() {
  const [walletAddress, setWalletAddress] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  // Use React Query hooks instead of setInterval polling
  const { data: userInfo, isLoading: userInfoLoading, error: userInfoError } = useUserInfo(walletAddress);
  const { data: minerStats, isLoading: minerStatsLoading, error: minerStatsError } = useMinerStats(walletAddress);
  const { data: recentBlocks = [], isLoading: blocksLoading } = useRecentBlocks(10);
  const { data: earningsData = [], isLoading: earningsLoading } = useEarningsHistory(walletAddress);
  const { data: hashrateData = [], isLoading: hashrateLoading } = useHashrateHistory(walletAddress);

  const isLoading = userInfoLoading || minerStatsLoading || blocksLoading || earningsLoading || hashrateLoading;

  useEffect(() => {
    setMounted(true);
    const address = localStorage.getItem("keplr_address");
    setWalletAddress(address);

    // Listen for wallet address changes
    const handleStorageChange = () => {
      const addr = localStorage.getItem("keplr_address");
      if (addr !== walletAddress) {
        setWalletAddress(addr);
      }
    };

    // Use storage event listener for cross-tab changes
    // Note: storage event only fires for changes from other tabs/windows
    const handleStorageEvent = (e: StorageEvent) => {
      if (e.key === "keplr_address") {
        handleStorageChange();
      }
    };
    
    window.addEventListener("storage", handleStorageEvent);
    
    // Also check periodically for same-tab changes (fallback)
    // But use longer interval since storage event handles cross-tab changes
    const interval = setInterval(() => {
      // Only check if wallet address changed in same tab
      const currentAddress = localStorage.getItem("keplr_address");
      if (currentAddress !== walletAddress) {
        handleStorageChange();
      }
    }, 30000); // Check every 30 seconds (longer interval since storage event handles most cases)

    return () => {
      window.removeEventListener("storage", handleStorageEvent);
      clearInterval(interval);
    };
  }, [walletAddress]);

  // Log errors from React Query
  useEffect(() => {
    if (userInfoError) {
      logger.error("Failed to fetch user info:", userInfoError);
    }
    if (minerStatsError) {
      logger.error("Failed to fetch miner stats:", minerStatsError);
    }
  }, [userInfoError, minerStatsError]);

  if (!mounted || isLoading) {
    return (
      <div className="min-h-screen bg-slate-900 text-slate-100 py-12 px-4">
        <div className="container mx-auto max-w-7xl">
          <div className="mb-8">
            <div className="h-10 bg-slate-800 rounded w-64 mb-4 animate-pulse"></div>
            <div className="h-6 bg-slate-800 rounded w-96 animate-pulse"></div>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <SkeletonStatCard key={i} />
            ))}
          </div>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <SkeletonChart />
            <SkeletonChart />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 py-8 px-4 sm:py-10 sm:px-6 md:py-12 md:px-8">
      <div className="container mx-auto max-w-full sm:max-w-2xl md:max-w-4xl lg:max-w-6xl xl:max-w-7xl">
        {/* Header Section */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between mb-6 sm:mb-8">
          <div>
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-2 gradient-text">Mining Dashboard</h1>
            {userInfo && (
              <div className="flex flex-col sm:flex-row items-start sm:items-center gap-2 sm:gap-4 mt-2">
                <div>
                  <span className="text-xs sm:text-sm text-slate-400">Your Earnings: </span>
                  <span className="text-xl sm:text-2xl font-bold text-[#06b6d4]">
                    {userInfo.credits.toFixed(2)} REMES
                  </span>
                </div>
                <StatusBadge
                  status={userInfo.is_miner ? "active" : "inactive"}
                  label={userInfo.is_miner ? "PRO MINER" : "GUEST"}
                />
              </div>
            )}
          </div>
          <a
            href="#"
            className="btn-primary inline-flex items-center gap-2 mt-4 md:mt-0 px-4 py-2 sm:px-6 sm:py-3 md:px-8 md:py-4 text-sm sm:text-base"
          >
            <Download className="w-4 h-4 sm:w-5 sm:h-5" />
            <span className="hidden sm:inline">Download Launcher v1.0</span>
            <span className="sm:hidden">Download</span>
          </a>
        </div>
        <p className="text-xs sm:text-sm text-slate-400 mb-6 sm:mb-8">
          Windows & Linux Support / Requires Python 3.10+
        </p>

        {/* Metrics Grid - 6 Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5 md:gap-6 mb-6 sm:mb-8">
          <StatCard
            label="Earnings"
            value={`${userInfo?.credits.toFixed(2) || "0.00"} REMES`}
            icon={<TrendingUp className="w-5 h-5" />}
            trend="up"
            trendValue="+12.5%"
          />
          <StatCard
            label="Hashrate"
            value={minerStats ? `${minerStats.hashrate.toFixed(1)} GH/s` : "0.0 GH/s"}
            icon={<Activity className="w-5 h-5" />}
            trend={minerStats && minerStats.hashrate > 0 ? "up" : "neutral"}
          />
          <StatCard
            label="GPU Temperature"
            value={minerStats ? `${Math.round(minerStats.gpu_temperature)}°C` : "N/A"}
            icon={<Thermometer className="w-5 h-5" />}
            trend="neutral"
          />
          <StatCard
            label="Blocks Found"
            value={minerStats ? minerStats.blocks_found.toString() : "0"}
            icon={<Cpu className="w-5 h-5" />}
          />
          <StatCard
            label="Uptime"
            value={minerStats ? `${minerStats.uptime_percentage.toFixed(1)}%` : "0%"}
            icon={<Clock className="w-5 h-5" />}
            trend={minerStats && minerStats.uptime_percentage > 90 ? "up" : "neutral"}
          />
          <StatCard
            label="Network Difficulty"
            value={minerStats ? Math.round(minerStats.network_difficulty).toLocaleString() : "1,234"}
            icon={<Activity className="w-5 h-5" />}
            trend="up"
          />
        </div>

        {/* Charts Section */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 sm:gap-5 md:gap-6 mb-6 sm:mb-8">
          {/* Earnings Chart */}
          <div className="card p-4 sm:p-5 md:p-6">
            <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 text-slate-200">
              Earnings Over Time
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={earningsData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="date"
                  stroke="#94a3b8"
                  fontSize={12}
                  tick={{ fill: "#94a3b8" }}
                />
                <YAxis
                  stroke="#94a3b8"
                  fontSize={12}
                  tick={{ fill: "#94a3b8" }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1e293b",
                    border: "1px solid #334155",
                    borderRadius: "8px",
                    color: "#f1f5f9",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="earnings"
                  stroke="#06b6d4"
                  strokeWidth={2}
                  dot={{ fill: "#06b6d4", r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Hashrate Chart */}
          <div className="card p-4 sm:p-5 md:p-6">
            <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 text-slate-200">
              Hashrate Over Time
            </h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={hashrateData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
                <XAxis
                  dataKey="date"
                  stroke="#94a3b8"
                  fontSize={12}
                  tick={{ fill: "#94a3b8" }}
                />
                <YAxis
                  stroke="#94a3b8"
                  fontSize={12}
                  tick={{ fill: "#94a3b8" }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#1e293b",
                    border: "1px solid #334155",
                    borderRadius: "8px",
                    color: "#f1f5f9",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="hashrate"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={{ fill: "#3b82f6", r: 4 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Recent Activity */}
        <div className="card p-4 sm:p-5 md:p-6">
          <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 text-slate-200">Recent Activity</h3>
          {recentBlocks.length === 0 ? (
            <div className="text-center py-6 sm:py-8 text-slate-400 text-sm sm:text-base">
              No recent activity. Start mining to see your blocks here!
            </div>
          ) : (
            <div className="space-y-2 sm:space-y-3">
              {recentBlocks.map((block, i) => (
                <div
                  key={i}
                  className="flex flex-col sm:flex-row items-start sm:items-center justify-between py-2 sm:py-3 px-3 sm:px-4 border-b border-slate-700/50 hover:bg-slate-800/50 rounded-lg transition-colors gap-2 sm:gap-0"
                >
                  <div className="flex items-center gap-3 sm:gap-4">
                    <div className="w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-br from-cyan-500 to-blue-500 flex items-center justify-center text-xs sm:text-sm font-bold shrink-0">
                      {i + 1}
                    </div>
                    <div>
                      <div className="font-semibold text-sm sm:text-base text-slate-200">
                        Block #{block.height?.toLocaleString() || "N/A"}
                      </div>
                      <div className="text-xs sm:text-sm text-slate-400">
                        {block.miner ? formatAddress(block.miner) : "Unknown"}
                      </div>
                    </div>
                  </div>
                  <div className="text-left sm:text-right ml-11 sm:ml-0">
                    <div className="text-xs sm:text-sm font-medium text-[#06b6d4]">
                      +{(block as any).reward || DEFAULT_BLOCK_REWARD} REMES
                    </div>
                    <div className="text-[10px] sm:text-xs text-slate-400">
                      {formatTimeAgo(block.timestamp)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function MinePage() {
  return (
    <WalletGuard>
      <MinePageContent />
    </WalletGuard>
  );
}
