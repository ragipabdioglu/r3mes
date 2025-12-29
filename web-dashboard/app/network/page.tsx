"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { getRoleStatistics, RoleStats } from "@/lib/api";
import { logger } from "@/lib/logger";
import { useNetworkStats, useRecentBlocks } from "@/hooks/useNetworkStats";
import StatCard from "@/components/StatCard";
import { Network, Database, TrendingUp, Cpu, Server, Layers, Shield } from "lucide-react";
import { SkeletonStatCard, SkeletonTable } from "@/components/SkeletonLoader";

// Lazy load heavy components
const NetworkExplorer = dynamic(() => import("@/components/NetworkExplorer"), {
  ssr: false,
  loading: () => <div className="h-screen flex items-center justify-center text-[var(--text-secondary)]">Loading network explorer...</div>
});

const MinersTable = dynamic(() => import("@/components/MinersTable"), {
  ssr: false,
  loading: () => <SkeletonTable />
});

export default function NetworkPage() {
  // Use React Query hooks instead of setInterval polling
  const { data: stats, isLoading: statsLoading, error: statsError } = useNetworkStats();
  const { data: blocks = [], isLoading: blocksLoading, error: blocksError } = useRecentBlocks(50);
  
  const [roleStats, setRoleStats] = useState<RoleStats[]>([]);
  const [selectedRoleFilter, setSelectedRoleFilter] = useState<string>("all");
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Fetch role statistics once (doesn't need frequent updates)
    getRoleStatistics()
      .then((data) => {
        setRoleStats(data.stats);
        setIsLoading(false);
      })
      .catch((err) => {
        logger.error("Failed to fetch role statistics:", err);
        setRoleStats([]);
      setIsLoading(false);
    });
  }, []);

  // Log errors from React Query
  useEffect(() => {
    if (statsError) {
      logger.error("Failed to fetch network stats:", statsError);
    }
    if (blocksError) {
      logger.error("Failed to fetch blocks:", blocksError);
    }
  }, [statsError, blocksError]);

  const isLoadingData = statsLoading || blocksLoading || isLoading;

  return (
    <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] py-8 px-4 sm:py-10 sm:px-6 md:py-12 md:px-8">
      <div className="container mx-auto max-w-full sm:max-w-2xl md:max-w-4xl lg:max-w-6xl xl:max-w-7xl">
        <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold mb-6 sm:mb-8 gradient-text">Network Explorer</h1>

        {/* Role Filter */}
        <div className="mb-6 flex flex-wrap gap-2">
          <button
            onClick={() => setSelectedRoleFilter("all")}
            className={`px-4 py-2 rounded-full text-sm font-medium transition-colors ${
              selectedRoleFilter === "all"
                ? "bg-[var(--accent-primary)] text-white"
                : "bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)]"
            }`}
          >
            All Nodes
          </button>
          {roleStats.map((stat) => {
            const icons: Record<string, React.ReactNode> = {
              Miner: <Cpu className="w-4 h-4" />,
              Serving: <Server className="w-4 h-4" />,
              Validator: <Shield className="w-4 h-4" />,
              Proposer: <Layers className="w-4 h-4" />,
            };
            return (
              <button
                key={stat.role_id}
                onClick={() => setSelectedRoleFilter(stat.role_name.toLowerCase())}
                className={`px-4 py-2 rounded-full text-sm font-medium transition-colors flex items-center gap-2 ${
                  selectedRoleFilter === stat.role_name.toLowerCase()
                    ? "bg-[var(--accent-primary)] text-white"
                    : "bg-[var(--bg-secondary)] text-[var(--text-secondary)] hover:bg-[var(--bg-tertiary)]"
                }`}
              >
                {icons[stat.role_name]}
                {stat.role_name} ({stat.active_nodes})
              </button>
            );
          })}
        </div>

        {/* Network Overview */}
        {isLoadingData ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5 md:gap-6 mb-6 sm:mb-8">
            {[1, 2, 3, 4].map((i) => (
              <SkeletonStatCard key={i} />
            ))}
          </div>
        ) : stats ? (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5 md:gap-6 mb-6 sm:mb-8">
            <StatCard
              label="Total Stake"
              value={`${(stats.total_credits / 1000000).toFixed(1)}M`}
              subtext="REMES"
              icon={<TrendingUp className="w-5 h-5" />}
            />
            <StatCard
              label="Active Miners"
              value={stats?.active_miners.toLocaleString() || "0"}
              icon={<Network className="w-5 h-5" />}
            />
            <StatCard
              label="Total Blocks"
              value={stats?.block_height?.toLocaleString() || "0"}
              icon={<Database className="w-5 h-5" />}
            />
            <StatCard
              label="Model Version"
              value="BitNet b1.58"
              subtext="Genesis"
              icon={<Cpu className="w-5 h-5" />}
            />
          </div>
        ) : null}

        {/* Role Statistics */}
        {roleStats.length > 0 && (
          <div className="card mb-6 sm:mb-8">
            <h2 className="text-xl sm:text-2xl font-semibold mb-4 sm:mb-6 text-[var(--text-primary)]">Role Distribution</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              {roleStats.map((stat) => {
                const icons: Record<string, React.ReactNode> = {
                  Miner: <Cpu className="w-5 h-5" />,
                  Serving: <Server className="w-5 h-5" />,
                  Validator: <Shield className="w-5 h-5" />,
                  Proposer: <Layers className="w-5 h-5" />,
                };
                return (
                  <div key={stat.role_id} className="p-4 rounded-xl bg-[var(--bg-secondary)]">
                    <div className="flex items-center gap-3 mb-2">
                      {icons[stat.role_name]}
                      <div className="text-sm font-semibold text-[var(--text-primary)]">{stat.role_name}</div>
                    </div>
                    <div className="text-2xl font-bold text-[var(--accent-primary)]">{stat.total_nodes}</div>
                    <div className="text-xs text-[var(--text-secondary)]">{stat.active_nodes} active</div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Miners Table */}
        <div className="mb-6 sm:mb-8">
          <h2 className="text-xl sm:text-2xl font-semibold mb-4 sm:mb-6 text-[var(--text-primary)]">Active Miners</h2>
          <MinersTable />
        </div>

        {/* Recent Blocks Table */}
        <div className="card p-4 sm:p-5 md:p-6">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between mb-4 sm:mb-6 gap-3 sm:gap-4">
            <h2 className="text-lg sm:text-xl font-semibold text-[var(--text-primary)]">Recent Blocks</h2>
            <div className="flex items-center gap-2 sm:gap-4 w-full sm:w-auto">
              <input
                type="text"
                placeholder="Search..."
                className="w-full sm:w-auto px-3 sm:px-4 py-1.5 sm:py-2 bg-[var(--bg-tertiary)] border border-[var(--border-color)] rounded-lg text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--accent-primary)] text-xs sm:text-sm"
              />
            </div>
          </div>

          {isLoadingData ? (
            <SkeletonTable />
          ) : blocks.length === 0 ? (
            <div className="text-center py-8 sm:py-12">
              <div className="text-sm sm:text-base text-[var(--text-secondary)]">No blocks available</div>
            </div>
          ) : (
            <div className="overflow-x-auto -mx-4 sm:-mx-5 md:-mx-6 px-4 sm:px-5 md:px-6">
              <table className="w-full min-w-[600px]">
                <thead>
                  <tr className="border-b border-[var(--border-color)]">
                    <th className="text-left py-3 sm:py-4 px-2 sm:px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                      Block
                    </th>
                    <th className="text-left py-3 sm:py-4 px-2 sm:px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                      Hash
                    </th>
                    <th className="text-left py-3 sm:py-4 px-2 sm:px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                      Miner
                    </th>
                    <th className="text-left py-3 sm:py-4 px-2 sm:px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                      Tx Count
                    </th>
                    <th className="text-left py-3 sm:py-4 px-2 sm:px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                      Time
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {blocks.map((block, i) => (
                    <tr
                      key={i}
                      className="border-b border-[var(--border-color)] hover:bg-[var(--bg-tertiary)] transition-colors cursor-pointer"
                      onClick={() => {
                        // Navigate to block detail page (to be implemented)
                        window.location.href = `/network/block/${block.height}`;
                      }}
                    >
                      <td className="py-3 sm:py-4 px-2 sm:px-4">
                        <span className="font-semibold text-xs sm:text-sm text-[var(--text-primary)]">
                          #{block.height?.toLocaleString() || "N/A"}
                        </span>
                      </td>
                      <td className="py-3 sm:py-4 px-2 sm:px-4">
                        <span className="text-xs sm:text-sm text-[var(--text-secondary)] font-mono">
                          {block.hash
                            ? `${block.hash.slice(0, 10)}...${block.hash.slice(-8)}`
                            : "N/A"}
                        </span>
                      </td>
                      <td className="py-3 sm:py-4 px-2 sm:px-4">
                        <span className="text-xs sm:text-sm text-[var(--text-secondary)] font-mono">
                          {block.miner
                            ? `${block.miner.slice(0, 10)}...${block.miner.slice(-8)}`
                            : "Unknown"}
                        </span>
                      </td>
                      <td className="py-3 sm:py-4 px-2 sm:px-4">
                        <span className="text-xs sm:text-sm text-[var(--text-secondary)]">-</span>
                      </td>
                      <td className="py-3 sm:py-4 px-2 sm:px-4">
                        <span className="text-xs sm:text-sm text-[var(--text-secondary)]">
                          {block.timestamp
                            ? new Date(block.timestamp).toLocaleString()
                            : "N/A"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
