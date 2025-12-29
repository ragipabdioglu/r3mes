"use client";

import { useEffect, useState } from "react";
import { Layers, TrendingUp, Clock, CheckCircle, Database, Activity } from "lucide-react";
import { useProposerNodes, useAggregations, useGradientPool } from "@/hooks/useProposerData";
import { logger } from "@/lib/logger";
import StatCard from "@/components/StatCard";
import WalletGuard from "@/components/WalletGuard";
import { SkeletonStatCard } from "@/components/SkeletonLoader";

function ProposerPageContent() {
  const [walletAddress, setWalletAddress] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  // Use React Query hooks instead of setInterval polling
  const { data: proposerNodesData, isLoading: nodesLoading, error: nodesError } = useProposerNodes(100, 0);
  const { data: aggregationsData, isLoading: aggregationsLoading, error: aggregationsError } = useAggregations(50, 0);
  const { data: gradientPool, isLoading: poolLoading, error: poolError } = useGradientPool(100, 0);
  
  const proposerNodes = proposerNodesData?.nodes ?? [];
  const aggregations = aggregationsData?.aggregations ?? [];
  const isLoading = nodesLoading || aggregationsLoading || poolLoading;

  useEffect(() => {
    setMounted(true);
    const address = localStorage.getItem("keplr_address");
    setWalletAddress(address);
  }, []);

  useEffect(() => {
    if (nodesError) {
      logger.error("Failed to fetch proposer nodes:", nodesError);
    }
    if (aggregationsError) {
      logger.error("Failed to fetch aggregations:", aggregationsError);
    }
    if (poolError) {
      logger.error("Failed to fetch gradient pool:", poolError);
    }
  }, [nodesError, aggregationsError, poolError]);

  const totalAggregations = proposerNodes.reduce((sum, n) => sum + n.total_aggregations, 0);
  const totalRewards = proposerNodes.reduce((sum, n) => {
    const rewards = parseFloat(n.total_rewards || "0");
    return sum + rewards;
  }, 0);

  return (
    <WalletGuard>
      <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] py-8 px-4 sm:py-10 sm:px-6 md:py-12 md:px-8">
        <div className="container mx-auto max-w-full sm:max-w-3xl md:max-w-5xl lg:max-w-7xl">
          <div className="mb-8">
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold gradient-text mb-2">Proposer Dashboard</h1>
            <p className="text-sm sm:text-base text-[var(--text-secondary)]">
              Manage gradient aggregation and proposer nodes
            </p>
          </div>

          {/* Overview Stats */}
          {isLoading ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5 md:gap-6 mb-8">
              {[1, 2, 3, 4].map((i) => (
                <SkeletonStatCard key={i} />
              ))}
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-5 md:gap-6 mb-8">
              <StatCard
                label="Proposer Nodes"
                value={proposerNodes.length.toString()}
                icon={<Layers className="w-5 h-5" />}
              />
              <StatCard
                label="Total Aggregations"
                value={totalAggregations.toLocaleString()}
                icon={<Activity className="w-5 h-5" />}
              />
              <StatCard
                label="Pending Gradients"
                value={gradientPool?.total_count.toString() || "0"}
                icon={<Database className="w-5 h-5" />}
              />
              <StatCard
                label="Total Rewards"
                value={`${totalRewards.toFixed(2)} REMES`}
                icon={<TrendingUp className="w-5 h-5" />}
              />
            </div>
          )}

          {/* Gradient Pool */}
          <div className="card mb-8">
            <h2 className="text-xl sm:text-2xl font-semibold mb-6 text-[var(--text-primary)]">Pending Gradients Pool</h2>
            
            {isLoading ? (
              <div className="text-center py-12">
                <div className="text-[var(--text-secondary)]">Loading gradient pool...</div>
              </div>
            ) : !gradientPool || gradientPool.total_count === 0 ? (
              <div className="text-center py-12">
                <div className="text-[var(--text-secondary)] mb-2">No pending gradients</div>
                <p className="text-sm text-[var(--text-muted)]">
                  Gradients will appear here when miners submit them
                </p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[600px]">
                  <thead>
                    <tr className="border-b border-[var(--border-color)]">
                      <th className="text-left py-4 px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                        Gradient ID
                      </th>
                      <th className="text-left py-4 px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                        Miner
                      </th>
                      <th className="text-left py-4 px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                        Training Round
                      </th>
                      <th className="text-left py-4 px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                        IPFS Hash
                      </th>
                      <th className="text-left py-4 px-4 text-xs sm:text-sm font-semibold text-[var(--text-secondary)]">
                        Status
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {gradientPool.pending_gradients.slice(0, 20).map((gradient, i) => (
                      <tr
                        key={i}
                        className="border-b border-[var(--border-color)] hover:bg-[var(--bg-tertiary)] transition-colors"
                      >
                        <td className="py-4 px-4">
                          <span className="font-semibold text-[var(--text-primary)] text-xs sm:text-sm">
                            #{gradient.id}
                          </span>
                        </td>
                        <td className="py-4 px-4">
                          <span className="text-xs text-[var(--text-secondary)] font-mono">
                            {gradient.miner ? `${gradient.miner.slice(0, 10)}...${gradient.miner.slice(-8)}` : "N/A"}
                          </span>
                        </td>
                        <td className="py-4 px-4">
                          <span className="text-xs text-[var(--text-secondary)]">
                            Round {gradient.training_round_id}
                          </span>
                        </td>
                        <td className="py-4 px-4">
                          <span className="text-xs text-[var(--text-secondary)] font-mono">
                            {gradient.ipfs_hash ? `${gradient.ipfs_hash.slice(0, 10)}...${gradient.ipfs_hash.slice(-8)}` : "N/A"}
                          </span>
                        </td>
                        <td className="py-4 px-4">
                          <span className="text-xs font-medium px-2 py-0.5 rounded-full bg-[var(--accent-primary)]/20 text-[var(--accent-primary)]">
                            {gradient.status}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Aggregation History */}
          <div className="card mb-8">
            <h2 className="text-xl sm:text-2xl font-semibold mb-6 text-[var(--text-primary)]">Recent Aggregations</h2>
            
            {isLoading ? (
              <div className="text-center py-12">
                <div className="text-[var(--text-secondary)]">Loading aggregations...</div>
              </div>
            ) : aggregations.length === 0 ? (
              <div className="text-center py-12">
                <div className="text-[var(--text-secondary)] mb-2">No aggregations found</div>
                <p className="text-sm text-[var(--text-muted)]">
                  Aggregation history will appear here
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {aggregations.slice(0, 10).map((agg) => (
                  <div
                    key={agg.aggregation_id}
                    className="p-4 rounded-xl border border-[var(--border-color)] hover:bg-[var(--bg-tertiary)] transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-semibold text-[var(--text-primary)] text-sm sm:text-base">
                          Aggregation #{agg.aggregation_id}
                        </div>
                        <div className="text-xs text-[var(--text-secondary)] mt-1">
                          Proposer: {agg.proposer.slice(0, 20)}...{agg.proposer.slice(-8)} | 
                          Participants: {agg.participant_count} | 
                          Round: {agg.training_round_id}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-xs text-[var(--text-muted)] font-mono">
                          {agg.aggregated_gradient_ipfs_hash.slice(0, 10)}...
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Proposer Nodes */}
          <div className="card">
            <h2 className="text-xl sm:text-2xl font-semibold mb-6 text-[var(--text-primary)]">Proposer Nodes</h2>
            
            {isLoading ? (
              <div className="text-center py-12">
                <div className="text-[var(--text-secondary)]">Loading proposer nodes...</div>
              </div>
            ) : proposerNodes.length === 0 ? (
              <div className="text-center py-12">
                <div className="text-[var(--text-secondary)] mb-2">No proposer nodes found</div>
                <p className="text-sm text-[var(--text-muted)]">
                  Register a proposer node to start aggregating gradients
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {proposerNodes.map((node) => (
                  <div
                    key={node.node_address}
                    className="p-4 rounded-xl border border-[var(--border-color)] hover:bg-[var(--bg-tertiary)] transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <div className="font-semibold text-[var(--text-primary)] text-sm sm:text-base">
                          {node.node_address.slice(0, 20)}...{node.node_address.slice(-8)}
                        </div>
                        <div className="text-xs text-[var(--text-secondary)] mt-1">
                          Status: {node.status} | Aggregations: {node.total_aggregations}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium text-[var(--accent-primary)]">
                          {parseFloat(node.total_rewards || "0").toFixed(2)} REMES
                        </div>
                        <div className="text-xs text-[var(--text-muted)]">Total Rewards</div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </WalletGuard>
  );
}

export default function ProposerPage() {
  return <ProposerPageContent />;
}

