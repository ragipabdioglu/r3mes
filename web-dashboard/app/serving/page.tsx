"use client";

import { useEffect, useState } from "react";
import { Server, Activity, Clock, CheckCircle, XCircle, AlertCircle } from "lucide-react";
import { useServingNodes, useServingNodeStats } from "@/hooks/useServingData";
import { useAnnouncer } from "@/hooks/useAccessibility";
import { formatAddress, formatLatency, formatNumber, formatPercentage } from "@/utils/formatters";
import { logger } from "@/lib/logger";
import StatCard from "@/components/StatCard";
import WalletGuard from "@/components/WalletGuard";
import { SkeletonStatCard } from "@/components/SkeletonLoader";

function ServingPageContent() {
  const [walletAddress, setWalletAddress] = useState<string | null>(null);
  const [selectedNode, setSelectedNode] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  // Accessibility announcer
  const { announce, announceError, announceSuccess, announceLoading } = useAnnouncer();

  // Use React Query hooks instead of setInterval polling
  const { data: servingNodesData, isLoading: nodesLoading, error: nodesError } = useServingNodes(100, 0);
  const { data: nodeStats, isLoading: statsLoading, error: statsError } = useServingNodeStats(selectedNode);
  
  const servingNodes = servingNodesData?.nodes ?? [];
  const isLoading = nodesLoading || statsLoading;

  useEffect(() => {
    setMounted(true);
    const address = localStorage.getItem("keplr_address");
    setWalletAddress(address);
  }, []);

  // Announce loading state changes
  useEffect(() => {
    if (nodesLoading) {
      announceLoading("serving nodes", true);
    } else if (servingNodes.length > 0) {
      announce(`Loaded ${servingNodes.length} serving nodes`);
    }
  }, [nodesLoading, servingNodes.length, announce, announceLoading]);

  useEffect(() => {
    if (nodesError) {
      logger.error("Failed to fetch serving nodes:", nodesError);
      announceError("Failed to load serving nodes");
    }
    if (statsError) {
      logger.error("Failed to fetch node stats:", statsError);
      announceError("Failed to load node statistics");
    }
  }, [nodesError, statsError, announceError]);

  const handleNodeSelect = (address: string) => {
    setSelectedNode(address);
    announce(`Selected node ${formatAddress(address)}`);
  };

  const activeNodes = servingNodes.filter((n) => n.is_available);
  const totalRequests = servingNodes.reduce((sum, n) => sum + n.total_requests, 0);
  const totalSuccessful = servingNodes.reduce((sum, n) => sum + n.successful_requests, 0);
  const avgLatency = servingNodes.length > 0
    ? servingNodes.reduce((sum, n) => sum + n.average_latency_ms, 0) / servingNodes.length
    : 0;

  return (
    <WalletGuard>
      <div className="min-h-screen bg-[var(--bg-primary)] text-[var(--text-primary)] py-8 px-4 sm:py-10 sm:px-6 md:py-12 md:px-8">
        <div className="container mx-auto max-w-full sm:max-w-3xl md:max-w-5xl lg:max-w-7xl">
          <div className="mb-8">
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold gradient-text mb-2">Serving Dashboard</h1>
            <p className="text-sm sm:text-base text-[var(--text-secondary)]">
              Manage AI model inference serving nodes
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
                label="Active Nodes"
                value={activeNodes.length.toString()}
                icon={<Server className="w-5 h-5" />}
              />
              <StatCard
                label="Total Requests"
                value={formatNumber(totalRequests)}
                icon={<Activity className="w-5 h-5" />}
              />
              <StatCard
                label="Success Rate"
                value={formatPercentage(totalRequests > 0 ? (totalSuccessful / totalRequests) * 100 : 0)}
                icon={<CheckCircle className="w-5 h-5" />}
              />
              <StatCard
                label="Avg Latency"
                value={formatLatency(avgLatency)}
                icon={<Clock className="w-5 h-5" />}
              />
            </div>
          )}

          {/* Serving Nodes List */}
          <div className="card mb-8">
            <h2 className="text-xl sm:text-2xl font-semibold mb-6 text-[var(--text-primary)]">Serving Nodes</h2>
            
            {isLoading ? (
              <div className="text-center py-12">
                <div className="text-[var(--text-secondary)]">Loading serving nodes...</div>
              </div>
            ) : servingNodes.length === 0 ? (
              <div className="text-center py-12">
                <div className="text-[var(--text-secondary)] mb-2">No serving nodes found</div>
                <p className="text-sm text-[var(--text-muted)]">
                  Register a serving node to start serving inference requests
                </p>
              </div>
            ) : (
              <div className="space-y-4">
                {servingNodes.map((node) => (
                  <div
                    key={node.node_address}
                    onClick={() => handleNodeSelect(node.node_address)}
                    className={`p-4 rounded-xl border transition-all cursor-pointer ${
                      selectedNode === node.node_address
                        ? "border-[var(--accent-primary)] bg-[var(--accent-primary)]/10"
                        : "border-[var(--border-color)] hover:bg-[var(--bg-tertiary)]"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div className={`w-3 h-3 rounded-full ${
                          node.is_available ? "bg-[var(--success)]" : "bg-[var(--error)]"
                        }`} />
                        <div>
                          <div className="font-semibold text-[var(--text-primary)] text-sm sm:text-base">
                            {formatAddress(node.node_address, 20, 8)}
                          </div>
                          <div className="text-xs text-[var(--text-secondary)] mt-1">
                            Model: {node.model_version || "N/A"} | Requests: {formatNumber(node.total_requests)}
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-sm font-medium text-[var(--accent-primary)]">
                          {formatPercentage(node.total_requests > 0 ? (node.successful_requests / node.total_requests) * 100 : 0)}
                        </div>
                        <div className="text-xs text-[var(--text-muted)]">
                          {formatLatency(node.average_latency_ms)} avg
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Node Statistics */}
          {selectedNode && nodeStats && (
            <div className="card">
              <h2 className="text-xl sm:text-2xl font-semibold mb-6 text-[var(--text-primary)]">
                Node Statistics
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5 md:gap-6">
                <div className="p-4 rounded-xl bg-[var(--bg-secondary)]">
                  <div className="text-sm text-[var(--text-secondary)] mb-1">Total Requests</div>
                  <div className="text-2xl font-bold text-[var(--text-primary)]">
                    {formatNumber(nodeStats.total_requests)}
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-[var(--bg-secondary)]">
                  <div className="text-sm text-[var(--text-secondary)] mb-1">Success Rate</div>
                  <div className="text-2xl font-bold text-[var(--success)]">
                    {formatPercentage(nodeStats.success_rate)}
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-[var(--bg-secondary)]">
                  <div className="text-sm text-[var(--text-secondary)] mb-1">Average Latency</div>
                  <div className="text-2xl font-bold text-[var(--text-primary)]">
                    {formatLatency(nodeStats.average_latency_ms)}
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-[var(--bg-secondary)]">
                  <div className="text-sm text-[var(--text-secondary)] mb-1">Successful</div>
                  <div className="text-2xl font-bold text-[var(--success)]">
                    {formatNumber(nodeStats.successful_requests)}
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-[var(--bg-secondary)]">
                  <div className="text-sm text-[var(--text-secondary)] mb-1">Failed</div>
                  <div className="text-2xl font-bold text-[var(--error)]">
                    {formatNumber(nodeStats.failed_requests)}
                  </div>
                </div>
                <div className="p-4 rounded-xl bg-[var(--bg-secondary)]">
                  <div className="text-sm text-[var(--text-secondary)] mb-1">Status</div>
                  <div className="flex items-center gap-2 mt-2">
                    {nodeStats.is_available ? (
                      <>
                        <CheckCircle className="w-5 h-5 text-[var(--success)]" />
                        <span className="text-lg font-semibold text-[var(--success)]">Available</span>
                      </>
                    ) : (
                      <>
                        <XCircle className="w-5 h-5 text-[var(--error)]" />
                        <span className="text-lg font-semibold text-[var(--error)]">Unavailable</span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </WalletGuard>
  );
}

export default function ServingPage() {
  return <ServingPageContent />;
}

