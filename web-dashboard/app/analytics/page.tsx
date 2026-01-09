"use client";

import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, PieChart, Pie, Cell } from "recharts";
import { TrendingUp, Users, Activity, Zap, Network, DollarSign, Gauge } from "lucide-react";
import { getAnalytics } from "@/lib/api";
import { useState, useEffect } from "react";
import { useAnnouncer } from "@/hooks/useAccessibility";
import { formatNumber, formatPercentage, formatLatency } from "@/utils/formatters";

export default function AnalyticsPage() {
  const [activeTab, setActiveTab] = useState<"overview" | "network" | "mining" | "economics">("overview");
  const { announce, announceLoading, announceError } = useAnnouncer();
  
  const { data: analytics, isLoading, error } = useQuery({
    queryKey: ["analytics"],
    queryFn: () => getAnalytics(),
    refetchInterval: 30000,
  });

  // Announce loading and error states
  useEffect(() => {
    if (isLoading) {
      announceLoading("analytics data", true);
    } else if (analytics) {
      announce("Analytics data loaded");
    }
    if (error) {
      announceError("Failed to load analytics data");
    }
  }, [isLoading, analytics, error, announce, announceLoading, announceError]);

  const handleTabChange = (tab: "overview" | "network" | "mining" | "economics") => {
    setActiveTab(tab);
    announce(`Switched to ${tab} tab`);
  };

  const { data: networkGrowth, isLoading: networkLoading } = useQuery({
    queryKey: ["analytics", "network-growth"],
    queryFn: async () => {
      const res = await fetch("/api/analytics/network-growth?days=30");
      return res.json();
    },
    enabled: activeTab === "network" || activeTab === "overview",
  });

  const { data: miningEfficiency, isLoading: miningLoading } = useQuery({
    queryKey: ["analytics", "mining-efficiency"],
    queryFn: async () => {
      const res = await fetch("/api/analytics/mining-efficiency?days=7");
      return res.json();
    },
    enabled: activeTab === "mining" || activeTab === "overview",
  });

  const { data: economicAnalysis, isLoading: economicLoading } = useQuery({
    queryKey: ["analytics", "economic-analysis"],
    queryFn: async () => {
      const res = await fetch("/api/analytics/economic-analysis?days=30");
      return res.json();
    },
    enabled: activeTab === "economics" || activeTab === "overview",
  });

  if (isLoading) {
    return (
      <div className="min-h-screen bg-slate-900 text-slate-100 flex items-center justify-center">
        <div className="animate-pulse text-slate-400">Loading analytics...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100">
      <div className="container mx-auto px-4 py-16">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-green-400 to-cyan-400 bg-clip-text text-transparent">
            Analytics Dashboard
          </h1>
          <p className="text-slate-400">Insights into R3MES network performance and usage</p>
        </motion.div>

        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-12">
          <StatCard
            icon={<Users className="w-6 h-6" />}
            label="Active Users"
            value={formatNumber(analytics?.user_engagement?.active_users || 0)}
          />
          <StatCard
            icon={<Activity className="w-6 h-6" />}
            label="API Calls"
            value={formatNumber(analytics?.api_usage?.total_requests || 0)}
          />
          <StatCard
            icon={<Zap className="w-6 h-6" />}
            label="Avg Latency"
            value={formatLatency(analytics?.model_performance?.average_latency || 0)}
          />
          <StatCard
            icon={<TrendingUp className="w-6 h-6" />}
            label="Success Rate"
            value={formatPercentage((analytics?.model_performance?.success_rate || 0) * 100)}
          />
        </div>

        {/* Tabs */}
        <div className="flex gap-4 mb-8 border-b border-slate-800" role="tablist" aria-label="Analytics categories">
          {[
            { id: "overview", label: "Overview" },
            { id: "network", label: "Network Growth" },
            { id: "mining", label: "Mining Efficiency" },
            { id: "economics", label: "Economics" },
          ].map((tab) => (
            <button
              key={tab.id}
              onClick={() => handleTabChange(tab.id as any)}
              role="tab"
              aria-selected={activeTab === tab.id}
              aria-controls={`${tab.id}-panel`}
              className={`px-6 py-3 font-semibold border-b-2 transition-colors ${
                activeTab === tab.id
                  ? "border-green-500 text-green-400"
                  : "border-transparent text-slate-400 hover:text-slate-200"
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === "overview" && (
          <div id="overview-panel" role="tabpanel" aria-labelledby="overview-tab" className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ChartCard title="API Usage by Endpoint">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={analytics?.api_usage?.endpoints_data || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="endpoint" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #374151" }} />
                  <Bar dataKey="count" fill="#10b981" />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>

            <ChartCard title="Model Performance Trend">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={analytics?.model_performance?.trend || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="date" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #374151" }} />
                  <Line type="monotone" dataKey="latency" stroke="#10b981" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
        )}

        {activeTab === "network" && networkGrowth && (
          <div id="network-panel" role="tabpanel" aria-labelledby="network-tab" className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ChartCard title="Network Growth Metrics">
              <div className="space-y-4">
                <MetricRow label="Total Miners" value={networkGrowth.metrics?.total_miners?.current} growth={networkGrowth.metrics?.total_miners?.growth_rate} />
                <MetricRow label="Total Validators" value={networkGrowth.metrics?.total_validators?.current} growth={networkGrowth.metrics?.total_validators?.growth_rate} />
                <MetricRow label="Network Hashrate" value={`${networkGrowth.metrics?.network_hashrate?.current?.toFixed(1)} H/s`} growth={networkGrowth.metrics?.network_hashrate?.growth_rate} />
              </div>
            </ChartCard>
            <ChartCard title="Growth Timeline">
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={networkGrowth.timeline || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="date" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #374151" }} />
                  <Line type="monotone" dataKey="miners" stroke="#10b981" name="Miners" />
                  <Line type="monotone" dataKey="validators" stroke="#06b6d4" name="Validators" />
                </LineChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
        )}

        {activeTab === "mining" && miningEfficiency && (
          <div id="mining-panel" role="tabpanel" aria-labelledby="mining-tab" className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ChartCard title="Mining Efficiency">
              <div className="space-y-4">
                <MetricRow label="Hashrate" value={`${miningEfficiency.hashrate?.toFixed(1)} H/s`} />
                <MetricRow label="GPU Utilization" value={`${miningEfficiency.gpu_utilization?.toFixed(1)}%`} />
                <MetricRow label="Power Efficiency" value={`${miningEfficiency.power_efficiency?.toFixed(2)} G/W`} />
                <MetricRow label="Uptime" value={`${miningEfficiency.uptime_percentage?.toFixed(1)}%`} />
              </div>
            </ChartCard>
            <ChartCard title="Earnings vs Hashrate">
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[{ name: "Efficiency", value: miningEfficiency.earnings_per_hashrate || 0 }]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip contentStyle={{ backgroundColor: "#1e293b", border: "1px solid #374151" }} />
                  <Bar dataKey="value" fill="#10b981" />
                </BarChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
        )}

        {activeTab === "economics" && economicAnalysis && (
          <div id="economics-panel" role="tabpanel" aria-labelledby="economics-tab" className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ChartCard title="Tokenomics">
              <div className="space-y-4">
                <MetricRow label="Total Supply" value={economicAnalysis.tokenomics?.total_supply?.toLocaleString()} />
                <MetricRow label="Circulating Supply" value={economicAnalysis.tokenomics?.circulating_supply?.toLocaleString()} />
                <MetricRow label="Staking Ratio" value={`${economicAnalysis.tokenomics?.staking_ratio?.toFixed(1)}%`} />
              </div>
            </ChartCard>
            <ChartCard title="Rewards Distribution">
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={[
                      { name: "Miners", value: economicAnalysis.rewards?.miner_rewards || 0 },
                      { name: "Validators", value: economicAnalysis.rewards?.validator_rewards || 0 },
                      { name: "Treasury", value: economicAnalysis.rewards?.treasury || 0 },
                    ]}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {[0, 1, 2].map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={["#10b981", "#06b6d4", "#8b5cf6"][index]} />
                    ))}
                  </Pie>
                  <Tooltip />
                </PieChart>
              </ResponsiveContainer>
            </ChartCard>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ icon, label, value }: { icon: JSX.Element; label: string; value: string | number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card"
    >
      <div className="flex items-center gap-4">
        <div className="text-green-400">{icon}</div>
        <div>
          <div className="text-2xl font-bold">{value}</div>
          <div className="text-sm text-slate-400">{label}</div>
        </div>
      </div>
    </motion.div>
  );
}

function ChartCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="card"
    >
      <h3 className="text-xl font-bold mb-4">{title}</h3>
      {children}
    </motion.div>
  );
}

function MetricRow({ label, value, growth }: { label: string; value: string | number; growth?: number }) {
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-800">
      <span className="text-slate-400">{label}</span>
      <div className="flex items-center gap-2">
        <span className="font-bold">{value}</span>
        {growth !== undefined && growth > 0 && (
          <span className="text-green-400 text-sm flex items-center gap-1">
            <TrendingUp className="w-4 h-4" />
            +{growth.toFixed(1)}%
          </span>
        )}
      </div>
    </div>
  );
}

