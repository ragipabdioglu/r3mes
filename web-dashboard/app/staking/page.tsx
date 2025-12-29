"use client";

import { useState } from "react";
import StakingDashboard from "@/components/StakingDashboard";
import WalletGuard from "@/components/WalletGuard";
import StatCard from "@/components/StatCard";
import { useQuery } from "@tanstack/react-query";
import { Wallet, TrendingUp, Clock } from "lucide-react";
import { useWallet } from "@/contexts/WalletContext";

function StakingPageContent() {
  const { walletAddress } = useWallet();

  // Fetch staking analytics
  const { data: stakingAnalytics, isLoading: analyticsLoading } = useQuery({
    queryKey: ["staking", "analytics", walletAddress],
    queryFn: async () => {
      if (!walletAddress) return null;

      const [stakingResponse, rewardsResponse] = await Promise.all([
        fetch(`/api/blockchain/cosmos/staking/v1beta1/delegations/${walletAddress}`),
        fetch(`/api/blockchain/cosmos/distribution/v1beta1/delegators/${walletAddress}/rewards`),
      ]);

      if (!stakingResponse.ok || !rewardsResponse.ok) {
        throw new Error("Failed to fetch staking analytics");
      }

      const stakingData = await stakingResponse.json();
      const rewardsData = await rewardsResponse.json();

      // Calculate total rewards
      const totalRewards = rewardsData.rewards?.reduce((sum: number, reward: any) => {
        const rewardAmount = reward.reward?.find((r: any) => r.denom === "uremes")?.amount || "0";
        return sum + parseFloat(rewardAmount);
      }, 0) || 0;

      return {
        totalStaked: stakingData.total_staked || "0",
        pendingRewards: totalRewards.toString(),
        totalDelegations: stakingData.delegation_responses?.length || 0,
      };
    },
    enabled: !!walletAddress,
    refetchInterval: 30000,
  });

  const formatAmount = (amount: string): string => {
    const num = parseFloat(amount) / 1e6; // Convert from uremes to REMES
    if (isNaN(num)) return "0.00";
    return num.toFixed(2);
  };

  return (
    <div className="min-h-screen bg-slate-900 text-slate-100 py-8 px-4 sm:py-10 sm:px-6 md:py-12 md:px-8">
      <div className="container mx-auto max-w-full sm:max-w-2xl md:max-w-4xl lg:max-w-6xl xl:max-w-7xl">
        <div className="mb-6 sm:mb-8">
          <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold gradient-text mb-2">Staking & Validators</h1>
          <p className="text-sm sm:text-base text-slate-400">
            Delegate your tokens to validators and earn staking rewards
          </p>
        </div>

        {/* Analytics Cards */}
        {!analyticsLoading && stakingAnalytics && (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-5 md:gap-6 mb-6 sm:mb-8">
            <StatCard
              label="Total Staked"
              value={`${formatAmount(stakingAnalytics.totalStaked)} REMES`}
              icon={<Wallet className="w-5 h-5" />}
            />
            <StatCard
              label="Pending Rewards"
              value={`${formatAmount(stakingAnalytics.pendingRewards)} REMES`}
              icon={<TrendingUp className="w-5 h-5" />}
            />
            <StatCard
              label="Active Delegations"
              value={stakingAnalytics.totalDelegations.toString()}
              icon={<Clock className="w-5 h-5" />}
            />
          </div>
        )}

        {/* Staking Dashboard */}
        <div className="card p-4 sm:p-5 md:p-6">
          <StakingDashboard />
        </div>
      </div>
    </div>
  );
}

export default function StakingPage() {
  return (
    <WalletGuard>
      <StakingPageContent />
    </WalletGuard>
  );
}

