"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import ValidatorList from "./ValidatorList";
import DelegateForm from "./DelegateForm";

interface StakingInfo {
  total_staked: string;
  pending_rewards: string;
  unbonding: string;
  unbonding_end_time: string;
}

interface UnbondingEntry {
  validator_address: string;
  validator_name: string;
  amount: string;
  completion_time: string;
  remaining_days: number;
}

interface Delegation {
  validator_address: string;
  validator_name: string;
  amount: string;
  rewards: string;
}

export default function StakingDashboard() {
  const [selectedValidator, setSelectedValidator] = useState<string | null>(null);
  const [showDelegateForm, setShowDelegateForm] = useState(false);
  const [delegateAction, setDelegateAction] = useState<"delegate" | "undelegate" | "redelegate" | null>(null);
  const [activeTab, setActiveTab] = useState<"validators" | "delegations" | "unbonding">("validators");
  const queryClient = useQueryClient();

  const { data: stakingInfo, refetch: refetchStaking } = useQuery<StakingInfo>({
    queryKey: ["staking", "info"],
    queryFn: async () => {
      const walletAddress = localStorage.getItem("keplr_address");
      if (!walletAddress) throw new Error("Wallet not connected");
      const response = await fetch(`/api/blockchain/cosmos/staking/v1beta1/delegations/${walletAddress}`);
      if (!response.ok) throw new Error("Failed to fetch staking info");
      const data = await response.json();
      return { total_staked: data.total_staked || "0", pending_rewards: data.pending_rewards || "0", unbonding: data.unbonding || "0", unbonding_end_time: data.unbonding_end_time || "" };
    },
    refetchInterval: 30000,
  });

  const { data: unbondingEntries = [] } = useQuery<UnbondingEntry[]>({
    queryKey: ["staking", "unbonding"],
    queryFn: async () => {
      const walletAddress = localStorage.getItem("keplr_address");
      if (!walletAddress) return [];
      const response = await fetch(`/api/blockchain/cosmos/staking/v1beta1/delegators/${walletAddress}/unbonding_delegations`);
      if (!response.ok) return [];
      const data = await response.json();
      const entries: UnbondingEntry[] = [];
      for (const unbonding of data.unbonding_responses || []) {
        for (const entry of unbonding.entries || []) {
          const completionTime = new Date(entry.completion_time);
          const remainingMs = completionTime.getTime() - Date.now();
          entries.push({
            validator_address: unbonding.validator_address,
            validator_name: unbonding.validator_name || unbonding.validator_address.slice(0, 12) + "...",
            amount: entry.balance,
            completion_time: entry.completion_time,
            remaining_days: Math.max(0, Math.ceil(remainingMs / (1000 * 60 * 60 * 24))),
          });
        }
      }
      return entries.sort((a, b) => a.remaining_days - b.remaining_days);
    },
    refetchInterval: 60000,
  });

  const { data: delegations = [] } = useQuery<Delegation[]>({
    queryKey: ["staking", "delegations"],
    queryFn: async () => {
      const walletAddress = localStorage.getItem("keplr_address");
      if (!walletAddress) return [];
      const response = await fetch(`/api/blockchain/cosmos/staking/v1beta1/delegations/${walletAddress}`);
      if (!response.ok) return [];
      const data = await response.json();
      return (data.delegation_responses || []).map((d: any) => ({
        validator_address: d.delegation.validator_address,
        validator_name: d.validator_name || d.delegation.validator_address.slice(0, 12) + "...",
        amount: d.balance.amount,
        rewards: d.rewards || "0",
      }));
    },
    refetchInterval: 30000,
  });

  const handleDelegate = (addr: string) => { setSelectedValidator(addr); setDelegateAction("delegate"); setShowDelegateForm(true); };
  const handleUndelegate = (addr: string) => { setSelectedValidator(addr); setDelegateAction("undelegate"); setShowDelegateForm(true); };
  const handleRedelegate = (addr: string) => { setSelectedValidator(addr); setDelegateAction("redelegate"); setShowDelegateForm(true); };

  const handleClaimRewards = async () => {
    try {
      const walletAddress = localStorage.getItem("keplr_address");
      if (!walletAddress) { alert("Please connect your wallet first"); return; }
      const rewardsResponse = await fetch(`/api/blockchain/cosmos/distribution/v1beta1/delegators/${walletAddress}/rewards`);
      if (!rewardsResponse.ok) throw new Error("Failed to fetch delegator rewards");
      const rewardsData = await rewardsResponse.json();
      const rewards = rewardsData.rewards || [];
      if (rewards.length === 0) { alert("No pending rewards to claim"); return; }
      const { signAndBroadcastTransaction } = await import("@/lib/keplr");
      const messages = rewards.map((reward: any) => ({ typeUrl: "/cosmos.distribution.v1beta1.MsgWithdrawDelegatorReward", value: { delegatorAddress: walletAddress, validatorAddress: reward.validator_address } }));
      await signAndBroadcastTransaction(messages, "Claim staking rewards");
      refetchStaking();
      queryClient.invalidateQueries({ queryKey: ["staking"] });
    } catch (error: any) {
      console.error("Failed to claim rewards:", error);
      alert(error.message || "Failed to claim rewards. Please try again.");
    }
  };

  const formatAmount = (amount: string) => (parseFloat(amount) / 1e6).toFixed(2);
  const getUnbondingProgress = (remainingDays: number) => ((21 - remainingDays) / 21) * 100;

  const handleSuccess = () => {
    setShowDelegateForm(false);
    setSelectedValidator(null);
    setDelegateAction(null);
    refetchStaking();
    queryClient.invalidateQueries({ queryKey: ["staking"] });
  };

  return (
    <div className="p-6 text-slate-100">
      {/* Header */}
      <div className="mb-8">
        <h2 className="text-[28px] font-semibold mb-2 bg-gradient-to-r from-blue-500 to-violet-500 bg-clip-text text-transparent">Staking & Validators</h2>
        <p className="text-slate-400 text-sm">Delegate your tokens to validators and earn rewards</p>
      </div>

      {/* Overview Cards */}
      <div className="grid grid-cols-[repeat(auto-fit,minmax(250px,1fr))] gap-5 mb-8">
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Total Staked</div>
          <div className="text-2xl font-semibold text-slate-100 mb-3">{stakingInfo ? formatAmount(stakingInfo.total_staked) : "0.00"} REMES</div>
          <div className="text-xs text-slate-500">{delegations.length} validator(s)</div>
        </div>
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Pending Rewards</div>
          <div className="text-2xl font-semibold text-green-500 mb-3">{stakingInfo ? formatAmount(stakingInfo.pending_rewards) : "0.00"} REMES</div>
          <button onClick={handleClaimRewards} className="px-4 py-2 bg-green-500 hover:bg-green-600 text-white text-xs font-medium rounded-lg transition-colors mt-2">Claim Rewards</button>
        </div>
        <div className="bg-slate-800 border border-slate-700 rounded-xl p-5">
          <div className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Unbonding</div>
          <div className="text-2xl font-semibold text-slate-100 mb-3">{stakingInfo ? formatAmount(stakingInfo.unbonding) : "0.00"} REMES</div>
          <div className="text-xs text-slate-500">{unbondingEntries.length > 0 ? `${unbondingEntries.length} pending unbonding(s)` : "No unbonding tokens"}</div>
        </div>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 mb-6 pb-4 border-b border-slate-700">
        {[{ key: "validators", label: "All Validators" }, { key: "delegations", label: `My Delegations (${delegations.length})` }, { key: "unbonding", label: `Unbonding (${unbondingEntries.length})` }].map(tab => (
          <button key={tab.key} onClick={() => setActiveTab(tab.key as any)} className={`px-5 py-2.5 rounded-lg text-sm font-medium transition-all border ${activeTab === tab.key ? "bg-blue-500 border-blue-500 text-white" : "bg-transparent border-slate-700 text-slate-400 hover:border-slate-600 hover:text-slate-100"}`}>{tab.label}</button>
        ))}
      </div>

      {/* Validators Tab */}
      {activeTab === "validators" && <ValidatorList onDelegate={handleDelegate} onUndelegate={handleUndelegate} onRedelegate={handleRedelegate} />}

      {/* Delegations Tab */}
      {activeTab === "delegations" && (
        <div className="flex flex-col gap-4">
          {delegations.length === 0 ? (
            <div className="text-center py-16 text-slate-400">
              <p className="mb-4">You haven't delegated to any validators yet.</p>
              <button onClick={() => setActiveTab("validators")} className="px-5 py-2.5 bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium rounded-lg transition-colors">Browse Validators</button>
            </div>
          ) : (
            delegations.map(d => (
              <div key={d.validator_address} className="bg-slate-800 border border-slate-700 rounded-xl p-5 grid grid-cols-[2fr_1fr_1fr_auto] gap-5 items-center max-md:grid-cols-1 max-md:gap-4">
                <div>
                  <div className="text-base font-semibold text-slate-100 mb-1">{d.validator_name}</div>
                  <div className="text-xs text-slate-500 font-mono">{d.validator_address}</div>
                </div>
                <div className="max-md:text-left">
                  <div className="text-xs text-slate-500 mb-1">Staked</div>
                  <div className="text-base font-semibold text-slate-100">{formatAmount(d.amount)} REMES</div>
                </div>
                <div className="max-md:text-left">
                  <div className="text-xs text-slate-500 mb-1">Rewards</div>
                  <div className="text-base font-semibold text-green-500">{formatAmount(d.rewards)} REMES</div>
                </div>
                <div className="flex gap-2 flex-wrap">
                  <button onClick={() => handleDelegate(d.validator_address)} className="px-3 py-2 bg-blue-500/10 hover:bg-blue-500/20 text-blue-500 text-xs font-medium rounded-md transition-colors">Delegate More</button>
                  <button onClick={() => handleUndelegate(d.validator_address)} className="px-3 py-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 text-xs font-medium rounded-md transition-colors">Undelegate</button>
                  <button onClick={() => handleRedelegate(d.validator_address)} className="px-3 py-2 bg-amber-500/10 hover:bg-amber-500/20 text-amber-500 text-xs font-medium rounded-md transition-colors">Redelegate</button>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* Unbonding Tab */}
      {activeTab === "unbonding" && (
        <div className="flex flex-col gap-4">
          {unbondingEntries.length === 0 ? (
            <div className="text-center py-16 text-slate-400"><p>No tokens currently unbonding.</p></div>
          ) : (
            <>
              <div className="flex items-center gap-3 px-4 py-3 bg-blue-500/10 border border-blue-500/20 rounded-lg text-slate-400 text-sm">
                <span>ℹ️</span>
                <span>Unbonding period is 21 days. Tokens will be available after the unbonding period completes.</span>
              </div>
              {unbondingEntries.map((entry, i) => (
                <div key={`${entry.validator_address}-${i}`} className="bg-slate-800 border border-slate-700 rounded-xl p-5">
                  <div className="flex justify-between items-start mb-4">
                    <div>
                      <div className="text-base font-semibold text-slate-100 mb-1">{entry.validator_name}</div>
                      <div className="text-xs text-slate-500 font-mono">{entry.validator_address}</div>
                    </div>
                    <div className="text-lg font-semibold text-amber-500">{formatAmount(entry.amount)} REMES</div>
                  </div>
                  <div className="mt-3">
                    <div className="h-2 bg-slate-900 rounded overflow-hidden mb-2">
                      <div className="h-full bg-gradient-to-r from-blue-500 to-violet-500 rounded transition-all duration-300" style={{ width: `${getUnbondingProgress(entry.remaining_days)}%` }} />
                    </div>
                    <div className="flex justify-between text-xs">
                      <span className="text-amber-500 font-medium">{entry.remaining_days === 0 ? "Ready to claim" : `${entry.remaining_days} day${entry.remaining_days !== 1 ? "s" : ""} remaining`}</span>
                      <span className="text-slate-500">Completes: {new Date(entry.completion_time).toLocaleDateString()}</span>
                    </div>
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      )}

      {/* Delegate Form Modal */}
      {showDelegateForm && selectedValidator && delegateAction && (
        <DelegateForm validatorAddress={selectedValidator} action={delegateAction} onClose={() => { setShowDelegateForm(false); setSelectedValidator(null); setDelegateAction(null); }} onSuccess={handleSuccess} />
      )}
    </div>
  );
}
