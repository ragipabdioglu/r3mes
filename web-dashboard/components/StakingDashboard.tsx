"use client";

import { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import ValidatorList from "./ValidatorList";
import DelegateForm from "./DelegateForm";
import "./StakingDashboard.css";

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

  const { data: stakingInfo, isLoading, refetch: refetchStaking } = useQuery<StakingInfo>({
    queryKey: ["staking", "info"],
    queryFn: async () => {
      const walletAddress = localStorage.getItem("keplr_address");
      if (!walletAddress) {
        throw new Error("Wallet not connected");
      }
      const response = await fetch(`/api/blockchain/cosmos/staking/v1beta1/delegations/${walletAddress}`);
      if (!response.ok) {
        throw new Error("Failed to fetch staking info");
      }
      const data = await response.json();
      return {
        total_staked: data.total_staked || "0",
        pending_rewards: data.pending_rewards || "0",
        unbonding: data.unbonding || "0",
        unbonding_end_time: data.unbonding_end_time || "",
      };
    },
    refetchInterval: 30000,
  });

  // Fetch unbonding delegations
  const { data: unbondingEntries = [] } = useQuery<UnbondingEntry[]>({
    queryKey: ["staking", "unbonding"],
    queryFn: async () => {
      const walletAddress = localStorage.getItem("keplr_address");
      if (!walletAddress) return [];
      
      const response = await fetch(
        `/api/blockchain/cosmos/staking/v1beta1/delegators/${walletAddress}/unbonding_delegations`
      );
      if (!response.ok) return [];
      
      const data = await response.json();
      const entries: UnbondingEntry[] = [];
      
      for (const unbonding of data.unbonding_responses || []) {
        for (const entry of unbonding.entries || []) {
          const completionTime = new Date(entry.completion_time);
          const now = new Date();
          const remainingMs = completionTime.getTime() - now.getTime();
          const remainingDays = Math.max(0, Math.ceil(remainingMs / (1000 * 60 * 60 * 24)));
          
          entries.push({
            validator_address: unbonding.validator_address,
            validator_name: unbonding.validator_name || unbonding.validator_address.slice(0, 12) + "...",
            amount: entry.balance,
            completion_time: entry.completion_time,
            remaining_days: remainingDays,
          });
        }
      }
      
      return entries.sort((a, b) => a.remaining_days - b.remaining_days);
    },
    refetchInterval: 60000,
  });

  // Fetch current delegations
  const { data: delegations = [] } = useQuery<Delegation[]>({
    queryKey: ["staking", "delegations"],
    queryFn: async () => {
      const walletAddress = localStorage.getItem("keplr_address");
      if (!walletAddress) return [];
      
      const response = await fetch(
        `/api/blockchain/cosmos/staking/v1beta1/delegations/${walletAddress}`
      );
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

  const handleDelegate = (validatorAddress: string) => {
    setSelectedValidator(validatorAddress);
    setDelegateAction("delegate");
    setShowDelegateForm(true);
  };

  const handleUndelegate = (validatorAddress: string) => {
    setSelectedValidator(validatorAddress);
    setDelegateAction("undelegate");
    setShowDelegateForm(true);
  };

  const handleRedelegate = (validatorAddress: string) => {
    setSelectedValidator(validatorAddress);
    setDelegateAction("redelegate");
    setShowDelegateForm(true);
  };

  const handleClaimRewards = async () => {
    try {
      const walletAddress = localStorage.getItem("keplr_address");
      if (!walletAddress) {
        alert("Please connect your wallet first");
        return;
      }

      // Query delegator rewards to get all validators with pending rewards
      const rewardsResponse = await fetch(
        `/api/blockchain/cosmos/distribution/v1beta1/delegators/${walletAddress}/rewards`
      );

      if (!rewardsResponse.ok) {
        throw new Error("Failed to fetch delegator rewards");
      }

      const rewardsData = await rewardsResponse.json();
      const rewards = rewardsData.rewards || [];

      if (rewards.length === 0) {
        alert("No pending rewards to claim");
        return;
      }

      // Import signAndBroadcastTransaction
      const { signAndBroadcastTransaction } = await import("@/lib/keplr");

      // Create MsgWithdrawDelegatorReward messages for each validator
      const messages = rewards.map((reward: any) => ({
        typeUrl: "/cosmos.distribution.v1beta1.MsgWithdrawDelegatorReward",
        value: {
          delegatorAddress: walletAddress,
          validatorAddress: reward.validator_address,
        },
      }));

      // Sign and broadcast transaction
      await signAndBroadcastTransaction(messages, "Claim staking rewards");

      // Refetch staking info
      refetchStaking();
      queryClient.invalidateQueries({ queryKey: ["staking"] });
    } catch (error: any) {
      console.error("Failed to claim rewards:", error);
      alert(error.message || "Failed to claim rewards. Please try again.");
    }
  };

  const formatAmount = (amount: string) => {
    return (parseFloat(amount) / 1e6).toFixed(2);
  };

  const getUnbondingProgress = (remainingDays: number) => {
    const totalDays = 21; // Standard unbonding period
    return ((totalDays - remainingDays) / totalDays) * 100;
  };

  const handleSuccess = () => {
    setShowDelegateForm(false);
    setSelectedValidator(null);
    setDelegateAction(null);
    refetchStaking();
    queryClient.invalidateQueries({ queryKey: ["staking"] });
  };

  return (
    <div className="staking-dashboard">
      <div className="staking-header">
        <h2>Staking & Validators</h2>
        <p className="subtitle">Delegate your tokens to validators and earn rewards</p>
      </div>

      {/* Staking Overview */}
      <div className="staking-overview">
        <div className="overview-card">
          <div className="card-label">Total Staked</div>
          <div className="card-value">
            {stakingInfo ? formatAmount(stakingInfo.total_staked) : "0.00"} REMES
          </div>
          <div className="card-subtext">{delegations.length} validator(s)</div>
        </div>
        <div className="overview-card">
          <div className="card-label">Pending Rewards</div>
          <div className="card-value rewards">
            {stakingInfo ? formatAmount(stakingInfo.pending_rewards) : "0.00"} REMES
          </div>
          <button onClick={handleClaimRewards} className="claim-btn">
            Claim Rewards
          </button>
        </div>
        <div className="overview-card">
          <div className="card-label">Unbonding</div>
          <div className="card-value">
            {stakingInfo ? formatAmount(stakingInfo.unbonding) : "0.00"} REMES
          </div>
          <div className="card-subtext">
            {unbondingEntries.length > 0 
              ? `${unbondingEntries.length} pending unbonding(s)`
              : "No unbonding tokens"
            }
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="staking-tabs">
        <button 
          className={`staking-tab ${activeTab === "validators" ? "active" : ""}`}
          onClick={() => setActiveTab("validators")}
        >
          All Validators
        </button>
        <button 
          className={`staking-tab ${activeTab === "delegations" ? "active" : ""}`}
          onClick={() => setActiveTab("delegations")}
        >
          My Delegations ({delegations.length})
        </button>
        <button 
          className={`staking-tab ${activeTab === "unbonding" ? "active" : ""}`}
          onClick={() => setActiveTab("unbonding")}
        >
          Unbonding ({unbondingEntries.length})
        </button>
      </div>

      {/* Validators Tab */}
      {activeTab === "validators" && (
        <ValidatorList
          onDelegate={handleDelegate}
          onUndelegate={handleUndelegate}
          onRedelegate={handleRedelegate}
        />
      )}

      {/* My Delegations Tab */}
      {activeTab === "delegations" && (
        <div className="delegations-list">
          {delegations.length === 0 ? (
            <div className="empty-state">
              <p>You haven't delegated to any validators yet.</p>
              <button onClick={() => setActiveTab("validators")} className="browse-btn">
                Browse Validators
              </button>
            </div>
          ) : (
            delegations.map((delegation) => (
              <div key={delegation.validator_address} className="delegation-card">
                <div className="delegation-info">
                  <div className="validator-name">{delegation.validator_name}</div>
                  <div className="validator-address">{delegation.validator_address}</div>
                </div>
                <div className="delegation-amount">
                  <div className="amount-label">Staked</div>
                  <div className="amount-value">{formatAmount(delegation.amount)} REMES</div>
                </div>
                <div className="delegation-rewards">
                  <div className="rewards-label">Rewards</div>
                  <div className="rewards-value">{formatAmount(delegation.rewards)} REMES</div>
                </div>
                <div className="delegation-actions">
                  <button onClick={() => handleDelegate(delegation.validator_address)} className="action-btn delegate">
                    Delegate More
                  </button>
                  <button onClick={() => handleUndelegate(delegation.validator_address)} className="action-btn undelegate">
                    Undelegate
                  </button>
                  <button onClick={() => handleRedelegate(delegation.validator_address)} className="action-btn redelegate">
                    Redelegate
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* Unbonding Tab */}
      {activeTab === "unbonding" && (
        <div className="unbonding-list">
          {unbondingEntries.length === 0 ? (
            <div className="empty-state">
              <p>No tokens currently unbonding.</p>
            </div>
          ) : (
            <>
              <div className="unbonding-info-banner">
                <span className="info-icon">ℹ️</span>
                <span>Unbonding period is 21 days. Tokens will be available after the unbonding period completes.</span>
              </div>
              {unbondingEntries.map((entry, index) => (
                <div key={`${entry.validator_address}-${index}`} className="unbonding-card">
                  <div className="unbonding-header">
                    <div className="validator-info">
                      <div className="validator-name">{entry.validator_name}</div>
                      <div className="validator-address">{entry.validator_address}</div>
                    </div>
                    <div className="unbonding-amount">
                      {formatAmount(entry.amount)} REMES
                    </div>
                  </div>
                  <div className="unbonding-progress">
                    <div className="progress-bar">
                      <div 
                        className="progress-fill" 
                        style={{ width: `${getUnbondingProgress(entry.remaining_days)}%` }}
                      />
                    </div>
                    <div className="progress-info">
                      <span className="remaining-days">
                        {entry.remaining_days === 0 
                          ? "Ready to claim" 
                          : `${entry.remaining_days} day${entry.remaining_days !== 1 ? 's' : ''} remaining`
                        }
                      </span>
                      <span className="completion-date">
                        Completes: {new Date(entry.completion_time).toLocaleDateString()}
                      </span>
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
        <DelegateForm
          validatorAddress={selectedValidator}
          action={delegateAction}
          onClose={() => {
            setShowDelegateForm(false);
            setSelectedValidator(null);
            setDelegateAction(null);
          }}
          onSuccess={handleSuccess}
        />
      )}
    </div>
  );
}

