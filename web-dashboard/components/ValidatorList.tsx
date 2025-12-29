"use client";

import { useQuery } from "@tanstack/react-query";
import "./ValidatorList.css";

interface Validator {
  operator_address: string;
  moniker: string;
  commission: string;
  voting_power: string;
  uptime: number;
  status: "active" | "jailed" | "unbonding";
  self_delegation: string;
  total_delegations: string;
}

interface TrustScoreData {
  trust_score: number;
  total_verifications: number;
  successful_verifications: number;
  false_verdicts: number;
  lazy_validation_count: number;
}

interface ValidatorWithTrustScore extends Validator {
  trust_score?: TrustScoreData;
}

interface ValidatorListProps {
  onDelegate: (validatorAddress: string) => void;
  onUndelegate: (validatorAddress: string) => void;
  onRedelegate: (validatorAddress: string) => void;
}

// Trust Score Badge Component
function TrustScoreBadge({ score, verifications }: { score: number; verifications: number }) {
  const getScoreColor = (score: number) => {
    if (score >= 90) return "#22c55e"; // Green - Excellent
    if (score >= 70) return "#84cc16"; // Lime - Good
    if (score >= 50) return "#eab308"; // Yellow - Fair
    if (score >= 30) return "#f97316"; // Orange - Poor
    return "#ef4444"; // Red - Very Poor
  };

  const getScoreLabel = (score: number) => {
    if (score >= 90) return "Excellent";
    if (score >= 70) return "Good";
    if (score >= 50) return "Fair";
    if (score >= 30) return "Poor";
    return "Very Poor";
  };

  return (
    <div className="trust-score-badge" title={`${verifications} total verifications`}>
      <div 
        className="score-circle"
        style={{ 
          borderColor: getScoreColor(score),
          background: `conic-gradient(${getScoreColor(score)} ${score * 3.6}deg, #1e293b ${score * 3.6}deg)`
        }}
      >
        <span className="score-value" style={{ color: getScoreColor(score) }}>
          {score.toFixed(0)}
        </span>
      </div>
      <div className="score-info">
        <span className="score-label" style={{ color: getScoreColor(score) }}>
          {getScoreLabel(score)}
        </span>
        <span className="verification-count">{verifications} verifications</span>
      </div>
    </div>
  );
}

export default function ValidatorList({
  onDelegate,
  onUndelegate,
  onRedelegate,
}: ValidatorListProps) {
  const { data: validators, isLoading, error } = useQuery<Validator[]>({
    queryKey: ["staking", "validators"],
    queryFn: async () => {
      const response = await fetch("/api/blockchain/cosmos/staking/v1beta1/validators");
      if (!response.ok) {
        throw new Error("Failed to fetch validators");
      }
      const data = await response.json();
      // Map API response to Validator interface with defaults
      const validatorsList = (data.validators || []).map((v: any) => ({
        operator_address: v.operator_address || "",
        moniker: v.description?.moniker || v.moniker || "Unknown",
        commission: v.commission?.commission_rates?.rate || v.commission || "0",
        voting_power: v.tokens || v.voting_power || "0",
        uptime: v.uptime !== undefined ? v.uptime : 100.0, // Default to 100% if not provided
        status: v.status === "BOND_STATUS_BONDED" ? "active" : 
                v.status === "BOND_STATUS_JAILED" ? "jailed" : 
                v.status === "BOND_STATUS_UNBONDING" ? "unbonding" : 
                (v.status || "active"),
        self_delegation: v.self_delegation || "0",
        total_delegations: v.delegator_shares || v.total_delegations || "0",
      }));
      return validatorsList;
    },
    refetchInterval: 30000,
  });

  // Fetch trust scores from R3MES keeper
  const { data: trustScores } = useQuery<Record<string, TrustScoreData>>({
    queryKey: ["validators", "trust-scores"],
    queryFn: async () => {
      const response = await fetch("/api/validators/trust-scores");
      if (!response.ok) {
        // Return empty object if endpoint not available
        return {};
      }
      return response.json();
    },
    refetchInterval: 60000, // Refetch every minute
    retry: false,
  });

  // Combine validators with trust scores
  const validatorsWithScores: ValidatorWithTrustScore[] = validators?.map(v => ({
    ...v,
    trust_score: trustScores?.[v.operator_address],
  })) || [];

  const formatCommission = (commission: string) => {
    if (!commission) return "0.00%";
    const percent = parseFloat(commission) * 100;
    if (isNaN(percent)) return "0.00%";
    return `${percent.toFixed(2)}%`;
  };

  const formatVotingPower = (power: string) => {
    if (!power) return "0";
    const num = parseFloat(power);
    if (isNaN(num)) return "0";
    if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
    if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
    return num.toFixed(2);
  };

  if (isLoading) {
    return (
      <div className="validator-list">
        <div className="loading">Loading validators...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="validator-list">
        <div className="error">Failed to load validators. Please try again.</div>
      </div>
    );
  }

  return (
    <div className="validator-list">
      <h3 className="list-title">Validators</h3>
      <div className="validators-table">
        <div className="table-header">
          <div className="col-moniker">Validator</div>
          <div className="col-trust-score">Trust Score</div>
          <div className="col-voting-power">Voting Power</div>
          <div className="col-commission">Commission</div>
          <div className="col-uptime">Uptime</div>
          <div className="col-status">Status</div>
          <div className="col-actions">Actions</div>
        </div>
        <div className="table-body">
          {validatorsWithScores?.map((validator, index) => (
            <div key={validator.operator_address || `validator-${index}`} className="table-row">
              <div className="col-moniker">
                <div className="validator-name">{validator.moniker || "Unknown"}</div>
                <div className="validator-address">
                  {validator.operator_address ? `${validator.operator_address.slice(0, 20)}...` : "N/A"}
                </div>
              </div>
              <div className="col-trust-score">
                {validator.trust_score ? (
                  <TrustScoreBadge 
                    score={validator.trust_score.trust_score} 
                    verifications={validator.trust_score.total_verifications} 
                  />
                ) : (
                  <span className="no-score">N/A</span>
                )}
              </div>
              <div className="col-voting-power">
                {formatVotingPower(validator.voting_power)} REMES
              </div>
              <div className="col-commission">
                {formatCommission(validator.commission)}
              </div>
              <div className="col-uptime">
                {(validator.uptime ?? 100.0).toFixed(1)}%
              </div>
              <div className="col-status">
                <span className={`status-badge status-${validator.status || "active"}`}>
                  {validator.status || "active"}
                </span>
              </div>
              <div className="col-actions">
                <div className="action-buttons">
                  <button
                    onClick={() => onDelegate(validator.operator_address)}
                    className="btn-action btn-delegate"
                  >
                    Delegate
                  </button>
                  <button
                    onClick={() => onUndelegate(validator.operator_address)}
                    className="btn-action btn-undelegate"
                  >
                    Undelegate
                  </button>
                  <button
                    onClick={() => onRedelegate(validator.operator_address)}
                    className="btn-action btn-redelegate"
                  >
                    Redelegate
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

