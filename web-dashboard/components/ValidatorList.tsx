"use client";

import { useQuery } from "@tanstack/react-query";

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

function TrustScoreBadge({ score, verifications }: { score: number; verifications: number }) {
  const getScoreColor = (score: number) => {
    if (score >= 90) return "#22c55e";
    if (score >= 70) return "#84cc16";
    if (score >= 50) return "#eab308";
    if (score >= 30) return "#f97316";
    return "#ef4444";
  };

  const getScoreLabel = (score: number) => {
    if (score >= 90) return "Excellent";
    if (score >= 70) return "Good";
    if (score >= 50) return "Fair";
    if (score >= 30) return "Poor";
    return "Very Poor";
  };

  return (
    <div className="flex items-center gap-2.5 px-2.5 py-1.5 bg-slate-900/60 rounded-lg border border-slate-700" title={`${verifications} total verifications`}>
      <div 
        className="relative w-10 h-10 rounded-full border-[3px] flex items-center justify-center bg-slate-900"
        style={{ borderColor: getScoreColor(score) }}
      >
        <span className="text-xs font-bold z-10" style={{ color: getScoreColor(score) }}>{score.toFixed(0)}</span>
      </div>
      <div className="flex flex-col gap-0.5">
        <span className="text-xs font-semibold uppercase tracking-tight" style={{ color: getScoreColor(score) }}>{getScoreLabel(score)}</span>
        <span className="text-[10px] text-slate-500">{verifications} verifications</span>
      </div>
    </div>
  );
}

export default function ValidatorList({ onDelegate, onUndelegate, onRedelegate }: ValidatorListProps) {
  const { data: validators, isLoading, error } = useQuery<Validator[]>({
    queryKey: ["staking", "validators"],
    queryFn: async () => {
      const response = await fetch("/api/blockchain/cosmos/staking/v1beta1/validators");
      if (!response.ok) throw new Error("Failed to fetch validators");
      const data = await response.json();
      return (data.validators || []).map((v: any) => ({
        operator_address: v.operator_address || "",
        moniker: v.description?.moniker || v.moniker || "Unknown",
        commission: v.commission?.commission_rates?.rate || v.commission || "0",
        voting_power: v.tokens || v.voting_power || "0",
        uptime: v.uptime !== undefined ? v.uptime : 100.0,
        status: v.status === "BOND_STATUS_BONDED" ? "active" : v.status === "BOND_STATUS_JAILED" ? "jailed" : v.status === "BOND_STATUS_UNBONDING" ? "unbonding" : (v.status || "active"),
        self_delegation: v.self_delegation || "0",
        total_delegations: v.delegator_shares || v.total_delegations || "0",
      }));
    },
    refetchInterval: 30000,
  });

  const { data: trustScores } = useQuery<Record<string, TrustScoreData>>({
    queryKey: ["validators", "trust-scores"],
    queryFn: async () => {
      const response = await fetch("/api/validators/trust-scores");
      if (!response.ok) return {};
      return response.json();
    },
    refetchInterval: 60000,
    retry: false,
  });

  const validatorsWithScores: ValidatorWithTrustScore[] = validators?.map(v => ({ ...v, trust_score: trustScores?.[v.operator_address] })) || [];

  const formatCommission = (commission: string) => {
    if (!commission) return "0.00%";
    const percent = parseFloat(commission) * 100;
    return isNaN(percent) ? "0.00%" : `${percent.toFixed(2)}%`;
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

  const getStatusClass = (status: string) => {
    const base = "px-3 py-1 rounded-md text-xs font-semibold uppercase";
    if (status === "active") return `${base} bg-green-500/10 text-green-500`;
    if (status === "jailed") return `${base} bg-red-500/10 text-red-500`;
    if (status === "unbonding") return `${base} bg-amber-500/10 text-amber-500`;
    return `${base} bg-slate-400/10 text-slate-400`;
  };

  if (isLoading) return <div className="mt-8"><div className="text-center py-10 text-slate-400">Loading validators...</div></div>;
  if (error) return <div className="mt-8"><div className="text-center py-10 text-red-500">Failed to load validators. Please try again.</div></div>;

  return (
    <div className="mt-8">
      <h3 className="text-xl font-semibold text-slate-100 mb-5">Validators</h3>
      <div className="bg-slate-800 border border-slate-700 rounded-xl overflow-hidden">
        {/* Header */}
        <div className="grid grid-cols-[2fr_1.5fr_1fr_1fr_0.8fr_0.8fr_2fr] gap-4 px-5 py-4 bg-slate-900 border-b border-slate-700 text-xs font-semibold text-slate-400 uppercase tracking-wide">
          <div>Validator</div>
          <div>Trust Score</div>
          <div>Voting Power</div>
          <div>Commission</div>
          <div>Uptime</div>
          <div>Status</div>
          <div>Actions</div>
        </div>
        {/* Body */}
        <div className="flex flex-col">
          {validatorsWithScores?.map((validator, index) => (
            <div key={validator.operator_address || `validator-${index}`} className="grid grid-cols-[2fr_1.5fr_1fr_1fr_0.8fr_0.8fr_2fr] gap-4 px-5 py-5 border-b border-slate-700 last:border-b-0 transition-colors hover:bg-slate-900">
              <div className="flex flex-col gap-1">
                <div className="text-sm font-semibold text-slate-100">{validator.moniker || "Unknown"}</div>
                <div className="text-xs font-mono text-slate-500">{validator.operator_address ? `${validator.operator_address.slice(0, 20)}...` : "N/A"}</div>
              </div>
              <div className="flex items-center">
                {validator.trust_score ? (
                  <TrustScoreBadge score={validator.trust_score.trust_score} verifications={validator.trust_score.total_verifications} />
                ) : (
                  <span className="text-xs text-slate-500 italic">N/A</span>
                )}
              </div>
              <div className="flex items-center text-sm text-slate-100">{formatVotingPower(validator.voting_power)} REMES</div>
              <div className="flex items-center text-sm text-slate-100">{formatCommission(validator.commission)}</div>
              <div className="flex items-center text-sm text-slate-100">{(validator.uptime ?? 100.0).toFixed(1)}%</div>
              <div className="flex items-center">
                <span className={getStatusClass(validator.status || "active")}>{validator.status || "active"}</span>
              </div>
              <div className="flex items-center">
                <div className="flex gap-2 flex-wrap">
                  <button onClick={() => onDelegate(validator.operator_address)} className="px-3 py-1.5 bg-blue-500 hover:bg-blue-600 text-white text-xs font-medium rounded-md transition-colors">Delegate</button>
                  <button onClick={() => onUndelegate(validator.operator_address)} className="px-3 py-1.5 bg-amber-500 hover:bg-amber-600 text-white text-xs font-medium rounded-md transition-colors">Undelegate</button>
                  <button onClick={() => onRedelegate(validator.operator_address)} className="px-3 py-1.5 bg-violet-500 hover:bg-violet-600 text-white text-xs font-medium rounded-md transition-colors">Redelegate</button>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
