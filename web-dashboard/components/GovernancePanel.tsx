"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import VoteForm from "./VoteForm";
import CreateProposalModal from "./CreateProposalModal";

interface Proposal {
  id: string;
  title: string;
  description: string;
  type: string;
  status: "deposit_period" | "voting_period" | "passed" | "rejected";
  voting_end_time: string;
  votes: { yes: string; no: string; abstain: string; no_with_veto: string };
  total_votes: string;
}

export default function GovernancePanel() {
  const [selectedProposal, setSelectedProposal] = useState<Proposal | null>(null);
  const [showVoteForm, setShowVoteForm] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [filter, setFilter] = useState<"all" | "active" | "passed" | "rejected">("all");

  const { data: proposals, isLoading, error, refetch } = useQuery<Proposal[]>({
    queryKey: ["governance", "proposals"],
    queryFn: async () => {
      const response = await fetch("/api/blockchain/cosmos/gov/v1beta1/proposals");
      if (!response.ok) throw new Error("Failed to fetch proposals");
      const data = await response.json();
      return data.proposals || [];
    },
    refetchInterval: 30000,
  });

  const filteredProposals = proposals?.filter(p => {
    if (filter === "all") return true;
    if (filter === "active") return p.status === "voting_period" || p.status === "deposit_period";
    return p.status === filter;
  });

  const handleVote = (proposal: Proposal) => { setSelectedProposal(proposal); setShowVoteForm(true); };
  const handleCreateSuccess = () => { setShowCreateModal(false); refetch(); };

  const getStatusColor = (status: string) => {
    const colors: Record<string, string> = { passed: "text-green-500", rejected: "text-red-500", voting_period: "text-blue-500", deposit_period: "text-amber-500" };
    return colors[status] || "text-slate-400";
  };

  const getStatusLabel = (status: string) => {
    const labels: Record<string, string> = { passed: "Passed", rejected: "Rejected", voting_period: "Voting Period", deposit_period: "Deposit Period" };
    return labels[status] || status;
  };

  const formatVotePercentage = (votes: string, total: string) => {
    if (!total || total === "0") return "0%";
    return `${((parseFloat(votes) / parseFloat(total)) * 100).toFixed(1)}%`;
  };

  if (isLoading) return <div className="p-6 text-slate-100"><div className="text-center py-10 text-slate-400">Loading proposals...</div></div>;
  if (error) return <div className="p-6 text-slate-100"><div className="text-center py-10 text-red-500">Failed to load proposals. Please try again.</div></div>;

  return (
    <div className="p-6 text-slate-100">
      {/* Header */}
      <div className="flex justify-between items-start mb-6">
        <div>
          <h2 className="text-[28px] font-semibold mb-2 bg-gradient-to-r from-blue-500 to-violet-500 bg-clip-text text-transparent">Governance</h2>
          <p className="text-slate-400 text-sm">Vote on proposals and model upgrades</p>
        </div>
        <button onClick={() => setShowCreateModal(true)} className="px-5 py-2.5 bg-gradient-to-br from-blue-500 to-violet-500 text-white text-sm font-medium rounded-lg transition-all hover:-translate-y-0.5 hover:shadow-lg hover:shadow-blue-500/40 whitespace-nowrap">+ Create Proposal</button>
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-2 mb-6 flex-wrap">
        {[{ key: "all", label: `All (${proposals?.length || 0})` }, { key: "active", label: `Active (${proposals?.filter(p => p.status === "voting_period" || p.status === "deposit_period").length || 0})` }, { key: "passed", label: `Passed (${proposals?.filter(p => p.status === "passed").length || 0})` }, { key: "rejected", label: `Rejected (${proposals?.filter(p => p.status === "rejected").length || 0})` }].map(tab => (
          <button key={tab.key} onClick={() => setFilter(tab.key as any)} className={`px-4 py-2 rounded-lg text-sm font-medium transition-all border ${filter === tab.key ? "bg-blue-500 border-blue-500 text-white" : "bg-transparent border-slate-700 text-slate-400 hover:border-slate-600 hover:text-slate-100"}`}>{tab.label}</button>
        ))}
      </div>

      {/* Proposals */}
      {filteredProposals && filteredProposals.length === 0 ? (
        <div className="text-center py-16 text-slate-400">
          <p>No {filter === "all" ? "" : filter} proposals</p>
          {filter === "all" && <button onClick={() => setShowCreateModal(true)} className="mt-4 px-5 py-2.5 bg-transparent border border-dashed border-slate-600 hover:border-blue-500 hover:text-blue-500 text-slate-400 text-sm rounded-lg transition-colors">Create the first proposal</button>}
        </div>
      ) : (
        <div className="flex flex-col gap-5">
          {filteredProposals?.map(proposal => (
            <div key={proposal.id} className="bg-slate-800 border border-slate-700 rounded-xl p-6 transition-all hover:border-slate-600 hover:shadow-lg hover:shadow-black/30">
              {/* Header */}
              <div className="flex justify-between items-center mb-4">
                <div className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Proposal #{proposal.id}</div>
                <div className={`text-xs font-semibold px-3 py-1 rounded-md bg-blue-500/10 ${getStatusColor(proposal.status)}`}>{getStatusLabel(proposal.status)}</div>
              </div>
              {/* Content */}
              <div className="mb-5">
                <h3 className="text-xl font-semibold text-slate-100 mb-3">{proposal.title}</h3>
                <p className="text-slate-400 text-sm leading-relaxed mb-3">{proposal.description}</p>
                <div className="flex items-center gap-2 text-xs">
                  <span className="text-slate-500 font-semibold">Type:</span>
                  <span className="text-blue-500 font-medium">{proposal.type}</span>
                </div>
              </div>
              {/* Votes */}
              {proposal.status === "voting_period" && (
                <div className="mb-5 p-4 bg-slate-900 rounded-lg">
                  <div className="flex flex-col gap-3">
                    {[{ key: "yes", label: "Yes", bg: "bg-green-500/10" }, { key: "no", label: "No", bg: "bg-red-500/10" }, { key: "abstain", label: "Abstain", bg: "bg-slate-400/10" }, { key: "no_with_veto", label: "No with Veto", bg: "bg-amber-500/10" }].map(v => (
                      <div key={v.key} className={`flex justify-between items-center px-3 py-2 rounded-md ${v.bg}`}>
                        <span className="text-sm font-medium text-slate-100">{v.label}</span>
                        <span className="text-sm font-semibold text-slate-400">{formatVotePercentage(proposal.votes[v.key as keyof typeof proposal.votes], proposal.total_votes)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
              {/* Footer */}
              <div className="flex justify-between items-center pt-4 border-t border-slate-700">
                <div className="text-xs text-slate-500">Voting ends: {new Date(proposal.voting_end_time).toLocaleString()}</div>
                {proposal.status === "voting_period" && <button onClick={() => handleVote(proposal)} className="px-5 py-2 bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium rounded-lg transition-colors">Vote</button>}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Modals */}
      {showVoteForm && selectedProposal && <VoteForm proposal={selectedProposal} onClose={() => { setShowVoteForm(false); setSelectedProposal(null); }} onSuccess={() => { setShowVoteForm(false); setSelectedProposal(null); refetch(); }} />}
      {showCreateModal && <CreateProposalModal isOpen={showCreateModal} onClose={() => setShowCreateModal(false)} onSuccess={handleCreateSuccess} />}
    </div>
  );
}
