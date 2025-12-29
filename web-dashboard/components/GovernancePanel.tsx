"use client";

import { useState, useEffect } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import VoteForm from "./VoteForm";
import CreateProposalModal from "./CreateProposalModal";
import "./GovernancePanel.css";

interface Proposal {
  id: string;
  title: string;
  description: string;
  type: string;
  status: "deposit_period" | "voting_period" | "passed" | "rejected";
  voting_end_time: string;
  votes: {
    yes: string;
    no: string;
    abstain: string;
    no_with_veto: string;
  };
  total_votes: string;
}

export default function GovernancePanel() {
  const [selectedProposal, setSelectedProposal] = useState<Proposal | null>(null);
  const [showVoteForm, setShowVoteForm] = useState(false);
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [filter, setFilter] = useState<"all" | "active" | "passed" | "rejected">("all");
  const queryClient = useQueryClient();

  const { data: proposals, isLoading, error, refetch } = useQuery<Proposal[]>({
    queryKey: ["governance", "proposals"],
    queryFn: async () => {
      const response = await fetch("/api/blockchain/cosmos/gov/v1beta1/proposals");
      if (!response.ok) {
        throw new Error("Failed to fetch proposals");
      }
      const data = await response.json();
      return data.proposals || [];
    },
    refetchInterval: 30000, // Refetch every 30 seconds
  });

  const filteredProposals = proposals?.filter(proposal => {
    if (filter === "all") return true;
    if (filter === "active") return proposal.status === "voting_period" || proposal.status === "deposit_period";
    return proposal.status === filter;
  });

  const handleVote = (proposal: Proposal) => {
    setSelectedProposal(proposal);
    setShowVoteForm(true);
  };

  const handleCreateSuccess = () => {
    setShowCreateModal(false);
    refetch();
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case "passed":
        return "#22c55e";
      case "rejected":
        return "#ef4444";
      case "voting_period":
        return "#3b82f6";
      case "deposit_period":
        return "#f59e0b";
      default:
        return "#94a3b8";
    }
  };

  const getStatusLabel = (status: string) => {
    switch (status) {
      case "passed":
        return "Passed";
      case "rejected":
        return "Rejected";
      case "voting_period":
        return "Voting Period";
      case "deposit_period":
        return "Deposit Period";
      default:
        return status;
    }
  };

  const formatVotePercentage = (votes: string, total: string) => {
    if (!total || total === "0") return "0%";
    const percent = (parseFloat(votes) / parseFloat(total)) * 100;
    return `${percent.toFixed(1)}%`;
  };

  if (isLoading) {
    return (
      <div className="governance-panel">
        <div className="loading">Loading proposals...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="governance-panel">
        <div className="error">Failed to load proposals. Please try again.</div>
      </div>
    );
  }

  return (
    <div className="governance-panel">
      <div className="governance-header">
        <div className="header-content">
          <h2>Governance</h2>
          <p className="subtitle">Vote on proposals and model upgrades</p>
        </div>
        <button 
          className="create-proposal-btn"
          onClick={() => setShowCreateModal(true)}
        >
          + Create Proposal
        </button>
      </div>

      <div className="filter-tabs">
        <button 
          className={`filter-tab ${filter === "all" ? "active" : ""}`}
          onClick={() => setFilter("all")}
        >
          All ({proposals?.length || 0})
        </button>
        <button 
          className={`filter-tab ${filter === "active" ? "active" : ""}`}
          onClick={() => setFilter("active")}
        >
          Active ({proposals?.filter(p => p.status === "voting_period" || p.status === "deposit_period").length || 0})
        </button>
        <button 
          className={`filter-tab ${filter === "passed" ? "active" : ""}`}
          onClick={() => setFilter("passed")}
        >
          Passed ({proposals?.filter(p => p.status === "passed").length || 0})
        </button>
        <button 
          className={`filter-tab ${filter === "rejected" ? "active" : ""}`}
          onClick={() => setFilter("rejected")}
        >
          Rejected ({proposals?.filter(p => p.status === "rejected").length || 0})
        </button>
      </div>

      {filteredProposals && filteredProposals.length === 0 ? (
        <div className="no-proposals">
          <p>No {filter === "all" ? "" : filter} proposals</p>
          {filter === "all" && (
            <button 
              className="create-first-btn"
              onClick={() => setShowCreateModal(true)}
            >
              Create the first proposal
            </button>
          )}
        </div>
      ) : (
        <div className="proposals-list">
          {filteredProposals?.map((proposal) => (
            <div key={proposal.id} className="proposal-card">
              <div className="proposal-header">
                <div className="proposal-id">Proposal #{proposal.id}</div>
                <div
                  className="proposal-status"
                  style={{ color: getStatusColor(proposal.status) }}
                >
                  {getStatusLabel(proposal.status)}
                </div>
              </div>

              <div className="proposal-content">
                <h3 className="proposal-title">{proposal.title}</h3>
                <p className="proposal-description">{proposal.description}</p>
                <div className="proposal-type">
                  <span className="type-label">Type:</span>
                  <span className="type-value">{proposal.type}</span>
                </div>
              </div>

              {proposal.status === "voting_period" && (
                <div className="proposal-votes">
                  <div className="vote-bar">
                    <div className="vote-item yes">
                      <span className="vote-label">Yes</span>
                      <span className="vote-percentage">
                        {formatVotePercentage(proposal.votes.yes, proposal.total_votes)}
                      </span>
                    </div>
                    <div className="vote-item no">
                      <span className="vote-label">No</span>
                      <span className="vote-percentage">
                        {formatVotePercentage(proposal.votes.no, proposal.total_votes)}
                      </span>
                    </div>
                    <div className="vote-item abstain">
                      <span className="vote-label">Abstain</span>
                      <span className="vote-percentage">
                        {formatVotePercentage(proposal.votes.abstain, proposal.total_votes)}
                      </span>
                    </div>
                    <div className="vote-item veto">
                      <span className="vote-label">No with Veto</span>
                      <span className="vote-percentage">
                        {formatVotePercentage(proposal.votes.no_with_veto, proposal.total_votes)}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              <div className="proposal-footer">
                <div className="proposal-meta">
                  <span>Voting ends: {new Date(proposal.voting_end_time).toLocaleString()}</span>
                </div>
                {proposal.status === "voting_period" && (
                  <button
                    onClick={() => handleVote(proposal)}
                    className="vote-button"
                  >
                    Vote
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {showVoteForm && selectedProposal && (
        <VoteForm
          proposal={selectedProposal}
          onClose={() => {
            setShowVoteForm(false);
            setSelectedProposal(null);
          }}
          onSuccess={() => {
            setShowVoteForm(false);
            setSelectedProposal(null);
            refetch();
          }}
        />
      )}

      {showCreateModal && (
        <CreateProposalModal
          isOpen={showCreateModal}
          onClose={() => setShowCreateModal(false)}
          onSuccess={handleCreateSuccess}
        />
      )}
    </div>
  );
}

