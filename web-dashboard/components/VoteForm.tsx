"use client";

import { useState } from "react";
import { signAndBroadcastTransaction } from "@/lib/keplr";

interface Proposal {
  id: string;
  title: string;
  description: string;
  type: string;
  status: string;
}

interface VoteFormProps {
  proposal: Proposal;
  onClose: () => void;
  onSuccess: () => void;
}

type VoteOption = "yes" | "no" | "abstain" | "no_with_veto";

export default function VoteForm({ proposal, onClose, onSuccess }: VoteFormProps) {
  const [selectedVote, setSelectedVote] = useState<VoteOption | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!selectedVote) {
      setError("Please select a vote option");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      // Create governance vote message
      const voteMessage = {
        typeUrl: "/cosmos.gov.v1beta1.MsgVote",
        value: {
          proposalId: proposal.id,
          voter: "", // Will be filled by Keplr
          option: getVoteOptionNumber(selectedVote),
        },
      };

      await signAndBroadcastTransaction([voteMessage], `Vote ${selectedVote} on proposal ${proposal.id}`);

      onSuccess();
    } catch (err: any) {
      setError(err.message || "Failed to submit vote");
    } finally {
      setIsSubmitting(false);
    }
  };

  const getVoteOptionNumber = (option: VoteOption): number => {
    switch (option) {
      case "yes":
        return 1; // VOTE_OPTION_YES
      case "no":
        return 3; // VOTE_OPTION_NO
      case "abstain":
        return 2; // VOTE_OPTION_ABSTAIN
      case "no_with_veto":
        return 4; // VOTE_OPTION_NO_WITH_VETO
      default:
        return 0;
    }
  };

  return (
    <div className="vote-form-overlay" onClick={onClose}>
      <div className="vote-form" onClick={(e) => e.stopPropagation()}>
        <div className="vote-form-header">
          <h3>Vote on Proposal #{proposal.id}</h3>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        <div className="vote-form-content">
          <div className="proposal-info">
            <h4>{proposal.title}</h4>
            <p className="proposal-type">Type: {proposal.type}</p>
          </div>

          <div className="vote-options">
            <label className={`vote-option ${selectedVote === "yes" ? "selected" : ""}`}>
              <input
                type="radio"
                name="vote"
                value="yes"
                checked={selectedVote === "yes"}
                onChange={() => setSelectedVote("yes")}
              />
              <div className="vote-option-content">
                <span className="vote-option-icon">✅</span>
                <span className="vote-option-label">Yes</span>
                <span className="vote-option-desc">Approve this proposal</span>
              </div>
            </label>

            <label className={`vote-option ${selectedVote === "no" ? "selected" : ""}`}>
              <input
                type="radio"
                name="vote"
                value="no"
                checked={selectedVote === "no"}
                onChange={() => setSelectedVote("no")}
              />
              <div className="vote-option-content">
                <span className="vote-option-icon">❌</span>
                <span className="vote-option-label">No</span>
                <span className="vote-option-desc">Reject this proposal</span>
              </div>
            </label>

            <label className={`vote-option ${selectedVote === "abstain" ? "selected" : ""}`}>
              <input
                type="radio"
                name="vote"
                value="abstain"
                checked={selectedVote === "abstain"}
                onChange={() => setSelectedVote("abstain")}
              />
              <div className="vote-option-content">
                <span className="vote-option-icon">⚪</span>
                <span className="vote-option-label">Abstain</span>
                <span className="vote-option-desc">Neutral position</span>
              </div>
            </label>

            <label className={`vote-option ${selectedVote === "no_with_veto" ? "selected" : ""}`}>
              <input
                type="radio"
                name="vote"
                value="no_with_veto"
                checked={selectedVote === "no_with_veto"}
                onChange={() => setSelectedVote("no_with_veto")}
              />
              <div className="vote-option-content">
                <span className="vote-option-icon">🚫</span>
                <span className="vote-option-label">No with Veto</span>
                <span className="vote-option-desc">Strongly reject (may invalidate proposal)</span>
              </div>
            </label>
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>

        <div className="vote-form-footer">
          <button onClick={onClose} className="btn-cancel" disabled={isSubmitting}>
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            className="btn-submit"
            disabled={!selectedVote || isSubmitting}
          >
            {isSubmitting ? "Submitting..." : "Submit Vote"}
          </button>
        </div>
      </div>
    </div>
  );
}

