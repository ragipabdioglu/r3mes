"use client";

import { useState } from "react";
import { useWallet } from "@/contexts/WalletContext";
import "./CreateProposalModal.css";

interface CreateProposalModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

type ProposalType = "parameter_change" | "software_upgrade" | "model_upgrade" | "dataset_proposal" | "text";

export default function CreateProposalModal({ isOpen, onClose, onSuccess }: CreateProposalModalProps) {
  const { walletAddress } = useWallet();
  const [proposalType, setProposalType] = useState<ProposalType>("text");
  const [title, setTitle] = useState("");
  const [description, setDescription] = useState("");
  const [depositAmount, setDepositAmount] = useState("100");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Model Upgrade specific fields
  const [modelVersion, setModelVersion] = useState("");
  const [modelIpfsHash, setModelIpfsHash] = useState("");
  const [migrationPlan, setMigrationPlan] = useState("");
  const [backwardCompatible, setBackwardCompatible] = useState(true);
  
  // Dataset Proposal specific fields
  const [datasetName, setDatasetName] = useState("");
  const [datasetIpfsHash, setDatasetIpfsHash] = useState("");
  const [datasetSize, setDatasetSize] = useState("");
  const [datasetCategory, setDatasetCategory] = useState("");
  const [datasetDescription, setDatasetDescription] = useState("");

  // Parameter Change specific fields
  const [parameterKey, setParameterKey] = useState("");
  const [parameterValue, setParameterValue] = useState("");

  const resetForm = () => {
    setTitle("");
    setDescription("");
    setDepositAmount("100");
    setModelVersion("");
    setModelIpfsHash("");
    setMigrationPlan("");
    setBackwardCompatible(true);
    setDatasetName("");
    setDatasetIpfsHash("");
    setDatasetSize("");
    setDatasetCategory("");
    setDatasetDescription("");
    setParameterKey("");
    setParameterValue("");
    setError(null);
  };

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const validateForm = (): boolean => {
    if (!title.trim()) {
      setError("Title is required");
      return false;
    }
    if (!description.trim()) {
      setError("Description is required");
      return false;
    }
    if (parseFloat(depositAmount) < 100) {
      setError("Minimum deposit is 100 REMES");
      return false;
    }

    if (proposalType === "model_upgrade") {
      if (!modelVersion.trim()) {
        setError("Model version is required");
        return false;
      }
      if (!modelIpfsHash.trim() || !modelIpfsHash.startsWith("Qm")) {
        setError("Valid IPFS hash is required (starts with Qm)");
        return false;
      }
    }

    if (proposalType === "dataset_proposal") {
      if (!datasetName.trim()) {
        setError("Dataset name is required");
        return false;
      }
      if (!datasetIpfsHash.trim() || !datasetIpfsHash.startsWith("Qm")) {
        setError("Valid IPFS hash is required (starts with Qm)");
        return false;
      }
      if (!datasetCategory) {
        setError("Dataset category is required");
        return false;
      }
    }

    if (proposalType === "parameter_change") {
      if (!parameterKey.trim()) {
        setError("Parameter key is required");
        return false;
      }
      if (!parameterValue.trim()) {
        setError("Parameter value is required");
        return false;
      }
    }

    return true;
  };

  const handleSubmit = async () => {
    if (!walletAddress) {
      setError("Please connect your wallet first");
      return;
    }

    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);
    setError(null);

    const proposalData: Record<string, any> = {
      type: proposalType,
      title,
      description,
      deposit: {
        denom: "uremes",
        amount: (parseFloat(depositAmount) * 1e6).toString(),
      },
      proposer: walletAddress,
    };

    // Add type-specific fields
    if (proposalType === "model_upgrade") {
      proposalData.content = {
        "@type": "/remes.remes.v1.ModelUpgradeProposal",
        model_version: modelVersion,
        model_ipfs_hash: modelIpfsHash,
        migration_plan: migrationPlan,
        backward_compatible: backwardCompatible,
      };
    } else if (proposalType === "dataset_proposal") {
      proposalData.content = {
        "@type": "/remes.remes.v1.DatasetProposal",
        dataset_name: datasetName,
        dataset_ipfs_hash: datasetIpfsHash,
        dataset_size: datasetSize,
        dataset_category: datasetCategory,
        dataset_description: datasetDescription,
      };
    } else if (proposalType === "parameter_change") {
      proposalData.content = {
        "@type": "/cosmos.params.v1beta1.ParameterChangeProposal",
        changes: [{
          subspace: "remes",
          key: parameterKey,
          value: parameterValue,
        }],
      };
    } else {
      proposalData.content = {
        "@type": "/cosmos.gov.v1beta1.TextProposal",
        title,
        description,
      };
    }

    try {
      const response = await fetch("/api/governance/proposals", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(proposalData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || "Failed to create proposal");
      }

      onSuccess();
      handleClose();
    } catch (err) {
      console.error("Error creating proposal:", err);
      setError(err instanceof Error ? err.message : "Failed to create proposal");
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={handleClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Create New Proposal</h2>
          <button className="close-btn" onClick={handleClose}>×</button>
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}
        
        <div className="modal-body">
          <div className="form-group">
            <label>Proposal Type</label>
            <select 
              value={proposalType} 
              onChange={(e) => setProposalType(e.target.value as ProposalType)}
              disabled={isSubmitting}
            >
              <option value="text">Text Proposal</option>
              <option value="parameter_change">Parameter Change</option>
              <option value="software_upgrade">Software Upgrade</option>
              <option value="model_upgrade">Model Upgrade</option>
              <option value="dataset_proposal">Dataset Proposal</option>
            </select>
          </div>

          <div className="form-group">
            <label>Title *</label>
            <input 
              type="text" 
              value={title} 
              onChange={(e) => setTitle(e.target.value)} 
              placeholder="Enter proposal title"
              disabled={isSubmitting}
              maxLength={140}
            />
            <small>{title.length}/140 characters</small>
          </div>

          <div className="form-group">
            <label>Description *</label>
            <textarea 
              value={description} 
              onChange={(e) => setDescription(e.target.value)} 
              placeholder="Detailed description of the proposal"
              rows={4}
              disabled={isSubmitting}
            />
          </div>

          {proposalType === "model_upgrade" && (
            <div className="type-specific-fields">
              <h4>Model Upgrade Details</h4>
              <div className="form-group">
                <label>Model Version *</label>
                <input 
                  type="text" 
                  value={modelVersion} 
                  onChange={(e) => setModelVersion(e.target.value)} 
                  placeholder="e.g., BitNet v2.0"
                  disabled={isSubmitting}
                />
              </div>
              <div className="form-group">
                <label>Model IPFS Hash *</label>
                <input 
                  type="text" 
                  value={modelIpfsHash} 
                  onChange={(e) => setModelIpfsHash(e.target.value)} 
                  placeholder="Qm..."
                  disabled={isSubmitting}
                />
              </div>
              <div className="form-group">
                <label>Migration Plan</label>
                <textarea 
                  value={migrationPlan} 
                  onChange={(e) => setMigrationPlan(e.target.value)} 
                  placeholder="Describe the migration process for miners"
                  rows={3}
                  disabled={isSubmitting}
                />
              </div>
              <div className="form-group checkbox-group">
                <label>
                  <input 
                    type="checkbox" 
                    checked={backwardCompatible} 
                    onChange={(e) => setBackwardCompatible(e.target.checked)}
                    disabled={isSubmitting}
                  />
                  Backward Compatible
                </label>
              </div>
            </div>
          )}

          {proposalType === "dataset_proposal" && (
            <div className="type-specific-fields">
              <h4>Dataset Details</h4>
              <div className="form-group">
                <label>Dataset Name *</label>
                <input 
                  type="text" 
                  value={datasetName} 
                  onChange={(e) => setDatasetName(e.target.value)} 
                  placeholder="Dataset name"
                  disabled={isSubmitting}
                />
              </div>
              <div className="form-group">
                <label>Dataset IPFS Hash *</label>
                <input 
                  type="text" 
                  value={datasetIpfsHash} 
                  onChange={(e) => setDatasetIpfsHash(e.target.value)} 
                  placeholder="Qm..."
                  disabled={isSubmitting}
                />
              </div>
              <div className="form-row">
                <div className="form-group">
                  <label>Size</label>
                  <input 
                    type="text" 
                    value={datasetSize} 
                    onChange={(e) => setDatasetSize(e.target.value)} 
                    placeholder="e.g., 10GB"
                    disabled={isSubmitting}
                  />
                </div>
                <div className="form-group">
                  <label>Category *</label>
                  <select 
                    value={datasetCategory} 
                    onChange={(e) => setDatasetCategory(e.target.value)}
                    disabled={isSubmitting}
                  >
                    <option value="">Select category</option>
                    <option value="text">Text</option>
                    <option value="code">Code</option>
                    <option value="multimodal">Multimodal</option>
                    <option value="scientific">Scientific</option>
                    <option value="conversational">Conversational</option>
                  </select>
                </div>
              </div>
              <div className="form-group">
                <label>Dataset Description</label>
                <textarea 
                  value={datasetDescription} 
                  onChange={(e) => setDatasetDescription(e.target.value)} 
                  placeholder="Describe the dataset content and quality"
                  rows={2}
                  disabled={isSubmitting}
                />
              </div>
            </div>
          )}

          {proposalType === "parameter_change" && (
            <div className="type-specific-fields">
              <h4>Parameter Change Details</h4>
              <div className="form-group">
                <label>Parameter Key *</label>
                <input 
                  type="text" 
                  value={parameterKey} 
                  onChange={(e) => setParameterKey(e.target.value)} 
                  placeholder="e.g., base_reward_per_gradient"
                  disabled={isSubmitting}
                />
              </div>
              <div className="form-group">
                <label>New Value *</label>
                <input 
                  type="text" 
                  value={parameterValue} 
                  onChange={(e) => setParameterValue(e.target.value)} 
                  placeholder="New parameter value"
                  disabled={isSubmitting}
                />
              </div>
            </div>
          )}

          <div className="form-group">
            <label>Deposit Amount (REMES) *</label>
            <input 
              type="number" 
              value={depositAmount} 
              onChange={(e) => setDepositAmount(e.target.value)} 
              min="100"
              disabled={isSubmitting}
            />
            <small>Minimum deposit: 100 REMES. Deposit is returned if proposal passes or is rejected (not vetoed).</small>
          </div>
        </div>

        <div className="modal-footer">
          <button onClick={handleClose} className="btn-secondary" disabled={isSubmitting}>
            Cancel
          </button>
          <button onClick={handleSubmit} className="btn-primary" disabled={isSubmitting || !walletAddress}>
            {isSubmitting ? "Submitting..." : "Submit Proposal"}
          </button>
        </div>
      </div>
    </div>
  );
}
