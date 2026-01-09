"use client";

import { useState } from "react";
import { useWallet } from "@/contexts/WalletContext";

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
  
  const [modelVersion, setModelVersion] = useState("");
  const [modelIpfsHash, setModelIpfsHash] = useState("");
  const [migrationPlan, setMigrationPlan] = useState("");
  const [backwardCompatible, setBackwardCompatible] = useState(true);
  
  const [datasetName, setDatasetName] = useState("");
  const [datasetIpfsHash, setDatasetIpfsHash] = useState("");
  const [datasetSize, setDatasetSize] = useState("");
  const [datasetCategory, setDatasetCategory] = useState("");
  const [datasetDescription, setDatasetDescription] = useState("");

  const [parameterKey, setParameterKey] = useState("");
  const [parameterValue, setParameterValue] = useState("");

  const resetForm = () => {
    setTitle(""); setDescription(""); setDepositAmount("100"); setModelVersion(""); setModelIpfsHash("");
    setMigrationPlan(""); setBackwardCompatible(true); setDatasetName(""); setDatasetIpfsHash("");
    setDatasetSize(""); setDatasetCategory(""); setDatasetDescription(""); setParameterKey(""); setParameterValue(""); setError(null);
  };

  const handleClose = () => { resetForm(); onClose(); };

  const validateForm = (): boolean => {
    if (!title.trim()) { setError("Title is required"); return false; }
    if (!description.trim()) { setError("Description is required"); return false; }
    if (parseFloat(depositAmount) < 100) { setError("Minimum deposit is 100 REMES"); return false; }
    if (proposalType === "model_upgrade") {
      if (!modelVersion.trim()) { setError("Model version is required"); return false; }
      if (!modelIpfsHash.trim() || !modelIpfsHash.startsWith("Qm")) { setError("Valid IPFS hash is required (starts with Qm)"); return false; }
    }
    if (proposalType === "dataset_proposal") {
      if (!datasetName.trim()) { setError("Dataset name is required"); return false; }
      if (!datasetIpfsHash.trim() || !datasetIpfsHash.startsWith("Qm")) { setError("Valid IPFS hash is required (starts with Qm)"); return false; }
      if (!datasetCategory) { setError("Dataset category is required"); return false; }
    }
    if (proposalType === "parameter_change") {
      if (!parameterKey.trim()) { setError("Parameter key is required"); return false; }
      if (!parameterValue.trim()) { setError("Parameter value is required"); return false; }
    }
    return true;
  };

  const handleSubmit = async () => {
    if (!walletAddress) { setError("Please connect your wallet first"); return; }
    if (!validateForm()) return;
    setIsSubmitting(true); setError(null);

    const proposalData: Record<string, any> = { type: proposalType, title, description, deposit: { denom: "uremes", amount: (parseFloat(depositAmount) * 1e6).toString() }, proposer: walletAddress };
    if (proposalType === "model_upgrade") proposalData.content = { "@type": "/remes.remes.v1.ModelUpgradeProposal", model_version: modelVersion, model_ipfs_hash: modelIpfsHash, migration_plan: migrationPlan, backward_compatible: backwardCompatible };
    else if (proposalType === "dataset_proposal") proposalData.content = { "@type": "/remes.remes.v1.DatasetProposal", dataset_name: datasetName, dataset_ipfs_hash: datasetIpfsHash, dataset_size: datasetSize, dataset_category: datasetCategory, dataset_description: datasetDescription };
    else if (proposalType === "parameter_change") proposalData.content = { "@type": "/cosmos.params.v1beta1.ParameterChangeProposal", changes: [{ subspace: "remes", key: parameterKey, value: parameterValue }] };
    else proposalData.content = { "@type": "/cosmos.gov.v1beta1.TextProposal", title, description };

    try {
      const response = await fetch("/api/governance/proposals", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(proposalData) });
      if (!response.ok) { const errorData = await response.json(); throw new Error(errorData.message || "Failed to create proposal"); }
      onSuccess(); handleClose();
    } catch (err) { console.error("Error creating proposal:", err); setError(err instanceof Error ? err.message : "Failed to create proposal"); }
    finally { setIsSubmitting(false); }
  };

  if (!isOpen) return null;

  const inputClass = "w-full px-4 py-3 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 text-sm transition-all focus:outline-none focus:border-indigo-500 focus:ring-[3px] focus:ring-indigo-500/10 disabled:opacity-60 disabled:cursor-not-allowed";
  const labelClass = "block mb-2 text-sm text-slate-400 font-medium";

  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-[1000]" onClick={handleClose}>
      <div className="bg-slate-800 rounded-xl w-[90%] max-w-[600px] max-h-[90vh] overflow-y-auto border border-slate-700 shadow-2xl" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex justify-between items-center px-6 py-5 border-b border-slate-700">
          <h2 className="text-xl font-semibold text-slate-100">Create New Proposal</h2>
          <button onClick={handleClose} className="text-2xl text-slate-400 hover:text-slate-100 transition-colors leading-none">×</button>
        </div>

        {error && <div className="mx-6 mt-4 px-4 py-3 bg-red-500/10 border border-red-500/30 text-red-500 rounded-lg text-sm">{error}</div>}
        
        {/* Body */}
        <div className="p-6 space-y-5">
          <div>
            <label className={labelClass}>Proposal Type</label>
            <select value={proposalType} onChange={e => setProposalType(e.target.value as ProposalType)} disabled={isSubmitting} className={inputClass}>
              <option value="text">Text Proposal</option>
              <option value="parameter_change">Parameter Change</option>
              <option value="software_upgrade">Software Upgrade</option>
              <option value="model_upgrade">Model Upgrade</option>
              <option value="dataset_proposal">Dataset Proposal</option>
            </select>
          </div>

          <div>
            <label className={labelClass}>Title *</label>
            <input type="text" value={title} onChange={e => setTitle(e.target.value)} placeholder="Enter proposal title" disabled={isSubmitting} maxLength={140} className={inputClass} />
            <small className="block mt-1.5 text-xs text-slate-500">{title.length}/140 characters</small>
          </div>

          <div>
            <label className={labelClass}>Description *</label>
            <textarea value={description} onChange={e => setDescription(e.target.value)} placeholder="Detailed description of the proposal" rows={4} disabled={isSubmitting} className={`${inputClass} resize-y min-h-[80px]`} />
          </div>

          {proposalType === "model_upgrade" && (
            <div className="bg-slate-900 rounded-lg p-5 border border-slate-700 space-y-4">
              <h4 className="text-sm font-semibold text-indigo-400 mb-4">Model Upgrade Details</h4>
              <div><label className={labelClass}>Model Version *</label><input type="text" value={modelVersion} onChange={e => setModelVersion(e.target.value)} placeholder="e.g., BitNet v2.0" disabled={isSubmitting} className={inputClass} /></div>
              <div><label className={labelClass}>Model IPFS Hash *</label><input type="text" value={modelIpfsHash} onChange={e => setModelIpfsHash(e.target.value)} placeholder="Qm..." disabled={isSubmitting} className={inputClass} /></div>
              <div><label className={labelClass}>Migration Plan</label><textarea value={migrationPlan} onChange={e => setMigrationPlan(e.target.value)} placeholder="Describe the migration process for miners" rows={3} disabled={isSubmitting} className={`${inputClass} resize-y`} /></div>
              <label className="flex items-center gap-2 cursor-pointer"><input type="checkbox" checked={backwardCompatible} onChange={e => setBackwardCompatible(e.target.checked)} disabled={isSubmitting} className="cursor-pointer" /><span className="text-sm text-slate-400">Backward Compatible</span></label>
            </div>
          )}

          {proposalType === "dataset_proposal" && (
            <div className="bg-slate-900 rounded-lg p-5 border border-slate-700 space-y-4">
              <h4 className="text-sm font-semibold text-indigo-400 mb-4">Dataset Details</h4>
              <div><label className={labelClass}>Dataset Name *</label><input type="text" value={datasetName} onChange={e => setDatasetName(e.target.value)} placeholder="Dataset name" disabled={isSubmitting} className={inputClass} /></div>
              <div><label className={labelClass}>Dataset IPFS Hash *</label><input type="text" value={datasetIpfsHash} onChange={e => setDatasetIpfsHash(e.target.value)} placeholder="Qm..." disabled={isSubmitting} className={inputClass} /></div>
              <div className="grid grid-cols-2 gap-4 max-sm:grid-cols-1">
                <div><label className={labelClass}>Size</label><input type="text" value={datasetSize} onChange={e => setDatasetSize(e.target.value)} placeholder="e.g., 10GB" disabled={isSubmitting} className={inputClass} /></div>
                <div><label className={labelClass}>Category *</label><select value={datasetCategory} onChange={e => setDatasetCategory(e.target.value)} disabled={isSubmitting} className={inputClass}><option value="">Select category</option><option value="text">Text</option><option value="code">Code</option><option value="multimodal">Multimodal</option><option value="scientific">Scientific</option><option value="conversational">Conversational</option></select></div>
              </div>
              <div><label className={labelClass}>Dataset Description</label><textarea value={datasetDescription} onChange={e => setDatasetDescription(e.target.value)} placeholder="Describe the dataset content and quality" rows={2} disabled={isSubmitting} className={`${inputClass} resize-y`} /></div>
            </div>
          )}

          {proposalType === "parameter_change" && (
            <div className="bg-slate-900 rounded-lg p-5 border border-slate-700 space-y-4">
              <h4 className="text-sm font-semibold text-indigo-400 mb-4">Parameter Change Details</h4>
              <div><label className={labelClass}>Parameter Key *</label><input type="text" value={parameterKey} onChange={e => setParameterKey(e.target.value)} placeholder="e.g., base_reward_per_gradient" disabled={isSubmitting} className={inputClass} /></div>
              <div><label className={labelClass}>New Value *</label><input type="text" value={parameterValue} onChange={e => setParameterValue(e.target.value)} placeholder="New parameter value" disabled={isSubmitting} className={inputClass} /></div>
            </div>
          )}

          <div>
            <label className={labelClass}>Deposit Amount (REMES) *</label>
            <input type="number" value={depositAmount} onChange={e => setDepositAmount(e.target.value)} min="100" disabled={isSubmitting} className={inputClass} />
            <small className="block mt-1.5 text-xs text-slate-500">Minimum deposit: 100 REMES. Deposit is returned if proposal passes or is rejected (not vetoed).</small>
          </div>
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 px-6 py-4 border-t border-slate-700 max-sm:flex-col">
          <button onClick={handleClose} disabled={isSubmitting} className="px-6 py-3 bg-transparent border border-slate-700 text-slate-400 hover:bg-slate-900 hover:text-slate-100 text-sm font-medium rounded-lg transition-all disabled:opacity-60 disabled:cursor-not-allowed max-sm:w-full">Cancel</button>
          <button onClick={handleSubmit} disabled={isSubmitting || !walletAddress} className="px-6 py-3 bg-indigo-500 hover:bg-indigo-600 text-white text-sm font-medium rounded-lg transition-all disabled:opacity-60 disabled:cursor-not-allowed max-sm:w-full">{isSubmitting ? "Submitting..." : "Submit Proposal"}</button>
        </div>
      </div>
    </div>
  );
}
