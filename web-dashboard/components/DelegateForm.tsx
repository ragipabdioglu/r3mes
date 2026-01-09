"use client";

import { useState } from "react";
import { signAndBroadcastTransaction } from "@/lib/keplr";

interface DelegateFormProps {
  validatorAddress: string;
  action: "delegate" | "undelegate" | "redelegate";
  onClose: () => void;
  onSuccess: () => void;
}

export default function DelegateForm({ validatorAddress, action, onClose, onSuccess }: DelegateFormProps) {
  const [amount, setAmount] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [targetValidator, setTargetValidator] = useState("");

  const handleSubmit = async () => {
    if (!amount || parseFloat(amount) <= 0) { setError("Please enter a valid amount"); return; }
    if (action === "redelegate" && !targetValidator) { setError("Please enter target validator address"); return; }

    setIsSubmitting(true); setError(null);

    try {
      const amountInUremes = (parseFloat(amount) * 1e6).toString();
      let message: any;

      switch (action) {
        case "delegate":
          message = { typeUrl: "/cosmos.staking.v1beta1.MsgDelegate", value: { delegatorAddress: "", validatorAddress, amount: { denom: "uremes", amount: amountInUremes } } };
          break;
        case "undelegate":
          message = { typeUrl: "/cosmos.staking.v1beta1.MsgUndelegate", value: { delegatorAddress: "", validatorAddress, amount: { denom: "uremes", amount: amountInUremes } } };
          break;
        case "redelegate":
          message = { typeUrl: "/cosmos.staking.v1beta1.MsgBeginRedelegate", value: { delegatorAddress: "", validatorSrcAddress: validatorAddress, validatorDstAddress: targetValidator, amount: { denom: "uremes", amount: amountInUremes } } };
          break;
      }

      await signAndBroadcastTransaction([message], `${action.charAt(0).toUpperCase() + action.slice(1)} ${amount} REMES`);
      onSuccess();
    } catch (err: any) { setError(err.message || "Transaction failed"); }
    finally { setIsSubmitting(false); }
  };

  const getActionLabel = () => ({ delegate: "Delegate", undelegate: "Undelegate", redelegate: "Redelegate" }[action]);

  const inputClass = "w-full px-3 py-3 bg-slate-900 border border-slate-700 rounded-lg text-slate-100 text-sm transition-colors focus:outline-none focus:border-blue-500 disabled:opacity-50 disabled:cursor-not-allowed";

  return (
    <div className="fixed inset-0 bg-black/70 flex items-center justify-center z-[2000]" onClick={onClose}>
      <div className="bg-slate-800 border border-slate-700 rounded-2xl p-6 max-w-[500px] w-[90%]" onClick={e => e.stopPropagation()}>
        {/* Header */}
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-xl font-semibold text-slate-100">{getActionLabel()} Tokens</h3>
          <button onClick={onClose} className="text-slate-400 hover:text-slate-100 hover:bg-slate-700 text-3xl leading-none w-8 h-8 flex items-center justify-center rounded transition-colors">×</button>
        </div>

        {/* Content */}
        <div className="mb-6 space-y-5">
          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Validator Address</label>
            <input type="text" value={validatorAddress} disabled className={`${inputClass} opacity-50 cursor-not-allowed`} />
          </div>

          {action === "redelegate" && (
            <div>
              <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Target Validator Address</label>
              <input type="text" value={targetValidator} onChange={e => setTargetValidator(e.target.value)} placeholder="Enter target validator address" className={inputClass} />
            </div>
          )}

          <div>
            <label className="block text-xs font-semibold text-slate-400 uppercase tracking-wide mb-2">Amount (REMES)</label>
            <input type="number" value={amount} onChange={e => setAmount(e.target.value)} placeholder="0.00" min="0" step="0.01" className={inputClass} />
            {action === "undelegate" && <div className="text-xs text-amber-500 mt-1">⚠️ Unbonding period: 21 days</div>}
          </div>

          {error && <div className="bg-red-900/50 border border-red-500 rounded-lg px-3 py-3 text-red-300 text-sm">{error}</div>}
        </div>

        {/* Footer */}
        <div className="flex justify-end gap-3 pt-4 border-t border-slate-700">
          <button onClick={onClose} disabled={isSubmitting} className="px-6 py-2.5 bg-slate-700 hover:bg-slate-600 text-slate-100 text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed">Cancel</button>
          <button onClick={handleSubmit} disabled={!amount || isSubmitting} className="px-6 py-2.5 bg-blue-500 hover:bg-blue-600 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed">{isSubmitting ? "Processing..." : getActionLabel()}</button>
        </div>
      </div>
    </div>
  );
}
