"use client";

import { useState } from "react";
import { signAndBroadcastTransaction } from "@/lib/keplr";

interface DelegateFormProps {
  validatorAddress: string;
  action: "delegate" | "undelegate" | "redelegate";
  onClose: () => void;
  onSuccess: () => void;
}

export default function DelegateForm({
  validatorAddress,
  action,
  onClose,
  onSuccess,
}: DelegateFormProps) {
  const [amount, setAmount] = useState("");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [targetValidator, setTargetValidator] = useState(""); // For redelegate

  const handleSubmit = async () => {
    if (!amount || parseFloat(amount) <= 0) {
      setError("Please enter a valid amount");
      return;
    }

    if (action === "redelegate" && !targetValidator) {
      setError("Please enter target validator address");
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      const amountInUremes = (parseFloat(amount) * 1e6).toString();

      let message: any;

      switch (action) {
        case "delegate":
          message = {
            typeUrl: "/cosmos.staking.v1beta1.MsgDelegate",
            value: {
              delegatorAddress: "", // Will be filled by Keplr
              validatorAddress: validatorAddress,
              amount: {
                denom: "uremes",
                amount: amountInUremes,
              },
            },
          };
          break;

        case "undelegate":
          message = {
            typeUrl: "/cosmos.staking.v1beta1.MsgUndelegate",
            value: {
              delegatorAddress: "", // Will be filled by Keplr
              validatorAddress: validatorAddress,
              amount: {
                denom: "uremes",
                amount: amountInUremes,
              },
            },
          };
          break;

        case "redelegate":
          message = {
            typeUrl: "/cosmos.staking.v1beta1.MsgBeginRedelegate",
            value: {
              delegatorAddress: "", // Will be filled by Keplr
              validatorSrcAddress: validatorAddress,
              validatorDstAddress: targetValidator,
              amount: {
                denom: "uremes",
                amount: amountInUremes,
              },
            },
          };
          break;
      }

      await signAndBroadcastTransaction(
        [message],
        `${action.charAt(0).toUpperCase() + action.slice(1)} ${amount} REMES`
      );

      onSuccess();
    } catch (err: any) {
      setError(err.message || "Transaction failed");
    } finally {
      setIsSubmitting(false);
    }
  };

  const getActionLabel = () => {
    switch (action) {
      case "delegate":
        return "Delegate";
      case "undelegate":
        return "Undelegate";
      case "redelegate":
        return "Redelegate";
    }
  };

  return (
    <div className="delegate-form-overlay" onClick={onClose}>
      <div className="delegate-form" onClick={(e) => e.stopPropagation()}>
        <div className="form-header">
          <h3>{getActionLabel()} Tokens</h3>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        <div className="form-content">
          <div className="form-field">
            <label>Validator Address</label>
            <input
              type="text"
              value={validatorAddress}
              disabled
              className="form-input disabled"
            />
          </div>

          {action === "redelegate" && (
            <div className="form-field">
              <label>Target Validator Address</label>
              <input
                type="text"
                value={targetValidator}
                onChange={(e) => setTargetValidator(e.target.value)}
                placeholder="Enter target validator address"
                className="form-input"
              />
            </div>
          )}

          <div className="form-field">
            <label>Amount (REMES)</label>
            <input
              type="number"
              value={amount}
              onChange={(e) => setAmount(e.target.value)}
              placeholder="0.00"
              min="0"
              step="0.01"
              className="form-input"
            />
            {action === "undelegate" && (
              <div className="form-hint">
                ⚠️ Unbonding period: 21 days
              </div>
            )}
          </div>

          {error && (
            <div className="error-message">
              {error}
            </div>
          )}
        </div>

        <div className="form-footer">
          <button onClick={onClose} className="btn-cancel" disabled={isSubmitting}>
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            className="btn-submit"
            disabled={!amount || isSubmitting}
          >
            {isSubmitting ? "Processing..." : getActionLabel()}
          </button>
        </div>
      </div>
    </div>
  );
}

