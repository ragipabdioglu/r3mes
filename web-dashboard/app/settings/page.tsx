"use client";

import { useState, useEffect } from "react";
import { Settings, Save, AlertCircle, CheckCircle } from "lucide-react";
import { useWallet } from "@/contexts/WalletContext";
import WalletGuard from "@/components/WalletGuard";

interface AppConfig {
  base_model_path: string;
  model_download_dir: string;
  database_path: string;
  chain_json_path: string;
  mining_difficulty: number;
  gpu_memory_limit_mb: number | null;
  p2p_port: number;
  rate_limit_chat: string;
  rate_limit_get: string;
  blockchain_rpc_url: string;
  blockchain_grpc_url: string;
  auto_start_mining: boolean;
  enable_notifications: boolean;
}

export default function SettingsPage() {
  const { walletAddress } = useWallet();
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      setLoading(true);
      const response = await fetch("/api/config");
      if (!response.ok) {
        throw new Error("Failed to load configuration");
      }
      const data = await response.json();
      setConfig(data.config);
      setError(null);
    } catch (err: any) {
      setError(err.message || "Failed to load configuration");
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    if (!config) return;

    try {
      setSaving(true);
      setError(null);
      setSuccess(false);

      const response = await fetch("/api/config", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(config),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to save configuration");
      }

      setSuccess(true);
      setTimeout(() => setSuccess(false), 3000);
    } catch (err: any) {
      setError(err.message || "Failed to save configuration");
    } finally {
      setSaving(false);
    }
  };

  const handleChange = (field: keyof AppConfig, value: any) => {
    if (!config) return;
    setConfig({ ...config, [field]: value });
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-900 text-slate-100 py-8 px-4 sm:py-10 sm:px-6 md:py-12 md:px-8">
        <div className="container mx-auto max-w-full sm:max-w-xl md:max-w-2xl lg:max-w-4xl">
          <div className="card p-4 sm:p-5 md:p-6">
            <div className="text-center py-8 sm:py-10 md:py-12">
              <div className="text-sm sm:text-base text-slate-400">Loading configuration...</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!config) {
    return (
      <div className="min-h-screen bg-slate-900 text-slate-100 py-8 px-4 sm:py-10 sm:px-6 md:py-12 md:px-8">
        <div className="container mx-auto max-w-full sm:max-w-xl md:max-w-2xl lg:max-w-4xl">
          <div className="card p-4 sm:p-5 md:p-6 bg-red-900/20 border-red-500/50">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-4 h-4 sm:w-5 sm:h-5 text-red-400" />
              <div>
                <h3 className="text-base sm:text-lg font-semibold text-red-400 mb-1">
                  Failed to Load Configuration
                </h3>
                <p className="text-xs sm:text-sm text-red-300">
                  {error || "An error occurred while loading configuration"}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <WalletGuard>
      <div className="min-h-screen bg-slate-900 text-slate-100 py-8 px-4 sm:py-10 sm:px-6 md:py-12 md:px-8">
        <div className="container mx-auto max-w-full sm:max-w-xl md:max-w-2xl lg:max-w-4xl">
          <div className="flex items-center gap-2 sm:gap-3 mb-6 sm:mb-8">
            <Settings className="w-6 h-6 sm:w-8 sm:h-8 text-primary" />
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-bold gradient-text">Settings</h1>
          </div>

          {error && (
            <div className="card bg-red-900/20 border-red-500/50 mb-6">
              <div className="flex items-center gap-3">
                <AlertCircle className="w-5 h-5 text-red-400" />
                <p className="text-red-300">{error}</p>
              </div>
            </div>
          )}

          {success && (
            <div className="card bg-green-900/20 border-green-500/50 mb-6">
              <div className="flex items-center gap-3">
                <CheckCircle className="w-5 h-5 text-green-400" />
                <p className="text-green-300">Configuration saved successfully!</p>
              </div>
            </div>
          )}

          <div className="card mb-6">
            <h2 className="text-xl font-semibold mb-4">Model Settings</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Base Model Path
                </label>
                <input
                  type="text"
                  value={config.base_model_path}
                  onChange={(e) => handleChange("base_model_path", e.target.value)}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:outline-none focus:border-primary"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Model Download Directory
                </label>
                <input
                  type="text"
                  value={config.model_download_dir}
                  onChange={(e) => handleChange("model_download_dir", e.target.value)}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:outline-none focus:border-primary"
                />
              </div>
            </div>
          </div>

          <div className="card mb-6">
            <h2 className="text-xl font-semibold mb-4">Mining Settings</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Mining Difficulty
                </label>
                <input
                  type="number"
                  step="0.1"
                  value={config.mining_difficulty}
                  onChange={(e) => handleChange("mining_difficulty", parseFloat(e.target.value))}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:outline-none focus:border-primary"
                />
                <p className="text-xs text-slate-400 mt-1">
                  Usually fetched from blockchain. Set manually for testing.
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  GPU Memory Limit (MB)
                </label>
                <input
                  type="number"
                  value={config.gpu_memory_limit_mb || ""}
                  onChange={(e) =>
                    handleChange(
                      "gpu_memory_limit_mb",
                      e.target.value ? parseInt(e.target.value) : null
                    )
                  }
                  placeholder="Auto-detect"
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:outline-none focus:border-primary"
                />
                <p className="text-xs text-slate-400 mt-1">
                  Leave empty to auto-detect GPU memory
                </p>
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  P2P Port
                </label>
                <input
                  type="number"
                  min="1024"
                  max="65535"
                  value={config.p2p_port}
                  onChange={(e) => handleChange("p2p_port", parseInt(e.target.value))}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:outline-none focus:border-primary"
                />
              </div>
            </div>
          </div>

          <div className="card mb-6">
            <h2 className="text-xl font-semibold mb-4">Network Settings</h2>
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Blockchain RPC URL
                </label>
                <input
                  type="text"
                  value={config.blockchain_rpc_url}
                  onChange={(e) => handleChange("blockchain_rpc_url", e.target.value)}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:outline-none focus:border-primary"
                />
              </div>
              <div>
                <label className="block text-sm font-medium text-slate-300 mb-2">
                  Blockchain gRPC URL
                </label>
                <input
                  type="text"
                  value={config.blockchain_grpc_url}
                  onChange={(e) => handleChange("blockchain_grpc_url", e.target.value)}
                  className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 focus:outline-none focus:border-primary"
                />
              </div>
            </div>
          </div>

          <div className="card mb-6">
            <h2 className="text-xl font-semibold mb-4">Feature Flags</h2>
            <div className="space-y-4">
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.auto_start_mining}
                  onChange={(e) => handleChange("auto_start_mining", e.target.checked)}
                  className="w-5 h-5 rounded bg-slate-800 border-slate-700 text-primary focus:ring-primary"
                />
                <span className="text-slate-300">Auto-start mining on launch</span>
              </label>
              <label className="flex items-center gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.enable_notifications}
                  onChange={(e) => handleChange("enable_notifications", e.target.checked)}
                  className="w-5 h-5 rounded bg-slate-800 border-slate-700 text-primary focus:ring-primary"
                />
                <span className="text-slate-300">Enable desktop notifications</span>
              </label>
            </div>
          </div>

          <div className="flex justify-end gap-4">
            <button
              onClick={loadConfig}
              className="px-6 py-2 bg-slate-800 border border-slate-700 rounded-lg text-slate-200 hover:bg-slate-700 transition-colors"
            >
              Reset
            </button>
            <button
              onClick={handleSave}
              disabled={saving}
              className="px-6 py-2 bg-primary text-slate-900 rounded-lg font-medium hover:bg-primary/90 transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
            >
              <Save className="w-4 h-4" />
              {saving ? "Saving..." : "Save Configuration"}
            </button>
          </div>
        </div>
      </div>
    </WalletGuard>
  );
}

