import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/tauri";
import { 
  NETWORK_PRESETS, 
  type MinerConfig,
  type NetworkConfig,
  type AdvancedConfig,
  getDefaultMinerConfig,
  getDefaultNetworkConfig,
  getDefaultAdvancedConfig,
} from "../types/config";
import "./ConfigurationPanel.css";

interface ConfigurationPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ConfigurationPanel({ isOpen, onClose }: ConfigurationPanelProps) {
  const [activeTab, setActiveTab] = useState<"miner" | "network" | "advanced">("miner");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Miner Config State - using typed defaults
  const [minerConfig, setMinerConfig] = useState<MinerConfig>(getDefaultMinerConfig());

  // Network Config State - using typed defaults
  const [networkConfig, setNetworkConfig] = useState<NetworkConfig>(getDefaultNetworkConfig());

  // Advanced Config State - using typed defaults
  const [advancedConfig, setAdvancedConfig] = useState<AdvancedConfig>(getDefaultAdvancedConfig());

  useEffect(() => {
    if (isOpen) {
      loadConfig();
    }
  }, [isOpen]);

  const loadConfig = async () => {
    try {
      const config = await invoke<{
        miner: MinerConfig;
        network: NetworkConfig;
        advanced: AdvancedConfig;
      }>("get_config");
      
      if (config.miner) setMinerConfig(config.miner);
      if (config.network) setNetworkConfig(config.network);
      if (config.advanced) setAdvancedConfig(config.advanced);
    } catch (err) {
      console.error("Failed to load config:", err);
      // Use defaults if config doesn't exist
    }
  };

  const saveConfig = async () => {
    setSaving(true);
    setError(null);
    setSaved(false);

    try {
      await invoke("save_config", {
        config: {
          miner: minerConfig,
          network: networkConfig,
          advanced: advancedConfig,
        },
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      setError(err as string);
    } finally {
      setSaving(false);
    }
  };

  const resetToDefaults = async () => {
    try {
      await invoke("reset_config_to_defaults");
      await loadConfig();
    } catch (err) {
      setError(err as string);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="config-overlay">
      <div className="config-panel">
        <div className="config-header">
          <h2>Configuration</h2>
          <button className="close-btn" onClick={onClose}>×</button>
        </div>

        <div className="config-tabs">
          <button
            className={`tab ${activeTab === "miner" ? "active" : ""}`}
            onClick={() => setActiveTab("miner")}
          >
            ⚡ Miner
          </button>
          <button
            className={`tab ${activeTab === "network" ? "active" : ""}`}
            onClick={() => setActiveTab("network")}
          >
            🌐 Network
          </button>
          <button
            className={`tab ${activeTab === "advanced" ? "active" : ""}`}
            onClick={() => setActiveTab("advanced")}
          >
            ⚙️ Advanced
          </button>
        </div>

        <div className="config-content">
          {activeTab === "miner" && (
            <div className="config-section">
              <div className="form-group">
                <label>Wallet Address</label>
                <input
                  type="text"
                  value={minerConfig.wallet_address}
                  onChange={(e) => setMinerConfig({ ...minerConfig, wallet_address: e.target.value })}
                  placeholder="remes1..."
                />
                <span className="hint">Your R3MES wallet address for receiving rewards</span>
              </div>

              <div className="form-group">
                <label>Backend URL</label>
                <input
                  type="text"
                  value={minerConfig.backend_url}
                  onChange={(e) => setMinerConfig({ ...minerConfig, backend_url: e.target.value })}
                  placeholder="http://localhost:8000"
                />
              </div>

              <div className="form-group">
                <label>IPFS Gateway</label>
                <input
                  type="text"
                  value={minerConfig.ipfs_gateway}
                  onChange={(e) => setMinerConfig({ ...minerConfig, ipfs_gateway: e.target.value })}
                  placeholder="http://localhost:5001"
                />
              </div>

              <div className="form-row">
                <div className="form-group half">
                  <label>GPU Memory Limit (%)</label>
                  <input
                    type="range"
                    min="50"
                    max="100"
                    value={minerConfig.gpu_memory_limit}
                    onChange={(e) => setMinerConfig({ ...minerConfig, gpu_memory_limit: parseInt(e.target.value) })}
                  />
                  <span className="range-value">{minerConfig.gpu_memory_limit}%</span>
                </div>

                <div className="form-group half">
                  <label>Batch Size</label>
                  <select
                    value={minerConfig.batch_size}
                    onChange={(e) => setMinerConfig({ ...minerConfig, batch_size: parseInt(e.target.value) })}
                  >
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="4">4</option>
                    <option value="8">8</option>
                    <option value="16">16</option>
                  </select>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group half">
                  <label>Gradient Accumulation Steps</label>
                  <select
                    value={minerConfig.gradient_accumulation_steps}
                    onChange={(e) => setMinerConfig({ ...minerConfig, gradient_accumulation_steps: parseInt(e.target.value) })}
                  >
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="4">4</option>
                    <option value="8">8</option>
                    <option value="16">16</option>
                  </select>
                </div>

                <div className="form-group half">
                  <label>Log Level</label>
                  <select
                    value={minerConfig.log_level}
                    onChange={(e) => setMinerConfig({ ...minerConfig, log_level: e.target.value as MinerConfig['log_level'] })}
                  >
                    <option value="DEBUG">Debug</option>
                    <option value="INFO">Info</option>
                    <option value="WARNING">Warning</option>
                    <option value="ERROR">Error</option>
                  </select>
                </div>
              </div>

              <div className="form-group checkbox-group">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={minerConfig.mixed_precision}
                    onChange={(e) => setMinerConfig({ ...minerConfig, mixed_precision: e.target.checked })}
                  />
                  <span>Enable Mixed Precision (FP16)</span>
                </label>
                <span className="hint">Reduces memory usage and improves performance</span>
              </div>

              <div className="form-group checkbox-group">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={minerConfig.auto_restart}
                    onChange={(e) => setMinerConfig({ ...minerConfig, auto_restart: e.target.checked })}
                  />
                  <span>Auto-restart on crash</span>
                </label>
              </div>
            </div>
          )}

          {activeTab === "network" && (
            <div className="config-section">
              <div className="form-group">
                <label>Chain ID</label>
                <input
                  type="text"
                  value={networkConfig.chain_id}
                  onChange={(e) => setNetworkConfig({ ...networkConfig, chain_id: e.target.value })}
                  placeholder="remes-1"
                />
              </div>

              <div className="form-group">
                <label>RPC Endpoint</label>
                <input
                  type="text"
                  value={networkConfig.rpc_endpoint}
                  onChange={(e) => setNetworkConfig({ ...networkConfig, rpc_endpoint: e.target.value })}
                  placeholder="http://localhost:26657"
                />
                <span className="hint">Tendermint RPC endpoint for blockchain queries</span>
              </div>

              <div className="form-group">
                <label>REST Endpoint</label>
                <input
                  type="text"
                  value={networkConfig.rest_endpoint}
                  onChange={(e) => setNetworkConfig({ ...networkConfig, rest_endpoint: e.target.value })}
                  placeholder="http://localhost:1317"
                />
                <span className="hint">Cosmos SDK REST API endpoint</span>
              </div>

              <div className="form-group">
                <label>gRPC Endpoint</label>
                <input
                  type="text"
                  value={networkConfig.grpc_endpoint}
                  onChange={(e) => setNetworkConfig({ ...networkConfig, grpc_endpoint: e.target.value })}
                  placeholder="localhost:9090"
                />
              </div>

              <div className="form-group">
                <label>WebSocket Endpoint</label>
                <input
                  type="text"
                  value={networkConfig.websocket_endpoint}
                  onChange={(e) => setNetworkConfig({ ...networkConfig, websocket_endpoint: e.target.value })}
                  placeholder="ws://localhost:26657/websocket"
                />
                <span className="hint">For real-time block and event subscriptions</span>
              </div>

              <div className="network-presets">
                <h4>Quick Presets</h4>
                <div className="preset-buttons">
                  <button
                    className="preset-btn"
                    onClick={() => setNetworkConfig({ ...NETWORK_PRESETS.local })}
                  >
                    Local Node
                  </button>
                  <button
                    className="preset-btn"
                    onClick={() => setNetworkConfig({ ...NETWORK_PRESETS.testnet })}
                  >
                    Testnet
                  </button>
                  <button
                    className="preset-btn"
                    onClick={() => setNetworkConfig({ ...NETWORK_PRESETS.mainnet })}
                  >
                    Mainnet
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === "advanced" && (
            <div className="config-section">
              <div className="form-row">
                <div className="form-group half">
                  <label>Max Workers</label>
                  <input
                    type="number"
                    min="1"
                    max="8"
                    value={advancedConfig.max_workers}
                    onChange={(e) => setAdvancedConfig({ ...advancedConfig, max_workers: parseInt(e.target.value) })}
                  />
                  <span className="hint">Number of parallel training workers</span>
                </div>

                <div className="form-group half">
                  <label>Checkpoint Interval</label>
                  <input
                    type="number"
                    min="10"
                    max="1000"
                    value={advancedConfig.checkpoint_interval}
                    onChange={(e) => setAdvancedConfig({ ...advancedConfig, checkpoint_interval: parseInt(e.target.value) })}
                  />
                  <span className="hint">Steps between checkpoints</span>
                </div>
              </div>

              <div className="form-group">
                <label>Update Channel</label>
                <select
                  value={advancedConfig.update_channel}
                  onChange={(e) => setAdvancedConfig({ ...advancedConfig, update_channel: e.target.value as AdvancedConfig['update_channel'] })}
                >
                  <option value="stable">Stable</option>
                  <option value="beta">Beta</option>
                  <option value="nightly">Nightly</option>
                </select>
                <span className="hint">Software update release channel</span>
              </div>

              <div className="form-group checkbox-group">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={advancedConfig.auto_update}
                    onChange={(e) => setAdvancedConfig({ ...advancedConfig, auto_update: e.target.checked })}
                  />
                  <span>Enable Auto-Updates</span>
                </label>
                <span className="hint">Automatically download and install updates</span>
              </div>

              <div className="form-group checkbox-group">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={advancedConfig.telemetry_enabled}
                    onChange={(e) => setAdvancedConfig({ ...advancedConfig, telemetry_enabled: e.target.checked })}
                  />
                  <span>Enable Telemetry</span>
                </label>
                <span className="hint">Send anonymous usage statistics to improve R3MES</span>
              </div>

              <div className="form-group checkbox-group">
                <label className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={advancedConfig.debug_mode}
                    onChange={(e) => setAdvancedConfig({ ...advancedConfig, debug_mode: e.target.checked })}
                  />
                  <span>Debug Mode</span>
                </label>
                <span className="hint">Enable verbose logging and debugging features</span>
              </div>

              <div className="danger-zone">
                <h4>Danger Zone</h4>
                <button className="btn-danger" onClick={resetToDefaults}>
                  Reset to Defaults
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="config-footer">
          {error && <div className="error-message">{error}</div>}
          {saved && <div className="success-message">✓ Configuration saved</div>}
          
          <div className="footer-actions">
            <button className="btn-secondary" onClick={onClose}>
              Cancel
            </button>
            <button className="btn-primary" onClick={saveConfig} disabled={saving}>
              {saving ? "Saving..." : "Save Configuration"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
