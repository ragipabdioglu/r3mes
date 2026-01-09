import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/tauri";
import { Activity, CheckCircle, XCircle, AlertCircle, RefreshCw } from "lucide-react";

interface SystemStatus {
  chain_sync: {
    synced: boolean;
    percentage: number;
    block_height: number | null;
    latest_block_height: number | null;
  };
  ipfs: {
    connected: boolean;
    peers: number;
    status: string;
  };
  model: {
    downloaded: boolean;
    progress: number;
    file_name: string | null;
    file_size_gb: number | null;
    integrity_verified: boolean;
  };
  node: {
    running: boolean;
    rpc_endpoint: string;
    grpc_endpoint: string;
    last_block_time: number | null;
  };
}

export default function SystemStatusPanel() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchStatus = async () => {
    try {
      setLoading(true);
      setError(null);
      const result = await invoke<SystemStatus>("get_system_status");
      setStatus(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch system status");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000); // Poll every 5 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading && !status) {
    return (
      <div className="card p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-slate-700 rounded w-1/4 mb-4"></div>
          <div className="h-4 bg-slate-700 rounded w-full mb-2"></div>
          <div className="h-4 bg-slate-700 rounded w-5/6"></div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="card p-6">
        <div className="flex items-center gap-3 text-red-400">
          <XCircle className="w-5 h-5" />
          <p>{error}</p>
        </div>
        <button onClick={fetchStatus} className="btn-secondary mt-4">
          Retry
        </button>
      </div>
    );
  }

  if (!status) {
    return null;
  }

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-2xl font-bold flex items-center gap-2">
          <Activity className="w-6 h-6 text-green-400" />
          System Status
        </h2>
        <button
          onClick={fetchStatus}
          className="p-2 hover:bg-slate-800 rounded-lg transition-colors"
          title="Refresh"
        >
          <RefreshCw className="w-5 h-5" />
        </button>
      </div>

      <div className="space-y-4">
        {/* Chain Sync Status */}
        <div className="border-l-2 border-slate-700 pl-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-bold">Chain Sync</h3>
            {status.chain_sync.synced ? (
              <CheckCircle className="w-5 h-5 text-green-400" />
            ) : (
              <AlertCircle className="w-5 h-5 text-yellow-400" />
            )}
          </div>
          <div className="text-sm text-slate-400">
            {status.chain_sync.synced ? (
              <span>Synced ({status.chain_sync.percentage.toFixed(1)}%)</span>
            ) : (
              <span>Syncing... {status.chain_sync.percentage.toFixed(1)}%</span>
            )}
          </div>
          {status.chain_sync.block_height && (
            <div className="text-xs text-slate-500 mt-1">
              Block Height: {status.chain_sync.block_height}
            </div>
          )}
        </div>

        {/* IPFS Status */}
        <div className="border-l-2 border-slate-700 pl-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-bold">IPFS Node</h3>
            {status.ipfs.connected ? (
              <CheckCircle className="w-5 h-5 text-green-400" />
            ) : (
              <XCircle className="w-5 h-5 text-red-400" />
            )}
          </div>
          <div className="text-sm text-slate-400">
            {status.ipfs.connected ? (
              <span>Connected ({status.ipfs.peers} peers)</span>
            ) : (
              <span>Offline</span>
            )}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            Status: {status.ipfs.status}
          </div>
        </div>

        {/* Model Status */}
        <div className="border-l-2 border-slate-700 pl-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-bold">Model</h3>
            {status.model.downloaded ? (
              <CheckCircle className="w-5 h-5 text-green-400" />
            ) : (
              <AlertCircle className="w-5 h-5 text-yellow-400" />
            )}
          </div>
          {status.model.downloaded ? (
            <div className="text-sm text-slate-400">
              <div>Downloaded: {status.model.file_name || "Unknown"}</div>
              {status.model.file_size_gb && (
                <div className="text-xs text-slate-500 mt-1">
                  Size: {status.model.file_size_gb.toFixed(2)} GB
                </div>
              )}
            </div>
          ) : (
            <div className="text-sm text-slate-400">
              <div>Not downloaded</div>
              {status.model.progress > 0 && (
                <div className="mt-2">
                  <div className="w-full bg-slate-800 rounded-full h-2">
                    <div
                      className="bg-green-500 h-2 rounded-full transition-all"
                      style={{ width: `${status.model.progress * 100}%` }}
                    />
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    {Math.round(status.model.progress * 100)}%
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {/* Node Status */}
        <div className="border-l-2 border-slate-700 pl-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-bold">Blockchain Node</h3>
            {status.node.running ? (
              <CheckCircle className="w-5 h-5 text-green-400" />
            ) : (
              <XCircle className="w-5 h-5 text-red-400" />
            )}
          </div>
          <div className="text-sm text-slate-400">
            {status.node.running ? (
              <span>Running</span>
            ) : (
              <span>Stopped</span>
            )}
          </div>
          <div className="text-xs text-slate-500 mt-1">
            RPC: {status.node.rpc_endpoint} | gRPC: {status.node.grpc_endpoint}
          </div>
        </div>
      </div>
    </div>
  );
}
