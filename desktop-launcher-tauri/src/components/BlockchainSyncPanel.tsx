/**
 * Blockchain Sync Panel Component
 * 
 * Displays blockchain synchronization status for:
 * - Models
 * - Adapters
 * - Datasets
 * 
 * Features:
 * - Real-time sync status
 * - Manual sync trigger
 * - Event notifications
 * - Progress tracking
 */

import React, { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { listen } from '@tauri-apps/api/event';

interface ModelSyncStatus {
  model_name: string;
  version: string;
  ipfs_hash: string;
  local_version: string | null;
  up_to_date: boolean;
  size_gb: number;
  last_checked: string;
}

interface AdapterSyncStatus {
  adapter_id: string;
  name: string;
  domain: string;
  version: string;
  ipfs_hash: string;
  synced: boolean;
  last_synced: string | null;
}

interface DatasetSyncStatus {
  dataset_id: string;
  name: string;
  ipfs_hash: string;
  size_gb: number;
  downloaded: boolean;
  last_updated: string | null;
}

interface SyncResult {
  success: boolean;
  message: string;
  new_items: number;
  updated_items: number;
  failed_items: number;
}

export const BlockchainSyncPanel: React.FC = () => {
  const [modelStatus, setModelStatus] = useState<ModelSyncStatus | null>(null);
  const [adapters, setAdapters] = useState<AdapterSyncStatus[]>([]);
  const [datasets, setDatasets] = useState<DatasetSyncStatus[]>([]);
  const [syncing, setSyncing] = useState(false);
  const [lastSyncTime, setLastSyncTime] = useState<string | null>(null);
  const [notifications, setNotifications] = useState<string[]>([]);
  const [autoSync, setAutoSync] = useState(true);

  // Load initial status
  useEffect(() => {
    loadSyncStatus();
    
    // Setup event listeners for blockchain events
    const unlistenModel = listen('blockchain:model_upgraded', (event: any) => {
      addNotification(`Model upgraded: ${event.payload.model_name} v${event.payload.version}`);
      loadSyncStatus();
    });

    const unlistenAdapter = listen('blockchain:adapter_approved', (event: any) => {
      addNotification(`New adapter approved: ${event.payload.adapter_id}`);
      loadSyncStatus();
    });

    const unlistenDataset = listen('blockchain:dataset_approved', (event: any) => {
      addNotification(`New dataset approved: ${event.payload.dataset_id}`);
      loadSyncStatus();
    });

    return () => {
      unlistenModel.then(fn => fn());
      unlistenAdapter.then(fn => fn());
      unlistenDataset.then(fn => fn());
    };
  }, []);

  // Auto-sync every 5 minutes
  useEffect(() => {
    if (!autoSync) return;

    const interval = setInterval(() => {
      syncAll();
    }, 5 * 60 * 1000);

    return () => clearInterval(interval);
  }, [autoSync]);

  const loadSyncStatus = async () => {
    try {
      // Load model status
      const modelData = await invoke<ModelSyncStatus>('check_model_update');
      setModelStatus(modelData);

      // Load adapter status
      const adapterData = await invoke<AdapterSyncStatus[]>('get_synced_adapters');
      setAdapters(adapterData);

      // Load dataset status
      const datasetData = await invoke<DatasetSyncStatus[]>('get_synced_datasets');
      setDatasets(datasetData);
    } catch (error) {
      console.error('Failed to load sync status:', error);
      addNotification(`Error loading sync status: ${error}`);
    }
  };

  const syncAll = async () => {
    setSyncing(true);
    try {
      // Sync models
      const modelResult = await invoke<SyncResult>('sync_model_from_blockchain');
      
      // Sync adapters
      const adapterResult = await invoke<SyncResult>('sync_all_adapters');
      
      // Sync datasets
      const datasetResult = await invoke<SyncResult>('sync_all_datasets');

      setLastSyncTime(new Date().toISOString());
      
      const totalNew = modelResult.new_items + adapterResult.new_items + datasetResult.new_items;
      const totalUpdated = modelResult.updated_items + adapterResult.updated_items + datasetResult.updated_items;
      
      if (totalNew > 0 || totalUpdated > 0) {
        addNotification(`Sync complete: ${totalNew} new, ${totalUpdated} updated`);
      }

      await loadSyncStatus();
    } catch (error) {
      console.error('Sync failed:', error);
      addNotification(`Sync failed: ${error}`);
    } finally {
      setSyncing(false);
    }
  };

  const syncModel = async () => {
    setSyncing(true);
    try {
      const result = await invoke<SyncResult>('sync_model_from_blockchain');
      addNotification(result.message);
      await loadSyncStatus();
    } catch (error) {
      addNotification(`Model sync failed: ${error}`);
    } finally {
      setSyncing(false);
    }
  };

  const syncAdapters = async () => {
    setSyncing(true);
    try {
      const result = await invoke<SyncResult>('sync_all_adapters');
      addNotification(`Adapters synced: ${result.new_items} new, ${result.updated_items} updated`);
      await loadSyncStatus();
    } catch (error) {
      addNotification(`Adapter sync failed: ${error}`);
    } finally {
      setSyncing(false);
    }
  };

  const syncDatasets = async () => {
    setSyncing(true);
    try {
      const result = await invoke<SyncResult>('sync_all_datasets');
      addNotification(`Datasets synced: ${result.new_items} new`);
      await loadSyncStatus();
    } catch (error) {
      addNotification(`Dataset sync failed: ${error}`);
    } finally {
      setSyncing(false);
    }
  };

  const addNotification = (message: string) => {
    setNotifications(prev => [
      `[${new Date().toLocaleTimeString()}] ${message}`,
      ...prev.slice(0, 9) // Keep last 10 notifications
    ]);
  };

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return 'Never';
    return new Date(dateStr).toLocaleString();
  };

  return (
    <div className="blockchain-sync-panel">
      <div className="panel-header">
        <h2>Blockchain Synchronization</h2>
        <div className="header-actions">
          <label className="auto-sync-toggle">
            <input
              type="checkbox"
              checked={autoSync}
              onChange={(e) => setAutoSync(e.target.checked)}
            />
            Auto-sync (5 min)
          </label>
          <button
            onClick={syncAll}
            disabled={syncing}
            className="btn-primary"
          >
            {syncing ? 'Syncing...' : 'Sync All'}
          </button>
        </div>
      </div>

      {lastSyncTime && (
        <div className="last-sync">
          Last sync: {formatDate(lastSyncTime)}
        </div>
      )}

      {/* Model Status */}
      <section className="sync-section">
        <div className="section-header">
          <h3>Model</h3>
          <button onClick={syncModel} disabled={syncing} className="btn-secondary">
            Sync Model
          </button>
        </div>
        {modelStatus ? (
          <div className="status-card">
            <div className="status-row">
              <span className="label">Name:</span>
              <span className="value">{modelStatus.model_name}</span>
            </div>
            <div className="status-row">
              <span className="label">Blockchain Version:</span>
              <span className="value">{modelStatus.version}</span>
            </div>
            <div className="status-row">
              <span className="label">Local Version:</span>
              <span className="value">{modelStatus.local_version || 'Not installed'}</span>
            </div>
            <div className="status-row">
              <span className="label">Status:</span>
              <span className={`badge ${modelStatus.up_to_date ? 'badge-success' : 'badge-warning'}`}>
                {modelStatus.up_to_date ? 'Up to date' : 'Update available'}
              </span>
            </div>
            <div className="status-row">
              <span className="label">Size:</span>
              <span className="value">{modelStatus.size_gb.toFixed(2)} GB</span>
            </div>
            <div className="status-row">
              <span className="label">IPFS Hash:</span>
              <span className="value code">{modelStatus.ipfs_hash}</span>
            </div>
          </div>
        ) : (
          <div className="loading">Loading model status...</div>
        )}
      </section>

      {/* Adapters Status */}
      <section className="sync-section">
        <div className="section-header">
          <h3>Adapters ({adapters.length})</h3>
          <button onClick={syncAdapters} disabled={syncing} className="btn-secondary">
            Sync Adapters
          </button>
        </div>
        <div className="adapter-list">
          {adapters.length > 0 ? (
            adapters.map(adapter => (
              <div key={adapter.adapter_id} className="adapter-card">
                <div className="adapter-header">
                  <h4>{adapter.name}</h4>
                  <span className={`badge ${adapter.synced ? 'badge-success' : 'badge-warning'}`}>
                    {adapter.synced ? 'Synced' : 'Pending'}
                  </span>
                </div>
                <div className="adapter-details">
                  <div className="detail-row">
                    <span className="label">Domain:</span>
                    <span className="value">{adapter.domain}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Version:</span>
                    <span className="value">{adapter.version}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Last Synced:</span>
                    <span className="value">{formatDate(adapter.last_synced)}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">IPFS:</span>
                    <span className="value code">{adapter.ipfs_hash.substring(0, 20)}...</span>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="empty-state">No adapters synced yet</div>
          )}
        </div>
      </section>

      {/* Datasets Status */}
      <section className="sync-section">
        <div className="section-header">
          <h3>Datasets ({datasets.length})</h3>
          <button onClick={syncDatasets} disabled={syncing} className="btn-secondary">
            Sync Datasets
          </button>
        </div>
        <div className="dataset-list">
          {datasets.length > 0 ? (
            datasets.map(dataset => (
              <div key={dataset.dataset_id} className="dataset-card">
                <div className="dataset-header">
                  <h4>{dataset.name}</h4>
                  <span className={`badge ${dataset.downloaded ? 'badge-success' : 'badge-warning'}`}>
                    {dataset.downloaded ? 'Downloaded' : 'Not downloaded'}
                  </span>
                </div>
                <div className="dataset-details">
                  <div className="detail-row">
                    <span className="label">Size:</span>
                    <span className="value">{dataset.size_gb.toFixed(2)} GB</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">Last Updated:</span>
                    <span className="value">{formatDate(dataset.last_updated)}</span>
                  </div>
                  <div className="detail-row">
                    <span className="label">IPFS:</span>
                    <span className="value code">{dataset.ipfs_hash.substring(0, 20)}...</span>
                  </div>
                </div>
              </div>
            ))
          ) : (
            <div className="empty-state">No datasets available</div>
          )}
        </div>
      </section>

      {/* Notifications */}
      <section className="sync-section">
        <h3>Recent Activity</h3>
        <div className="notifications">
          {notifications.length > 0 ? (
            notifications.map((notification, index) => (
              <div key={index} className="notification">
                {notification}
              </div>
            ))
          ) : (
            <div className="empty-state">No recent activity</div>
          )}
        </div>
      </section>

      <style jsx>{`
        .blockchain-sync-panel {
          padding: 20px;
          max-width: 1200px;
          margin: 0 auto;
        }

        .panel-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .panel-header h2 {
          margin: 0;
          font-size: 24px;
          font-weight: 600;
        }

        .header-actions {
          display: flex;
          gap: 15px;
          align-items: center;
        }

        .auto-sync-toggle {
          display: flex;
          align-items: center;
          gap: 8px;
          cursor: pointer;
        }

        .last-sync {
          padding: 10px;
          background: #f5f5f5;
          border-radius: 4px;
          margin-bottom: 20px;
          font-size: 14px;
          color: #666;
        }

        .sync-section {
          margin-bottom: 30px;
          padding: 20px;
          background: white;
          border-radius: 8px;
          box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .section-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 15px;
        }

        .section-header h3 {
          margin: 0;
          font-size: 18px;
          font-weight: 600;
        }

        .status-card {
          padding: 15px;
          background: #f9f9f9;
          border-radius: 6px;
        }

        .status-row {
          display: flex;
          justify-content: space-between;
          padding: 8px 0;
          border-bottom: 1px solid #eee;
        }

        .status-row:last-child {
          border-bottom: none;
        }

        .label {
          font-weight: 500;
          color: #666;
        }

        .value {
          color: #333;
        }

        .value.code {
          font-family: monospace;
          font-size: 12px;
          background: #f0f0f0;
          padding: 2px 6px;
          border-radius: 3px;
        }

        .badge {
          padding: 4px 12px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 500;
        }

        .badge-success {
          background: #d4edda;
          color: #155724;
        }

        .badge-warning {
          background: #fff3cd;
          color: #856404;
        }

        .adapter-list, .dataset-list {
          display: grid;
          grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
          gap: 15px;
        }

        .adapter-card, .dataset-card {
          padding: 15px;
          background: #f9f9f9;
          border-radius: 6px;
          border: 1px solid #e0e0e0;
        }

        .adapter-header, .dataset-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 10px;
        }

        .adapter-header h4, .dataset-header h4 {
          margin: 0;
          font-size: 16px;
          font-weight: 600;
        }

        .adapter-details, .dataset-details {
          font-size: 14px;
        }

        .detail-row {
          display: flex;
          justify-content: space-between;
          padding: 4px 0;
        }

        .notifications {
          max-height: 300px;
          overflow-y: auto;
        }

        .notification {
          padding: 10px;
          background: #f9f9f9;
          border-left: 3px solid #007bff;
          margin-bottom: 8px;
          font-size: 14px;
          border-radius: 4px;
        }

        .loading, .empty-state {
          padding: 20px;
          text-align: center;
          color: #999;
        }

        .btn-primary, .btn-secondary {
          padding: 8px 16px;
          border: none;
          border-radius: 4px;
          cursor: pointer;
          font-size: 14px;
          font-weight: 500;
          transition: all 0.2s;
        }

        .btn-primary {
          background: #007bff;
          color: white;
        }

        .btn-primary:hover:not(:disabled) {
          background: #0056b3;
        }

        .btn-secondary {
          background: #6c757d;
          color: white;
        }

        .btn-secondary:hover:not(:disabled) {
          background: #545b62;
        }

        .btn-primary:disabled, .btn-secondary:disabled {
          opacity: 0.6;
          cursor: not-allowed;
        }
      `}</style>
    </div>
  );
};

export default BlockchainSyncPanel;
