/**
 * R3MES Desktop Launcher - Configuration Hook
 * 
 * React hook for managing application configuration with Tauri backend.
 */

import { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import {
  FullConfig,
  MinerConfig,
  NetworkConfig,
  AdvancedConfig,
  NetworkPreset,
  getDefaultFullConfig,
  getNetworkPreset,
  NETWORK_PRESETS,
} from '../types/config';

interface UseConfigReturn {
  config: FullConfig;
  loading: boolean;
  error: string | null;
  saved: boolean;
  
  // Update functions
  updateMinerConfig: (updates: Partial<MinerConfig>) => void;
  updateNetworkConfig: (updates: Partial<NetworkConfig>) => void;
  updateAdvancedConfig: (updates: Partial<AdvancedConfig>) => void;
  
  // Actions
  saveConfig: () => Promise<void>;
  resetToDefaults: () => Promise<void>;
  applyNetworkPreset: (preset: NetworkPreset) => void;
  reloadConfig: () => Promise<void>;
}

export function useConfig(): UseConfigReturn {
  const [config, setConfig] = useState<FullConfig>(getDefaultFullConfig());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  // Load config from backend
  const loadConfig = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      
      const loadedConfig = await invoke<FullConfig>('get_full_config');
      setConfig(loadedConfig);
    } catch (err) {
      console.error('Failed to load config:', err);
      setError(err instanceof Error ? err.message : 'Failed to load configuration');
      // Use defaults on error
      setConfig(getDefaultFullConfig());
    } finally {
      setLoading(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    loadConfig();
  }, [loadConfig]);

  // Update miner config
  const updateMinerConfig = useCallback((updates: Partial<MinerConfig>) => {
    setConfig(prev => ({
      ...prev,
      miner: { ...prev.miner, ...updates },
    }));
    setSaved(false);
  }, []);

  // Update network config
  const updateNetworkConfig = useCallback((updates: Partial<NetworkConfig>) => {
    setConfig(prev => ({
      ...prev,
      network: { ...prev.network, ...updates },
    }));
    setSaved(false);
  }, []);

  // Update advanced config
  const updateAdvancedConfig = useCallback((updates: Partial<AdvancedConfig>) => {
    setConfig(prev => ({
      ...prev,
      advanced: { ...prev.advanced, ...updates },
    }));
    setSaved(false);
  }, []);

  // Apply network preset
  const applyNetworkPreset = useCallback((preset: NetworkPreset) => {
    const presetConfig = getNetworkPreset(preset);
    setConfig(prev => ({
      ...prev,
      network: presetConfig,
      // Also update miner backend URLs based on preset
      miner: {
        ...prev.miner,
        backend_url: preset === 'local' 
          ? 'http://localhost:8000'
          : preset === 'testnet'
            ? 'https://testnet-api.r3mes.network'
            : 'https://api.r3mes.network',
        blockchain_rpc: presetConfig.rpc_endpoint,
      },
    }));
    setSaved(false);
  }, []);

  // Save config to backend
  const saveConfig = useCallback(async () => {
    try {
      setError(null);
      await invoke('save_config', { config });
      setSaved(true);
      
      // Clear saved indicator after 3 seconds
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      console.error('Failed to save config:', err);
      setError(err instanceof Error ? err.message : 'Failed to save configuration');
      throw err;
    }
  }, [config]);

  // Reset to defaults
  const resetToDefaults = useCallback(async () => {
    try {
      setError(null);
      await invoke('reset_config_to_defaults');
      await loadConfig();
      setSaved(true);
      setTimeout(() => setSaved(false), 3000);
    } catch (err) {
      console.error('Failed to reset config:', err);
      setError(err instanceof Error ? err.message : 'Failed to reset configuration');
      throw err;
    }
  }, [loadConfig]);

  return {
    config,
    loading,
    error,
    saved,
    updateMinerConfig,
    updateNetworkConfig,
    updateAdvancedConfig,
    saveConfig,
    resetToDefaults,
    applyNetworkPreset,
    reloadConfig: loadConfig,
  };
}

// Utility hook for detecting network preset
export function useNetworkPresetDetection(networkConfig: NetworkConfig): NetworkPreset | null {
  const [detectedPreset, setDetectedPreset] = useState<NetworkPreset | null>(null);

  useEffect(() => {
    // Check if current config matches any preset
    for (const [presetName, preset] of Object.entries(NETWORK_PRESETS)) {
      if (
        networkConfig.chain_id === preset.chain_id &&
        networkConfig.rpc_endpoint === preset.rpc_endpoint &&
        networkConfig.rest_endpoint === preset.rest_endpoint
      ) {
        setDetectedPreset(presetName as NetworkPreset);
        return;
      }
    }
    setDetectedPreset(null);
  }, [networkConfig]);

  return detectedPreset;
}

export default useConfig;
