import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { AlertTriangle, CheckCircle, XCircle, ExternalLink, Shield } from 'lucide-react';

interface PortCheck {
  port: number;
  description: string;
  isOpen: boolean;
}

interface FirewallCheckResult {
  allOk: boolean;
  portChecks: PortCheck[];
  warnings: string[];
}

interface FirewallWarningProps {
  onContinue?: () => void;
  onDismiss?: () => void;
  requiredPorts?: Array<{ port: number; description: string }>;
}

export default function FirewallWarning({
  onContinue,
  onDismiss,
  requiredPorts = [
    { port: 26656, description: 'Blockchain P2P (Tendermint)' },
    { port: 4001, description: 'IPFS P2P' },
  ],
}: FirewallWarningProps) {
  const [checkResult, setCheckResult] = useState<FirewallCheckResult | null>(null);
  const [isChecking, setIsChecking] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkFirewallPorts();
  }, []);

  const checkFirewallPorts = async () => {
    setIsChecking(true);
    setError(null);
    
    try {
      // Check ports via Tauri command
      const result = await invoke<FirewallCheckResult>('check_firewall_ports', {
        ports: requiredPorts,
      });
      
      setCheckResult(result);
    } catch (err) {
      console.error('Firewall check failed:', err);
      setError(err instanceof Error ? err.message : 'Failed to check firewall ports');
      
      // Create fallback result with warnings
      setCheckResult({
        allOk: false,
        portChecks: requiredPorts.map((p) => ({ ...p, isOpen: false })),
        warnings: ['Could not check firewall ports. Please ensure ports are open manually.'],
      });
    } finally {
      setIsChecking(false);
    }
  };

  const handleContinue = () => {
    if (onContinue) {
      onContinue();
    }
  };

  const handleDismiss = () => {
    if (onDismiss) {
      onDismiss();
    }
  };

  const handleLearnMore = () => {
    // Open documentation in browser
    invoke('open_url', { url: 'https://docs.r3mes.network/installation/firewall' }).catch(
      console.error
    );
  };

  if (isChecking) {
    return (
      <div className="flex flex-col items-center justify-center p-8 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-600 dark:border-yellow-400 mb-4"></div>
        <p className="text-yellow-800 dark:text-yellow-200">Checking firewall ports...</p>
      </div>
    );
  }

  if (error && !checkResult) {
    return (
      <div className="p-6 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
        <div className="flex items-center gap-3 mb-4">
          <XCircle className="w-6 h-6 text-red-600 dark:text-red-400" />
          <h3 className="text-lg font-semibold text-red-800 dark:text-red-200">
            Firewall Check Error
          </h3>
        </div>
        <p className="text-red-700 dark:text-red-300 mb-4">{error}</p>
        <button
          onClick={checkFirewallPorts}
          className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md transition-colors"
        >
          Retry Check
        </button>
      </div>
    );
  }

  if (!checkResult) {
    return null;
  }

  const { allOk, portChecks, warnings } = checkResult;

  return (
    <div className="p-6 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <Shield className="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
        <h3 className="text-lg font-semibold text-yellow-800 dark:text-yellow-200">
          Firewall Warning
        </h3>
      </div>

      {/* Status Summary */}
      {allOk ? (
        <div className="flex items-center gap-2 mb-4 text-green-700 dark:text-green-300">
          <CheckCircle className="w-5 h-5" />
          <p className="font-medium">All required ports are accessible.</p>
        </div>
      ) : (
        <div className="flex items-center gap-2 mb-4 text-yellow-700 dark:text-yellow-300">
          <AlertTriangle className="w-5 h-5" />
          <p className="font-medium">Some ports may be blocked by your firewall.</p>
        </div>
      )}

      {/* Port Status List */}
      <div className="mb-4 space-y-2">
        {portChecks.map((portCheck) => (
          <div
            key={portCheck.port}
            className="flex items-center justify-between p-3 bg-white dark:bg-gray-800 rounded-md border border-gray-200 dark:border-gray-700"
          >
            <div className="flex items-center gap-3">
              {portCheck.isOpen ? (
                <CheckCircle className="w-5 h-5 text-green-500" />
              ) : (
                <XCircle className="w-5 h-5 text-red-500" />
              )}
              <div>
                <p className="font-medium text-gray-900 dark:text-gray-100">
                  Port {portCheck.port}
                </p>
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  {portCheck.description}
                </p>
              </div>
            </div>
            <span
              className={`px-3 py-1 rounded-full text-xs font-medium ${
                portCheck.isOpen
                  ? 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-300'
                  : 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300'
              }`}
            >
              {portCheck.isOpen ? 'Open' : 'Blocked'}
            </span>
          </div>
        ))}
      </div>

      {/* Warnings */}
      {warnings.length > 0 && (
        <div className="mb-4 p-4 bg-yellow-100 dark:bg-yellow-900/30 rounded-md border border-yellow-300 dark:border-yellow-700">
          <h4 className="font-semibold text-yellow-800 dark:text-yellow-200 mb-2">
            Important Notes:
          </h4>
          <ul className="list-disc list-inside space-y-1 text-sm text-yellow-700 dark:text-yellow-300">
            {warnings.map((warning, index) => (
              <li key={index}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Windows-specific instructions */}
      {!allOk && (
        <div className="mb-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-md border border-blue-200 dark:border-blue-800">
          <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">
            Windows Firewall Instructions:
          </h4>
          <ol className="list-decimal list-inside space-y-1 text-sm text-blue-700 dark:text-blue-300">
            <li>When the miner starts, Windows Firewall may prompt for access.</li>
            <li>Click &quot;Allow Access&quot; to enable P2P connectivity.</li>
            <li>
              If no prompt appears, you may need to manually add firewall rules (see documentation).
            </li>
          </ol>
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-3 pt-4 border-t border-yellow-200 dark:border-yellow-700">
        <button
          onClick={handleContinue}
          className="flex-1 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-md transition-colors font-medium"
        >
          Continue Anyway
        </button>
        {onDismiss && (
          <button
            onClick={handleDismiss}
            className="px-4 py-2 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 rounded-md transition-colors"
          >
            Dismiss
          </button>
        )}
        <button
          onClick={handleLearnMore}
          className="flex items-center gap-2 px-4 py-2 bg-transparent hover:bg-yellow-100 dark:hover:bg-yellow-900/30 text-yellow-700 dark:text-yellow-300 rounded-md transition-colors border border-yellow-300 dark:border-yellow-700"
        >
          <ExternalLink className="w-4 h-4" />
          Learn More
        </button>
      </div>
    </div>
  );
}

