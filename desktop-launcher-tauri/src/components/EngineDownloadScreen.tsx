import { useState, useEffect } from 'react';
import { invoke } from '@tauri-apps/api/tauri';
import { Download, CheckCircle, XCircle, AlertCircle, Loader2 } from 'lucide-react';

interface EngineStatus {
  installed: boolean;
  version: string | null;
  path: string | null;
  checksum_valid: boolean;
}

interface EngineDownloadScreenProps {
  onComplete?: () => void;
  onCancel?: () => void;
}

export default function EngineDownloadScreen({
  onComplete,
  onCancel,
}: EngineDownloadScreenProps) {
  const [status, setStatus] = useState<EngineStatus | null>(null);
  const [isChecking, setIsChecking] = useState(true);
  const [isDownloading, setIsDownloading] = useState(false);
  const [downloadProgress, setDownloadProgress] = useState({
    percentage: 0,
    downloadedBytes: 0,
    totalBytes: 0,
    speed: 0,
    eta: 0,
  });
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    checkEngineStatus();
  }, []);

  const checkEngineStatus = async () => {
    setIsChecking(true);
    setError(null);

    try {
      const result = await invoke<EngineStatus>('ensure_engine_ready');
      setStatus(result);

      if (result.installed && result.checksum_valid) {
        // Engine already installed and valid
        if (onComplete) {
          onComplete();
        }
      }
    } catch (err) {
      console.error('Failed to check engine status:', err);
      setError(err instanceof Error ? err.message : 'Failed to check engine status');
    } finally {
      setIsChecking(false);
    }
  };

  const handleDownload = async () => {
    setIsDownloading(true);
    setError(null);
    setDownloadProgress({
      percentage: 0,
      downloadedBytes: 0,
      totalBytes: 0,
      speed: 0,
      eta: 0,
    });

    try {
      // Note: In a real implementation, we would use Tauri events to track progress
      // For now, we call the download function which logs progress to console
      await invoke<string>('download_engine');
      
      // Download complete, verify status
      await checkEngineStatus();
      
      if (onComplete) {
        onComplete();
      }
    } catch (err) {
      console.error('Engine download failed:', err);
      setError(err instanceof Error ? err.message : 'Failed to download engine');
      setIsDownloading(false);
    }
  };

  const formatBytes = (bytes: number): string => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
  };

  const formatSpeed = (bytesPerSec: number): string => {
    return `${formatBytes(bytesPerSec)}/s`;
  };

  const formatETA = (seconds: number): string => {
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds}s`;
  };

  if (isChecking) {
    return (
      <div className="flex flex-col items-center justify-center p-8 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600 dark:text-blue-400 mb-4" />
        <p className="text-blue-800 dark:text-blue-200">Checking engine status...</p>
      </div>
    );
  }

  if (status?.installed && status.checksum_valid) {
    return (
      <div className="p-6 bg-green-50 dark:bg-green-900/20 rounded-lg border border-green-200 dark:border-green-800">
        <div className="flex items-center gap-3 mb-4">
          <CheckCircle className="w-6 h-6 text-green-600 dark:text-green-400" />
          <h3 className="text-lg font-semibold text-green-800 dark:text-green-200">
            Engine Ready
          </h3>
        </div>
        <p className="text-green-700 dark:text-green-300 mb-4">
          Engine is installed and ready to use.
        </p>
        {status.version && (
          <p className="text-sm text-green-600 dark:text-green-400">
            Version: {status.version}
          </p>
        )}
        {onComplete && (
          <button
            onClick={onComplete}
            className="mt-4 px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-md transition-colors"
          >
            Continue
          </button>
        )}
      </div>
    );
  }

  if (error && !isDownloading) {
    return (
      <div className="p-6 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
        <div className="flex items-center gap-3 mb-4">
          <XCircle className="w-6 h-6 text-red-600 dark:text-red-400" />
          <h3 className="text-lg font-semibold text-red-800 dark:text-red-200">
            Error
          </h3>
        </div>
        <p className="text-red-700 dark:text-red-300 mb-4">{error}</p>
        <div className="flex gap-3">
          <button
            onClick={handleDownload}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-md transition-colors"
          >
            Retry Download
          </button>
          {onCancel && (
            <button
              onClick={onCancel}
              className="px-4 py-2 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 rounded-md transition-colors"
            >
              Cancel
            </button>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="p-6 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg border border-yellow-200 dark:border-yellow-800">
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <Download className="w-6 h-6 text-yellow-600 dark:text-yellow-400" />
        <h3 className="text-lg font-semibold text-yellow-800 dark:text-yellow-200">
          Engine Download Required
        </h3>
      </div>

      {/* Status Message */}
      {!isDownloading && (
        <div className="mb-4">
          <p className="text-yellow-700 dark:text-yellow-300 mb-2">
            The mining engine is not installed. Please download it to continue.
          </p>
          {status && !status.checksum_valid && (
            <div className="flex items-center gap-2 text-sm text-yellow-600 dark:text-yellow-400">
              <AlertCircle className="w-4 h-4" />
              <span>Existing engine file is invalid or corrupted.</span>
            </div>
          )}
        </div>
      )}

      {/* Download Progress */}
      {isDownloading && (
        <div className="mb-4 space-y-3">
          <div className="flex items-center justify-between text-sm text-yellow-700 dark:text-yellow-300">
            <span>Downloading...</span>
            <span>{downloadProgress.percentage.toFixed(1)}%</span>
          </div>

          {/* Progress Bar */}
          <div className="w-full bg-yellow-200 dark:bg-yellow-800 rounded-full h-2">
            <div
              className="bg-yellow-600 dark:bg-yellow-400 h-2 rounded-full transition-all duration-300"
              style={{ width: `${downloadProgress.percentage}%` }}
            />
          </div>

          {/* Download Stats */}
          <div className="grid grid-cols-3 gap-4 text-xs text-yellow-600 dark:text-yellow-400">
            <div>
              <div className="font-medium">Downloaded</div>
              <div>{formatBytes(downloadProgress.downloadedBytes)}</div>
            </div>
            {downloadProgress.totalBytes > 0 && (
              <>
                <div>
                  <div className="font-medium">Total</div>
                  <div>{formatBytes(downloadProgress.totalBytes)}</div>
                </div>
                <div>
                  <div className="font-medium">Speed</div>
                  <div>{formatSpeed(downloadProgress.speed)}</div>
                </div>
              </>
            )}
          </div>

          {downloadProgress.eta > 0 && (
            <div className="text-xs text-yellow-600 dark:text-yellow-400">
              Estimated time remaining: {formatETA(downloadProgress.eta)}
            </div>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="flex items-center gap-3 pt-4 border-t border-yellow-200 dark:border-yellow-700">
        {!isDownloading ? (
          <>
            <button
              onClick={handleDownload}
              className="flex-1 px-4 py-2 bg-yellow-600 hover:bg-yellow-700 text-white rounded-md transition-colors font-medium flex items-center justify-center gap-2"
            >
              <Download className="w-4 h-4" />
              Download Engine
            </button>
            {onCancel && (
              <button
                onClick={onCancel}
                className="px-4 py-2 bg-gray-200 hover:bg-gray-300 dark:bg-gray-700 dark:hover:bg-gray-600 text-gray-800 dark:text-gray-200 rounded-md transition-colors"
              >
                Cancel
              </button>
            )}
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center gap-2 text-yellow-700 dark:text-yellow-300">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Downloading engine... Please wait.</span>
          </div>
        )}
      </div>

      {/* Info */}
      <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-md border border-blue-200 dark:border-blue-800">
        <p className="text-xs text-blue-700 dark:text-blue-300">
          <strong>Note:</strong> The engine download is approximately 2.5-3GB. Ensure you have a
          stable internet connection and sufficient disk space.
        </p>
      </div>
    </div>
  );
}

