/**
 * R3MES Inference Hook
 * 
 * React hook for AI inference using BitNet + DoRA + RAG pipeline.
 * Supports both standard and streaming inference modes.
 * 
 * @module hooks/useInference
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import {
  generateInference,
  generateInferenceStream,
  getInferenceHealth,
  getInferenceMetrics,
  warmupInferencePipeline,
  type InferenceRequest,
  type InferenceResponse,
  type InferenceHealth,
  type InferenceMetrics,
} from '@/lib/api';
import { logger } from '@/lib/logger';

/** Inference state */
export interface InferenceState {
  /** Whether inference is in progress */
  isLoading: boolean;
  /** Whether streaming is in progress */
  isStreaming: boolean;
  /** Current response text (for streaming) */
  streamedText: string;
  /** Final inference response */
  response: InferenceResponse | null;
  /** Error message if any */
  error: string | null;
  /** Inference health status */
  health: InferenceHealth | null;
  /** Inference metrics */
  metrics: InferenceMetrics | null;
}

/** Inference options */
export interface UseInferenceOptions {
  /** Auto-fetch health on mount */
  autoFetchHealth?: boolean;
  /** Health polling interval in ms (0 to disable) */
  healthPollingInterval?: number;
  /** Default wallet address */
  defaultWalletAddress?: string;
  /** Default inference parameters */
  defaultParams?: Partial<InferenceRequest>;
}

/** Inference hook return type */
export interface UseInferenceReturn extends InferenceState {
  /** Generate inference (non-streaming) */
  generate: (prompt: string, options?: Partial<InferenceRequest>) => Promise<InferenceResponse | null>;
  /** Generate streaming inference */
  generateStream: (prompt: string, options?: Partial<InferenceRequest>) => Promise<void>;
  /** Stop streaming */
  stopStream: () => void;
  /** Refresh health status */
  refreshHealth: () => Promise<void>;
  /** Refresh metrics */
  refreshMetrics: () => Promise<void>;
  /** Warmup pipeline */
  warmup: () => Promise<boolean>;
  /** Clear error */
  clearError: () => void;
  /** Reset state */
  reset: () => void;
}

/**
 * Hook for AI inference with BitNet + DoRA + RAG pipeline
 * 
 * @example
 * ```tsx
 * const { generate, generateStream, isLoading, response, streamedText } = useInference();
 * 
 * // Non-streaming
 * const result = await generate("What is R3MES?");
 * 
 * // Streaming
 * await generateStream("Explain blockchain", { max_tokens: 1024 });
 * ```
 */
export function useInference(options: UseInferenceOptions = {}): UseInferenceReturn {
  const {
    autoFetchHealth = false,
    healthPollingInterval = 0,
    defaultWalletAddress,
    defaultParams = {},
  } = options;

  const [state, setState] = useState<InferenceState>({
    isLoading: false,
    isStreaming: false,
    streamedText: '',
    response: null,
    error: null,
    health: null,
    metrics: null,
  });

  const abortControllerRef = useRef<AbortController | null>(null);
  const streamingRef = useRef(false);

  // Fetch health status
  const refreshHealth = useCallback(async () => {
    try {
      const health = await getInferenceHealth();
      setState(prev => ({ ...prev, health }));
    } catch (error) {
      logger.error('Failed to fetch inference health:', error);
    }
  }, []);

  // Fetch metrics
  const refreshMetrics = useCallback(async () => {
    try {
      const metrics = await getInferenceMetrics();
      setState(prev => ({ ...prev, metrics }));
    } catch (error) {
      logger.error('Failed to fetch inference metrics:', error);
    }
  }, []);

  // Auto-fetch health on mount
  useEffect(() => {
    if (autoFetchHealth) {
      refreshHealth();
    }
  }, [autoFetchHealth, refreshHealth]);

  // Health polling
  useEffect(() => {
    if (healthPollingInterval <= 0) return;

    const interval = setInterval(refreshHealth, healthPollingInterval);
    return () => clearInterval(interval);
  }, [healthPollingInterval, refreshHealth]);

  // Generate inference (non-streaming)
  const generate = useCallback(async (
    prompt: string,
    options: Partial<InferenceRequest> = {}
  ): Promise<InferenceResponse | null> => {
    setState(prev => ({
      ...prev,
      isLoading: true,
      error: null,
      response: null,
      streamedText: '',
    }));

    try {
      const request: InferenceRequest = {
        prompt,
        wallet_address: options.wallet_address ?? defaultWalletAddress,
        ...defaultParams,
        ...options,
      };

      const response = await generateInference(request);
      
      setState(prev => ({
        ...prev,
        isLoading: false,
        response,
      }));

      return response;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Inference failed';
      logger.error('Inference error:', error);
      
      setState(prev => ({
        ...prev,
        isLoading: false,
        error: errorMessage,
      }));

      return null;
    }
  }, [defaultWalletAddress, defaultParams]);

  // Generate streaming inference
  const generateStream = useCallback(async (
    prompt: string,
    options: Partial<InferenceRequest> = {}
  ): Promise<void> => {
    // Cancel any existing stream
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();
    streamingRef.current = true;

    setState(prev => ({
      ...prev,
      isLoading: true,
      isStreaming: true,
      error: null,
      response: null,
      streamedText: '',
    }));

    try {
      const request: InferenceRequest = {
        prompt,
        wallet_address: options.wallet_address ?? defaultWalletAddress,
        ...defaultParams,
        ...options,
        stream: true,
      };

      await generateInferenceStream(
        request,
        // onToken
        (token) => {
          if (!streamingRef.current) return;
          setState(prev => ({
            ...prev,
            streamedText: prev.streamedText + token,
          }));
        },
        // onDone
        () => {
          streamingRef.current = false;
          setState(prev => ({
            ...prev,
            isLoading: false,
            isStreaming: false,
          }));
        },
        // onError
        (error) => {
          streamingRef.current = false;
          setState(prev => ({
            ...prev,
            isLoading: false,
            isStreaming: false,
            error: error.message,
          }));
        }
      );
    } catch (error) {
      streamingRef.current = false;
      const errorMessage = error instanceof Error ? error.message : 'Streaming failed';
      
      setState(prev => ({
        ...prev,
        isLoading: false,
        isStreaming: false,
        error: errorMessage,
      }));
    }
  }, [defaultWalletAddress, defaultParams]);

  // Stop streaming
  const stopStream = useCallback(() => {
    streamingRef.current = false;
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setState(prev => ({
      ...prev,
      isLoading: false,
      isStreaming: false,
    }));
  }, []);

  // Warmup pipeline
  const warmup = useCallback(async (): Promise<boolean> => {
    try {
      const result = await warmupInferencePipeline();
      return result.status === 'success';
    } catch (error) {
      logger.error('Pipeline warmup failed:', error);
      return false;
    }
  }, []);

  // Clear error
  const clearError = useCallback(() => {
    setState(prev => ({ ...prev, error: null }));
  }, []);

  // Reset state
  const reset = useCallback(() => {
    stopStream();
    setState({
      isLoading: false,
      isStreaming: false,
      streamedText: '',
      response: null,
      error: null,
      health: null,
      metrics: null,
    });
  }, [stopStream]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      streamingRef.current = false;
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, []);

  return {
    ...state,
    generate,
    generateStream,
    stopStream,
    refreshHealth,
    refreshMetrics,
    warmup,
    clearError,
    reset,
  };
}

export default useInference;
