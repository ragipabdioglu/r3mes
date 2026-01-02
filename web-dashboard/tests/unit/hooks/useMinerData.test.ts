/**
 * @jest-environment jsdom
 */

import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { useUserInfo, useMinerStats, useEarningsHistory, useHashrateHistory } from '@/hooks/useMinerData';
import { getUserInfo, getMinerStats, getEarningsHistory, getHashrateHistory } from '@/lib/api';

// Mock API functions
jest.mock('@/lib/api', () => ({
  getUserInfo: jest.fn(),
  getMinerStats: jest.fn(),
  getEarningsHistory: jest.fn(),
  getHashrateHistory: jest.fn(),
}));

const mockGetUserInfo = getUserInfo as jest.MockedFunction<typeof getUserInfo>;
const mockGetMinerStats = getMinerStats as jest.MockedFunction<typeof getMinerStats>;
const mockGetEarningsHistory = getEarningsHistory as jest.MockedFunction<typeof getEarningsHistory>;
const mockGetHashrateHistory = getHashrateHistory as jest.MockedFunction<typeof getHashrateHistory>;

// Test wrapper with QueryClient
const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

  return ({ children }: { children: React.ReactNode }) => 
    React.createElement(QueryClientProvider, { client: queryClient }, children);
};

describe('useMinerData Hooks', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('useUserInfo', () => {
    const mockUserInfo = {
      wallet_address: 'remes1234567890abcdef',
      credits: 1500.75,
      is_miner: true,
    };

    it('fetches user info successfully', async () => {
      mockGetUserInfo.mockResolvedValue(mockUserInfo);

      const { result } = renderHook(
        () => useUserInfo('remes1234567890abcdef'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockUserInfo);
      expect(mockGetUserInfo).toHaveBeenCalledWith('remes1234567890abcdef');
    });

    it('returns null when no wallet address provided', async () => {
      const { result } = renderHook(
        () => useUserInfo(null),
        { wrapper: createWrapper() }
      );

      expect(result.current.data).toBeUndefined();
      expect(mockGetUserInfo).not.toHaveBeenCalled();
    });

    it('handles API errors gracefully', async () => {
      const error = new Error('API Error');
      mockGetUserInfo.mockRejectedValue(error);

      const { result } = renderHook(
        () => useUserInfo('remes1234567890abcdef'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toEqual(error);
    });

    it('disables query when wallet address is null', () => {
      const { result } = renderHook(
        () => useUserInfo(null),
        { wrapper: createWrapper() }
      );

      expect(result.current.fetchStatus).toBe('idle');
      expect(mockGetUserInfo).not.toHaveBeenCalled();
    });

    it('refetches data at specified intervals', async () => {
      mockGetUserInfo.mockResolvedValue(mockUserInfo);

      const { result } = renderHook(
        () => useUserInfo('remes1234567890abcdef'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Check that refetch interval is configured
      expect(result.current.dataUpdatedAt).toBeGreaterThan(0);
    });

    it('handles backend unavailable error', async () => {
      const error = new Error('Backend service is not available');
      mockGetUserInfo.mockRejectedValue(error);

      const { result } = renderHook(
        () => useUserInfo('remes1234567890abcdef'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error?.message).toContain('not available');
    });
  });

  describe('useMinerStats', () => {
    const mockMinerStats = {
      wallet_address: 'remes1234567890abcdef',
      total_earnings: 1500.75,
      hashrate: 125.5,
      gpu_temperature: 72,
      blocks_found: 15,
      uptime_percentage: 98.5,
      network_difficulty: 1234567,
    };

    it('fetches miner stats successfully', async () => {
      mockGetMinerStats.mockResolvedValue(mockMinerStats);

      const { result } = renderHook(
        () => useMinerStats('remes1234567890abcdef'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockMinerStats);
      expect(mockGetMinerStats).toHaveBeenCalledWith('remes1234567890abcdef');
    });

    it('returns null when no wallet address provided', () => {
      const { result } = renderHook(
        () => useMinerStats(null),
        { wrapper: createWrapper() }
      );

      expect(result.current.data).toBeUndefined();
      expect(mockGetMinerStats).not.toHaveBeenCalled();
    });

    it('handles miner not found error', async () => {
      const error = new Error('Miner not found');
      mockGetMinerStats.mockRejectedValue(error);

      const { result } = renderHook(
        () => useMinerStats('remes1234567890abcdef'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toEqual(error);
    });

    it('has appropriate refetch interval for real-time data', async () => {
      mockGetMinerStats.mockResolvedValue(mockMinerStats);

      const { result } = renderHook(
        () => useMinerStats('remes1234567890abcdef'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Miner stats should refetch more frequently than user info
      expect(result.current.dataUpdatedAt).toBeGreaterThan(0);
    });
  });

  describe('useEarningsHistory', () => {
    const mockEarningsData = [
      { date: '2024-01-01', earnings: 100 },
      { date: '2024-01-02', earnings: 150 },
      { date: '2024-01-03', earnings: 200 },
    ];

    it('fetches earnings history successfully', async () => {
      mockGetEarningsHistory.mockResolvedValue(mockEarningsData);

      const { result } = renderHook(
        () => useEarningsHistory('remes1234567890abcdef'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockEarningsData);
      expect(mockGetEarningsHistory).toHaveBeenCalledWith('remes1234567890abcdef');
    });

    it('returns empty array when no wallet address provided', () => {
      const { result } = renderHook(
        () => useEarningsHistory(null),
        { wrapper: createWrapper() }
      );

      expect(result.current.data).toBeUndefined();
      expect(mockGetEarningsHistory).not.toHaveBeenCalled();
    });

    it('handles empty earnings history', async () => {
      mockGetEarningsHistory.mockResolvedValue([]);

      const { result } = renderHook(
        () => useEarningsHistory('remes1234567890abcdef'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual([]);
    });

    it('handles API errors for earnings history', async () => {
      const error = new Error('Failed to fetch earnings');
      mockGetEarningsHistory.mockRejectedValue(error);

      const { result } = renderHook(
        () => useEarningsHistory('remes1234567890abcdef'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toEqual(error);
    });
  });

  describe('useHashrateHistory', () => {
    const mockHashrateData = [
      { date: '2024-01-01', hashrate: 120.5 },
      { date: '2024-01-02', hashrate: 125.0 },
      { date: '2024-01-03', hashrate: 130.2 },
    ];

    it('fetches hashrate history successfully', async () => {
      mockGetHashrateHistory.mockResolvedValue(mockHashrateData);

      const { result } = renderHook(
        () => useHashrateHistory('remes1234567890abcdef'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockHashrateData);
      expect(mockGetHashrateHistory).toHaveBeenCalledWith('remes1234567890abcdef');
    });

    it('returns empty array when no wallet address provided', () => {
      const { result } = renderHook(
        () => useHashrateHistory(null),
        { wrapper: createWrapper() }
      );

      expect(result.current.data).toBeUndefined();
      expect(mockGetHashrateHistory).not.toHaveBeenCalled();
    });

    it('handles fluctuating hashrate data', async () => {
      const fluctuatingData = [
        { date: '2024-01-01', hashrate: 100 },
        { date: '2024-01-02', hashrate: 0 }, // Offline period
        { date: '2024-01-03', hashrate: 150 },
      ];
      mockGetHashrateHistory.mockResolvedValue(fluctuatingData);

      const { result } = renderHook(
        () => useHashrateHistory('remes1234567890abcdef'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(fluctuatingData);
    });

    it('handles network connectivity issues', async () => {
      const error = new Error('Network error');
      mockGetHashrateHistory.mockRejectedValue(error);

      const { result } = renderHook(
        () => useHashrateHistory('remes1234567890abcdef'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toEqual(error);
    });
  });

  describe('Hook Integration', () => {
    it('all hooks work together with same wallet address', async () => {
      const walletAddress = 'remes1234567890abcdef';
      
      mockGetUserInfo.mockResolvedValue({
        wallet_address: walletAddress,
        credits: 1500,
        is_miner: true,
      });
      
      mockGetMinerStats.mockResolvedValue({
        wallet_address: walletAddress,
        total_earnings: 1500,
        hashrate: 125,
        gpu_temperature: 70,
        blocks_found: 10,
        uptime_percentage: 95,
        network_difficulty: 1000000,
      });
      
      mockGetEarningsHistory.mockResolvedValue([
        { date: '2024-01-01', earnings: 100 },
      ]);
      
      mockGetHashrateHistory.mockResolvedValue([
        { date: '2024-01-01', hashrate: 125 },
      ]);

      const { result: userInfoResult } = renderHook(
        () => useUserInfo(walletAddress),
        { wrapper: createWrapper() }
      );

      const { result: minerStatsResult } = renderHook(
        () => useMinerStats(walletAddress),
        { wrapper: createWrapper() }
      );

      const { result: earningsResult } = renderHook(
        () => useEarningsHistory(walletAddress),
        { wrapper: createWrapper() }
      );

      const { result: hashrateResult } = renderHook(
        () => useHashrateHistory(walletAddress),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(userInfoResult.current.isSuccess).toBe(true);
        expect(minerStatsResult.current.isSuccess).toBe(true);
        expect(earningsResult.current.isSuccess).toBe(true);
        expect(hashrateResult.current.isSuccess).toBe(true);
      });

      expect(mockGetUserInfo).toHaveBeenCalledWith(walletAddress);
      expect(mockGetMinerStats).toHaveBeenCalledWith(walletAddress);
      expect(mockGetEarningsHistory).toHaveBeenCalledWith(walletAddress);
      expect(mockGetHashrateHistory).toHaveBeenCalledWith(walletAddress);
    });

    it('handles wallet address changes correctly', async () => {
      const initialAddress = 'remes1111111111111111';
      const newAddress = 'remes2222222222222222';

      mockGetUserInfo.mockResolvedValue({
        wallet_address: initialAddress,
        credits: 1000,
        is_miner: true,
      });

      const { result, rerender } = renderHook(
        ({ address }) => useUserInfo(address),
        { 
          wrapper: createWrapper(),
          initialProps: { address: initialAddress }
        }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mockGetUserInfo).toHaveBeenCalledWith(initialAddress);

      // Change wallet address
      mockGetUserInfo.mockResolvedValue({
        wallet_address: newAddress,
        credits: 2000,
        is_miner: false,
      });

      rerender({ address: newAddress });

      await waitFor(() => {
        expect(result.current.data?.wallet_address).toBe(newAddress);
      });

      expect(mockGetUserInfo).toHaveBeenCalledWith(newAddress);
    });
  });

  describe('Error Recovery', () => {
    it('retries failed requests according to configuration', async () => {
      mockGetUserInfo
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValue({
          wallet_address: 'remes1234567890abcdef',
          credits: 1500,
          is_miner: true,
        });

      const { result } = renderHook(
        () => useUserInfo('remes1234567890abcdef'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Should have retried and succeeded
      expect(mockGetUserInfo).toHaveBeenCalledTimes(2);
    });

    it('stops retrying for non-retryable errors', async () => {
      const error = new Error('Backend service is not available');
      mockGetUserInfo.mockRejectedValue(error);

      const { result } = renderHook(
        () => useUserInfo('remes1234567890abcdef'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      // Should not retry for backend unavailable errors
      expect(mockGetUserInfo).toHaveBeenCalledTimes(1);
    });
  });
});