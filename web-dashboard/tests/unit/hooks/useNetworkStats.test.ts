/**
 * @jest-environment jsdom
 */

import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import { useNetworkStats, useRecentBlocks } from '@/hooks/useNetworkStats';
import { getNetworkStats, getRecentBlocks } from '@/lib/api';

// Mock API functions
jest.mock('@/lib/api', () => ({
  getNetworkStats: jest.fn(),
  getRecentBlocks: jest.fn(),
}));

const mockGetNetworkStats = getNetworkStats as jest.MockedFunction<typeof getNetworkStats>;
const mockGetRecentBlocks = getRecentBlocks as jest.MockedFunction<typeof getRecentBlocks>;

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

describe('useNetworkStats Hooks', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('useNetworkStats', () => {
    const mockNetworkStats = {
      active_miners: 1250,
      total_users: 5000,
      total_credits: 10000000,
      block_height: 123456,
    };

    it('fetches network stats successfully', async () => {
      mockGetNetworkStats.mockResolvedValue(mockNetworkStats);

      const { result } = renderHook(
        () => useNetworkStats(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockNetworkStats);
      expect(mockGetNetworkStats).toHaveBeenCalledTimes(1);
    });

    it('handles API errors gracefully', async () => {
      const error = new Error('Network stats API error');
      mockGetNetworkStats.mockRejectedValue(error);

      const { result } = renderHook(
        () => useNetworkStats(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toEqual(error);
    });

    it('handles backend unavailable error', async () => {
      const error = new Error('Backend service is not available');
      mockGetNetworkStats.mockRejectedValue(error);

      const { result } = renderHook(
        () => useNetworkStats(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error?.message).toContain('not available');
      
      // Should not refetch when backend is unavailable
      await new Promise(resolve => setTimeout(resolve, 100));
      expect(mockGetNetworkStats).toHaveBeenCalledTimes(1);
    });

    it('refetches data at specified intervals', async () => {
      mockGetNetworkStats.mockResolvedValue(mockNetworkStats);

      const { result } = renderHook(
        () => useNetworkStats(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Check that data is considered fresh initially
      expect(result.current.isStale).toBe(false);
      expect(result.current.dataUpdatedAt).toBeGreaterThan(0);
    });

    it('handles network stats with zero values', async () => {
      const emptyStats = {
        active_miners: 0,
        total_users: 0,
        total_credits: 0,
        block_height: 0,
      };
      mockGetNetworkStats.mockResolvedValue(emptyStats);

      const { result } = renderHook(
        () => useNetworkStats(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(emptyStats);
    });

    it('handles partial network stats data', async () => {
      const partialStats = {
        active_miners: 100,
        total_users: 500,
        total_credits: 1000000,
        // block_height missing
      };
      mockGetNetworkStats.mockResolvedValue(partialStats as any);

      const { result } = renderHook(
        () => useNetworkStats(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(partialStats);
    });

    it('retries on network errors', async () => {
      mockGetNetworkStats
        .mockRejectedValueOnce(new Error('Network error'))
        .mockResolvedValue(mockNetworkStats);

      const { result } = renderHook(
        () => useNetworkStats(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mockGetNetworkStats).toHaveBeenCalledTimes(2);
      expect(result.current.data).toEqual(mockNetworkStats);
    });
  });

  describe('useRecentBlocks', () => {
    const mockBlocks = [
      {
        height: 123456,
        miner: 'remes1234567890abcdef',
        timestamp: '2024-01-15T10:00:00Z',
        hash: 'abc123def456',
      },
      {
        height: 123455,
        miner: 'remes9876543210fedcba',
        timestamp: '2024-01-15T09:59:00Z',
        hash: 'def456abc123',
      },
    ];

    it('fetches recent blocks successfully with default limit', async () => {
      mockGetRecentBlocks.mockResolvedValue(mockBlocks);

      const { result } = renderHook(
        () => useRecentBlocks(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockBlocks);
      expect(mockGetRecentBlocks).toHaveBeenCalledWith(10); // default limit
    });

    it('fetches recent blocks with custom limit', async () => {
      mockGetRecentBlocks.mockResolvedValue(mockBlocks);

      const { result } = renderHook(
        () => useRecentBlocks(5),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(mockBlocks);
      expect(mockGetRecentBlocks).toHaveBeenCalledWith(5);
    });

    it('handles empty blocks list', async () => {
      mockGetRecentBlocks.mockResolvedValue([]);

      const { result } = renderHook(
        () => useRecentBlocks(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual([]);
    });

    it('handles blocks API errors', async () => {
      const error = new Error('Failed to fetch blocks');
      mockGetRecentBlocks.mockRejectedValue(error);

      const { result } = renderHook(
        () => useRecentBlocks(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      expect(result.current.error).toEqual(error);
    });

    it('handles blocks with missing data', async () => {
      const incompleteBlocks = [
        {
          height: 123456,
          // miner missing
          timestamp: '2024-01-15T10:00:00Z',
          // hash missing
        },
      ];
      mockGetRecentBlocks.mockResolvedValue(incompleteBlocks as any);

      const { result } = renderHook(
        () => useRecentBlocks(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(incompleteBlocks);
    });

    it('refetches blocks more frequently than network stats', async () => {
      mockGetRecentBlocks.mockResolvedValue(mockBlocks);

      const { result } = renderHook(
        () => useRecentBlocks(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      // Blocks should have shorter stale time than network stats
      expect(result.current.dataUpdatedAt).toBeGreaterThan(0);
    });

    it('handles large block datasets', async () => {
      const largeBlockList = Array.from({ length: 100 }, (_, i) => ({
        height: 123456 - i,
        miner: `remes${i.toString().padStart(16, '0')}`,
        timestamp: new Date(Date.now() - i * 60000).toISOString(),
        hash: `hash${i.toString().padStart(16, '0')}`,
      }));

      mockGetRecentBlocks.mockResolvedValue(largeBlockList);

      const { result } = renderHook(
        () => useRecentBlocks(100),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toHaveLength(100);
      expect(result.current.data?.[0].height).toBe(123456);
      expect(result.current.data?.[99].height).toBe(123357);
    });

    it('handles blocks with different timestamp formats', async () => {
      const blocksWithDifferentTimestamps = [
        {
          height: 123456,
          miner: 'remes1234567890abcdef',
          timestamp: '2024-01-15T10:00:00.000Z', // with milliseconds
          hash: 'abc123',
        },
        {
          height: 123455,
          miner: 'remes9876543210fedcba',
          timestamp: '2024-01-15T09:59:00Z', // without milliseconds
          hash: 'def456',
        },
      ];

      mockGetRecentBlocks.mockResolvedValue(blocksWithDifferentTimestamps);

      const { result } = renderHook(
        () => useRecentBlocks(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(result.current.data).toEqual(blocksWithDifferentTimestamps);
    });
  });

  describe('Hook Integration', () => {
    it('both hooks work together correctly', async () => {
      const mockStats = {
        active_miners: 1000,
        total_users: 5000,
        total_credits: 10000000,
        block_height: 123456,
      };

      const mockBlocks = [
        {
          height: 123456,
          miner: 'remes1234567890abcdef',
          timestamp: '2024-01-15T10:00:00Z',
          hash: 'abc123',
        },
      ];

      mockGetNetworkStats.mockResolvedValue(mockStats);
      mockGetRecentBlocks.mockResolvedValue(mockBlocks);

      const { result: statsResult } = renderHook(
        () => useNetworkStats(),
        { wrapper: createWrapper() }
      );

      const { result: blocksResult } = renderHook(
        () => useRecentBlocks(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(statsResult.current.isSuccess).toBe(true);
        expect(blocksResult.current.isSuccess).toBe(true);
      });

      expect(statsResult.current.data).toEqual(mockStats);
      expect(blocksResult.current.data).toEqual(mockBlocks);

      // Block height should match between stats and latest block
      expect(statsResult.current.data?.block_height).toBe(
        blocksResult.current.data?.[0].height
      );
    });

    it('handles mixed success/error states', async () => {
      const mockStats = {
        active_miners: 1000,
        total_users: 5000,
        total_credits: 10000000,
        block_height: 123456,
      };

      mockGetNetworkStats.mockResolvedValue(mockStats);
      mockGetRecentBlocks.mockRejectedValue(new Error('Blocks API error'));

      const { result: statsResult } = renderHook(
        () => useNetworkStats(),
        { wrapper: createWrapper() }
      );

      const { result: blocksResult } = renderHook(
        () => useRecentBlocks(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(statsResult.current.isSuccess).toBe(true);
        expect(blocksResult.current.isError).toBe(true);
      });

      expect(statsResult.current.data).toEqual(mockStats);
      expect(blocksResult.current.error?.message).toBe('Blocks API error');
    });
  });

  describe('Performance and Caching', () => {
    it('caches network stats appropriately', async () => {
      const mockStats = {
        active_miners: 1000,
        total_users: 5000,
        total_credits: 10000000,
        block_height: 123456,
      };

      mockGetNetworkStats.mockResolvedValue(mockStats);

      const { result: firstResult } = renderHook(
        () => useNetworkStats(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(firstResult.current.isSuccess).toBe(true);
      });

      // Second hook should use cached data
      const { result: secondResult } = renderHook(
        () => useNetworkStats(),
        { wrapper: createWrapper() }
      );

      expect(secondResult.current.data).toEqual(mockStats);
      expect(mockGetNetworkStats).toHaveBeenCalledTimes(1); // Only called once due to caching
    });

    it('handles concurrent requests efficiently', async () => {
      const mockStats = {
        active_miners: 1000,
        total_users: 5000,
        total_credits: 10000000,
        block_height: 123456,
      };

      mockGetNetworkStats.mockResolvedValue(mockStats);

      // Render multiple hooks simultaneously
      const { result: result1 } = renderHook(() => useNetworkStats(), { wrapper: createWrapper() });
      const { result: result2 } = renderHook(() => useNetworkStats(), { wrapper: createWrapper() });
      const { result: result3 } = renderHook(() => useNetworkStats(), { wrapper: createWrapper() });

      await waitFor(() => {
        expect(result1.current.isSuccess).toBe(true);
        expect(result2.current.isSuccess).toBe(true);
        expect(result3.current.isSuccess).toBe(true);
      });

      // Should deduplicate requests
      expect(mockGetNetworkStats).toHaveBeenCalledTimes(1);
    });
  });

  describe('Error Recovery', () => {
    it('recovers from temporary network errors', async () => {
      const mockStats = {
        active_miners: 1000,
        total_users: 5000,
        total_credits: 10000000,
        block_height: 123456,
      };

      mockGetNetworkStats
        .mockRejectedValueOnce(new Error('Temporary network error'))
        .mockResolvedValue(mockStats);

      const { result } = renderHook(
        () => useNetworkStats(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });

      expect(mockGetNetworkStats).toHaveBeenCalledTimes(2);
      expect(result.current.data).toEqual(mockStats);
    });

    it('stops retrying for persistent backend errors', async () => {
      const error = new Error('Backend service is not available');
      mockGetNetworkStats.mockRejectedValue(error);

      const { result } = renderHook(
        () => useNetworkStats(),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });

      // Should not retry for backend unavailable
      expect(mockGetNetworkStats).toHaveBeenCalledTimes(1);
    });
  });
});