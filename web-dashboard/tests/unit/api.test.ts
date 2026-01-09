/**
 * Unit tests for API client
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';

// Mock fetch
const mockFetch = jest.fn();
global.fetch = mockFetch;

// Mock logger
jest.mock('@/lib/logger', () => ({
  logger: {
    error: jest.fn(),
    debug: jest.fn(),
    info: jest.fn(),
    warn: jest.fn(),
  },
}));

describe('API Client', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    process.env.NEXT_PUBLIC_BACKEND_URL = 'http://localhost:8000';
  });

  afterEach(() => {
    delete process.env.NEXT_PUBLIC_BACKEND_URL;
  });

  it('should use environment variable for API base URL', () => {
    const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
    expect(API_BASE_URL).toBe('http://localhost:8000');
  });

  it('should throw error in production if API URL not set', () => {
    const originalEnv = process.env.NODE_ENV;
    process.env.NODE_ENV = 'production';
    delete process.env.NEXT_PUBLIC_BACKEND_URL;

    expect(() => {
      if (!process.env.NEXT_PUBLIC_BACKEND_URL && process.env.NODE_ENV === 'production') {
        throw new Error('NEXT_PUBLIC_BACKEND_URL must be set in production');
      }
    }).toThrow('NEXT_PUBLIC_BACKEND_URL must be set in production');

    process.env.NODE_ENV = originalEnv;
  });

  it('should use localhost fallback in development', () => {
    const originalEnv = process.env.NODE_ENV;
    process.env.NODE_ENV = 'development';
    delete process.env.NEXT_PUBLIC_BACKEND_URL;

    const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
    expect(API_BASE_URL).toBe('http://localhost:8000');

    process.env.NODE_ENV = originalEnv;
  });
});

describe('API Functions', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockFetch.mockReset();
  });

  describe('getUserInfo', () => {
    it('should fetch user info successfully', async () => {
      const mockUserInfo = {
        wallet_address: 'remes123...',
        credits: 100,
        is_miner: true,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockUserInfo),
      });

      const { getUserInfo } = await import('@/lib/api');
      const result = await getUserInfo('remes123...');

      expect(result).toEqual(mockUserInfo);
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/user/remes123...'),
        expect.any(Object)
      );
    });

    it('should throw error on failed request', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ detail: 'User not found' }),
      });

      const { getUserInfo } = await import('@/lib/api');
      
      await expect(getUserInfo('invalid')).rejects.toThrow();
    });
  });

  describe('getNetworkStats', () => {
    it('should fetch network stats successfully', async () => {
      const mockStats = {
        active_miners: 100,
        total_users: 1000,
        total_credits: 50000,
        block_height: 12345,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockStats),
      });

      const { getNetworkStats } = await import('@/lib/api');
      const result = await getNetworkStats();

      expect(result).toEqual(mockStats);
    });
  });

  describe('getFaucetStatus', () => {
    it('should fetch faucet status successfully', async () => {
      const mockStatus = {
        enabled: true,
        amount_per_claim: '1000000uremes',
        daily_limit: '5000000uremes',
        rate_limit: '1 request per day',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockStatus),
      });

      const { getFaucetStatus } = await import('@/lib/api');
      const result = await getFaucetStatus();

      expect(result.enabled).toBe(true);
      expect(result.amount_per_claim).toBe('1000000uremes');
    });
  });

  describe('claimFaucet', () => {
    it('should claim faucet successfully', async () => {
      const mockResponse = {
        success: true,
        tx_hash: '0xabc123...',
        amount: '1000000uremes',
        next_claim_available_at: '2026-01-03T12:00:00Z',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });

      const { claimFaucet } = await import('@/lib/api');
      const result = await claimFaucet({ address: 'remes123...' });

      expect(result.success).toBe(true);
      expect(result.tx_hash).toBe('0xabc123...');
    });

    it('should handle rate limit error', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
        json: () => Promise.resolve({ detail: 'Rate limit exceeded' }),
      });

      const { claimFaucet } = await import('@/lib/api');
      const result = await claimFaucet({ address: 'remes123...' });

      expect(result.success).toBe(false);
      expect(result.message).toContain('Rate limit');
    });
  });

  describe('getLeaderboard', () => {
    it('should fetch miners leaderboard', async () => {
      const mockMiners = {
        miners: [
          { address: 'remes1...', tier: 'diamond', total_submissions: 1000, reputation: 98 },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockMiners),
      });

      const { getLeaderboard } = await import('@/lib/api');
      const result = await getLeaderboard('miners');

      expect(result.miners).toBeDefined();
      expect(result.miners?.length).toBeGreaterThan(0);
    });

    it('should fetch validators leaderboard', async () => {
      const mockValidators = {
        validators: [
          { address: 'remes1...', tier: 'platinum', trust_score: 99, uptime: 99.9, voting_power: 50000 },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockValidators),
      });

      const { getLeaderboard } = await import('@/lib/api');
      const result = await getLeaderboard('validators');

      expect(result.validators).toBeDefined();
      expect(result.validators?.length).toBeGreaterThan(0);
    });
  });

  describe('getRoleStatistics', () => {
    it('should fetch role statistics', async () => {
      const mockStats = {
        stats: [
          { role_id: 1, role_name: 'Miner', total_nodes: 100, active_nodes: 80 },
          { role_id: 2, role_name: 'Serving', total_nodes: 50, active_nodes: 40 },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockStats),
      });

      const { getRoleStatistics } = await import('@/lib/api');
      const result = await getRoleStatistics();

      expect(result.stats).toBeDefined();
      expect(result.stats.length).toBe(2);
    });
  });

  describe('getProposerNodes', () => {
    it('should fetch proposer nodes with pagination', async () => {
      const mockNodes = {
        nodes: [
          { node_address: 'remes1...', status: 'active', total_aggregations: 50, total_rewards: '100' },
        ],
        total: 1,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockNodes),
      });

      const { getProposerNodes } = await import('@/lib/api');
      const result = await getProposerNodes(10, 0);

      expect(result.nodes).toBeDefined();
      expect(result.total).toBe(1);
    });
  });

  describe('getServingNodes', () => {
    it('should fetch serving nodes', async () => {
      const mockNodes = {
        nodes: [
          {
            node_address: 'remes1...',
            status: 'active',
            is_available: true,
            total_requests: 1000,
            successful_requests: 990,
            average_latency_ms: 150,
          },
        ],
        total: 1,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockNodes),
      });

      const { getServingNodes } = await import('@/lib/api');
      const result = await getServingNodes();

      expect(result.nodes).toBeDefined();
      expect(result.nodes[0].is_available).toBe(true);
    });
  });

  describe('getTransactionHistory', () => {
    it('should fetch transaction history', async () => {
      const mockHistory = {
        transactions: [
          { id: 1, type: 'send', amount: 100, timestamp: Date.now(), status: 'confirmed' },
        ],
        total: 1,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockHistory),
      });

      const { getTransactionHistory } = await import('@/lib/api');
      const result = await getTransactionHistory('remes123...', 50);

      expect(result.transactions).toBeDefined();
      expect(result.total).toBe(1);
    });

    it('should return empty array on error', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      const { getTransactionHistory } = await import('@/lib/api');
      const result = await getTransactionHistory('remes123...', 50);

      expect(result.transactions).toEqual([]);
      expect(result.total).toBe(0);
    });
  });
});
