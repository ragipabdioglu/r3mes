/**
 * R3MES JavaScript SDK Tests
 */

import {
  R3MESClient,
  MinerClient,
  BlockchainClient,
  R3MESError,
  ConnectionError,
  AuthenticationError,
  InsufficientCreditsError,
  NotFoundError,
  RateLimitError,
} from './index';

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('R3MESClient', () => {
  let client: R3MESClient;

  beforeEach(() => {
    client = new R3MESClient({
      rpcUrl: 'http://localhost:26657',
      restUrl: 'http://localhost:1317',
      backendUrl: 'http://localhost:8000',
      timeout: 5000,
    });
    mockFetch.mockClear();
  });

  describe('constructor', () => {
    it('should use default URLs when not provided', () => {
      const defaultClient = new R3MESClient();
      expect(defaultClient).toBeDefined();
    });

    it('should use custom URLs when provided', () => {
      const customClient = new R3MESClient({
        rpcUrl: 'http://custom-rpc:26657',
        restUrl: 'http://custom-rest:1317',
        backendUrl: 'http://custom-backend:8000',
      });
      expect(customClient).toBeDefined();
    });
  });

  describe('getNetworkStats', () => {
    it('should return network stats', async () => {
      const mockStats = {
        active_miners: 100,
        total_users: 1000,
        total_credits: 50000,
        block_height: 12345,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockStats,
      });

      const stats = await client.getNetworkStats();
      expect(stats).toEqual(mockStats);
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/network/stats',
        expect.any(Object)
      );
    });

    it('should throw RateLimitError on 429', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
      });

      await expect(client.getNetworkStats()).rejects.toThrow(RateLimitError);
    });
  });

  describe('getUserInfo', () => {
    it('should return user info', async () => {
      const mockUser = {
        wallet_address: 'remes1abc123',
        credits: 100.5,
        is_miner: true,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockUser,
      });

      const user = await client.getUserInfo('remes1abc123');
      expect(user).toEqual(mockUser);
    });

    it('should throw NotFoundError on 404', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      });

      await expect(client.getUserInfo('remes1notfound')).rejects.toThrow(NotFoundError);
    });
  });

  describe('getMinerStats', () => {
    it('should return miner stats', async () => {
      const mockStats = {
        wallet_address: 'remes1miner',
        hashrate: 1500.5,
        total_earnings: 250.75,
        blocks_found: 10,
        uptime_percentage: 99.5,
        is_active: true,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockStats,
      });

      const stats = await client.getMinerStats('remes1miner');
      expect(stats).toEqual(mockStats);
    });

    it('should return default stats on 404', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      });

      const stats = await client.getMinerStats('remes1notfound');
      expect(stats.is_active).toBe(false);
      expect(stats.hashrate).toBe(0);
    });
  });

  describe('getLatestBlock', () => {
    it('should return latest block info', async () => {
      const mockResponse = {
        result: {
          block: {
            header: {
              height: '12345',
              time: '2025-01-01T00:00:00Z',
              proposer_address: 'ABC123',
            },
            data: {
              txs: ['tx1', 'tx2'],
            },
          },
          block_id: {
            hash: 'BLOCKHASH123',
          },
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockResponse,
      });

      const block = await client.getLatestBlock();
      expect(block.height).toBe(12345);
      expect(block.hash).toBe('BLOCKHASH123');
      expect(block.tx_count).toBe(2);
    });
  });

  describe('getBalance', () => {
    it('should return account balances', async () => {
      const mockBalances = {
        balances: [
          { denom: 'uremes', amount: '1000000' },
          { denom: 'uatom', amount: '500000' },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockBalances,
      });

      const balances = await client.getBalance('remes1abc');
      expect(balances).toHaveLength(2);
      expect(balances[0].denom).toBe('uremes');
    });
  });

  describe('getValidators', () => {
    it('should return validators', async () => {
      const mockValidators = {
        validators: [
          {
            operator_address: 'remesvaloper1abc',
            description: { moniker: 'Validator1' },
            tokens: '1000000',
            status: 'BOND_STATUS_BONDED',
            jailed: false,
            commission: { commission_rates: { rate: '0.1' } },
          },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockValidators,
      });

      const validators = await client.getValidators();
      expect(validators).toHaveLength(1);
      expect(validators[0].moniker).toBe('Validator1');
    });
  });

  describe('getLeaderboard', () => {
    it('should return leaderboard', async () => {
      const mockLeaderboard = {
        miners: [
          { rank: 1, wallet_address: 'remes1top', total_earnings: 1000, hashrate: 5000, blocks_found: 50 },
          { rank: 2, wallet_address: 'remes1second', total_earnings: 800, hashrate: 4000, blocks_found: 40 },
        ],
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockLeaderboard,
      });

      const leaderboard = await client.getLeaderboard(10, 'week');
      expect(leaderboard).toHaveLength(2);
      expect(leaderboard[0].rank).toBe(1);
    });
  });

  describe('chat', () => {
    it('should throw InsufficientCreditsError on 402', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 402,
      });

      await expect(client.chat('Hello!')).rejects.toThrow(InsufficientCreditsError);
    });

    it('should throw RateLimitError on 429', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 429,
      });

      await expect(client.chat('Hello!')).rejects.toThrow(RateLimitError);
    });
  });
});

describe('Error Classes', () => {
  it('R3MESError should have correct properties', () => {
    const error = new R3MESError('Test error', 'TEST_CODE', { key: 'value' });
    expect(error.message).toBe('Test error');
    expect(error.code).toBe('TEST_CODE');
    expect(error.details).toEqual({ key: 'value' });
    expect(error.name).toBe('R3MESError');
  });

  it('ConnectionError should extend R3MESError', () => {
    const error = new ConnectionError('Connection failed');
    expect(error).toBeInstanceOf(R3MESError);
    expect(error.code).toBe('CONNECTION_ERROR');
  });

  it('AuthenticationError should extend R3MESError', () => {
    const error = new AuthenticationError('Auth failed');
    expect(error).toBeInstanceOf(R3MESError);
    expect(error.code).toBe('AUTHENTICATION_ERROR');
  });

  it('InsufficientCreditsError should extend R3MESError', () => {
    const error = new InsufficientCreditsError('No credits');
    expect(error).toBeInstanceOf(R3MESError);
    expect(error.code).toBe('INSUFFICIENT_CREDITS');
  });

  it('NotFoundError should extend R3MESError', () => {
    const error = new NotFoundError('Not found');
    expect(error).toBeInstanceOf(R3MESError);
    expect(error.code).toBe('NOT_FOUND');
  });

  it('RateLimitError should extend R3MESError', () => {
    const error = new RateLimitError('Rate limited');
    expect(error).toBeInstanceOf(R3MESError);
    expect(error.code).toBe('RATE_LIMIT');
  });
});

describe('BlockchainClient', () => {
  let blockchainClient: BlockchainClient;

  beforeEach(() => {
    blockchainClient = new BlockchainClient(
      'http://localhost:26657',
      'http://localhost:1317'
    );
    mockFetch.mockClear();
  });

  describe('getBlock', () => {
    it('should return block by height', async () => {
      const mockResponse = {
        result: {
          block: {
            header: {
              height: '100',
              time: '2025-01-01T00:00:00Z',
              proposer_address: 'ABC123',
            },
            data: { txs: [] },
          },
          block_id: { hash: 'HASH100' },
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockResponse,
      });

      const block = await blockchainClient.getBlock(100);
      expect(block.height).toBe(100);
    });
  });

  describe('getTransaction', () => {
    it('should throw NotFoundError on 404', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
      });

      await expect(blockchainClient.getTransaction('NOTFOUND')).rejects.toThrow(NotFoundError);
    });
  });

  describe('getStatus', () => {
    it('should return node status', async () => {
      const mockStatus = {
        result: {
          node_info: { network: 'remes-1' },
          sync_info: { latest_block_height: '12345' },
        },
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => mockStatus,
      });

      const status = await blockchainClient.getStatus();
      expect(status.node_info.network).toBe('remes-1');
    });
  });
});
