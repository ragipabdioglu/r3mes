/**
 * Unit tests for lib/rate-limiter.ts
 * 
 * Tests rate limiting functionality with in-memory backend.
 */

import {
  checkRateLimit,
  getRateLimitKey,
  getMaxRequestsForKey,
  RATE_LIMIT_CONFIG,
  InMemoryRateLimiter,
} from '@/lib/rate-limiter';

describe('Rate Limiter', () => {
  describe('getRateLimitKey', () => {
    it('should return auth key for auth routes', () => {
      expect(getRateLimitKey('192.168.1.1', '/api/auth/login')).toBe('auth:192.168.1.1');
    });

    it('should return auth key for faucet routes', () => {
      expect(getRateLimitKey('192.168.1.1', '/api/faucet/claim')).toBe('auth:192.168.1.1');
    });

    it('should return api key for other API routes', () => {
      expect(getRateLimitKey('192.168.1.1', '/api/user/info')).toBe('api:192.168.1.1');
    });

    it('should return general key for non-API routes', () => {
      expect(getRateLimitKey('192.168.1.1', '/dashboard')).toBe('general:192.168.1.1');
    });
  });

  describe('getMaxRequestsForKey', () => {
    it('should return auth limit for auth keys', () => {
      expect(getMaxRequestsForKey('auth:192.168.1.1')).toBe(RATE_LIMIT_CONFIG.authMaxRequests);
    });

    it('should return api limit for api keys', () => {
      expect(getMaxRequestsForKey('api:192.168.1.1')).toBe(RATE_LIMIT_CONFIG.apiMaxRequests);
    });

    it('should return general limit for general keys', () => {
      expect(getMaxRequestsForKey('general:192.168.1.1')).toBe(RATE_LIMIT_CONFIG.maxRequests);
    });
  });

  describe('InMemoryRateLimiter', () => {
    let limiter: InMemoryRateLimiter;

    beforeEach(() => {
      limiter = new InMemoryRateLimiter();
    });

    afterEach(() => {
      limiter.destroy();
    });

    it('should return null for non-existent key', async () => {
      const result = await limiter.get('test-key');
      expect(result).toBeNull();
    });

    it('should store and retrieve records', async () => {
      const record = { count: 5, resetTime: Date.now() + 60000 };
      await limiter.set('test-key', record, 60000);
      
      const result = await limiter.get('test-key');
      expect(result).toEqual(record);
    });

    it('should return null for expired records', async () => {
      const record = { count: 5, resetTime: Date.now() - 1000 }; // Already expired
      await limiter.set('test-key', record, 0);
      
      const result = await limiter.get('test-key');
      expect(result).toBeNull();
    });

    it('should increment count', async () => {
      const record = { count: 5, resetTime: Date.now() + 60000 };
      await limiter.set('test-key', record, 60000);
      
      const newCount = await limiter.increment('test-key');
      expect(newCount).toBe(6);
    });

    it('should return 1 for increment on non-existent key', async () => {
      const count = await limiter.increment('new-key');
      expect(count).toBe(1);
    });
  });

  describe('checkRateLimit', () => {
    beforeEach(() => {
      // Reset any cached state
      jest.clearAllMocks();
    });

    it('should allow first request', async () => {
      const result = await checkRateLimit('test-ip-1', 10);
      
      expect(result.isAllowed).toBe(true);
      expect(result.remaining).toBe(9);
      expect(result.retryAfter).toBe(0);
    });

    it('should track request count', async () => {
      // Make multiple requests
      await checkRateLimit('test-ip-2', 10);
      await checkRateLimit('test-ip-2', 10);
      const result = await checkRateLimit('test-ip-2', 10);
      
      expect(result.isAllowed).toBe(true);
      expect(result.remaining).toBe(7);
    });

    it('should block when limit exceeded', async () => {
      const maxRequests = 3;
      
      // Exhaust the limit
      for (let i = 0; i < maxRequests; i++) {
        await checkRateLimit('test-ip-3', maxRequests);
      }
      
      // Next request should be blocked
      const result = await checkRateLimit('test-ip-3', maxRequests);
      
      expect(result.isAllowed).toBe(false);
      expect(result.remaining).toBe(0);
      expect(result.retryAfter).toBeGreaterThan(0);
    });

    it('should use different limits for different keys', async () => {
      // These should be tracked separately
      const result1 = await checkRateLimit('ip-a', 5);
      const result2 = await checkRateLimit('ip-b', 5);
      
      expect(result1.remaining).toBe(4);
      expect(result2.remaining).toBe(4);
    });
  });
});

describe('RATE_LIMIT_CONFIG', () => {
  it('should have valid configuration values', () => {
    expect(RATE_LIMIT_CONFIG.windowMs).toBeGreaterThan(0);
    expect(RATE_LIMIT_CONFIG.maxRequests).toBeGreaterThan(0);
    expect(RATE_LIMIT_CONFIG.apiMaxRequests).toBeGreaterThan(0);
    expect(RATE_LIMIT_CONFIG.authMaxRequests).toBeGreaterThan(0);
  });

  it('should have stricter limits for auth routes', () => {
    expect(RATE_LIMIT_CONFIG.authMaxRequests).toBeLessThan(RATE_LIMIT_CONFIG.apiMaxRequests);
    expect(RATE_LIMIT_CONFIG.apiMaxRequests).toBeLessThan(RATE_LIMIT_CONFIG.maxRequests);
  });
});
