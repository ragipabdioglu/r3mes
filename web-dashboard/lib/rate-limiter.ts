/**
 * Rate Limiter with Optional Redis Support
 * 
 * Provides distributed rate limiting for multi-instance deployments.
 * Uses in-memory storage by default, can use Upstash Redis REST API in production.
 * 
 * @module lib/rate-limiter
 */

import { logger } from './logger';

// Rate limiting configuration
export const RATE_LIMIT_CONFIG = {
  windowMs: 15 * 60 * 1000, // 15 minutes
  maxRequests: 100, // requests per window for general routes
  apiMaxRequests: 50, // stricter limit for API routes
  authMaxRequests: 10, // very strict for auth/faucet routes
} as const;

// Rate limit record interface
interface RateLimitRecord {
  count: number;
  resetTime: number;
}

// Rate limiter interface for different backends
interface RateLimiterBackend {
  get(key: string): Promise<RateLimitRecord | null>;
  set(key: string, record: RateLimitRecord, ttlMs: number): Promise<void>;
  increment(key: string): Promise<number>;
}

/**
 * In-memory rate limiter backend
 * Used as fallback when Redis is not available
 */
class InMemoryRateLimiter implements RateLimiterBackend {
  private store = new Map<string, RateLimitRecord>();
  private cleanupInterval: ReturnType<typeof setInterval> | null = null;

  constructor() {
    // Clean up expired entries every 5 minutes
    if (typeof setInterval !== 'undefined') {
      this.cleanupInterval = setInterval(() => this.cleanup(), 5 * 60 * 1000);
    }
  }

  async get(key: string): Promise<RateLimitRecord | null> {
    const record = this.store.get(key);
    if (!record) return null;
    
    // Check if expired
    if (Date.now() > record.resetTime) {
      this.store.delete(key);
      return null;
    }
    
    return record;
  }

  async set(key: string, record: RateLimitRecord): Promise<void> {
    this.store.set(key, record);
  }

  async increment(key: string): Promise<number> {
    const record = this.store.get(key);
    if (record && Date.now() <= record.resetTime) {
      record.count++;
      return record.count;
    }
    return 1;
  }

  private cleanup(): void {
    const now = Date.now();
    for (const [key, record] of this.store.entries()) {
      if (now > record.resetTime) {
        this.store.delete(key);
      }
    }
  }

  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
    this.store.clear();
  }
}

/**
 * Upstash Redis REST API rate limiter backend
 * Uses Upstash REST API for distributed rate limiting (no native Redis client needed)
 */
class UpstashRateLimiter implements RateLimiterBackend {
  private restUrl: string;
  private restToken: string;

  constructor(restUrl: string, restToken: string) {
    this.restUrl = restUrl;
    this.restToken = restToken;
  }

  private async request(command: string[]): Promise<unknown> {
    try {
      const response = await fetch(this.restUrl, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.restToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(command),
      });
      
      if (!response.ok) {
        throw new Error(`Upstash request failed: ${response.status}`);
      }
      
      const data = await response.json();
      return data.result;
    } catch (error) {
      logger.error('Upstash request error:', error);
      throw error;
    }
  }

  async get(key: string): Promise<RateLimitRecord | null> {
    try {
      const data = await this.request(['GET', `ratelimit:${key}`]) as string | null;
      if (!data) return null;
      return JSON.parse(data) as RateLimitRecord;
    } catch (error) {
      logger.error('Upstash get error:', error);
      return null;
    }
  }

  async set(key: string, record: RateLimitRecord, ttlMs: number): Promise<void> {
    try {
      await this.request([
        'SET', 
        `ratelimit:${key}`, 
        JSON.stringify(record), 
        'PX', 
        ttlMs.toString()
      ]);
    } catch (error) {
      logger.error('Upstash set error:', error);
    }
  }

  async increment(key: string): Promise<number> {
    try {
      const result = await this.request(['INCR', `ratelimit:${key}:count`]) as number;
      return result;
    } catch (error) {
      logger.error('Upstash increment error:', error);
      return 1;
    }
  }
}

/**
 * Rate Limiter Factory
 * Creates appropriate rate limiter based on environment configuration
 */
class RateLimiterFactory {
  private static instance: RateLimiterBackend | null = null;
  private static hasWarnedAboutInMemory = false;

  static getInstance(): RateLimiterBackend {
    if (this.instance) {
      return this.instance;
    }

    const upstashUrl = process.env.UPSTASH_REDIS_REST_URL;
    const upstashToken = process.env.UPSTASH_REDIS_REST_TOKEN;

    if (upstashUrl && upstashToken) {
      logger.info('Using Upstash Redis for rate limiting');
      this.instance = new UpstashRateLimiter(upstashUrl, upstashToken);
    } else {
      if (process.env.NODE_ENV === 'production' && !this.hasWarnedAboutInMemory) {
        logger.warn(
          '[SECURITY WARNING] Using in-memory rate limiting in production. ' +
          'Set UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN for distributed rate limiting.'
        );
        this.hasWarnedAboutInMemory = true;
      }
      this.instance = new InMemoryRateLimiter();
    }

    return this.instance;
  }
}

/**
 * Check rate limit for a given key
 */
export async function checkRateLimit(
  key: string,
  maxRequests: number = RATE_LIMIT_CONFIG.maxRequests
): Promise<{
  isAllowed: boolean;
  remaining: number;
  resetTime: number;
  retryAfter: number;
}> {
  const limiter = RateLimiterFactory.getInstance();
  const now = Date.now();
  
  let record = await limiter.get(key);
  
  if (!record || now > record.resetTime) {
    record = {
      count: 1,
      resetTime: now + RATE_LIMIT_CONFIG.windowMs,
    };
    await limiter.set(key, record, RATE_LIMIT_CONFIG.windowMs);
    
    return {
      isAllowed: true,
      remaining: maxRequests - 1,
      resetTime: record.resetTime,
      retryAfter: 0,
    };
  }
  
  if (record.count >= maxRequests) {
    const retryAfter = Math.ceil((record.resetTime - now) / 1000);
    return {
      isAllowed: false,
      remaining: 0,
      resetTime: record.resetTime,
      retryAfter,
    };
  }
  
  record.count++;
  await limiter.set(key, record, record.resetTime - now);
  
  return {
    isAllowed: true,
    remaining: maxRequests - record.count,
    resetTime: record.resetTime,
    retryAfter: 0,
  };
}

/**
 * Get rate limit key based on request path
 */
export function getRateLimitKey(ip: string, pathname: string): string {
  if (pathname.startsWith('/api/auth') || pathname.startsWith('/api/faucet')) {
    return `auth:${ip}`;
  }
  if (pathname.startsWith('/api/')) {
    return `api:${ip}`;
  }
  return `general:${ip}`;
}

/**
 * Get max requests based on rate limit key type
 */
export function getMaxRequestsForKey(key: string): number {
  if (key.startsWith('auth:')) {
    return RATE_LIMIT_CONFIG.authMaxRequests;
  }
  if (key.startsWith('api:')) {
    return RATE_LIMIT_CONFIG.apiMaxRequests;
  }
  return RATE_LIMIT_CONFIG.maxRequests;
}

export { RateLimiterFactory, InMemoryRateLimiter, UpstashRateLimiter };
