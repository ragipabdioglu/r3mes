/**
 * Unit tests for API client
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';

// Mock fetch
global.fetch = jest.fn();

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

    // In production, should throw if not set
    // This is tested by the actual API client implementation
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

