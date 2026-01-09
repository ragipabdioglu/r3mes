/**
 * Unit tests for utils/formatters.ts
 * 
 * Tests all formatting utility functions used across the dashboard.
 */

import {
  formatAddress,
  formatTimeAgo,
  formatTimestamp,
  formatHash,
  formatNumber,
  formatBytes,
  formatPercentage,
  formatTokenAmount,
  formatDuration,
  formatLatency,
} from '@/utils/formatters';

describe('formatAddress', () => {
  it('should format a valid address with default parameters', () => {
    const address = 'remes1234567890abcdef1234567890abcdef12345678';
    const result = formatAddress(address);
    expect(result).toBe('remes12345...12345678');
  });

  it('should return "N/A" for null or undefined', () => {
    expect(formatAddress(null)).toBe('N/A');
    expect(formatAddress(undefined)).toBe('N/A');
    expect(formatAddress('')).toBe('N/A');
  });

  it('should return full address if shorter than truncation length', () => {
    const shortAddress = 'remes123';
    expect(formatAddress(shortAddress)).toBe('remes123');
  });

  it('should respect custom start and end character counts', () => {
    const address = 'remes1234567890abcdef1234567890abcdef12345678';
    expect(formatAddress(address, 6, 4)).toBe('remes1...5678');
  });
});

describe('formatTimeAgo', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2026-01-02T12:00:00Z'));
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should return "Just now" for very recent timestamps', () => {
    const now = Date.now();
    expect(formatTimeAgo(now)).toBe('Just now');
    expect(formatTimeAgo(now - 500)).toBe('Just now');
  });

  it('should format seconds ago', () => {
    const timestamp = Date.now() - 30000; // 30 seconds ago
    expect(formatTimeAgo(timestamp)).toBe('30s ago');
  });

  it('should format minutes ago', () => {
    const timestamp = Date.now() - 5 * 60 * 1000; // 5 minutes ago
    expect(formatTimeAgo(timestamp)).toBe('5m ago');
  });

  it('should format hours ago', () => {
    const timestamp = Date.now() - 3 * 60 * 60 * 1000; // 3 hours ago
    expect(formatTimeAgo(timestamp)).toBe('3h ago');
  });

  it('should format days ago', () => {
    const timestamp = Date.now() - 5 * 24 * 60 * 60 * 1000; // 5 days ago
    expect(formatTimeAgo(timestamp)).toBe('5d ago');
  });

  it('should return "N/A" for null or undefined', () => {
    expect(formatTimeAgo(null)).toBe('N/A');
    expect(formatTimeAgo(undefined)).toBe('N/A');
  });

  it('should handle string timestamps', () => {
    const timestamp = new Date(Date.now() - 60000).toISOString(); // 1 minute ago
    expect(formatTimeAgo(timestamp)).toBe('1m ago');
  });

  it('should handle Date objects', () => {
    const date = new Date(Date.now() - 2 * 60 * 60 * 1000); // 2 hours ago
    expect(formatTimeAgo(date)).toBe('2h ago');
  });
});

describe('formatTimestamp', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.setSystemTime(new Date('2026-01-02T12:00:00Z'));
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('should use relative time for recent timestamps', () => {
    const timestamp = Date.now() - 2 * 60 * 60 * 1000; // 2 hours ago
    expect(formatTimestamp(timestamp)).toBe('2h ago');
  });

  it('should return "N/A" for null or undefined', () => {
    expect(formatTimestamp(null)).toBe('N/A');
    expect(formatTimestamp(undefined)).toBe('N/A');
  });
});

describe('formatHash', () => {
  it('should format a valid hash', () => {
    const hash = '0x1234567890abcdef1234567890abcdef12345678';
    expect(formatHash(hash)).toBe('0x123456...12345678');
  });

  it('should return "N/A" for null or undefined', () => {
    expect(formatHash(null)).toBe('N/A');
    expect(formatHash(undefined)).toBe('N/A');
  });

  it('should return full hash if shorter than truncation length', () => {
    expect(formatHash('0x1234')).toBe('0x1234');
  });

  it('should respect custom character count', () => {
    const hash = '0x1234567890abcdef1234567890abcdef12345678';
    expect(formatHash(hash, 4)).toBe('0x12...5678');
  });
});

describe('formatNumber', () => {
  it('should format numbers with thousand separators', () => {
    expect(formatNumber(1234567)).toBe('1,234,567');
  });

  it('should handle decimal places', () => {
    expect(formatNumber(1234.5678, 2)).toBe('1,234.57');
  });

  it('should return "0" for null or undefined', () => {
    expect(formatNumber(null)).toBe('0');
    expect(formatNumber(undefined)).toBe('0');
  });

  it('should handle zero', () => {
    expect(formatNumber(0)).toBe('0');
  });
});

describe('formatBytes', () => {
  it('should format bytes', () => {
    expect(formatBytes(500)).toBe('500 Bytes');
  });

  it('should format kilobytes', () => {
    expect(formatBytes(1024)).toBe('1 KB');
    expect(formatBytes(1536)).toBe('1.5 KB');
  });

  it('should format megabytes', () => {
    expect(formatBytes(1048576)).toBe('1 MB');
  });

  it('should format gigabytes', () => {
    expect(formatBytes(1073741824)).toBe('1 GB');
  });

  it('should return "0 Bytes" for null, undefined, or zero', () => {
    expect(formatBytes(null)).toBe('0 Bytes');
    expect(formatBytes(undefined)).toBe('0 Bytes');
    expect(formatBytes(0)).toBe('0 Bytes');
  });
});

describe('formatPercentage', () => {
  it('should format percentage values', () => {
    expect(formatPercentage(95.6)).toBe('95.6%');
  });

  it('should handle decimal form (0-1)', () => {
    expect(formatPercentage(0.956, true)).toBe('95.6%');
  });

  it('should respect decimal places', () => {
    expect(formatPercentage(95.678, false, 2)).toBe('95.68%');
  });

  it('should return "0%" for null or undefined', () => {
    expect(formatPercentage(null)).toBe('0%');
    expect(formatPercentage(undefined)).toBe('0%');
  });
});

describe('formatTokenAmount', () => {
  it('should format token amounts from smallest unit', () => {
    expect(formatTokenAmount(1500000)).toBe('1.50 REMES');
  });

  it('should handle string amounts', () => {
    expect(formatTokenAmount('2000000')).toBe('2.00 REMES');
  });

  it('should use custom denomination', () => {
    expect(formatTokenAmount(1000000, 'TOKEN')).toBe('1.00 TOKEN');
  });

  it('should return "0 REMES" for null or undefined', () => {
    expect(formatTokenAmount(null)).toBe('0 REMES');
    expect(formatTokenAmount(undefined)).toBe('0 REMES');
  });
});

describe('formatDuration', () => {
  it('should format seconds', () => {
    expect(formatDuration(5000)).toBe('5s');
  });

  it('should format minutes and seconds', () => {
    expect(formatDuration(65000)).toBe('1m 5s');
  });

  it('should format hours, minutes, and seconds', () => {
    expect(formatDuration(3661000)).toBe('1h 1m 1s');
  });

  it('should format days', () => {
    expect(formatDuration(90061000)).toBe('1d 1h 1m 1s');
  });

  it('should return "0s" for null, undefined, or zero', () => {
    expect(formatDuration(null)).toBe('0s');
    expect(formatDuration(undefined)).toBe('0s');
    expect(formatDuration(0)).toBe('0s');
  });
});

describe('formatLatency', () => {
  it('should format milliseconds', () => {
    expect(formatLatency(150)).toBe('150ms');
  });

  it('should format seconds for values >= 1000ms', () => {
    expect(formatLatency(1500)).toBe('1.5s');
  });

  it('should return "N/A" for null or undefined', () => {
    expect(formatLatency(null)).toBe('N/A');
    expect(formatLatency(undefined)).toBe('N/A');
  });
});
