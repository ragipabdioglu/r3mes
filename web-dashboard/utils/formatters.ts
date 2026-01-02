/**
 * Common formatting utilities for the R3MES Web Dashboard
 * 
 * This module provides consistent formatting functions used across
 * multiple components to avoid code duplication.
 * 
 * @module utils/formatters
 */

/**
 * Format a wallet address for display
 * Shows first and last characters with ellipsis in between
 * 
 * @param address - Full wallet address
 * @param startChars - Number of characters to show at start (default: 10)
 * @param endChars - Number of characters to show at end (default: 8)
 * @returns Formatted address string or "N/A" if empty
 * 
 * @example
 * formatAddress("remes1234567890abcdef1234567890abcdef12345678")
 * // Returns: "remes12345...ef12345678"
 */
export function formatAddress(
  address: string | null | undefined,
  startChars: number = 10,
  endChars: number = 8
): string {
  if (!address) return "N/A";
  if (address.length <= startChars + endChars + 3) return address;
  return `${address.slice(0, startChars)}...${address.slice(-endChars)}`;
}

/**
 * Format a timestamp as relative time (e.g., "2m ago", "3h ago")
 * 
 * @param timestamp - Timestamp as string, number (ms), or Date
 * @returns Formatted relative time string
 * 
 * @example
 * formatTimeAgo(Date.now() - 120000) // Returns: "2m ago"
 * formatTimeAgo("2024-01-01T12:00:00Z") // Returns: "3d ago" (depending on current time)
 */
export function formatTimeAgo(timestamp: string | number | Date | undefined | null): string {
  if (!timestamp) return "N/A";
  
  const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  
  // Handle future dates
  if (diffMs < 0) return "Just now";
  
  const diffSecs = Math.floor(diffMs / 1000);
  
  if (diffSecs < 60) return diffSecs <= 1 ? "Just now" : `${diffSecs}s ago`;
  
  const diffMins = Math.floor(diffSecs / 60);
  if (diffMins < 60) return `${diffMins}m ago`;
  
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  
  const diffDays = Math.floor(diffHours / 24);
  if (diffDays < 30) return `${diffDays}d ago`;
  
  const diffMonths = Math.floor(diffDays / 30);
  if (diffMonths < 12) return `${diffMonths}mo ago`;
  
  const diffYears = Math.floor(diffMonths / 12);
  return `${diffYears}y ago`;
}

/**
 * Format a timestamp for display in tables/lists
 * Shows relative time for recent, absolute for older
 * 
 * @param timestamp - Timestamp as string, number (ms), or Date
 * @returns Formatted timestamp string
 */
export function formatTimestamp(timestamp: string | number | Date | undefined | null): string {
  if (!timestamp) return "N/A";
  
  const date = timestamp instanceof Date ? timestamp : new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffHours = diffMs / (1000 * 60 * 60);
  
  // Use relative time for last 24 hours
  if (diffHours < 24) {
    return formatTimeAgo(timestamp);
  }
  
  // Use absolute date for older timestamps
  return date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
  });
}

/**
 * Format a hash for display (transaction hash, block hash, etc.)
 * 
 * @param hash - Full hash string
 * @param chars - Number of characters to show on each side (default: 8)
 * @returns Formatted hash string
 * 
 * @example
 * formatHash("0x1234567890abcdef1234567890abcdef12345678")
 * // Returns: "0x123456...12345678"
 */
export function formatHash(hash: string | null | undefined, chars: number = 8): string {
  if (!hash) return "N/A";
  if (hash.length <= chars * 2 + 3) return hash;
  return `${hash.slice(0, chars)}...${hash.slice(-chars)}`;
}

/**
 * Format a number with thousand separators
 * 
 * @param value - Number to format
 * @param decimals - Number of decimal places (default: 0)
 * @returns Formatted number string
 * 
 * @example
 * formatNumber(1234567.89, 2) // Returns: "1,234,567.89"
 */
export function formatNumber(value: number | null | undefined, decimals: number = 0): string {
  if (value === null || value === undefined) return "0";
  return value.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

/**
 * Format bytes to human readable size
 * 
 * @param bytes - Number of bytes
 * @param decimals - Number of decimal places (default: 2)
 * @returns Formatted size string (e.g., "1.5 MB")
 */
export function formatBytes(bytes: number | null | undefined, decimals: number = 2): string {
  if (!bytes || bytes === 0) return "0 Bytes";
  
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB", "TB", "PB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(decimals))} ${sizes[i]}`;
}

/**
 * Format a percentage value
 * 
 * @param value - Percentage value (0-100 or 0-1)
 * @param isDecimal - Whether value is in decimal form (0-1)
 * @param decimals - Number of decimal places (default: 1)
 * @returns Formatted percentage string
 * 
 * @example
 * formatPercentage(0.956, true) // Returns: "95.6%"
 * formatPercentage(95.6, false) // Returns: "95.6%"
 */
export function formatPercentage(
  value: number | null | undefined,
  isDecimal: boolean = false,
  decimals: number = 1
): string {
  if (value === null || value === undefined) return "0%";
  const percentage = isDecimal ? value * 100 : value;
  return `${percentage.toFixed(decimals)}%`;
}

/**
 * Format token amount with proper denomination
 * 
 * @param amount - Amount in smallest unit (e.g., uremes)
 * @param denom - Token denomination (default: "REMES")
 * @param decimals - Number of decimal places (default: 2)
 * @returns Formatted token amount string
 * 
 * @example
 * formatTokenAmount(1500000, "REMES") // Returns: "1.50 REMES"
 */
export function formatTokenAmount(
  amount: number | string | null | undefined,
  denom: string = "REMES",
  decimals: number = 2
): string {
  if (amount === null || amount === undefined) return `0 ${denom}`;
  
  const numAmount = typeof amount === 'string' ? parseFloat(amount) : amount;
  
  // Convert from smallest unit (uremes) to main unit (REMES)
  // 1 REMES = 1,000,000 uremes
  const mainUnitAmount = numAmount / 1_000_000;
  
  return `${mainUnitAmount.toFixed(decimals)} ${denom}`;
}

/**
 * Format duration in milliseconds to human readable string
 * 
 * @param ms - Duration in milliseconds
 * @returns Formatted duration string
 * 
 * @example
 * formatDuration(3661000) // Returns: "1h 1m 1s"
 */
export function formatDuration(ms: number | null | undefined): string {
  if (!ms || ms <= 0) return "0s";
  
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);
  
  const parts: string[] = [];
  
  if (days > 0) parts.push(`${days}d`);
  if (hours % 24 > 0) parts.push(`${hours % 24}h`);
  if (minutes % 60 > 0) parts.push(`${minutes % 60}m`);
  if (seconds % 60 > 0 || parts.length === 0) parts.push(`${seconds % 60}s`);
  
  return parts.join(' ');
}

/**
 * Format latency in milliseconds
 * 
 * @param ms - Latency in milliseconds
 * @returns Formatted latency string with appropriate unit
 * 
 * @example
 * formatLatency(1500) // Returns: "1.5s"
 * formatLatency(150) // Returns: "150ms"
 */
export function formatLatency(ms: number | null | undefined): string {
  if (ms === null || ms === undefined) return "N/A";
  
  if (ms >= 1000) {
    return `${(ms / 1000).toFixed(1)}s`;
  }
  
  return `${Math.round(ms)}ms`;
}
