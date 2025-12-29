/**
 * Number formatting utilities with dynamic precision
 * 
 * Handles very small numbers (0.0000001) and large numbers appropriately
 */

/**
 * Format a number with dynamic precision based on its value
 * 
 * @param value - The number to format
 * @param minPrecision - Minimum decimal places (default: 2)
 * @param maxPrecision - Maximum decimal places (default: 8)
 * @returns Formatted number string
 */
export function formatNumber(
  value: number,
  minPrecision: number = 2,
  maxPrecision: number = 8
): string {
  if (value === 0) {
    return "0.00";
  }

  // For very small numbers, use more precision
  if (Math.abs(value) < 0.01) {
    // Find the first non-zero digit
    const absValue = Math.abs(value);
    let precision = minPrecision;
    
    // Calculate precision needed to show first significant digit
    if (absValue < 1) {
      const magnitude = Math.floor(Math.log10(absValue));
      precision = Math.min(Math.abs(magnitude) + 2, maxPrecision);
    }
    
    return value.toFixed(precision);
  }

  // For normal numbers, use standard precision
  return value.toFixed(minPrecision);
}

/**
 * Format currency (REMES) with appropriate precision
 * 
 * @param amount - Amount in REMES
 * @returns Formatted currency string
 */
export function formatCurrency(amount: number): string {
  return formatNumber(amount, 2, 8) + " REMES";
}

/**
 * Format credits with appropriate precision
 * 
 * @param credits - Credit amount
 * @returns Formatted credits string
 */
export function formatCredits(credits: number): string {
  if (credits >= 1000) {
    return credits.toFixed(0);
  }
  return formatNumber(credits, 2, 4);
}

/**
 * Format percentage with appropriate precision
 * 
 * @param percentage - Percentage value (0-100)
 * @returns Formatted percentage string
 */
export function formatPercentage(percentage: number): string {
  if (percentage >= 100) {
    return percentage.toFixed(0) + "%";
  }
  return formatNumber(percentage, 1, 2) + "%";
}

