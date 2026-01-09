/**
 * Comprehensive input validation utilities for R3MES Web Dashboard
 * Provides secure validation for wallet addresses, amounts, messages, and other inputs
 */

import DOMPurify from 'isomorphic-dompurify';

// Validation result interface
export interface ValidationResult {
  valid: boolean;
  error?: string;
  sanitized?: string;
}

/**
 * Validates R3MES wallet address with comprehensive checks
 * @param address - The wallet address to validate
 * @returns Validation result with error message if invalid
 */
export function validateWalletAddress(address: string): ValidationResult {
  if (!address) {
    return { valid: false, error: "Wallet address is required" };
  }

  // Trim whitespace
  const trimmedAddress = address.trim();

  // Check prefix
  if (!trimmedAddress.startsWith("remes")) {
    return { valid: false, error: "Address must start with 'remes'" };
  }

  // Check length (Bech32 addresses are typically 39-59 characters)
  if (trimmedAddress.length < 39 || trimmedAddress.length > 59) {
    return { valid: false, error: "Invalid address length" };
  }

  // Check for valid Bech32 characters (a-z, 0-9)
  const bech32Regex = /^remes[a-z0-9]+$/;
  if (!bech32Regex.test(trimmedAddress)) {
    return { valid: false, error: "Address contains invalid characters" };
  }

  return { valid: true, sanitized: trimmedAddress };
}

/**
 * Validates amount input with range and format checks
 * @param amount - The amount to validate
 * @param min - Minimum allowed value
 * @param max - Maximum allowed value
 * @param decimals - Maximum decimal places allowed
 * @returns Validation result
 */
export function validateAmount(
  amount: string, 
  min: number = 0, 
  max: number = Number.MAX_SAFE_INTEGER,
  decimals: number = 6
): ValidationResult {
  if (!amount) {
    return { valid: false, error: "Amount is required" };
  }

  const trimmedAmount = amount.trim();

  // Check for valid number format
  const numberRegex = /^\d+(\.\d+)?$/;
  if (!numberRegex.test(trimmedAmount)) {
    return { valid: false, error: "Invalid number format" };
  }

  const numValue = parseFloat(trimmedAmount);

  // Check if it's a valid number
  if (isNaN(numValue)) {
    return { valid: false, error: "Invalid number" };
  }

  // Check range
  if (numValue < min) {
    return { valid: false, error: `Amount must be at least ${min}` };
  }

  if (numValue > max) {
    return { valid: false, error: `Amount cannot exceed ${max}` };
  }

  // Check decimal places
  const decimalPart = trimmedAmount.split('.')[1];
  if (decimalPart && decimalPart.length > decimals) {
    return { valid: false, error: `Maximum ${decimals} decimal places allowed` };
  }

  return { valid: true, sanitized: trimmedAmount };
}

/**
 * Validates and sanitizes chat messages to prevent XSS
 * @param message - The message to validate and sanitize
 * @param maxLength - Maximum message length
 * @returns Validation result with sanitized message
 */
export function validateChatMessage(message: string, maxLength: number = 5000): ValidationResult {
  if (!message) {
    return { valid: false, error: "Message is required" };
  }

  const trimmedMessage = message.trim();

  if (trimmedMessage.length === 0) {
    return { valid: false, error: "Message cannot be empty" };
  }

  if (trimmedMessage.length > maxLength) {
    return { valid: false, error: `Message cannot exceed ${maxLength} characters` };
  }

  // Sanitize HTML to prevent XSS
  const sanitized = DOMPurify.sanitize(trimmedMessage, {
    ALLOWED_TAGS: [], // No HTML tags allowed
    ALLOWED_ATTR: [], // No attributes allowed
    KEEP_CONTENT: true // Keep text content
  });

  // Check for potential script injection attempts
  const dangerousPatterns = [
    /<script/i,
    /javascript:/i,
    /on\w+\s*=/i,
    /data:text\/html/i,
    /vbscript:/i
  ];

  for (const pattern of dangerousPatterns) {
    if (pattern.test(message)) {
      return { valid: false, error: "Message contains potentially dangerous content" };
    }
  }

  return { valid: true, sanitized };
}

/**
 * Validates URL parameters to prevent injection attacks
 * @param param - The URL parameter to validate
 * @param allowedChars - Regex pattern for allowed characters
 * @returns Validation result
 */
export function validateUrlParameter(param: string, allowedChars: RegExp = /^[a-zA-Z0-9_-]+$/): ValidationResult {
  if (!param) {
    return { valid: false, error: "Parameter is required" };
  }

  const trimmedParam = param.trim();

  if (!allowedChars.test(trimmedParam)) {
    return { valid: false, error: "Parameter contains invalid characters" };
  }

  if (trimmedParam.length > 100) {
    return { valid: false, error: "Parameter too long" };
  }

  return { valid: true, sanitized: trimmedParam };
}

/**
 * Validates email addresses
 * @param email - The email to validate
 * @returns Validation result
 */
export function validateEmail(email: string): ValidationResult {
  if (!email) {
    return { valid: false, error: "Email is required" };
  }

  const trimmedEmail = email.trim().toLowerCase();

  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(trimmedEmail)) {
    return { valid: false, error: "Invalid email format" };
  }

  if (trimmedEmail.length > 254) {
    return { valid: false, error: "Email too long" };
  }

  return { valid: true, sanitized: trimmedEmail };
}

/**
 * Validates transaction hash
 * @param hash - The transaction hash to validate
 * @returns Validation result
 */
export function validateTransactionHash(hash: string): ValidationResult {
  if (!hash) {
    return { valid: false, error: "Transaction hash is required" };
  }

  const trimmedHash = hash.trim().toUpperCase();

  // Check for valid hex format (64 characters)
  const hexRegex = /^[A-F0-9]{64}$/;
  if (!hexRegex.test(trimmedHash)) {
    return { valid: false, error: "Invalid transaction hash format" };
  }

  return { valid: true, sanitized: trimmedHash };
}

/**
 * Validates stake amount for role registration
 * @param amount - The stake amount
 * @param minStake - Minimum required stake
 * @returns Validation result
 */
export function validateStakeAmount(amount: string, minStake: number = 1000): ValidationResult {
  const amountValidation = validateAmount(amount, minStake, 1000000, 6);
  
  if (!amountValidation.valid) {
    return amountValidation;
  }

  const numValue = parseFloat(amount);
  
  if (numValue < minStake) {
    return { valid: false, error: `Minimum stake required: ${minStake} REMES` };
  }

  return { valid: true, sanitized: amount };
}

/**
 * Sanitizes HTML content for display
 * @param html - The HTML content to sanitize
 * @param allowedTags - Array of allowed HTML tags
 * @returns Sanitized HTML
 */
export function sanitizeHtml(html: string, allowedTags: string[] = ['b', 'i', 'em', 'strong']): string {
  return DOMPurify.sanitize(html, {
    ALLOWED_TAGS: allowedTags,
    ALLOWED_ATTR: ['href', 'target'],
    KEEP_CONTENT: true
  });
}

/**
 * Validates search query input
 * @param query - The search query
 * @param maxLength - Maximum query length
 * @returns Validation result
 */
export function validateSearchQuery(query: string, maxLength: number = 100): ValidationResult {
  if (!query) {
    return { valid: false, error: "Search query is required" };
  }

  const trimmedQuery = query.trim();

  if (trimmedQuery.length === 0) {
    return { valid: false, error: "Search query cannot be empty" };
  }

  if (trimmedQuery.length > maxLength) {
    return { valid: false, error: `Search query cannot exceed ${maxLength} characters` };
  }

  // Remove potentially dangerous characters
  const sanitized = trimmedQuery.replace(/[<>'"&]/g, '');

  return { valid: true, sanitized };
}