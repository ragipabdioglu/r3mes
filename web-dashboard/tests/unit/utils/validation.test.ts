/**
 * @jest-environment jsdom
 */

import {
  validateWalletAddress,
  validateAmount,
  validateChatMessage,
  validateUrlParameter,
  validateEmail,
  validateTransactionHash,
  validateStakeAmount,
  validateSearchQuery,
  sanitizeHtml,
} from '@/utils/validation';

describe('Validation Utilities', () => {
  describe('validateWalletAddress', () => {
    it('validates correct R3MES wallet addresses', () => {
      const validAddress = 'remes1234567890abcdef1234567890abcdef12';
      const result = validateWalletAddress(validAddress);
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe(validAddress);
      expect(result.error).toBeUndefined();
    });

    it('rejects empty addresses', () => {
      const result = validateWalletAddress('');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Wallet address is required');
    });

    it('rejects addresses without remes prefix', () => {
      const result = validateWalletAddress('cosmos1234567890abcdef1234567890abcdef12');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe("Address must start with 'remes'");
    });

    it('rejects addresses that are too short', () => {
      const result = validateWalletAddress('remes123');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid address length');
    });

    it('rejects addresses that are too long', () => {
      const longAddress = 'remes' + '1'.repeat(60);
      const result = validateWalletAddress(longAddress);
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid address length');
    });

    it('rejects addresses with invalid characters', () => {
      const result = validateWalletAddress('remes1234567890ABCDEF1234567890abcdef12');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Address contains invalid characters');
    });

    it('rejects addresses with wrong format length', () => {
      const result = validateWalletAddress('remes1234567890abcdef1234567890abcdef1'); // 41 chars
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid address length');
    });

    it('trims whitespace from addresses', () => {
      const address = '  remes1234567890abcdef1234567890abcdef12  ';
      const result = validateWalletAddress(address);
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('remes1234567890abcdef1234567890abcdef12');
    });
  });

  describe('validateAmount', () => {
    it('validates correct amounts', () => {
      const result = validateAmount('100.50');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('100.50');
    });

    it('validates integer amounts', () => {
      const result = validateAmount('1000');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('1000');
    });

    it('rejects empty amounts', () => {
      const result = validateAmount('');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Amount is required');
    });

    it('rejects invalid number formats', () => {
      const result = validateAmount('abc');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid number format');
    });

    it('rejects amounts below minimum', () => {
      const result = validateAmount('5', 10);
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Amount must be at least 10');
    });

    it('rejects amounts above maximum', () => {
      const result = validateAmount('1500', 0, 1000);
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Amount cannot exceed 1000');
    });

    it('rejects amounts with too many decimal places', () => {
      const result = validateAmount('100.1234567', 0, 1000, 6);
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Maximum 6 decimal places allowed');
    });

    it('accepts amounts with exact decimal places', () => {
      const result = validateAmount('100.123456', 0, 1000, 6);
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('100.123456');
    });

    it('handles zero amounts correctly', () => {
      const result = validateAmount('0', 0, 1000);
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('0');
    });

    it('trims whitespace from amounts', () => {
      const result = validateAmount('  100.50  ');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('100.50');
    });
  });

  describe('validateChatMessage', () => {
    it('validates normal chat messages', () => {
      const result = validateChatMessage('Hello, how are you?');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('Hello, how are you?');
    });

    it('rejects empty messages', () => {
      const result = validateChatMessage('');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Message is required');
    });

    it('rejects whitespace-only messages', () => {
      const result = validateChatMessage('   ');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Message cannot be empty');
    });

    it('rejects messages that are too long', () => {
      const longMessage = 'a'.repeat(5001);
      const result = validateChatMessage(longMessage);
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Message cannot exceed 5000 characters');
    });

    it('accepts messages at maximum length', () => {
      const maxMessage = 'a'.repeat(5000);
      const result = validateChatMessage(maxMessage);
      
      expect(result.valid).toBe(true);
    });

    it('sanitizes HTML content', () => {
      const result = validateChatMessage('<script>alert("xss")</script>Hello');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).not.toContain('<script>');
      expect(result.sanitized).toContain('Hello');
    });

    it('detects script injection attempts', () => {
      const result = validateChatMessage('Hello <script>alert("xss")</script>');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Message contains potentially dangerous content');
    });

    it('detects javascript: URLs', () => {
      const result = validateChatMessage('Click here: javascript:alert("xss")');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Message contains potentially dangerous content');
    });

    it('detects event handlers', () => {
      const result = validateChatMessage('Hello onclick="alert(1)"');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Message contains potentially dangerous content');
    });

    it('allows safe HTML entities', () => {
      const result = validateChatMessage('Hello &amp; goodbye');
      
      expect(result.valid).toBe(true);
    });

    it('trims whitespace from messages', () => {
      const result = validateChatMessage('  Hello world  ');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('Hello world');
    });
  });

  describe('validateUrlParameter', () => {
    it('validates safe URL parameters', () => {
      const result = validateUrlParameter('user123');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('user123');
    });

    it('allows hyphens and underscores', () => {
      const result = validateUrlParameter('user-name_123');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('user-name_123');
    });

    it('rejects empty parameters', () => {
      const result = validateUrlParameter('');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Parameter is required');
    });

    it('rejects parameters with special characters', () => {
      const result = validateUrlParameter('user@domain.com');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Parameter contains invalid characters');
    });

    it('rejects parameters that are too long', () => {
      const longParam = 'a'.repeat(101);
      const result = validateUrlParameter(longParam);
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Parameter too long');
    });

    it('accepts custom character patterns', () => {
      const result = validateUrlParameter('user@domain.com', /^[a-zA-Z0-9@.]+$/);
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('user@domain.com');
    });

    it('trims whitespace from parameters', () => {
      const result = validateUrlParameter('  user123  ');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('user123');
    });
  });

  describe('validateEmail', () => {
    it('validates correct email addresses', () => {
      const result = validateEmail('user@example.com');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('user@example.com');
    });

    it('validates emails with subdomains', () => {
      const result = validateEmail('user@mail.example.com');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('user@mail.example.com');
    });

    it('rejects empty emails', () => {
      const result = validateEmail('');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Email is required');
    });

    it('rejects invalid email formats', () => {
      const result = validateEmail('invalid-email');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid email format');
    });

    it('rejects emails without domain', () => {
      const result = validateEmail('user@');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid email format');
    });

    it('rejects emails without username', () => {
      const result = validateEmail('@example.com');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid email format');
    });

    it('rejects emails that are too long', () => {
      const longEmail = 'a'.repeat(250) + '@example.com';
      const result = validateEmail(longEmail);
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Email too long');
    });

    it('converts email to lowercase', () => {
      const result = validateEmail('USER@EXAMPLE.COM');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('user@example.com');
    });

    it('trims whitespace from emails', () => {
      const result = validateEmail('  user@example.com  ');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('user@example.com');
    });
  });

  describe('validateTransactionHash', () => {
    it('validates correct transaction hashes', () => {
      const hash = 'A'.repeat(64);
      const result = validateTransactionHash(hash);
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe(hash);
    });

    it('validates lowercase hex hashes', () => {
      const hash = 'a'.repeat(64);
      const result = validateTransactionHash(hash);
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe(hash.toUpperCase());
    });

    it('rejects empty hashes', () => {
      const result = validateTransactionHash('');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Transaction hash is required');
    });

    it('rejects hashes with wrong length', () => {
      const result = validateTransactionHash('ABC123');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid transaction hash format');
    });

    it('rejects hashes with invalid characters', () => {
      const hash = 'G'.repeat(64);
      const result = validateTransactionHash(hash);
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid transaction hash format');
    });

    it('converts hash to uppercase', () => {
      const hash = 'abcdef1234567890'.repeat(4);
      const result = validateTransactionHash(hash);
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe(hash.toUpperCase());
    });

    it('trims whitespace from hashes', () => {
      const hash = '  ' + 'A'.repeat(64) + '  ';
      const result = validateTransactionHash(hash);
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('A'.repeat(64));
    });
  });

  describe('validateStakeAmount', () => {
    it('validates amounts above minimum stake', () => {
      const result = validateStakeAmount('2000', 1000);
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('2000');
    });

    it('rejects amounts below minimum stake', () => {
      const result = validateStakeAmount('500', 1000);
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Minimum stake required: 1000 REMES');
    });

    it('uses default minimum stake', () => {
      const result = validateStakeAmount('500'); // default min is 1000
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Minimum stake required: 1000 REMES');
    });

    it('validates exact minimum stake', () => {
      const result = validateStakeAmount('1000', 1000);
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('1000');
    });

    it('inherits amount validation rules', () => {
      const result = validateStakeAmount('invalid', 1000);
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Invalid number format');
    });
  });

  describe('validateSearchQuery', () => {
    it('validates normal search queries', () => {
      const result = validateSearchQuery('bitcoin mining');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('bitcoin mining');
    });

    it('rejects empty queries', () => {
      const result = validateSearchQuery('');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Search query is required');
    });

    it('rejects whitespace-only queries', () => {
      const result = validateSearchQuery('   ');
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Search query cannot be empty');
    });

    it('rejects queries that are too long', () => {
      const longQuery = 'a'.repeat(101);
      const result = validateSearchQuery(longQuery);
      
      expect(result.valid).toBe(false);
      expect(result.error).toBe('Search query cannot exceed 100 characters');
    });

    it('sanitizes dangerous characters', () => {
      const result = validateSearchQuery('search<script>alert(1)</script>');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).not.toContain('<');
      expect(result.sanitized).not.toContain('>');
      expect(result.sanitized).toContain('searchscriptalert(1)/script');
    });

    it('accepts custom max length', () => {
      const result = validateSearchQuery('long query', 50);
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('long query');
    });

    it('trims whitespace from queries', () => {
      const result = validateSearchQuery('  search term  ');
      
      expect(result.valid).toBe(true);
      expect(result.sanitized).toBe('search term');
    });
  });

  describe('sanitizeHtml', () => {
    it('removes script tags', () => {
      const html = '<script>alert("xss")</script>Hello';
      const result = sanitizeHtml(html);
      
      expect(result).not.toContain('<script>');
      expect(result).toContain('Hello');
    });

    it('allows safe tags', () => {
      const html = '<b>Bold</b> and <i>italic</i> text';
      const result = sanitizeHtml(html, ['b', 'i']);
      
      expect(result).toContain('<b>Bold</b>');
      expect(result).toContain('<i>italic</i>');
    });

    it('removes dangerous attributes', () => {
      const html = '<div onclick="alert(1)">Content</div>';
      const result = sanitizeHtml(html, ['div']);
      
      expect(result).not.toContain('onclick');
      expect(result).toContain('Content');
    });

    it('allows safe attributes', () => {
      const html = '<a href="https://example.com" target="_blank">Link</a>';
      const result = sanitizeHtml(html, ['a']);
      
      expect(result).toContain('href="https://example.com"');
      expect(result).toContain('target="_blank"');
    });

    it('uses default allowed tags when none specified', () => {
      const html = '<b>Bold</b> <script>alert(1)</script> <em>Emphasis</em>';
      const result = sanitizeHtml(html);
      
      expect(result).toContain('<b>Bold</b>');
      expect(result).toContain('<em>Emphasis</em>');
      expect(result).not.toContain('<script>');
    });

    it('handles empty HTML', () => {
      const result = sanitizeHtml('');
      
      expect(result).toBe('');
    });

    it('preserves text content', () => {
      const html = 'Plain text without HTML';
      const result = sanitizeHtml(html);
      
      expect(result).toBe('Plain text without HTML');
    });
  });
});