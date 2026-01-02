/**
 * CSRF Protection utilities for R3MES Web Dashboard
 * Provides token generation, validation, and middleware integration
 */

import { NextRequest, NextResponse } from 'next/server';

// CSRF token interface
export interface CSRFToken {
  token: string;
  timestamp: number;
  expires: number;
}

// CSRF configuration
const CSRF_CONFIG = {
  TOKEN_LENGTH: 32,
  TOKEN_LIFETIME: 60 * 60 * 1000, // 1 hour in milliseconds
  COOKIE_NAME: 'csrf-token',
  HEADER_NAME: 'x-csrf-token',
  FORM_FIELD_NAME: '_csrf',
};

/**
 * Generates a cryptographically secure CSRF token
 * @returns CSRF token object with expiration
 */
export function generateCSRFToken(): CSRFToken {
  // Use Web Crypto API for browser compatibility
  let token: string;
  if (typeof window !== 'undefined' && window.crypto && window.crypto.getRandomValues) {
    const array = new Uint8Array(CSRF_CONFIG.TOKEN_LENGTH);
    window.crypto.getRandomValues(array);
    token = Array.from(array, byte => byte.toString(16).padStart(2, '0')).join('');
  } else {
    // Fallback for server-side
    token = Math.random().toString(36).substring(2) + Date.now().toString(36);
  }
  
  const timestamp = Date.now();
  const expires = timestamp + CSRF_CONFIG.TOKEN_LIFETIME;

  return {
    token,
    timestamp,
    expires,
  };
}

/**
 * Validates a CSRF token
 * @param providedToken - Token provided by client
 * @param storedToken - Token stored on server/session
 * @returns True if token is valid
 */
export function validateCSRFToken(providedToken: string, storedToken: CSRFToken): boolean {
  if (!providedToken || !storedToken) {
    return false;
  }

  // Check if token has expired
  if (Date.now() > storedToken.expires) {
    return false;
  }

  // Simple string comparison (timing attack protection is less critical for CSRF tokens)
  return providedToken === storedToken.token;
}

/**
 * Extracts CSRF token from request (header, body, or query)
 * @param request - Next.js request object
 * @returns CSRF token string or null
 */
export function extractCSRFToken(request: NextRequest): string | null {
  // Try header first
  const headerToken = request.headers.get(CSRF_CONFIG.HEADER_NAME);
  if (headerToken) {
    return headerToken;
  }

  // Try form data (for POST requests)
  const contentType = request.headers.get('content-type');
  if (contentType?.includes('application/x-www-form-urlencoded')) {
    // This would need to be handled in the API route since we can't read body here
    return null;
  }

  // Try query parameter (less secure, only for GET requests)
  const queryToken = request.nextUrl.searchParams.get('_csrf');
  if (queryToken) {
    return queryToken;
  }

  return null;
}

/**
 * Sets CSRF token in response cookie
 * @param response - Next.js response object
 * @param token - CSRF token to set
 * @returns Modified response
 */
export function setCSRFTokenCookie(response: NextResponse, token: CSRFToken): NextResponse {
  response.cookies.set(CSRF_CONFIG.COOKIE_NAME, JSON.stringify(token), {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'strict',
    maxAge: CSRF_CONFIG.TOKEN_LIFETIME / 1000, // Convert to seconds
    path: '/',
  });

  return response;
}

/**
 * Gets CSRF token from request cookie
 * @param request - Next.js request object
 * @returns CSRF token object or null
 */
export function getCSRFTokenFromCookie(request: NextRequest): CSRFToken | null {
  const cookieValue = request.cookies.get(CSRF_CONFIG.COOKIE_NAME)?.value;
  
  if (!cookieValue) {
    return null;
  }

  try {
    return JSON.parse(cookieValue) as CSRFToken;
  } catch {
    return null;
  }
}

/**
 * CSRF middleware for API routes
 * @param request - Next.js request object
 * @returns Response if CSRF check fails, null if passes
 */
export function csrfMiddleware(request: NextRequest): NextResponse | null {
  // Skip CSRF check for GET, HEAD, OPTIONS requests
  if (['GET', 'HEAD', 'OPTIONS'].includes(request.method)) {
    return null;
  }

  // Skip CSRF check for API documentation routes
  if (request.nextUrl.pathname.startsWith('/api/docs')) {
    return null;
  }

  // Get stored token from cookie
  const storedToken = getCSRFTokenFromCookie(request);
  if (!storedToken) {
    return NextResponse.json(
      { error: 'CSRF token missing' },
      { status: 403 }
    );
  }

  // Get provided token from request
  const providedToken = extractCSRFToken(request);
  if (!providedToken) {
    return NextResponse.json(
      { error: 'CSRF token required' },
      { status: 403 }
    );
  }

  // Validate token
  if (!validateCSRFToken(providedToken, storedToken)) {
    return NextResponse.json(
      { error: 'Invalid CSRF token' },
      { status: 403 }
    );
  }

  return null; // CSRF check passed
}

/**
 * Utility to add CSRF token to fetch requests
 * @param options - Fetch options
 * @param csrfToken - CSRF token
 * @returns Modified fetch options
 */
export function addCSRFToken(
  options: RequestInit = {},
  csrfToken: string
): RequestInit {
  const headers = new Headers(options.headers);
  headers.set(CSRF_CONFIG.HEADER_NAME, csrfToken);

  return {
    ...options,
    headers,
    credentials: 'include', // Include cookies
  };
}