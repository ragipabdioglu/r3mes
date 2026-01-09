import { NextRequest, NextResponse } from 'next/server';
import { csrfMiddleware, generateCSRFToken, setCSRFTokenCookie } from '@/lib/csrf';

/**
 * Enhanced Security Middleware for Web Dashboard
 * 
 * Features:
 * - Rate limiting with IP-based tracking
 * - CSRF protection for state-changing requests
 * - HTTPS enforcement in production
 * - Security headers
 * - Path-based rate limiting
 * 
 * IMPORTANT: In production with multiple instances, consider:
 * 1. Redis-based rate limiting
 * 2. CDN/reverse proxy rate limiting (Cloudflare, AWS WAF)
 * 3. Load balancer rate limiting
 */

// Rate limiting map: IP -> { count, resetTime }
const rateLimitMap = new Map<string, { count: number; resetTime: number }>();

// Rate limiting configuration
const RATE_LIMIT = {
  windowMs: 15 * 60 * 1000, // 15 minutes
  maxRequests: 100, // requests per window
  apiMaxRequests: 50, // stricter limit for API routes
  authMaxRequests: 10, // very strict for auth routes
};

// Track warnings
let hasWarnedAboutInMemory = false;

// Clean up old entries periodically
if (typeof setInterval !== 'undefined') {
  setInterval(() => {
    const now = Date.now();
    for (const [ip, record] of rateLimitMap.entries()) {
      if (now > record.resetTime) {
        rateLimitMap.delete(ip);
      }
    }
  }, 5 * 60 * 1000);
}

function getClientIP(req: NextRequest): string {
  const forwarded = req.headers.get('x-forwarded-for');
  if (forwarded) {
    return forwarded.split(',')[0].trim();
  }
  
  const realIP = req.headers.get('x-real-ip');
  if (realIP) {
    return realIP;
  }
  
  return req.ip || 'unknown';
}

function getRateLimitKey(request: NextRequest): string {
  const ip = getClientIP(request);
  const path = request.nextUrl.pathname;
  
  // Different limits for different endpoints
  if (path.startsWith('/api/auth') || path.startsWith('/api/faucet')) {
    return `auth:${ip}`;
  }
  if (path.startsWith('/api/')) {
    return `api:${ip}`;
  }
  return `general:${ip}`;
}

function getMaxRequests(key: string): number {
  if (key.startsWith('auth:')) {
    return RATE_LIMIT.authMaxRequests;
  }
  if (key.startsWith('api:')) {
    return RATE_LIMIT.apiMaxRequests;
  }
  return RATE_LIMIT.maxRequests;
}

function rateLimit(req: NextRequest): NextResponse | null {
  // Warn about in-memory rate limiting in production
  if (process.env.NODE_ENV === 'production' && !hasWarnedAboutInMemory) {
    console.warn(
      '[SECURITY WARNING] Using in-memory rate limiting in production. ' +
      'Consider using Redis-based rate limiting for multi-instance deployments.'
    );
    hasWarnedAboutInMemory = true;
  }
  
  const key = getRateLimitKey(req);
  const maxRequests = getMaxRequests(key);
  const now = Date.now();
  
  const record = rateLimitMap.get(key);
  
  if (!record || now > record.resetTime) {
    rateLimitMap.set(key, { count: 1, resetTime: now + RATE_LIMIT.windowMs });
    return null;
  }
  
  if (record.count >= maxRequests) {
    return NextResponse.json(
      { 
        error: 'Rate limit exceeded',
        message: `Too many requests. Please try again after ${Math.ceil((record.resetTime - now) / 1000)} seconds.`,
        retryAfter: Math.ceil((record.resetTime - now) / 1000),
      },
      { 
        status: 429,
        headers: {
          'Retry-After': String(Math.ceil((record.resetTime - now) / 1000)),
          'X-RateLimit-Limit': String(maxRequests),
          'X-RateLimit-Remaining': String(Math.max(0, maxRequests - record.count)),
          'X-RateLimit-Reset': String(Math.ceil(record.resetTime / 1000)),
        },
      }
    );
  }
  
  record.count++;
  return null;
}

function enforceHTTPS(req: NextRequest): NextResponse | null {
  if (process.env.NODE_ENV === 'production') {
    const hostname = req.headers.get('host') || req.nextUrl.hostname;
    if (hostname && (hostname.includes('localhost') || hostname.includes('127.0.0.1'))) {
      return null;
    }
    
    const protocol = req.headers.get('x-forwarded-proto') || req.nextUrl.protocol;
    
    if (protocol !== 'https') {
      const httpsUrl = req.nextUrl.clone();
      httpsUrl.protocol = 'https';
      return NextResponse.redirect(httpsUrl, 301);
    }
  }
  
  return null;
}

function addSecurityHeaders(response: NextResponse): NextResponse {
  // Enhanced security headers
  response.headers.set('X-DNS-Prefetch-Control', 'on');
  response.headers.set('X-Frame-Options', 'SAMEORIGIN');
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('Referrer-Policy', 'origin-when-cross-origin');
  response.headers.set('X-XSS-Protection', '1; mode=block');
  
  // Permissions Policy
  response.headers.set(
    'Permissions-Policy',
    'camera=(), microphone=(), geolocation=(), payment=(), usb=()'
  );

  // HSTS in production
  if (process.env.NODE_ENV === 'production') {
    response.headers.set(
      'Strict-Transport-Security',
      'max-age=31536000; includeSubDomains; preload'
    );
  }

  return response;
}

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  // Enforce HTTPS in production
  const httpsResponse = enforceHTTPS(req);
  if (httpsResponse) {
    return addSecurityHeaders(httpsResponse);
  }

  // Apply rate limiting
  const rateLimitResponse = rateLimit(req);
  if (rateLimitResponse) {
    return addSecurityHeaders(rateLimitResponse);
  }

  // Apply CSRF protection to API routes (except GET requests and docs)
  if (pathname.startsWith('/api/') && !pathname.startsWith('/api/docs') && !pathname.startsWith('/api/csrf-token')) {
    const csrfResponse = csrfMiddleware(req);
    if (csrfResponse) {
      return addSecurityHeaders(csrfResponse);
    }
  }

  // Generate CSRF token for first-time visitors
  const response = NextResponse.next();
  
  // Add CSRF token to response if it's a page request and no token exists
  if (!pathname.startsWith('/api/') && !pathname.startsWith('/_next/')) {
    const hasCsrfCookie = req.cookies.get('csrf-token');
    if (!hasCsrfCookie) {
      const csrfToken = generateCSRFToken();
      setCSRFTokenCookie(response, csrfToken);
    }
  }

  return addSecurityHeaders(response);
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public folder files
     */
    '/((?!_next/static|_next/image|favicon.ico|public/).*)',
  ],
};

