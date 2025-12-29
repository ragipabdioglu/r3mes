import { NextRequest, NextResponse } from 'next/server';

/**
 * Rate Limiting Middleware for Web Dashboard
 * 
 * IMPORTANT: In production with multiple instances, this in-memory rate limiting
 * will NOT work correctly. Each instance will have its own rate limit counter.
 * 
 * For production deployments:
 * 1. Use Redis-based rate limiting via API routes
 * 2. Or use a CDN/reverse proxy with built-in rate limiting (e.g., Cloudflare, AWS WAF)
 * 3. Or implement rate limiting at the load balancer level
 * 
 * This middleware is suitable for:
 * - Development environments
 * - Single-instance deployments
 * - As a first line of defense (with proper backend rate limiting)
 */

// Rate limiting map: IP -> { count, resetTime }
// WARNING: This is in-memory and will not work correctly in multi-instance deployments
const rateLimitMap = new Map<string, { count: number; resetTime: number }>();

// Track if we've warned about in-memory rate limiting in production
let hasWarnedAboutInMemory = false;

// Clean up old entries periodically (every 5 minutes)
if (typeof setInterval !== 'undefined') {
  setInterval(() => {
    const now = Date.now();
    for (const [ip, record] of rateLimitMap.entries()) {
      if (now > record.resetTime) {
        rateLimitMap.delete(ip);
      }
    }
  }, 5 * 60 * 1000); // 5 minutes
}

function getClientIP(req: NextRequest): string {
  // Try to get IP from various headers (for proxies/load balancers)
  const forwarded = req.headers.get('x-forwarded-for');
  if (forwarded) {
    return forwarded.split(',')[0].trim();
  }
  
  const realIP = req.headers.get('x-real-ip');
  if (realIP) {
    return realIP;
  }
  
  // Fallback to connection remote address (if available)
  return req.ip || 'unknown';
}

function rateLimit(req: NextRequest): NextResponse | null {
  // Get rate limit config from environment or use defaults
  const windowMs = parseInt(process.env.RATE_LIMIT_WINDOW_MS || '60000', 10); // 1 minute default
  const maxRequests = parseInt(process.env.RATE_LIMIT_MAX_REQUESTS || '100', 10); // 100 requests per minute default
  
  // Warn about in-memory rate limiting in production (once)
  if (process.env.NODE_ENV === 'production' && !hasWarnedAboutInMemory) {
    console.warn(
      '[SECURITY WARNING] Using in-memory rate limiting in production. ' +
      'This will NOT work correctly with multiple instances. ' +
      'Consider using Redis-based rate limiting or a CDN/WAF solution.'
    );
    hasWarnedAboutInMemory = true;
  }
  
  const ip = getClientIP(req);
  const now = Date.now();
  
  const record = rateLimitMap.get(ip);
  
  // If no record or window expired, create new record
  if (!record || now > record.resetTime) {
    rateLimitMap.set(ip, { count: 1, resetTime: now + windowMs });
    return null; // Allow request
  }
  
  // Check if limit exceeded
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
          'X-RateLimit-Remaining': '0',
          'X-RateLimit-Reset': String(Math.ceil(record.resetTime / 1000)),
        },
      }
    );
  }
  
  // Increment count
  record.count++;
  return null; // Allow request
}

function enforceHTTPS(req: NextRequest): NextResponse | null {
  // Only enforce HTTPS in production, but NOT on localhost
  if (process.env.NODE_ENV === 'production') {
    const hostname = req.headers.get('host') || req.nextUrl.hostname;
    // Skip HTTPS enforcement for localhost
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

export function middleware(req: NextRequest) {
  // Enforce HTTPS in production
  const httpsResponse = enforceHTTPS(req);
  if (httpsResponse) {
    return httpsResponse;
  }
  
  // Only apply rate limiting to API routes
  if (req.nextUrl.pathname.startsWith('/api/')) {
    const rateLimitResponse = rateLimit(req);
    if (rateLimitResponse) {
      return rateLimitResponse;
    }
  }
  
  // Continue with request
  return NextResponse.next();
}

// Configure which routes to apply middleware to
export const config = {
  matcher: [
    '/api/:path*',
  ],
};

