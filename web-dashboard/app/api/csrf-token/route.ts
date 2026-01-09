/**
 * CSRF Token API Endpoint
 * Generates and returns CSRF tokens for form protection
 */

import { NextResponse } from 'next/server';
import { generateCSRFToken, setCSRFTokenCookie } from '@/lib/csrf';

export async function GET() {
  try {
    const csrfToken = generateCSRFToken();
    
    const response = NextResponse.json({
      token: csrfToken.token,
      expires: csrfToken.expires,
    });
    
    // Set the token in an HTTP-only cookie
    setCSRFTokenCookie(response, csrfToken);
    
    return response;
  } catch (error) {
    return NextResponse.json(
      { error: 'Failed to generate CSRF token' },
      { status: 500 }
    );
  }
}
