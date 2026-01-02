/**
 * Backend API Proxy Route
 * 
 * Proxies requests to the Python backend API.
 * This allows the frontend to make requests to /api/backend/* 
 * which are forwarded to the actual backend server.
 */

import { NextRequest, NextResponse } from 'next/server';

// Backend API URL from environment variable
const BACKEND_URL = process.env.BACKEND_API_URL || 'http://localhost:8000';

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxyRequest(request, await params);
}

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxyRequest(request, await params);
}

export async function PUT(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxyRequest(request, await params);
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ path: string[] }> }
) {
  return proxyRequest(request, await params);
}

async function proxyRequest(
  request: NextRequest,
  params: { path: string[] }
) {
  try {
    const path = params.path.join('/');
    const searchParams = request.nextUrl.searchParams.toString();
    const url = `${BACKEND_URL}/${path}${searchParams ? `?${searchParams}` : ''}`;

    // Get request body for non-GET requests
    let body: string | undefined;
    if (request.method !== 'GET' && request.method !== 'HEAD') {
      try {
        body = await request.text();
      } catch {
        // No body
      }
    }

    // Forward the request to backend
    const response = await fetch(url, {
      method: request.method,
      headers: {
        'Content-Type': 'application/json',
        // Forward relevant headers
        ...(request.headers.get('authorization') && {
          'Authorization': request.headers.get('authorization')!,
        }),
        ...(request.headers.get('x-wallet-address') && {
          'X-Wallet-Address': request.headers.get('x-wallet-address')!,
        }),
      },
      body,
    });

    // Get response data
    const data = await response.json().catch(() => ({}));

    // Return response with same status
    return NextResponse.json(data, {
      status: response.status,
      headers: {
        'X-Proxied-From': BACKEND_URL,
      },
    });
  } catch (error) {
    console.error('Backend proxy error:', error);
    return NextResponse.json(
      { 
        error: 'Backend unavailable',
        message: error instanceof Error ? error.message : 'Failed to connect to backend',
      },
      { status: 503 }
    );
  }
}
