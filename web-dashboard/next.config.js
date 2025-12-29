/** @type {import('next').NextConfig} */
const withBundleAnalyzer = require('@next/bundle-analyzer')({
  enabled: process.env.ANALYZE === 'true',
});

const nextConfig = {
  reactStrictMode: true,
  output: 'standalone', // Enable standalone output for Docker

  // TypeScript: Ignore build errors from third-party libraries (react-globe.gl compatibility)
  typescript: {
    ignoreBuildErrors: true,
  },

  // CDN integration (production only)
  assetPrefix: process.env.NODE_ENV === 'production' && process.env.CDN_URL
    ? process.env.CDN_URL
    : undefined,

  // Security: Remove X-Powered-By header
  poweredByHeader: false,

  // Compression (Next.js handles this automatically, but we can verify)
  compress: true,

  // Performance optimizations
  swcMinify: true, // Use SWC for minification (faster than Terser)
  compiler: {
    removeConsole: process.env.NODE_ENV === 'production' ? {
      exclude: ['error', 'warn'], // Keep error and warn logs
    } : false,
  },

  // Image optimization
  images: {
    formats: ['image/avif', 'image/webp'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
    minimumCacheTTL: 60,
  },

  // Experimental features for better performance
  experimental: {
    // optimizeCss: true, // Disabled - requires 'critters' package
    optimizePackageImports: [
      'lucide-react', // Tree-shake lucide icons
      'recharts', // Tree-shake recharts components
      '@cosmos-kit/react', // Tree-shake cosmos kit
    ],
  },
  // Security headers
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on'
          },
          {
            key: 'Strict-Transport-Security',
            value: 'max-age=63072000; includeSubDomains; preload'
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN'
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff'
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block'
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin'
          },
          {
            key: 'Content-Security-Policy',
            value: (() => {
              const isProduction = process.env.NODE_ENV === 'production';

              if (isProduction) {
                // Production: Strict CSP without unsafe directives
                // Google Analytics is loaded via Next.js Script component which handles CSP correctly
                return [
                  "default-src 'self'",
                  "script-src 'self' https://www.googletagmanager.com https://www.google-analytics.com",
                  "style-src 'self' https://fonts.googleapis.com",
                  "img-src 'self' data: https: https://www.google-analytics.com https://www.googletagmanager.com",
                  "font-src 'self' data: https://fonts.gstatic.com",
                  "connect-src 'self' https: wss: https://www.google-analytics.com https://www.googletagmanager.com",
                  "frame-ancestors 'self'",
                  "object-src 'none'",
                  "base-uri 'self'",
                  "form-action 'self'",
                ].join('; ');
              } else {
                // Development: More permissive (webpack HMR, Next.js dev server require unsafe-eval)
                return [
                  "default-src 'self'",
                  "script-src 'self' 'unsafe-eval' 'unsafe-inline' https://www.googletagmanager.com https://www.google-analytics.com",
                  "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com",
                  "img-src 'self' data: https: https://www.google-analytics.com https://www.googletagmanager.com",
                  "font-src 'self' data: https://fonts.gstatic.com",
                  // In production, remove localhost from CSP. In development, allow localhost
                  `connect-src 'self' https: wss: ${!isProduction ? 'http://localhost:* ws://localhost:*' : ''} https://www.google-analytics.com https://www.googletagmanager.com`,
                  "frame-ancestors 'self'",
                ].join('; ');
              }
            })()
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()'
          }
        ],
      },
    ];
  },
  // CORS configuration for API calls
  async rewrites() {
    // Environment-based URL configuration
    const envMode = process.env.R3MES_ENV || 'development';
    const isProduction = envMode === 'production';

    // In production, environment variables are REQUIRED (no localhost fallback)
    if (isProduction) {
      if (!process.env.NEXT_PUBLIC_BACKEND_URL) {
        throw new Error(
          'NEXT_PUBLIC_BACKEND_URL environment variable must be set in production. ' +
          'Do not use localhost in production.'
        );
      }
      if (!process.env.NEXT_PUBLIC_API_URL) {
        throw new Error(
          'NEXT_PUBLIC_API_URL environment variable must be set in production. ' +
          'Do not use localhost in production.'
        );
      }
      // Validate no localhost in production URLs
      if (process.env.NEXT_PUBLIC_BACKEND_URL.includes('localhost') || 
          process.env.NEXT_PUBLIC_BACKEND_URL.includes('127.0.0.1')) {
        throw new Error(
          `NEXT_PUBLIC_BACKEND_URL cannot use localhost in production: ${process.env.NEXT_PUBLIC_BACKEND_URL}`
        );
      }
      if (process.env.NEXT_PUBLIC_API_URL.includes('localhost') || 
          process.env.NEXT_PUBLIC_API_URL.includes('127.0.0.1')) {
        throw new Error(
          `NEXT_PUBLIC_API_URL cannot use localhost in production: ${process.env.NEXT_PUBLIC_API_URL}`
        );
      }
    }

    // Development: allow localhost fallback
    const backendUrl = isProduction
      ? process.env.NEXT_PUBLIC_BACKEND_URL
      : (process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000');

    const apiUrl = isProduction
      ? process.env.NEXT_PUBLIC_API_URL
      : (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:1317');

    // List of backend API paths that should be proxied
    // Note: /api/docs/* is handled by Next.js API route (app/api/docs/[file]/route.ts)
    // and should NOT be included here
    const backendApiPaths = [
      'chat',
      'user',
      'network',
      'miner',
      'serving',
      'proposer',
      'roles',
      'faucet',
      'staking',
    ];

    const rewritesList = [
      {
        source: '/api/blockchain/:path*',
        destination: `${apiUrl}/:path*`,
      },
    ];

    // Add rewrites for each backend API path (excluding /api/docs)
    // Note: We don't include a catch-all /api/:path* rewrite because
    // Next.js rewrites can interfere with API routes. By being explicit,
    // we ensure /api/docs/* is handled by the Next.js API route.
    backendApiPaths.forEach(path => {
      rewritesList.push({
        source: `/api/${path}/:path*`,
        destination: `${backendUrl}/api/${path}/:path*`,
      });
    });

    return rewritesList;
  },
  // Webpack configuration for react-globe.gl (only for Network Explorer)
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        path: false,
        crypto: false,
      };
    }
    return config;
  },
  // Generate sitemap and robots.txt
  generateBuildId: async () => {
    return process.env.BUILD_ID || `build-${Date.now()}`;
  },
};

module.exports = withBundleAnalyzer(nextConfig);
