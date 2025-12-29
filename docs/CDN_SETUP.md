# CDN Integration Guide

This guide explains how to set up CDN (Content Delivery Network) for R3MES static assets.

## Overview

CDN integration improves performance by:
- Reducing latency through edge caching
- Offloading bandwidth from origin server
- Improving global accessibility
- DDoS protection

## Supported CDN Providers

### Cloudflare

1. **Sign up for Cloudflare**
   - Create account at https://cloudflare.com
   - Add your domain

2. **Configure DNS**
   - Point your domain to Cloudflare nameservers
   - Add A/CNAME records for your services

3. **Enable CDN**
   - Go to Speed → Optimization
   - Enable Auto Minify
   - Enable Brotli compression

4. **Configure Caching**
   - Go to Caching → Configuration
   - Set cache level to Standard
   - Browser Cache TTL: 4 hours
   - Edge Cache TTL: 2 hours

5. **Update Next.js Config**

   ```javascript
   // next.config.js
   const nextConfig = {
     assetPrefix: process.env.CDN_URL || '',
     // ... other config
   };
   ```

### AWS CloudFront

1. **Create S3 Bucket**
   ```bash
   aws s3 mb s3://r3mes-static-assets
   ```

2. **Upload Static Assets**
   ```bash
   npm run build
   aws s3 sync .next/static s3://r3mes-static-assets/static
   ```

3. **Create CloudFront Distribution**
   - Origin: S3 bucket
   - Viewer Protocol Policy: Redirect HTTP to HTTPS
   - Allowed HTTP Methods: GET, HEAD, OPTIONS
   - Cache Policy: CachingOptimized

4. **Update Environment Variables**
   ```bash
   export CDN_URL=https://d1234567890.cloudfront.net
   ```

## Configuration

### Environment Variables

```bash
# CDN URL for static assets
CDN_URL=https://cdn.r3mes.network

# CDN enabled flag
ENABLE_CDN=true
```

### Next.js Configuration

Update `web-dashboard/next.config.js`:

```javascript
const nextConfig = {
  assetPrefix: process.env.CDN_URL || '',
  images: {
    domains: ['cdn.r3mes.network'],
  },
  // ... other config
};
```

## Cache Headers

Static assets should have appropriate cache headers:

```nginx
# Nginx configuration
location /_next/static {
    add_header Cache-Control "public, max-age=31536000, immutable";
    proxy_pass http://localhost:3000;
}
```

## Monitoring

Monitor CDN performance:
- Cache hit ratio (target: >90%)
- Response times
- Bandwidth savings
- Error rates

## Best Practices

1. **Version Assets**: Use content hashing for cache busting
2. **Compression**: Enable Brotli/Gzip compression
3. **HTTPS**: Always use HTTPS for CDN
4. **Cache Control**: Set appropriate cache headers
5. **Monitoring**: Monitor CDN metrics regularly

