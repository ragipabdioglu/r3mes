# Frontend Performance Optimization Guide

This document outlines the frontend performance optimization strategy for the R3MES web dashboard.

## 1. Bundle Size Optimization

### 1.1. Code Splitting

The dashboard uses Next.js automatic code splitting with additional manual optimizations:

- **Route-based splitting**: Each page is automatically split into separate chunks
- **Component-based splitting**: Heavy components are lazy-loaded using `dynamic()` imports
- **Library-based splitting**: Large libraries are loaded only when needed

### 1.2. Lazy Loading Strategy

**Heavy Libraries (Lazy Loaded):**

- **react-globe.gl** (~500KB): Only loaded in Network Explorer
  ```typescript
  const Globe = dynamic(() => import("react-globe.gl"), {
    ssr: false,
    loading: () => <div>Loading globe...</div>
  });
  ```

- **recharts** (~200KB): Only loaded in Mine page for charts
  ```typescript
  const LineChart = dynamic(
    () => import("recharts").then((mod) => mod.LineChart),
    { ssr: false }
  );
  ```

- **framer-motion** (~150KB): Only loaded in marketing pages
  ```typescript
  const motion = dynamic(
    () => import("framer-motion").then((mod) => mod.motion),
    { ssr: false }
  );
  ```

- **three.js** (~600KB): Only loaded in HeroScene component
  ```typescript
  const HeroScene = dynamic(() => import("@/components/marketing/HeroScene"), {
    ssr: false
  });
  ```

### 1.3. Tree Shaking

Next.js automatically tree-shakes unused code. Additional optimizations:

- **lucide-react**: Icons are tree-shaken (only imported icons are included)
- **recharts**: Components are tree-shaken (only used chart components are included)
- **@cosmos-kit/react**: Only used wallet adapters are included

## 2. Image Optimization

### 2.1. Next.js Image Component

Always use `next/image` for images:

```typescript
import Image from 'next/image';

<Image
  src="/og-image.png"
  alt="R3MES"
  width={1200}
  height={630}
  priority // For above-the-fold images
/>
```

### 2.2. Image Formats

Next.js automatically serves:
- **AVIF**: Modern format with best compression
- **WebP**: Fallback for older browsers
- **Original**: Fallback for unsupported browsers

### 2.3. Image Sizing

Configure appropriate device sizes in `next.config.js`:

```javascript
images: {
  deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
  imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
}
```

## 3. Font Optimization

### 3.1. Google Fonts

Fonts are optimized using Next.js font optimization:

```typescript
import { Inter } from "next/font/google";

const inter = Inter({ 
  subsets: ["latin"],
  display: 'swap', // Use swap for better performance
  preload: true,
});
```

**Benefits:**
- Fonts are self-hosted (no external requests)
- Automatic subsetting (only used characters)
- Preloading for faster rendering
- Font-display: swap prevents invisible text

## 4. Script Optimization

### 4.1. Script Loading Strategy

Use Next.js `Script` component with appropriate strategies:

```typescript
// Load after page becomes interactive
<Script
  strategy="afterInteractive"
  src="https://www.googletagmanager.com/gtag/js?id=GA_ID"
/>

// Load on idle (for non-critical scripts)
<Script
  strategy="lazyOnload"
  src="https://example.com/script.js"
/>
```

**Strategies:**
- `beforeInteractive`: Load before page becomes interactive (critical scripts only)
- `afterInteractive`: Load after page becomes interactive (default for analytics)
- `lazyOnload`: Load during idle time (non-critical scripts)

## 5. Component Optimization

### 5.1. Dynamic Imports

Use dynamic imports for heavy components:

```typescript
// Heavy component (3D globe)
const NetworkExplorer = dynamic(() => import("@/components/NetworkExplorer"), {
  ssr: false, // Disable SSR for client-only components
  loading: () => <SkeletonLoader />, // Show loading state
});
```

### 5.2. Conditional Rendering

Only render heavy components when needed:

```typescript
{showGlobe && <NetworkExplorer />}
```

### 5.3. Memoization

Use React.memo for expensive components:

```typescript
export default React.memo(ExpensiveComponent);
```

## 6. Bundle Analysis

### 6.1. Analyze Bundle Size

Run bundle analyzer to identify large dependencies:

```bash
npm run analyze
```

This generates a visual report showing:
- Bundle sizes per route
- Largest dependencies
- Code splitting effectiveness

### 6.2. Target Bundle Sizes

**Target Metrics:**
- Initial JS bundle: < 200KB (gzipped)
- Total JS bundle: < 500KB (gzipped)
- CSS bundle: < 50KB (gzipped)
- Images: Optimized with WebP/AVIF

## 7. Performance Monitoring

### 7.1. Core Web Vitals

Monitor these metrics:

- **LCP (Largest Contentful Paint)**: < 2.5s
- **FID (First Input Delay)**: < 100ms
- **CLS (Cumulative Layout Shift)**: < 0.1

### 7.2. Lighthouse Scores

Target Lighthouse scores:
- Performance: > 90
- Accessibility: > 90
- Best Practices: > 90
- SEO: > 90

## 8. Caching Strategy

### 8.1. Static Assets

Static assets are cached with long TTL:
- JS/CSS: 1 year (with hash-based filenames)
- Images: 1 year (with hash-based filenames)
- Fonts: 1 year

### 8.2. API Responses

API responses are cached using:
- React Query for client-side caching
- Next.js API routes for server-side caching

## 9. Best Practices

### 9.1. Avoid Large Dependencies

- Use lightweight alternatives when possible
- Lazy load heavy libraries
- Tree-shake unused code

### 9.2. Optimize Imports

```typescript
// ❌ Bad: Import entire library
import * as recharts from 'recharts';

// ✅ Good: Import specific components
import { LineChart, Line } from 'recharts';
```

### 9.3. Use Production Builds

Always test with production builds:

```bash
npm run build
npm start
```

Development builds are not optimized and have larger bundle sizes.

### 9.4. Monitor Bundle Size

Set up CI/CD checks to prevent bundle size regressions:

```json
{
  "scripts": {
    "analyze": "ANALYZE=true next build",
    "check-bundle-size": "npm run build && node scripts/check-bundle-size.js"
  }
}
```

## 10. Troubleshooting

### 10.1. Large Bundle Size

1. Run bundle analyzer: `npm run analyze`
2. Identify large dependencies
3. Lazy load heavy components
4. Use tree-shaking for libraries
5. Consider code splitting

### 10.2. Slow Initial Load

1. Check bundle size
2. Enable compression (gzip/brotli)
3. Use CDN for static assets
4. Optimize images
5. Preload critical resources

### 10.3. Slow Runtime Performance

1. Profile with React DevTools Profiler
2. Identify expensive components
3. Use React.memo for expensive renders
4. Optimize re-renders with useMemo/useCallback
5. Lazy load heavy components

---

**Son Güncelleme**: 2025-12-24

