# R3MES Web Dashboard - Detaylı Aksiyon Öğeleri

## FAZE 1: KRİTİK SORUNLAR (1 Hafta)

### 1.1 Input Validation Iyileştirmeleri

**Dosya**: `web-dashboard/utils/validation.ts` (YENİ)

```typescript
// Wallet address validation
export function validateWalletAddress(address: string): { valid: boolean; error?: string } {
  if (!address) return { valid: false, error: "Address required" };
  if (!address.startsWith("remes")) return { valid: false, error: "Invalid prefix" };
  if (address.length !== 42) return { valid: false, error: "Invalid length" };
  // Bech32 validation ekle
  return { valid: true };
}

// Amount validation
export function validateAmount(amount: string, min: number, max: number) {
  const num = parseInt(amount);
  if (isNaN(num)) return { valid: false, error: "Invalid number" };
  if (num < min) return { valid: false, error: `Minimum: ${min}` };
  if (num > max) return { valid: false, error: `Maximum: ${max}` };
  return { valid: true };
}

// Chat message validation
export function validateChatMessage(message: string): { valid: boolean; error?: string } {
  if (!message.trim()) return { valid: false, error: "Message required" };
  if (message.length > 5000) return { valid: false, error: "Message too long" };
  return { valid: true };
}
```

**Kullanım Alanları**:
- `web-dashboard/app/faucet/page.tsx` - Amount validation
- `web-dashboard/app/roles/page.tsx` - Stake validation
- `web-dashboard/components/ChatInterface.tsx` - Message validation

### 1.2 API Error Handling Standardization

**Dosya**: `web-dashboard/lib/api-errors.ts` (YENİ)

```typescript
export class APIError extends Error {
  constructor(
    public statusCode: number,
    public userMessage: string,
    public technicalMessage: string,
    public retryable: boolean = false
  ) {
    super(userMessage);
  }
}

export function handleAPIError(error: any): APIError {
  if (error.response?.status === 429) {
    return new APIError(429, "Too many requests. Please try again later.", error.message, true);
  }
  if (error.response?.status === 503) {
    return new APIError(503, "Service unavailable. Please try again later.", error.message, true);
  }
  if (error.code === 'ECONNREFUSED') {
    return new APIError(0, "Backend service unavailable.", error.message, true);
  }
  return new APIError(500, "An error occurred. Please try again.", error.message, false);
}
```

**Kullanım**: Tüm API calls'ta standardize error handling

### 1.3 Security Fixes

**CSRF Protection** - `web-dashboard/middleware.ts`
```typescript
// CSRF token generation ve validation ekle
export function generateCSRFToken(): string {
  return crypto.randomUUID();
}

// POST requests'te CSRF token check et
```

**Signature Verification** - Backend'de implement et
```typescript
// Backend'de wallet signature verify et
// Frontend'de transaction'ı sign et
```

### 1.4 Test Coverage Artırma

**Target**: %50 → %80

```bash
# Yeni test dosyaları ekle:
tests/unit/hooks/useMinerData.test.ts
tests/unit/hooks/useNetworkStats.test.ts
tests/unit/hooks/useTransactionHistory.test.ts
tests/unit/components/Navbar.test.tsx
tests/unit/components/ChatInterface.test.tsx
tests/unit/components/ErrorBoundary.test.tsx
tests/unit/utils/validation.test.ts
tests/integration/wallet-connection.test.ts
tests/integration/faucet-claim.test.ts
tests/integration/role-registration.test.ts
```

---

## FAZE 2: ACCESSIBILITY & MOBILE (2 Hafta)

### 2.1 ARIA Attributes Ekleme

**Navbar.tsx**:
```typescript
// Dropdown menu
<button 
  aria-expanded={isOpen}
  aria-haspopup="menu"
  aria-label="More options"
>
  More
</button>

// Active link
<Link
  href={link.href}
  aria-current={isActive(link.href) ? "page" : undefined}
>
  {link.label}
</Link>
```

**Tüm Icon Buttons**:
```typescript
<button aria-label="Toggle theme">
  {resolvedTheme === "dark" ? <Sun /> : <Moon />}
</button>
```

### 2.2 Keyboard Navigation

**Mobile Menu**:
```typescript
// Escape key handling
useEffect(() => {
  const handleEscape = (e: KeyboardEvent) => {
    if (e.key === 'Escape') setMobileMenuOpen(false);
  };
  window.addEventListener('keydown', handleEscape);
  return () => window.removeEventListener('keydown', handleEscape);
}, []);

// Focus trap
useEffect(() => {
  if (mobileMenuOpen) {
    const focusableElements = menuRef.current?.querySelectorAll(
      'button, [href], input, select, textarea'
    );
    // Focus trap logic
  }
}, [mobileMenuOpen]);
```

### 2.3 Mobile UX Improvements

**Touch Targets**: Minimum 44x44px
```css
button, a {
  min-height: 44px;
  min-width: 44px;
  padding: 12px 16px; /* Ensure 44px minimum */
}
```

**Mobile Menu Scroll Lock**:
```typescript
useEffect(() => {
  if (mobileMenuOpen) {
    document.body.style.overflow = 'hidden';
  } else {
    document.body.style.overflow = 'unset';
  }
  return () => { document.body.style.overflow = 'unset'; };
}, [mobileMenuOpen]);
```

### 2.4 Color Contrast Fixes

**WCAG AA Compliance** (4.5:1 minimum)
- Text colors review ve update
- Disabled state colors iyileştir
- Dark mode contrast check

---

## FAZE 3: DOCUMENTATION & SEO (2 Hafta)

### 3.1 Code Documentation

**JSDoc Template**:
```typescript
/**
 * Fetches user information for a wallet address
 * 
 * @param walletAddress - The wallet address to fetch info for
 * @returns Promise resolving to user info or null if not found
 * @throws {APIError} If the request fails
 * 
 * @example
 * const userInfo = await getUserInfo('remes1...');
 * console.log(userInfo.credits);
 */
export async function getUserInfo(walletAddress: string): Promise<UserInfo | null> {
  // ...
}
```

**Dosyalar**:
- `web-dashboard/lib/api.ts` - Tüm functions'ta JSDoc ekle
- `web-dashboard/hooks/*.ts` - Hook documentation ekle
- `web-dashboard/components/*.tsx` - Component documentation ekle

### 3.2 User Documentation

**Yeni Dosyalar**:
- `docs/FEATURES.md` - Feature descriptions
- `docs/API_INTEGRATION.md` - Backend API integration guide
- `docs/DEPLOYMENT.md` - Production deployment guide
- `docs/TROUBLESHOOTING.md` - Common issues ve solutions

### 3.3 Meta Tags Implementation

**Page-specific Meta Tags**:
```typescript
// app/mine/page.tsx
export const metadata: Metadata = {
  title: "Mining Dashboard | R3MES",
  description: "Monitor your mining performance and earnings",
  openGraph: {
    title: "Mining Dashboard | R3MES",
    description: "Monitor your mining performance and earnings",
    url: "https://r3mes.network/mine",
  },
};
```

### 3.4 Structured Data

**JSON-LD Schema**:
```typescript
// app/layout.tsx
<script
  type="application/ld+json"
  dangerouslySetInnerHTML={{
    __html: JSON.stringify({
      "@context": "https://schema.org",
      "@type": "WebApplication",
      "name": "R3MES Dashboard",
      "description": "Decentralized AI training network",
      "url": "https://r3mes.network",
    }),
  }}
/>
```

---

## FAZE 4: PERFORMANCE & MONITORING (2 Hafta)

### 4.1 Bundle Size Optimization

**Analyze Bundle**:
```bash
npm run analyze
```

**Lazy Load Heavy Components**:
```typescript
// Already done for:
// - react-globe.gl (Network page)
// - recharts (Mine/Analytics pages)
// - three.js (Marketing pages)

// Verify and optimize further
```

### 4.2 Runtime Performance

**Scroll Listener Debounce**:
```typescript
// Navbar.tsx
const [scrolled, setScrolled] = useState(false);

useEffect(() => {
  let timeoutId: NodeJS.Timeout;
  const handleScroll = () => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => {
      setScrolled(window.scrollY > 20);
    }, 100); // Debounce 100ms
  };
  
  window.addEventListener("scroll", handleScroll);
  return () => {
    window.removeEventListener("scroll", handleScroll);
    clearTimeout(timeoutId);
  };
}, []);
```

**React Query Optimization**:
```typescript
// Reduce refetch intervals for non-critical data
const { data: minerStats } = useQuery({
  queryKey: ["minerStats", walletAddress],
  queryFn: () => getMinerStats(walletAddress),
  refetchInterval: 30000, // Increase from 10s to 30s
  staleTime: 10000, // Increase from 5s to 10s
});
```

### 4.3 Error Monitoring Setup

**Sentry Configuration**:
```typescript
// sentry.client.config.ts
import * as Sentry from "@sentry/nextjs";

Sentry.init({
  dsn: process.env.NEXT_PUBLIC_SENTRY_DSN,
  environment: process.env.NODE_ENV,
  tracesSampleRate: process.env.NODE_ENV === 'production' ? 0.1 : 1.0,
  integrations: [
    new Sentry.Replay({
      maskAllText: true,
      blockAllMedia: true,
    }),
  ],
});
```

### 4.4 Performance Monitoring

**Web Vitals Tracking**:
```typescript
// lib/analytics.ts
import { getCLS, getFID, getFCP, getLCP, getTTFB } from 'web-vitals';

export function reportWebVitals() {
  getCLS(metric => console.log('CLS:', metric.value));
  getFID(metric => console.log('FID:', metric.value));
  getFCP(metric => console.log('FCP:', metric.value));
  getLCP(metric => console.log('LCP:', metric.value));
  getTTFB(metric => console.log('TTFB:', metric.value));
}
```

---

## IMPLEMENTATION CHECKLIST

### Faz 1 Checklist
- [ ] Validation utility ekle
- [ ] API error handling standardize et
- [ ] CSRF protection implement et
- [ ] Signature verification backend'de implement et
- [ ] Test coverage %80'e çıkar
- [ ] CI/CD'ye test coverage check ekle

### Faz 2 Checklist
- [ ] ARIA attributes ekle (Navbar, buttons, forms)
- [ ] Keyboard navigation implement et
- [ ] Mobile menu scroll lock ekle
- [ ] Touch targets 44x44px'e çıkar
- [ ] Color contrast WCAG AA'ya uygun et
- [ ] Accessibility tests ekle

### Faz 3 Checklist
- [ ] JSDoc tüm functions'ta ekle
- [ ] User documentation yazı
- [ ] API integration guide yazı
- [ ] Deployment guide yazı
- [ ] Meta tags page-specific et
- [ ] JSON-LD structured data ekle
- [ ] Sitemap dynamic et

### Faz 4 Checklist
- [ ] Bundle size analyze et
- [ ] Heavy components lazy load et
- [ ] Scroll listener debounce et
- [ ] React Query intervals optimize et
- [ ] Sentry setup et
- [ ] Web Vitals tracking ekle
- [ ] Performance monitoring dashboard ekle

---

## KAYNAKLAR VE REFERANSLAR

- **WCAG 2.1 Guidelines**: https://www.w3.org/WAI/WCAG21/quickref/
- **Next.js Performance**: https://nextjs.org/docs/advanced-features/measuring-performance
- **React Query**: https://tanstack.com/query/latest
- **Sentry**: https://docs.sentry.io/platforms/javascript/guides/nextjs/
- **Web Vitals**: https://web.dev/vitals/

---

**Hazırlayan**: Context Gathering Agent  
**Tarih**: 2025-01-15  
**Versiyon**: 1.0
