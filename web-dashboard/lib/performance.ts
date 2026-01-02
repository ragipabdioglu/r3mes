// Performance monitoring and optimization utilities

export interface PerformanceMetrics {
  fcp: number; // First Contentful Paint
  lcp: number; // Largest Contentful Paint
  fid: number; // First Input Delay
  cls: number; // Cumulative Layout Shift
  ttfb: number; // Time to First Byte
}

export interface ResourceTiming {
  name: string;
  duration: number;
  size: number;
  type: 'script' | 'stylesheet' | 'image' | 'font' | 'other';
}

class PerformanceMonitor {
  private metrics: Partial<PerformanceMetrics> = {};
  private observers: PerformanceObserver[] = [];

  constructor() {
    if (typeof window !== 'undefined') {
      this.initializeObservers();
    }
  }

  private initializeObservers() {
    // Observe Core Web Vitals
    if ('PerformanceObserver' in window) {
      // LCP Observer
      const lcpObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        const lastEntry = entries[entries.length - 1] as any;
        this.metrics.lcp = lastEntry.startTime;
      });
      lcpObserver.observe({ entryTypes: ['largest-contentful-paint'] });
      this.observers.push(lcpObserver);

      // FID Observer
      const fidObserver = new PerformanceObserver((list) => {
        const entries = list.getEntries();
        entries.forEach((entry: any) => {
          this.metrics.fid = entry.processingStart - entry.startTime;
        });
      });
      fidObserver.observe({ entryTypes: ['first-input'] });
      this.observers.push(fidObserver);

      // CLS Observer
      const clsObserver = new PerformanceObserver((list) => {
        let clsValue = 0;
        const entries = list.getEntries();
        entries.forEach((entry: any) => {
          if (!entry.hadRecentInput) {
            clsValue += entry.value;
          }
        });
        this.metrics.cls = clsValue;
      });
      clsObserver.observe({ entryTypes: ['layout-shift'] });
      this.observers.push(clsObserver);
    }

    // FCP and TTFB from Navigation Timing
    window.addEventListener('load', () => {
      const navigation = performance.getEntriesByType('navigation')[0] as PerformanceNavigationTiming;
      this.metrics.ttfb = navigation.responseStart - navigation.requestStart;

      // FCP from Paint Timing
      const paintEntries = performance.getEntriesByType('paint');
      const fcpEntry = paintEntries.find(entry => entry.name === 'first-contentful-paint');
      if (fcpEntry) {
        this.metrics.fcp = fcpEntry.startTime;
      }
    });
  }

  getMetrics(): Partial<PerformanceMetrics> {
    return { ...this.metrics };
  }

  getResourceTimings(): ResourceTiming[] {
    if (typeof window === 'undefined') return [];

    const resources = performance.getEntriesByType('resource') as PerformanceResourceTiming[];
    return resources.map(resource => ({
      name: resource.name,
      duration: resource.duration,
      size: resource.transferSize || 0,
      type: this.getResourceType(resource.name),
    }));
  }

  private getResourceType(url: string): ResourceTiming['type'] {
    if (url.includes('.js')) return 'script';
    if (url.includes('.css')) return 'stylesheet';
    if (url.match(/\.(jpg|jpeg|png|gif|webp|svg)$/)) return 'image';
    if (url.match(/\.(woff|woff2|ttf|otf)$/)) return 'font';
    return 'other';
  }

  // Performance budget checker
  checkPerformanceBudget(): {
    passed: boolean;
    violations: string[];
  } {
    const violations: string[] = [];
    const metrics = this.getMetrics();

    // Core Web Vitals thresholds
    if (metrics.lcp && metrics.lcp > 2500) {
      violations.push(`LCP too slow: ${metrics.lcp}ms (should be < 2500ms)`);
    }
    if (metrics.fid && metrics.fid > 100) {
      violations.push(`FID too slow: ${metrics.fid}ms (should be < 100ms)`);
    }
    if (metrics.cls && metrics.cls > 0.1) {
      violations.push(`CLS too high: ${metrics.cls} (should be < 0.1)`);
    }
    if (metrics.fcp && metrics.fcp > 1800) {
      violations.push(`FCP too slow: ${metrics.fcp}ms (should be < 1800ms)`);
    }
    if (metrics.ttfb && metrics.ttfb > 600) {
      violations.push(`TTFB too slow: ${metrics.ttfb}ms (should be < 600ms)`);
    }

    return {
      passed: violations.length === 0,
      violations,
    };
  }

  // Send metrics to analytics
  sendMetrics() {
    const metrics = this.getMetrics();
    const budget = this.checkPerformanceBudget();

    // Send to analytics service (e.g., Google Analytics, custom endpoint)
    if (typeof window !== 'undefined' && (window as any).gtag) {
      (window as any).gtag('event', 'performance_metrics', {
        custom_map: {
          lcp: metrics.lcp,
          fid: metrics.fid,
          cls: metrics.cls,
          fcp: metrics.fcp,
          ttfb: metrics.ttfb,
          budget_passed: budget.passed,
        },
      });
    }
  }

  cleanup() {
    this.observers.forEach(observer => observer.disconnect());
    this.observers = [];
  }
}

// Image optimization utilities
export class ImageOptimizer {
  static getOptimalFormat(userAgent: string): 'webp' | 'avif' | 'jpeg' {
    if (userAgent.includes('Chrome') && !userAgent.includes('Edge')) {
      return 'avif'; // Chrome supports AVIF
    }
    if (userAgent.includes('Chrome') || userAgent.includes('Firefox') || userAgent.includes('Safari')) {
      return 'webp';
    }
    return 'jpeg';
  }

  static generateSrcSet(basePath: string, sizes: number[]): string {
    return sizes
      .map(size => `${basePath}?w=${size}&q=75 ${size}w`)
      .join(', ');
  }

  static generateSizes(breakpoints: { [key: string]: number }): string {
    const entries = Object.entries(breakpoints);
    return entries
      .map(([media, size], index) => {
        if (index === entries.length - 1) {
          return `${size}px`; // Default size
        }
        return `(max-width: ${media}) ${size}px`;
      })
      .join(', ');
  }
}

// Code splitting utilities
export class CodeSplitter {
  private static loadedChunks = new Set<string>();

  static async loadChunk(chunkName: string): Promise<any> {
    if (this.loadedChunks.has(chunkName)) {
      return Promise.resolve();
    }

    try {
      let module;
      switch (chunkName) {
        case 'chart':
          module = await import('recharts');
          break;
        case 'wallet':
          module = await import('@/components/WalletButton');
          break;
        case 'mining':
          module = await import('@/components/MinersTable');
          break;
        default:
          throw new Error(`Unknown chunk: ${chunkName}`);
      }

      this.loadedChunks.add(chunkName);
      return module;
    } catch (error) {
      console.error(`Failed to load chunk ${chunkName}:`, error);
      throw error;
    }
  }

  static preloadChunk(chunkName: string) {
    // Preload chunk without blocking
    this.loadChunk(chunkName).catch(() => {
      // Silently fail for preloading
    });
  }
}

// Memory management
export class MemoryManager {
  private static intervals = new Set<NodeJS.Timeout>();
  private static observers = new Set<IntersectionObserver | MutationObserver>();

  static addInterval(interval: NodeJS.Timeout) {
    this.intervals.add(interval);
  }

  static addObserver(observer: IntersectionObserver | MutationObserver) {
    this.observers.add(observer);
  }

  static cleanup() {
    // Clear all intervals
    this.intervals.forEach(interval => clearInterval(interval));
    this.intervals.clear();

    // Disconnect all observers
    this.observers.forEach(observer => observer.disconnect());
    this.observers.clear();
  }

  static getMemoryUsage(): {
    used: number;
    total: number;
    percentage: number;
  } | null {
    if (typeof window === 'undefined' || !(performance as any).memory) {
      return null;
    }

    const memory = (performance as any).memory;
    return {
      used: memory.usedJSHeapSize,
      total: memory.totalJSHeapSize,
      percentage: (memory.usedJSHeapSize / memory.totalJSHeapSize) * 100,
    };
  }
}

// Export singleton instance
export const performanceMonitor = new PerformanceMonitor();

// Cleanup on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    performanceMonitor.cleanup();
    MemoryManager.cleanup();
  });
}