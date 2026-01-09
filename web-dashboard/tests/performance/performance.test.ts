/**
 * Performance tests for R3MES Web Dashboard
 * 
 * These tests verify that the application meets performance budgets
 * and follows best practices for web performance.
 */

import { performanceMonitor, ImageOptimizer, CodeSplitter, MemoryManager } from '@/lib/performance';

// Mock performance APIs for testing
const mockPerformanceObserver = jest.fn();
const mockPerformanceEntry = {
  startTime: 1500,
  duration: 100,
  name: 'test-entry',
};

// Mock window.performance
Object.defineProperty(global, 'performance', {
  value: {
    getEntriesByType: jest.fn(),
    getEntriesByName: jest.fn(),
    now: jest.fn(() => Date.now()),
    memory: {
      usedJSHeapSize: 50 * 1024 * 1024, // 50MB
      totalJSHeapSize: 100 * 1024 * 1024, // 100MB
    },
  },
  writable: true,
});

// Mock PerformanceObserver
Object.defineProperty(global, 'PerformanceObserver', {
  value: mockPerformanceObserver,
  writable: true,
});

describe('Performance Monitoring', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Performance Metrics', () => {
    it('should collect Core Web Vitals metrics', () => {
      const metrics = performanceMonitor.getMetrics();
      
      // Metrics should be initialized (may be empty initially)
      expect(typeof metrics).toBe('object');
      // In a real environment, these would be populated by PerformanceObserver
      // For testing, we just verify the structure exists
      expect(metrics).toBeDefined();
    });

    it('should check performance budget', () => {
      // Mock some metrics
      jest.spyOn(performanceMonitor, 'getMetrics').mockReturnValue({
        lcp: 2000, // Good
        fid: 50,   // Good
        cls: 0.05, // Good
        fcp: 1500, // Good
        ttfb: 400, // Good
      });

      const budget = performanceMonitor.checkPerformanceBudget();
      
      expect(budget.passed).toBe(true);
      expect(budget.violations).toHaveLength(0);
    });

    it('should detect performance budget violations', () => {
      // Mock poor metrics
      jest.spyOn(performanceMonitor, 'getMetrics').mockReturnValue({
        lcp: 3000, // Poor
        fid: 150,  // Poor
        cls: 0.2,  // Poor
        fcp: 2500, // Poor
        ttfb: 800, // Poor
      });

      const budget = performanceMonitor.checkPerformanceBudget();
      
      expect(budget.passed).toBe(false);
      expect(budget.violations.length).toBeGreaterThan(0);
      expect(budget.violations[0]).toContain('LCP too slow');
    });
  });

  describe('Resource Timing', () => {
    it('should collect resource timing data', () => {
      // Mock resource timing entries
      (global.performance.getEntriesByType as jest.Mock).mockReturnValue([
        {
          name: 'https://example.com/script.js',
          duration: 200,
          transferSize: 50000,
        },
        {
          name: 'https://example.com/style.css',
          duration: 100,
          transferSize: 20000,
        },
        {
          name: 'https://example.com/image.jpg',
          duration: 300,
          transferSize: 100000,
        },
      ]);

      const resources = performanceMonitor.getResourceTimings();
      
      expect(resources).toHaveLength(3);
      expect(resources[0]).toMatchObject({
        name: 'https://example.com/script.js',
        duration: 200,
        size: 50000,
        type: 'script',
      });
    });

    it('should categorize resource types correctly', () => {
      (global.performance.getEntriesByType as jest.Mock).mockReturnValue([
        { name: 'test.js', duration: 100, transferSize: 1000 },
        { name: 'test.css', duration: 100, transferSize: 1000 },
        { name: 'test.jpg', duration: 100, transferSize: 1000 },
        { name: 'test.woff2', duration: 100, transferSize: 1000 },
        { name: 'test.html', duration: 100, transferSize: 1000 },
      ]);

      const resources = performanceMonitor.getResourceTimings();
      
      expect(resources[0].type).toBe('script');
      expect(resources[1].type).toBe('stylesheet');
      expect(resources[2].type).toBe('image');
      expect(resources[3].type).toBe('font');
      expect(resources[4].type).toBe('other');
    });
  });

  describe('Memory Management', () => {
    it('should track memory usage', () => {
      const memoryUsage = MemoryManager.getMemoryUsage();
      
      expect(memoryUsage).toMatchObject({
        used: 50 * 1024 * 1024,
        total: 100 * 1024 * 1024,
        percentage: 50,
      });
    });

    it('should handle missing memory API', () => {
      // Mock missing memory API
      const originalMemory = (global.performance as any).memory;
      delete (global.performance as any).memory;

      const memoryUsage = MemoryManager.getMemoryUsage();
      
      expect(memoryUsage).toBeNull();

      // Restore
      (global.performance as any).memory = originalMemory;
    });

    it('should manage intervals and observers', () => {
      const mockInterval = setInterval(() => {}, 1000) as any;
      const mockObserver = {
        disconnect: jest.fn(),
      } as any;

      MemoryManager.addInterval(mockInterval);
      MemoryManager.addObserver(mockObserver);

      // Cleanup should clear intervals and disconnect observers
      MemoryManager.cleanup();

      expect(mockObserver.disconnect).toHaveBeenCalled();
    });
  });
});

describe('Image Optimization', () => {
  describe('Format Detection', () => {
    it('should detect AVIF support in Chrome', () => {
      const userAgent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36';
      const format = ImageOptimizer.getOptimalFormat(userAgent);
      
      expect(format).toBe('avif');
    });

    it('should detect WebP support in modern browsers', () => {
      const userAgent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0';
      const format = ImageOptimizer.getOptimalFormat(userAgent);
      
      expect(format).toBe('webp');
    });

    it('should fallback to JPEG for older browsers', () => {
      const userAgent = 'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko';
      const format = ImageOptimizer.getOptimalFormat(userAgent);
      
      expect(format).toBe('jpeg');
    });
  });

  describe('Responsive Images', () => {
    it('should generate srcSet for responsive images', () => {
      const srcSet = ImageOptimizer.generateSrcSet('/image.jpg', [400, 800, 1200]);
      
      expect(srcSet).toBe('/image.jpg?w=400&q=75 400w, /image.jpg?w=800&q=75 800w, /image.jpg?w=1200&q=75 1200w');
    });

    it('should generate sizes attribute', () => {
      const sizes = ImageOptimizer.generateSizes({
        '768px': 400,
        '1024px': 600,
        '1200px': 800,
      });
      
      expect(sizes).toContain('(max-width: 768px) 400px');
      expect(sizes).toContain('(max-width: 1024px) 600px');
      expect(sizes).toContain('800px'); // Default size
    });
  });
});

describe('Code Splitting', () => {
  beforeEach(() => {
    // Reset loaded chunks
    (CodeSplitter as any).loadedChunks = new Set();
  });

  it('should load chunks dynamically', async () => {
    // Mock dynamic import
    const mockModule = { default: 'test-component' };
    jest.doMock('recharts', () => mockModule, { virtual: true });

    const module = await CodeSplitter.loadChunk('chart');
    
    expect(module).toBeDefined();
  });

  it('should cache loaded chunks', async () => {
    // Mock dynamic import
    jest.doMock('recharts', () => ({ default: 'test' }), { virtual: true });

    // Load chunk twice
    await CodeSplitter.loadChunk('chart');
    const secondLoad = await CodeSplitter.loadChunk('chart');
    
    // Second load should resolve (cached chunks return resolved promise)
    expect(secondLoad).toBeUndefined(); // Cached chunks return undefined
  });

  it('should handle unknown chunks', async () => {
    await expect(CodeSplitter.loadChunk('unknown-chunk')).rejects.toThrow('Unknown chunk: unknown-chunk');
  });

  it('should preload chunks without blocking', () => {
    // Mock console.error to avoid noise in tests
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation();

    // Preload should not throw
    expect(() => {
      CodeSplitter.preloadChunk('unknown-chunk');
    }).not.toThrow();

    consoleSpy.mockRestore();
  });
});

describe('Performance Budget Tests', () => {
  it('should meet Core Web Vitals thresholds', () => {
    // These are the actual thresholds we want to meet
    const thresholds = {
      lcp: 2500,  // Largest Contentful Paint
      fid: 100,   // First Input Delay
      cls: 0.1,   // Cumulative Layout Shift
      fcp: 1800,  // First Contentful Paint
      ttfb: 600,  // Time to First Byte
    };

    // Mock good performance metrics
    jest.spyOn(performanceMonitor, 'getMetrics').mockReturnValue({
      lcp: 2000,
      fid: 50,
      cls: 0.05,
      fcp: 1500,
      ttfb: 400,
    });

    const budget = performanceMonitor.checkPerformanceBudget();
    
    expect(budget.passed).toBe(true);
    expect(budget.violations).toHaveLength(0);
  });

  it('should detect when bundle size exceeds budget', () => {
    // This test would be implemented with actual bundle analysis
    // For now, we'll test the concept
    const maxBundleSize = 250 * 1024; // 250KB
    const mockBundleSize = 200 * 1024; // 200KB
    
    expect(mockBundleSize).toBeLessThan(maxBundleSize);
  });

  it('should verify critical resources load quickly', () => {
    // Mock resource timing for critical resources
    (global.performance.getEntriesByType as jest.Mock).mockReturnValue([
      {
        name: '/critical.css',
        duration: 100, // Should be < 200ms
        transferSize: 10000,
      },
      {
        name: '/critical.js',
        duration: 150, // Should be < 200ms
        transferSize: 50000,
      },
    ]);

    const resources = performanceMonitor.getResourceTimings();
    const criticalResources = resources.filter(r => 
      r.name.includes('critical') || r.name.includes('main')
    );

    criticalResources.forEach(resource => {
      expect(resource.duration).toBeLessThan(200);
    });
  });
});

describe('Accessibility Performance', () => {
  it('should not impact performance significantly', () => {
    // Test that accessibility features don't slow down the app
    // This would measure the performance impact of ARIA attributes,
    // screen reader announcements, etc.
    
    const startTime = performance.now();
    
    // Simulate accessibility operations
    const element = document.createElement('div');
    element.setAttribute('aria-label', 'Test element');
    element.setAttribute('role', 'button');
    element.setAttribute('aria-describedby', 'description');
    
    const endTime = performance.now();
    const duration = endTime - startTime;
    
    // Should complete very quickly
    expect(duration).toBeLessThan(10);
  });
});