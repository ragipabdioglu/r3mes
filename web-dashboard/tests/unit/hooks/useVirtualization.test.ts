/**
 * Unit tests for useVirtualization hook
 */

import { renderHook, act } from '@testing-library/react';
import { useVirtualization, useInfiniteScroll, useRecycledList } from '@/hooks/useVirtualization';

describe('useVirtualization', () => {
  const mockItems = Array.from({ length: 100 }, (_, i) => ({ id: i, name: `Item ${i}` }));
  const defaultOptions = {
    itemHeight: 50,
    containerHeight: 300,
    overscan: 5,
  };

  it('should calculate visible range correctly', () => {
    const { result } = renderHook(() => useVirtualization(mockItems, defaultOptions));

    // Initial state should show items from start
    expect(result.current.visibleRange.start).toBe(0);
    // Should show enough items to fill container + overscan
    expect(result.current.visibleRange.end).toBeGreaterThan(0);
  });

  it('should generate virtual items', () => {
    const { result } = renderHook(() => useVirtualization(mockItems, defaultOptions));

    expect(result.current.virtualItems.length).toBeGreaterThan(0);
    expect(result.current.virtualItems[0]).toHaveProperty('index');
    expect(result.current.virtualItems[0]).toHaveProperty('start');
    expect(result.current.virtualItems[0]).toHaveProperty('end');
  });

  it('should calculate total height correctly', () => {
    const { result } = renderHook(() => useVirtualization(mockItems, defaultOptions));

    expect(result.current.totalHeight).toBe(mockItems.length * defaultOptions.itemHeight);
  });

  it('should update visible range on scroll', () => {
    const { result } = renderHook(() => useVirtualization(mockItems, defaultOptions));

    act(() => {
      result.current.handleScroll(500); // Scroll down 500px
    });

    // After scrolling, start index should be greater than 0
    expect(result.current.visibleRange.start).toBeGreaterThan(0);
  });

  it('should set isScrolling state during scroll', () => {
    jest.useFakeTimers();
    
    const { result } = renderHook(() => useVirtualization(mockItems, {
      ...defaultOptions,
      scrollingDelay: 150,
    }));

    act(() => {
      result.current.handleScroll(100);
    });

    expect(result.current.isScrolling).toBe(true);

    // After delay, isScrolling should be false
    act(() => {
      jest.advanceTimersByTime(200);
    });

    expect(result.current.isScrolling).toBe(false);

    jest.useRealTimers();
  });

  it('should handle empty items array', () => {
    const { result } = renderHook(() => useVirtualization([], defaultOptions));

    expect(result.current.virtualItems).toEqual([]);
    expect(result.current.totalHeight).toBe(0);
  });

  it('should respect overscan option', () => {
    const { result: resultWithOverscan } = renderHook(() => 
      useVirtualization(mockItems, { ...defaultOptions, overscan: 10 })
    );
    
    const { result: resultWithoutOverscan } = renderHook(() => 
      useVirtualization(mockItems, { ...defaultOptions, overscan: 0 })
    );

    // With overscan, should have more virtual items
    expect(resultWithOverscan.current.virtualItems.length)
      .toBeGreaterThanOrEqual(resultWithoutOverscan.current.virtualItems.length);
  });
});

describe('useRecycledList', () => {
  const mockItems = Array.from({ length: 50 }, (_, i) => ({ id: i }));
  const options = {
    itemHeight: 40,
    containerHeight: 200,
  };

  it('should return visible items', () => {
    const { result } = renderHook(() => useRecycledList(mockItems, options));

    expect(result.current.visibleItems.length).toBeGreaterThan(0);
    expect(result.current.visibleItems.length).toBeLessThanOrEqual(mockItems.length);
  });

  it('should calculate offset correctly', () => {
    const { result } = renderHook(() => useRecycledList(mockItems, options));

    // Initial offset should be 0
    expect(result.current.offsetY).toBe(0);

    // After scrolling
    act(() => {
      result.current.handleScroll(200);
    });

    expect(result.current.offsetY).toBeGreaterThan(0);
  });

  it('should provide getItemStyle function', () => {
    const { result } = renderHook(() => useRecycledList(mockItems, options));

    const style = result.current.getItemStyle(0);
    
    expect(style).toHaveProperty('position', 'absolute');
    expect(style).toHaveProperty('height', options.itemHeight);
    expect(style).toHaveProperty('width', '100%');
  });

  it('should track start and end indices', () => {
    const { result } = renderHook(() => useRecycledList(mockItems, options));

    expect(result.current.startIndex).toBe(0);
    expect(result.current.endIndex).toBeGreaterThan(0);
    expect(result.current.endIndex).toBeLessThanOrEqual(mockItems.length);
  });
});

describe('useInfiniteScroll', () => {
  const mockLoadMore = jest.fn().mockResolvedValue(undefined);
  const mockItems = Array.from({ length: 20 }, (_, i) => ({ id: i }));

  beforeEach(() => {
    mockLoadMore.mockClear();
  });

  it('should return containerRef setter', () => {
    const { result } = renderHook(() => 
      useInfiniteScroll(mockItems, mockLoadMore, {
        hasMore: true,
        isLoading: false,
      })
    );

    expect(result.current.containerRef).toBeDefined();
    expect(typeof result.current.containerRef).toBe('function');
  });

  it('should not call loadMore when isLoading is true', () => {
    renderHook(() => 
      useInfiniteScroll(mockItems, mockLoadMore, {
        hasMore: true,
        isLoading: true,
      })
    );

    expect(mockLoadMore).not.toHaveBeenCalled();
  });

  it('should not call loadMore when hasMore is false', () => {
    renderHook(() => 
      useInfiniteScroll(mockItems, mockLoadMore, {
        hasMore: false,
        isLoading: false,
      })
    );

    expect(mockLoadMore).not.toHaveBeenCalled();
  });
});
