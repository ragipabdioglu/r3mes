// Virtual scrolling hook for large lists

import { useState, useEffect, useMemo, useCallback } from 'react';

interface VirtualizationOptions {
  itemHeight: number;
  containerHeight: number;
  overscan?: number; // Number of items to render outside visible area
  scrollingDelay?: number; // Delay before stopping scroll state
}

interface VirtualItem {
  index: number;
  start: number;
  end: number;
}

export function useVirtualization<T>(
  items: T[],
  options: VirtualizationOptions
) {
  const {
    itemHeight,
    containerHeight,
    overscan = 5,
    scrollingDelay = 150,
  } = options;

  const [scrollTop, setScrollTop] = useState(0);
  const [isScrolling, setIsScrolling] = useState(false);

  // Calculate visible range
  const visibleRange = useMemo(() => {
    const startIndex = Math.floor(scrollTop / itemHeight);
    const endIndex = Math.min(
      startIndex + Math.ceil(containerHeight / itemHeight),
      items.length - 1
    );

    return {
      start: Math.max(0, startIndex - overscan),
      end: Math.min(items.length - 1, endIndex + overscan),
    };
  }, [scrollTop, itemHeight, containerHeight, overscan, items.length]);

  // Generate virtual items
  const virtualItems = useMemo(() => {
    const virtualItemsList: VirtualItem[] = [];
    for (let i = visibleRange.start; i <= visibleRange.end; i++) {
      virtualItemsList.push({
        index: i,
        start: i * itemHeight,
        end: (i + 1) * itemHeight,
      });
    }
    return virtualItemsList;
  }, [visibleRange, itemHeight]);

  // Total height for scrollbar
  const totalHeight = items.length * itemHeight;

  // Scroll handler with debouncing
  const handleScroll = useCallback((scrollTop: number) => {
    setScrollTop(scrollTop);
    setIsScrolling(true);

    // Debounce scrolling state
    const timeoutId = setTimeout(() => {
      setIsScrolling(false);
    }, scrollingDelay);

    return () => clearTimeout(timeoutId);
  }, [scrollingDelay]);

  return {
    virtualItems,
    totalHeight,
    isScrolling,
    handleScroll,
    visibleRange,
  };
}

// Hook for infinite scrolling
export function useInfiniteScroll<T>(
  items: T[],
  loadMore: () => Promise<void>,
  options: {
    threshold?: number; // Distance from bottom to trigger load
    hasMore: boolean;
    isLoading: boolean;
  }
) {
  const { threshold = 100, hasMore, isLoading } = options;
  const [containerRef, setContainerRef] = useState<HTMLElement | null>(null);

  useEffect(() => {
    if (!containerRef || !hasMore || isLoading) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = containerRef;
      const distanceFromBottom = scrollHeight - scrollTop - clientHeight;

      if (distanceFromBottom < threshold) {
        loadMore();
      }
    };

    containerRef.addEventListener('scroll', handleScroll);
    return () => containerRef.removeEventListener('scroll', handleScroll);
  }, [containerRef, hasMore, isLoading, threshold, loadMore]);

  return { containerRef: setContainerRef };
}

// Hook for optimized list rendering with recycling
export function useRecycledList<T>(
  items: T[],
  options: {
    itemHeight: number;
    containerHeight: number;
  }
) {
  const { itemHeight, containerHeight } = options;
  const [scrollTop, setScrollTop] = useState(0);

  // Calculate visible items
  const startIndex = Math.floor(scrollTop / itemHeight);
  const endIndex = Math.min(
    startIndex + Math.ceil(containerHeight / itemHeight) + 1,
    items.length
  );

  const visibleItems = items.slice(startIndex, endIndex);
  const offsetY = startIndex * itemHeight;
  const totalHeight = items.length * itemHeight;

  const handleScroll = useCallback((scrollTopValue: number) => {
    setScrollTop(scrollTopValue);
  }, []);

  // Return data for component to use
  return {
    visibleItems,
    offsetY,
    totalHeight,
    handleScroll,
    startIndex,
    endIndex,
    getItemStyle: (index: number) => ({
      position: 'absolute' as const,
      top: offsetY + index * itemHeight,
      height: itemHeight,
      width: '100%',
    }),
  };
}