/**
 * Accessibility hooks for R3MES Web Dashboard
 * Custom React hooks for accessibility features
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { 
  announceToScreenReader, 
  trapFocus, 
  prefersReducedMotion, 
  prefersHighContrast,
  generateId 
} from '@/utils/accessibility';

// Hook for managing focus trap in modals/dialogs
export const useFocusTrap = (isActive: boolean) => {
  const containerRef = useRef<HTMLElement>(null);
  const cleanupRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    if (isActive && containerRef.current) {
      cleanupRef.current = trapFocus(containerRef.current);
    } else if (cleanupRef.current) {
      cleanupRef.current();
      cleanupRef.current = null;
    }

    return () => {
      if (cleanupRef.current) {
        cleanupRef.current();
      }
    };
  }, [isActive]);

  return containerRef;
};

// Hook for managing screen reader announcements
export const useAnnouncer = () => {
  const announce = useCallback((
    message: string, 
    priority: 'polite' | 'assertive' = 'polite',
    timeout?: number
  ) => {
    announceToScreenReader(message, priority, timeout);
  }, []);

  const announceError = useCallback((message: string) => {
    announce(message, 'assertive', 3000);
  }, [announce]);

  const announceSuccess = useCallback((message: string) => {
    announce(message, 'polite', 2000);
  }, [announce]);

  const announceLoading = useCallback((context: string, loading: boolean) => {
    if (loading) {
      announce(`Loading ${context}`, 'polite');
    } else {
      announce(`${context} loaded`, 'polite');
    }
  }, [announce]);

  return {
    announce,
    announceError,
    announceSuccess,
    announceLoading,
  };
};

// Hook for keyboard navigation in lists/menus
export const useKeyboardNavigation = (
  items: HTMLElement[],
  options: {
    loop?: boolean;
    orientation?: 'horizontal' | 'vertical';
    onSelect?: (index: number) => void;
  } = {}
) => {
  const [activeIndex, setActiveIndex] = useState(-1);
  const { loop = true, orientation = 'vertical', onSelect } = options;

  const handleKeyDown = useCallback((event: KeyboardEvent) => {
    const { key } = event;
    let newIndex = activeIndex;

    switch (key) {
      case orientation === 'vertical' ? 'ArrowDown' : 'ArrowRight':
        event.preventDefault();
        newIndex = activeIndex + 1;
        if (newIndex >= items.length) {
          newIndex = loop ? 0 : items.length - 1;
        }
        break;

      case orientation === 'vertical' ? 'ArrowUp' : 'ArrowLeft':
        event.preventDefault();
        newIndex = activeIndex - 1;
        if (newIndex < 0) {
          newIndex = loop ? items.length - 1 : 0;
        }
        break;

      case 'Home':
        event.preventDefault();
        newIndex = 0;
        break;

      case 'End':
        event.preventDefault();
        newIndex = items.length - 1;
        break;

      case 'Enter':
      case ' ':
        event.preventDefault();
        if (activeIndex >= 0 && onSelect) {
          onSelect(activeIndex);
        }
        break;

      default:
        return;
    }

    if (newIndex !== activeIndex && newIndex >= 0 && newIndex < items.length) {
      setActiveIndex(newIndex);
      items[newIndex]?.focus();
    }
  }, [activeIndex, items, loop, orientation, onSelect]);

  useEffect(() => {
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  return {
    activeIndex,
    setActiveIndex,
  };
};

// Hook for managing ARIA attributes
export const useAriaAttributes = (initialState: Record<string, string | boolean> = {}) => {
  const [attributes, setAttributes] = useState(initialState);
  const elementRef = useRef<HTMLElement>(null);

  const updateAttribute = useCallback((name: string, value: string | boolean) => {
    setAttributes(prev => ({ ...prev, [name]: value }));
  }, []);

  const toggleAttribute = useCallback((name: string) => {
    setAttributes(prev => ({ 
      ...prev, 
      [name]: prev[name] === 'true' ? 'false' : 'true' 
    }));
  }, []);

  useEffect(() => {
    if (elementRef.current) {
      Object.entries(attributes).forEach(([name, value]) => {
        elementRef.current!.setAttribute(`aria-${name}`, value.toString());
      });
    }
  }, [attributes]);

  return {
    elementRef,
    attributes,
    updateAttribute,
    toggleAttribute,
  };
};

// Hook for detecting user preferences
export const useAccessibilityPreferences = () => {
  const [preferences, setPreferences] = useState({
    reducedMotion: false,
    highContrast: false,
  });

  useEffect(() => {
    const updatePreferences = () => {
      setPreferences({
        reducedMotion: prefersReducedMotion(),
        highContrast: prefersHighContrast(),
      });
    };

    updatePreferences();

    const reducedMotionQuery = window.matchMedia('(prefers-reduced-motion: reduce)');
    const highContrastQuery = window.matchMedia('(prefers-contrast: high)');

    reducedMotionQuery.addEventListener('change', updatePreferences);
    highContrastQuery.addEventListener('change', updatePreferences);

    return () => {
      reducedMotionQuery.removeEventListener('change', updatePreferences);
      highContrastQuery.removeEventListener('change', updatePreferences);
    };
  }, []);

  return preferences;
};

// Hook for managing unique IDs for accessibility
export const useAccessibleId = (prefix: string = 'element') => {
  const [id] = useState(() => generateId(prefix));
  return id;
};

// Hook for managing live regions
export const useLiveRegion = (initialMessage: string = '') => {
  const [message, setMessage] = useState(initialMessage);
  const [priority, setPriority] = useState<'polite' | 'assertive'>('polite');
  const regionRef = useRef<HTMLDivElement>(null);

  const announce = useCallback((
    newMessage: string, 
    newPriority: 'polite' | 'assertive' = 'polite'
  ) => {
    setMessage(newMessage);
    setPriority(newPriority);
  }, []);

  const clear = useCallback(() => {
    setMessage('');
  }, []);

  useEffect(() => {
    if (regionRef.current) {
      regionRef.current.setAttribute('aria-live', priority);
      regionRef.current.setAttribute('aria-atomic', 'true');
    }
  }, [priority]);

  return {
    regionRef,
    message,
    announce,
    clear,
    priority,
  };
};

// Hook for managing skip links
export const useSkipLinks = (links: Array<{ target: string; label: string }>) => {
  const skipLinksRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (skipLinksRef.current) {
      skipLinksRef.current.innerHTML = '';
      
      links.forEach(({ target, label }) => {
        const link = document.createElement('a');
        link.href = `#${target}`;
        link.textContent = label;
        link.className = 'sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 z-[9999] px-4 py-2 rounded-md font-medium transition-all';
        link.style.backgroundColor = 'var(--accent-primary)';
        link.style.color = 'white';
        
        skipLinksRef.current!.appendChild(link);
      });
    }
  }, [links]);

  return skipLinksRef;
};

// Hook for managing form accessibility
export const useFormAccessibility = () => {
  const { announceError, announceSuccess } = useAnnouncer();

  const announceFieldError = useCallback((fieldName: string, error: string) => {
    announceError(`Error in ${fieldName}: ${error}`);
  }, [announceError]);

  const announceFormSubmission = useCallback((success: boolean, message?: string) => {
    if (success) {
      announceSuccess(message || 'Form submitted successfully');
    } else {
      announceError(message || 'Form submission failed');
    }
  }, [announceSuccess, announceError]);

  const getFieldProps = useCallback((
    fieldName: string,
    error?: string,
    description?: string
  ) => {
    const fieldId = generateId(fieldName);
    const errorId = error ? generateId(`${fieldName}-error`) : undefined;
    const descriptionId = description ? generateId(`${fieldName}-description`) : undefined;

    const describedBy = [errorId, descriptionId].filter(Boolean).join(' ');

    return {
      id: fieldId,
      'aria-invalid': error ? 'true' : 'false',
      'aria-describedby': describedBy || undefined,
      errorId,
      descriptionId,
    };
  }, []);

  return {
    announceFieldError,
    announceFormSubmission,
    getFieldProps,
  };
};

// Hook for managing loading states with accessibility
export const useAccessibleLoading = (initialState: boolean = false) => {
  const [loadingState, setLoadingState] = useState(initialState);
  const { announceLoading } = useAnnouncer();
  const loadingElementId = useAccessibleId('loading');

  const updateLoading = useCallback((active: boolean, context?: string) => {
    setLoadingState(active);
    if (context) {
      announceLoading(context, active);
    }
  }, [announceLoading]);

  const getLoadingAttributes = useCallback(() => ({
    'aria-busy': loadingState,
    'aria-describedby': loadingState ? loadingElementId : undefined,
  }), [loadingState, loadingElementId]);

  const getLoadingAnnouncementProps = useCallback(() => ({
    id: loadingElementId,
    'aria-live': 'polite' as const,
    'aria-atomic': 'true',
    className: loadingState ? 'sr-only' : 'hidden'
  }), [loadingElementId, loadingState]);

  return {
    isLoading: loadingState,
    setLoading: updateLoading,
    getLoadingProps: getLoadingAttributes,
    getLoadingAnnouncementProps,
    loadingElementId,
  };
};