/**
 * Accessibility utilities for R3MES Web Dashboard
 * Provides helper functions for WCAG AA compliance
 */

// Screen reader announcements
export const announceToScreenReader = (
  message: string,
  priority: 'polite' | 'assertive' = 'polite',
  timeout: number = 1000
) => {
  const announcement = document.createElement('div');
  announcement.setAttribute('aria-live', priority);
  announcement.setAttribute('aria-atomic', 'true');
  announcement.className = 'sr-only';
  announcement.textContent = message;
  
  document.body.appendChild(announcement);
  
  setTimeout(() => {
    if (document.body.contains(announcement)) {
      document.body.removeChild(announcement);
    }
  }, timeout);
};

// Focus management
export const trapFocus = (element: HTMLElement) => {
  const focusableElements = element.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  
  const firstElement = focusableElements[0] as HTMLElement;
  const lastElement = focusableElements[focusableElements.length - 1] as HTMLElement;
  
  const handleTabKey = (e: KeyboardEvent) => {
    if (e.key !== 'Tab') return;
    
    if (e.shiftKey) {
      if (document.activeElement === firstElement) {
        e.preventDefault();
        lastElement.focus();
      }
    } else {
      if (document.activeElement === lastElement) {
        e.preventDefault();
        firstElement.focus();
      }
    }
  };
  
  element.addEventListener('keydown', handleTabKey);
  
  // Return cleanup function
  return () => {
    element.removeEventListener('keydown', handleTabKey);
  };
};

// Color contrast utilities
export const getContrastRatio = (color1: string, color2: string): number => {
  const getLuminance = (color: string): number => {
    // Convert hex to RGB
    const hex = color.replace('#', '');
    const r = parseInt(hex.substr(0, 2), 16) / 255;
    const g = parseInt(hex.substr(2, 2), 16) / 255;
    const b = parseInt(hex.substr(4, 2), 16) / 255;
    
    // Calculate relative luminance
    const sRGB = [r, g, b].map(c => {
      return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
    });
    
    return 0.2126 * sRGB[0] + 0.7152 * sRGB[1] + 0.0722 * sRGB[2];
  };
  
  const lum1 = getLuminance(color1);
  const lum2 = getLuminance(color2);
  
  const brightest = Math.max(lum1, lum2);
  const darkest = Math.min(lum1, lum2);
  
  return (brightest + 0.05) / (darkest + 0.05);
};

export const meetsWCAGAA = (color1: string, color2: string): boolean => {
  return getContrastRatio(color1, color2) >= 4.5;
};

export const meetsWCAGAAA = (color1: string, color2: string): boolean => {
  return getContrastRatio(color1, color2) >= 7;
};

// Keyboard navigation helpers
export const isNavigationKey = (key: string): boolean => {
  return ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(key);
};

export const isActionKey = (key: string): boolean => {
  return ['Enter', ' ', 'Space'].includes(key);
};

// ARIA helpers
export const generateId = (prefix: string = 'element'): string => {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

export const setAriaExpanded = (element: HTMLElement, expanded: boolean) => {
  element.setAttribute('aria-expanded', expanded.toString());
};

export const setAriaSelected = (element: HTMLElement, selected: boolean) => {
  element.setAttribute('aria-selected', selected.toString());
};

export const setAriaPressed = (element: HTMLElement, pressed: boolean) => {
  element.setAttribute('aria-pressed', pressed.toString());
};

// Reduced motion detection
export const prefersReducedMotion = (): boolean => {
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
};

// High contrast detection
export const prefersHighContrast = (): boolean => {
  return window.matchMedia('(prefers-contrast: high)').matches;
};

// Touch target size validation
export const validateTouchTarget = (element: HTMLElement): boolean => {
  const rect = element.getBoundingClientRect();
  return rect.width >= 44 && rect.height >= 44;
};

// Screen reader text utilities
export const createScreenReaderText = (text: string): HTMLElement => {
  const span = document.createElement('span');
  span.className = 'sr-only';
  span.textContent = text;
  return span;
};

// Skip link utilities
export const createSkipLink = (targetId: string, text: string): HTMLElement => {
  const link = document.createElement('a');
  link.href = `#${targetId}`;
  link.textContent = text;
  link.className = 'sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 z-[9999] px-4 py-2 rounded-md font-medium transition-all';
  link.style.backgroundColor = 'var(--accent-primary)';
  link.style.color = 'white';
  
  return link;
};

// Form validation helpers
export const announceFormError = (fieldName: string, error: string) => {
  announceToScreenReader(`Error in ${fieldName}: ${error}`, 'assertive', 3000);
};

export const announceFormSuccess = (message: string) => {
  announceToScreenReader(message, 'polite', 2000);
};

// Loading state announcements
export const announceLoadingStart = (context: string) => {
  announceToScreenReader(`Loading ${context}`, 'polite');
};

export const announceLoadingComplete = (context: string) => {
  announceToScreenReader(`${context} loaded`, 'polite');
};

export const announceLoadingError = (context: string, error?: string) => {
  const message = error ? `Error loading ${context}: ${error}` : `Error loading ${context}`;
  announceToScreenReader(message, 'assertive', 3000);
};

// Modal/Dialog helpers
export const openModal = (modalElement: HTMLElement, triggerElement?: HTMLElement) => {
  // Store the trigger element for return focus
  if (triggerElement) {
    modalElement.setAttribute('data-trigger-element', triggerElement.id || '');
  }
  
  // Set up focus trap
  const cleanup = trapFocus(modalElement);
  modalElement.setAttribute('data-focus-cleanup', 'true');
  
  // Focus first focusable element
  const firstFocusable = modalElement.querySelector(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  ) as HTMLElement;
  
  if (firstFocusable) {
    firstFocusable.focus();
  }
  
  // Announce modal opening
  const modalTitle = modalElement.querySelector('h1, h2, h3, [role="heading"]')?.textContent;
  if (modalTitle) {
    announceToScreenReader(`Dialog opened: ${modalTitle}`, 'polite');
  }
  
  return cleanup;
};

export const closeModal = (modalElement: HTMLElement) => {
  // Return focus to trigger element
  const triggerElementId = modalElement.getAttribute('data-trigger-element');
  if (triggerElementId) {
    const triggerElement = document.getElementById(triggerElementId);
    if (triggerElement) {
      triggerElement.focus();
    }
  }
  
  // Announce modal closing
  announceToScreenReader('Dialog closed', 'polite');
};

// Debounced scroll announcements for long lists
export const createScrollAnnouncer = (container: HTMLElement, itemSelector: string) => {
  let timeoutId: NodeJS.Timeout;
  
  const announcePosition = () => {
    const items = container.querySelectorAll(itemSelector);
    const containerRect = container.getBoundingClientRect();
    const visibleItems = Array.from(items).filter(item => {
      const itemRect = item.getBoundingClientRect();
      return itemRect.top >= containerRect.top && itemRect.bottom <= containerRect.bottom;
    });
    
    if (visibleItems.length > 0) {
      const firstVisible = Array.from(items).indexOf(visibleItems[0]) + 1;
      const lastVisible = Array.from(items).indexOf(visibleItems[visibleItems.length - 1]) + 1;
      const total = items.length;
      
      announceToScreenReader(
        `Showing items ${firstVisible} to ${lastVisible} of ${total}`,
        'polite'
      );
    }
  };
  
  const handleScroll = () => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(announcePosition, 500);
  };
  
  container.addEventListener('scroll', handleScroll);
  
  return () => {
    clearTimeout(timeoutId);
    container.removeEventListener('scroll', handleScroll);
  };
};