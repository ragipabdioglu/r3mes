# R3MES Web Dashboard - Accessibility Guide

## Overview

This guide outlines the accessibility features and best practices implemented in the R3MES Web Dashboard to ensure WCAG AA compliance and provide an inclusive user experience for all users, including those using assistive technologies.

## Table of Contents

1. [Accessibility Standards](#accessibility-standards)
2. [Key Features](#key-features)
3. [Component Guidelines](#component-guidelines)
4. [Testing](#testing)
5. [Development Guidelines](#development-guidelines)
6. [Utilities and Hooks](#utilities-and-hooks)
7. [Common Patterns](#common-patterns)
8. [Troubleshooting](#troubleshooting)

## Accessibility Standards

### WCAG AA Compliance

The R3MES Web Dashboard follows the Web Content Accessibility Guidelines (WCAG) 2.1 Level AA standards:

- **Perceivable**: Information and UI components must be presentable to users in ways they can perceive
- **Operable**: UI components and navigation must be operable
- **Understandable**: Information and the operation of UI must be understandable
- **Robust**: Content must be robust enough to be interpreted by a wide variety of user agents

### Key Principles

1. **Semantic HTML**: Use proper HTML elements for their intended purpose
2. **Keyboard Navigation**: All interactive elements must be keyboard accessible
3. **Screen Reader Support**: Provide proper ARIA labels and descriptions
4. **Color Contrast**: Maintain WCAG AA contrast ratios (4.5:1 for normal text)
5. **Focus Management**: Provide clear focus indicators and logical tab order
6. **Responsive Design**: Ensure accessibility across all device sizes

## Key Features

### 1. Screen Reader Support

- **ARIA Labels**: All interactive elements have descriptive labels
- **Live Regions**: Dynamic content changes are announced to screen readers
- **Semantic Structure**: Proper heading hierarchy and landmark roles
- **Alternative Text**: Meaningful descriptions for images and icons

```tsx
// Example: Accessible button with ARIA label
<button
  aria-label="Send message to R3MES AI"
  aria-describedby="send-help"
  onClick={handleSend}
>
  <Send className="w-4 h-4" aria-hidden="true" />
  Send
</button>
<div id="send-help" className="sr-only">
  Sends your message to the AI assistant
</div>
```

### 2. Keyboard Navigation

- **Tab Order**: Logical tab sequence through interactive elements
- **Focus Trapping**: Modal dialogs trap focus within their boundaries
- **Keyboard Shortcuts**: Standard keyboard interactions (Enter, Space, Escape)
- **Skip Links**: Allow users to skip to main content

```tsx
// Example: Keyboard navigation in dropdown
const handleKeyDown = (e: KeyboardEvent) => {
  switch (e.key) {
    case 'ArrowDown':
      e.preventDefault();
      focusNextItem();
      break;
    case 'ArrowUp':
      e.preventDefault();
      focusPreviousItem();
      break;
    case 'Escape':
      e.preventDefault();
      closeDropdown();
      break;
  }
};
```

### 3. Visual Accessibility

- **High Contrast Support**: Adapts to user's high contrast preferences
- **Reduced Motion**: Respects user's motion preferences
- **Color Independence**: Information is not conveyed by color alone
- **Focus Indicators**: Clear visual focus indicators for keyboard users

```css
/* High contrast mode support */
@media (prefers-contrast: high) {
  :root {
    --text-primary: #000000;
    --bg-primary: #ffffff;
    --border-color: #000000;
  }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

### 4. Mobile Accessibility

- **Touch Targets**: Minimum 44px touch target size
- **Responsive Design**: Accessible across all screen sizes
- **Mobile Screen Readers**: Optimized for mobile assistive technologies
- **Gesture Support**: Alternative methods for gesture-based interactions

## Component Guidelines

### Navbar Component

```tsx
// Accessibility features:
// - Semantic navigation structure
// - ARIA labels for all interactive elements
// - Keyboard navigation support
// - Mobile menu with focus trapping
// - Screen reader announcements

<nav role="navigation" aria-label="Main navigation">
  <Link href="/" aria-label="R3MES Home">
    R3MES
  </Link>
  
  <button
    aria-expanded={mobileMenuOpen}
    aria-controls="mobile-menu"
    aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
    onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
  >
    <Menu aria-hidden="true" />
  </button>
</nav>
```

### ChatInterface Component

```tsx
// Accessibility features:
// - Form labels and descriptions
// - Live region for message updates
// - Keyboard shortcuts (Enter to send)
// - Screen reader announcements
// - Message history with proper structure

<div role="log" aria-live="polite" aria-label="Chat messages">
  {messages.map((message, index) => (
    <div
      key={message.id}
      role="article"
      aria-labelledby={`message-${index}-label`}
    >
      <div id={`message-${index}-label`}>
        {message.role === 'user' ? 'You' : 'R3MES AI'}
      </div>
      <div aria-label={`${message.role} message: ${message.content}`}>
        {message.content}
      </div>
    </div>
  ))}
</div>
```

### ErrorBoundary Component

```tsx
// Accessibility features:
// - Alert role for error announcements
// - Descriptive error messages
// - Recovery action buttons with clear labels
// - Screen reader announcements

<div role="alert" aria-labelledby="error-title" aria-describedby="error-description">
  <h2 id="error-title">Application Error</h2>
  <p id="error-description">
    The application encountered an error. Please try one of the recovery options below.
  </p>
  <button
    onClick={handleRetry}
    aria-describedby="retry-help"
  >
    Try Again
  </button>
  <div id="retry-help" className="sr-only">
    Attempts to recover the component without reloading the page
  </div>
</div>
```

## Testing

### Automated Testing

The project includes comprehensive accessibility testing using jest-axe:

```bash
# Run accessibility tests
npm run test:accessibility

# Run all tests with coverage
npm run test:coverage

# Run specific accessibility test
npm test -- tests/accessibility/accessibility.test.ts
```

### Manual Testing Checklist

1. **Keyboard Navigation**
   - [ ] Tab through all interactive elements
   - [ ] Verify logical tab order
   - [ ] Test keyboard shortcuts (Enter, Space, Escape)
   - [ ] Ensure focus is visible

2. **Screen Reader Testing**
   - [ ] Test with NVDA (Windows) or VoiceOver (Mac)
   - [ ] Verify all content is announced
   - [ ] Check ARIA labels and descriptions
   - [ ] Test live region announcements

3. **Visual Testing**
   - [ ] Check color contrast ratios
   - [ ] Test with high contrast mode
   - [ ] Verify focus indicators are visible
   - [ ] Test with 200% zoom

4. **Mobile Testing**
   - [ ] Verify touch target sizes (minimum 44px)
   - [ ] Test with mobile screen readers
   - [ ] Check responsive behavior
   - [ ] Test gesture alternatives

### Testing Tools

- **jest-axe**: Automated accessibility testing
- **axe-core**: Accessibility rule engine
- **Lighthouse**: Performance and accessibility audits
- **WAVE**: Web accessibility evaluation tool
- **Color Contrast Analyzers**: For checking contrast ratios

## Development Guidelines

### 1. Semantic HTML

Always use semantic HTML elements:

```tsx
// Good
<nav>
  <ul>
    <li><a href="/home">Home</a></li>
    <li><a href="/about">About</a></li>
  </ul>
</nav>

// Bad
<div className="nav">
  <div className="nav-item" onClick={goHome}>Home</div>
  <div className="nav-item" onClick={goAbout}>About</div>
</div>
```

### 2. ARIA Labels and Descriptions

Provide meaningful labels for all interactive elements:

```tsx
// Good
<button
  aria-label="Delete item"
  aria-describedby="delete-help"
  onClick={handleDelete}
>
  <Trash aria-hidden="true" />
</button>
<div id="delete-help" className="sr-only">
  This action cannot be undone
</div>

// Bad
<button onClick={handleDelete}>
  <Trash />
</button>
```

### 3. Focus Management

Manage focus appropriately, especially in dynamic content:

```tsx
// Good - Focus management in modal
const openModal = () => {
  setIsOpen(true);
  // Focus first focusable element in modal
  setTimeout(() => {
    const firstFocusable = modalRef.current?.querySelector('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])');
    firstFocusable?.focus();
  }, 0);
};

const closeModal = () => {
  setIsOpen(false);
  // Return focus to trigger element
  triggerRef.current?.focus();
};
```

### 4. Color and Contrast

Ensure sufficient color contrast and don't rely on color alone:

```tsx
// Good - Status with icon and color
<div className="status-success">
  <CheckCircle aria-hidden="true" />
  <span>Success: Operation completed</span>
</div>

// Bad - Status with color only
<div className="text-green-500">
  Operation completed
</div>
```

## Utilities and Hooks

### Accessibility Utilities

```tsx
import {
  announceToScreenReader,
  trapFocus,
  meetsWCAGAA,
  generateId
} from '@/utils/accessibility';

// Announce to screen reader
announceToScreenReader('Form submitted successfully', 'polite');

// Check color contrast
const hasGoodContrast = meetsWCAGAA('#000000', '#ffffff'); // true

// Generate unique ID for ARIA relationships
const fieldId = generateId('email-field');
```

### Accessibility Hooks

```tsx
import {
  useAnnouncer,
  useFocusTrap,
  useAccessibilityPreferences,
  useFormAccessibility
} from '@/hooks/useAccessibility';

// Screen reader announcements
const { announce, announceError } = useAnnouncer();
announce('Data loaded successfully');

// Focus trapping for modals
const containerRef = useFocusTrap(isModalOpen);

// User preferences
const { reducedMotion, highContrast } = useAccessibilityPreferences();

// Form accessibility
const { getFieldProps, announceFieldError } = useFormAccessibility();
const fieldProps = getFieldProps('email', error, 'Enter your email address');
```

## Common Patterns

### 1. Accessible Forms

```tsx
const AccessibleForm = () => {
  const { getFieldProps, announceFieldError } = useFormAccessibility();
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');

  const emailProps = getFieldProps('email', error, 'We will never share your email');

  return (
    <form onSubmit={handleSubmit}>
      <div className="form-field">
        <label htmlFor={emailProps.id} className="form-label">
          Email Address *
        </label>
        <input
          {...emailProps}
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          className="form-input"
          required
        />
        {emailProps.descriptionId && (
          <div id={emailProps.descriptionId} className="form-description">
            We will never share your email
          </div>
        )}
        {error && (
          <div id={emailProps.errorId} className="form-error" role="alert">
            {error}
          </div>
        )}
      </div>
      <button type="submit">Submit</button>
    </form>
  );
};
```

### 2. Accessible Modals

```tsx
const AccessibleModal = ({ isOpen, onClose, children }) => {
  const containerRef = useFocusTrap(isOpen);
  const { announce } = useAnnouncer();

  useEffect(() => {
    if (isOpen) {
      announce('Dialog opened');
    }
  }, [isOpen, announce]);

  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div
        ref={containerRef}
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="modal-title"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 id="modal-title">Modal Title</h2>
        {children}
        <button onClick={onClose} aria-label="Close dialog">
          <X aria-hidden="true" />
        </button>
      </div>
    </div>
  );
};
```

### 3. Accessible Data Tables

```tsx
const AccessibleTable = ({ data, columns }) => {
  return (
    <table className="accessible-table" role="table">
      <caption>User Data Table</caption>
      <thead>
        <tr>
          {columns.map((column) => (
            <th
              key={column.key}
              scope="col"
              aria-sort={getSortDirection(column.key)}
            >
              {column.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.map((row, index) => (
          <tr key={row.id}>
            {columns.map((column) => (
              <td key={column.key}>
                {column.key === 'name' ? (
                  <th scope="row">{row[column.key]}</th>
                ) : (
                  row[column.key]
                )}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
};
```

## Troubleshooting

### Common Issues

1. **Focus Not Visible**
   - Ensure focus indicators are not removed with CSS
   - Use `:focus-visible` for better UX
   - Test with keyboard navigation

2. **Screen Reader Not Announcing Changes**
   - Check if live regions are properly configured
   - Verify ARIA attributes are correct
   - Use `aria-live="polite"` or `aria-live="assertive"`

3. **Poor Color Contrast**
   - Use contrast checking tools
   - Test with high contrast mode
   - Ensure text meets WCAG AA standards (4.5:1)

4. **Keyboard Navigation Issues**
   - Check tab order with Tab key
   - Ensure all interactive elements are focusable
   - Implement proper keyboard event handlers

### Debugging Tools

1. **Browser DevTools**
   - Accessibility panel in Chrome/Edge DevTools
   - Lighthouse accessibility audit
   - Elements panel for ARIA inspection

2. **Screen Reader Testing**
   - NVDA (free, Windows)
   - JAWS (Windows)
   - VoiceOver (macOS/iOS)
   - TalkBack (Android)

3. **Automated Testing**
   - Run `npm run test:accessibility`
   - Use axe browser extension
   - Integrate with CI/CD pipeline

### Getting Help

- **Documentation**: Refer to WCAG 2.1 guidelines
- **Community**: WebAIM community forums
- **Tools**: Use axe-core documentation
- **Testing**: Screen reader user guides

## Resources

- [WCAG 2.1 Guidelines](https://www.w3.org/WAI/WCAG21/quickref/)
- [ARIA Authoring Practices Guide](https://www.w3.org/WAI/ARIA/apg/)
- [WebAIM Resources](https://webaim.org/)
- [axe-core Documentation](https://github.com/dequelabs/axe-core)
- [Color Contrast Checker](https://webaim.org/resources/contrastchecker/)

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Maintainer**: R3MES Development Team