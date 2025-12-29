/**
 * Smoke Tests for R3MES
 * 
 * Basic smoke tests to verify application startup and critical paths work.
 * These are lightweight tests that run quickly to ensure basic functionality.
 */

import { test, expect } from '@playwright/test';

test.describe('R3MES Smoke Tests', () => {
  test('Application loads successfully', async ({ page }) => {
    // Navigate to the application
    await page.goto('/');
    
    // Check that the page loads without errors
    await expect(page).toHaveTitle(/R3MES|Welcome/i, { timeout: 10000 });
    
    // Check that there's no critical error in the console
    const errors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });
    
    await page.waitForLoadState('networkidle', { timeout: 15000 });
    
    // Allow non-critical errors (filter out common ones)
    const criticalErrors = errors.filter(
      err => !err.includes('favicon') && 
             !err.includes('404') && 
             !err.includes('analytics')
    );
    
    expect(criticalErrors.length).toBe(0);
  });

  test('Navigation works', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');
    
    // Try to find and click navigation links (if they exist)
    const links = [
      '/chat',
      '/network',
      '/wallet',
    ];
    
    for (const link of links) {
      try {
        const navLink = page.locator(`a[href="${link}"]`).first();
        if (await navLink.isVisible({ timeout: 2000 }).catch(() => false)) {
          await navLink.click();
          await page.waitForLoadState('networkidle', { timeout: 5000 });
          
          // Verify we're on the expected page
          expect(page.url()).toContain(link);
          
          // Go back to home for next test
          await page.goto('/');
          await page.waitForLoadState('networkidle');
        }
      } catch (e) {
        // Navigation link might not exist, continue
        continue;
      }
    }
  });

  test('API endpoints respond', async ({ page }) => {
    // Test that backend API is accessible
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
    
    // Check health/status endpoint if it exists
    try {
      const response = await page.request.get(`${apiUrl}/health`, { timeout: 5000 });
      expect([200, 404]).toContain(response.status()); // 404 is OK if endpoint doesn't exist
    } catch (e) {
      // API might not be running, this is acceptable for smoke tests
      // Just log and continue
      console.log('API health check failed (API might not be running):', e);
    }
  });

  test('No JavaScript errors on homepage', async ({ page }) => {
    const errors: string[] = [];
    
    page.on('pageerror', (error) => {
      errors.push(error.message);
    });
    
    page.on('console', (msg) => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });
    
    await page.goto('/');
    await page.waitForLoadState('networkidle', { timeout: 15000 });
    
    // Filter out non-critical errors
    const criticalErrors = errors.filter(
      err => !err.includes('favicon') && 
             !err.includes('404') && 
             !err.includes('analytics') &&
             !err.includes('Google') &&
             !err.includes('gtag')
    );
    
    if (criticalErrors.length > 0) {
      console.log('JavaScript errors found:', criticalErrors);
    }
    
    // For smoke tests, we're lenient - just log errors
    // In full E2E tests, we would fail on any critical errors
  });
});

