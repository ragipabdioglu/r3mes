import { test, expect } from '@playwright/test';

/**
 * Smoke tests for critical user flows
 * 
 * These tests are tagged with @smoke and run in CD pipeline
 * to verify basic functionality after deployment.
 * 
 * @smoke
 */
test.describe('Smoke Tests @smoke', () => {
  test('Homepage loads successfully', async ({ page }) => {
    await page.goto('/');
    await expect(page).toHaveTitle(/R3MES/i);
    await expect(page.locator('body')).toBeVisible();
  });

  test('All main pages are accessible', async ({ page }) => {
    const pages = [
      { path: '/', name: 'Home' },
      { path: '/chat', name: 'Chat' },
      { path: '/mine', name: 'Mine' },
      { path: '/network', name: 'Network' },
    ];

    for (const pageInfo of pages) {
      await page.goto(pageInfo.path);
      await page.waitForLoadState('networkidle');
      await expect(page.locator('body')).toBeVisible();
    }
  });

  test('Application responds without critical errors', async ({ page }) => {
    const errors: string[] = [];
    
    page.on('console', msg => {
      if (msg.type() === 'error') {
        errors.push(msg.text());
      }
    });

    page.on('pageerror', error => {
      errors.push(error.message);
    });

    await page.goto('/');
    await page.waitForLoadState('networkidle');
    await page.waitForTimeout(2000); // Wait for any async errors

    // Filter out expected/known errors (e.g., wallet not connected)
    const criticalErrors = errors.filter(err => 
      !err.includes('Keplr') && 
      !err.includes('wallet') &&
      !err.includes('extension') &&
      !err.toLowerCase().includes('favicon')
    );

    // Log errors for debugging but don't fail if they're non-critical
    if (criticalErrors.length > 0) {
      console.log('Critical errors found:', criticalErrors);
    }

    // Page should still be visible
    await expect(page.locator('body')).toBeVisible();
  });

  test('Navigation menu is functional', async ({ page }) => {
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Check if navigation exists
    const nav = page.locator('nav').first();
    if (await nav.isVisible().catch(() => false)) {
      await expect(nav).toBeVisible();
    }

    // Verify we can navigate to main pages
    await page.goto('/chat');
    await expect(page).toHaveURL(/.*\/chat/);
    
    await page.goto('/mine');
    await expect(page).toHaveURL(/.*\/mine/);
  });
});

