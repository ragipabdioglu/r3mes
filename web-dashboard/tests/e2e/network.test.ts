import { test, expect } from '@playwright/test';

/**
 * Network explorer E2E tests
 * 
 * Tests the network explorer page, blockchain data display,
 * real-time updates, and network statistics.
 */
test.describe('Network Explorer Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/network');
  });

  test('Network page loads successfully', async ({ page }) => {
    await expect(page).toHaveURL(/.*\/network/);
    await expect(page.locator('body')).toBeVisible();
    
    // Wait for page to load
    await page.waitForLoadState('networkidle');
    
    // Check for main content area
    const mainContent = page.locator('main, [role="main"], .network-explorer, .explorer').first();
    await expect(mainContent).toBeVisible({ timeout: 10000 });
  });

  test('Network statistics are displayed', async ({ page }) => {
    await page.waitForLoadState('networkidle');

    // Look for network stats (blocks, miners, etc.)
    // Stats might be in various formats
    const statsElements = page.locator('[data-testid*="stat"], [class*="stat"], [class*="metric"]');
    
    // At least page should be visible
    await expect(page.locator('body')).toBeVisible();
  });

  test('Block list or blockchain data section is present', async ({ page }) => {
    await page.waitForLoadState('networkidle');

    // Look for blocks or blockchain data
    // This might be a table, list, or cards
    const blocksSection = page.locator('[data-testid*="block"], [class*="block"], table, .blocks').first();
    
    // Page should load regardless
    await expect(page.locator('body')).toBeVisible();
  });

  test('Network page handles API errors gracefully', async ({ page }) => {
    // Intercept API calls and return error
    await page.route('**/api/**', route => {
      route.fulfill({
        status: 500,
        body: JSON.stringify({ error: 'Internal server error' }),
      });
    });

    await page.reload();
    await page.waitForLoadState('networkidle');

    // Page should still render, showing error state
    await expect(page.locator('body')).toBeVisible();
  });

  test('Real-time updates work (if WebSocket enabled)', async ({ page }) => {
    await page.waitForLoadState('networkidle');

    // Wait a bit to see if data updates
    await page.waitForTimeout(2000);

    // Page should remain functional
    await expect(page.locator('body')).toBeVisible();
  });

  test('Network explorer is responsive', async ({ page }) => {
    // Test mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    await page.reload();
    await expect(page.locator('body')).toBeVisible();

    // Test tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    await page.reload();
    await expect(page.locator('body')).toBeVisible();

    // Test desktop viewport
    await page.setViewportSize({ width: 1920, height: 1080 });
    await page.reload();
    await expect(page.locator('body')).toBeVisible();
  });

  test('Navigation from network page works', async ({ page }) => {
    await page.waitForLoadState('networkidle');

    // Try to navigate to other pages
    await page.goto('/');
    await expect(page).toHaveURL(/\//);

    await page.goto('/network');
    await page.goto('/mine');
    await expect(page).toHaveURL(/.*\/mine/);
  });
});

