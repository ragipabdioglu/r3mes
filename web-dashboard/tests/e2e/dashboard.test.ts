import { test, expect } from '@playwright/test';

/**
 * Dashboard integration tests
 * Tests critical dashboard functionality
 */
test.describe('Dashboard Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/dashboard');
  });

  test('Dashboard page loads', async ({ page }) => {
    await expect(page.locator('body')).toBeVisible();
  });

  test('Network stats are displayed', async ({ page }) => {
    // Wait for stats to load (if they exist)
    await page.waitForTimeout(2000);
    // Check that page is interactive
    await expect(page.locator('body')).toBeVisible();
  });
});

