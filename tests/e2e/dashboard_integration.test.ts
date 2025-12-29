/**
 * Dashboard Integration Test
 * 
 * Tests dashboard pages and components integration
 */

import { test, expect } from '@playwright/test';
import { waitForNetworkIdle, waitForLoadingComplete, expectNoErrorMessages } from '../utils/test-helpers';
import { setupMockServer } from '../utils/mock-server';

test.describe('Dashboard Integration', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockServer(page);
  });

  test('Dashboard home page loads', async ({ page }) => {
    await page.goto('/app');
    await waitForNetworkIdle(page);
    await waitForLoadingComplete(page);

    // Check for dashboard content
    await expect(page.locator('text=/dashboard|network|mining/i').first()).toBeVisible({ timeout: 10000 });
    await expectNoErrorMessages(page);
  });

  test('Network page displays stats', async ({ page }) => {
    await page.goto('/network');
    await waitForNetworkIdle(page);
    await waitForLoadingComplete(page);

    // Check for network stats
    await expect(
      page.locator('text=/miners|validators|blocks|hashrate/i').first()
    ).toBeVisible({ timeout: 10000 });
    await expectNoErrorMessages(page);
  });

  test('Mine page displays miner stats', async ({ page }) => {
    await page.goto('/mine');
    await waitForNetworkIdle(page);
    await waitForLoadingComplete(page);

    // Check for miner-related content
    await expect(
      page.locator('text=/mining|earnings|hashrate|blocks/i').first()
    ).toBeVisible({ timeout: 10000 });
    await expectNoErrorMessages(page);
  });

  test('Wallet page displays wallet info', async ({ page }) => {
    await page.goto('/wallet');
    await waitForNetworkIdle(page);
    await waitForLoadingComplete(page);

    // Check for wallet-related content
    await expect(
      page.locator('text=/wallet|balance|credits|transactions/i').first()
    ).toBeVisible({ timeout: 10000 });
    await expectNoErrorMessages(page);
  });

  test('Settings page loads', async ({ page }) => {
    await page.goto('/settings');
    await waitForNetworkIdle(page);
    await waitForLoadingComplete(page);

    // Check for settings content
    await expect(
      page.locator('text=/settings|config|configuration/i').first()
    ).toBeVisible({ timeout: 10000 });
    await expectNoErrorMessages(page);
  });
});

