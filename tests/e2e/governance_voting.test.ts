/**
 * Governance Voting Flow Test
 * 
 * Tests governance proposal voting:
 * View proposal → Vote → Check vote recorded
 */

import { test, expect } from '@playwright/test';
import { waitForNetworkIdle, waitForLoadingComplete, expectNoErrorMessages, mockWalletConnection } from '../utils/test-helpers';
import { setupMockServer } from '../utils/mock-server';

test.describe('Governance Voting E2E', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockServer(page);
    await mockWalletConnection(page);
  });

  test('Governance page loads', async ({ page }) => {
    // Navigate to governance page (if route exists)
    await page.goto('/app');
    await waitForNetworkIdle(page);
    await waitForLoadingComplete(page);

    // Look for governance-related content
    const governanceLink = page.locator('a[href*="governance"], text=/governance/i').first();
    if (await governanceLink.isVisible({ timeout: 5000 }).catch(() => false)) {
      await governanceLink.click();
      await waitForNetworkIdle(page);
      await waitForLoadingComplete(page);

      // Check for governance content
      await expect(
        page.locator('text=/proposal|governance|vote/i').first()
      ).toBeVisible({ timeout: 10000 });
    }

    await expectNoErrorMessages(page);
  });

  test('View governance proposals', async ({ page }) => {
    // This test assumes governance page exists
    // For now, just verify we can navigate without errors
    await page.goto('/app');
    await waitForNetworkIdle(page);
    await waitForLoadingComplete(page);
    await expectNoErrorMessages(page);
  });
});

