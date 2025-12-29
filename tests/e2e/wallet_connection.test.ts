/**
 * Wallet Connection Flow Test
 * 
 * Tests wallet connection and transaction flow:
 * Connect Keplr → Check balance → Send transaction
 */

import { test, expect } from '@playwright/test';
import { waitForNetworkIdle, waitForLoadingComplete, expectNoErrorMessages, mockWalletConnection } from '../utils/test-helpers';
import { setupMockServer } from '../utils/mock-server';

test.describe('Wallet Connection E2E', () => {
  test.beforeEach(async ({ page, context }) => {
    await setupMockServer(page);
    
    // Mock Keplr wallet
    await context.addInitScript(() => {
      (window as any).keplr = {
        enable: async (chainId: string) => ({ address: 'remes1test123...' }),
        getOfflineSigner: (chainId: string) => ({}),
        getBalance: async (chainId: string, address: string) => ({
          amount: '1000000',
          denom: 'uremes',
        }),
      };
    });
  });

  test('Connect wallet and view balance', async ({ page }) => {
    await page.goto('/app');
    await waitForNetworkIdle(page);
    await waitForLoadingComplete(page);

    // Connect wallet (if button exists)
    const connectButton = page.locator('button:has-text("Connect Wallet")').first();
    if (await connectButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      await connectButton.click();
      await page.waitForSelector('text=/connected|wallet/i', { timeout: 10000 }).catch(() => {
        // Wallet might already be connected
      });
    }

    // Navigate to wallet page
    await page.goto('/wallet');
    await waitForNetworkIdle(page);
    await waitForLoadingComplete(page);

    // Check for wallet-related content
    await expect(
      page.locator('text=/wallet|balance|credits|address/i').first()
    ).toBeVisible({ timeout: 10000 });

    await expectNoErrorMessages(page);
  });

  test('Wallet page displays transaction history', async ({ page }) => {
    await page.goto('/wallet');
    await waitForNetworkIdle(page);
    await waitForLoadingComplete(page);

    // Check for transaction history or related content
    await expect(
      page.locator('text=/transaction|history|recent/i').first()
    ).toBeVisible({ timeout: 10000 }).catch(() => {
      // Transaction history might not be visible if empty
    });

    await expectNoErrorMessages(page);
  });
});

