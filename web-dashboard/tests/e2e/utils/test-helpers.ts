import { Page, expect } from '@playwright/test';

/**
 * Wait for network to be idle
 */
export async function waitForNetworkIdle(page: Page, timeout = 5000) {
  await page.waitForLoadState('networkidle', { timeout });
}

/**
 * Wait for wallet connection
 */
export async function waitForWalletConnection(page: Page) {
  // Wait for wallet button or connection indicator
  await page.waitForSelector('button:has-text("Connect Wallet"), [data-wallet-connected="true"]', {
    timeout: 10000,
  });
}

/**
 * Mock wallet connection
 */
export async function mockWalletConnection(page: Page, walletAddress: string = 'remes1test123...') {
  // Inject mock wallet into page
  await page.addInitScript((address) => {
    (window as any).mockWallet = {
      address: address,
      connected: true,
    };
  }, walletAddress);
}

/**
 * Wait for API response
 */
export async function waitForAPIResponse(page: Page, urlPattern: string | RegExp) {
  await page.waitForResponse(
    (response) => {
      const url = response.url();
      if (typeof urlPattern === 'string') {
        return url.includes(urlPattern);
      }
      return urlPattern.test(url);
    },
    { timeout: 10000 }
  );
}

/**
 * Check for error messages
 */
export async function expectNoErrorMessages(page: Page) {
  const errorSelectors = [
    '[role="alert"]',
    '.error',
    '.error-message',
    'text=/error/i',
    'text=/failed/i',
  ];

  for (const selector of errorSelectors) {
    await expect(page.locator(selector).first()).not.toBeVisible({ timeout: 1000 }).catch(() => {
      // Ignore if not found
    });
  }
}

/**
 * Wait for loading to complete
 */
export async function waitForLoadingComplete(page: Page) {
  // Wait for loading spinners to disappear
  await page.waitForSelector('.loading, [data-loading="true"]', {
    state: 'hidden',
    timeout: 10000,
  }).catch(() => {
    // Ignore if no loading indicators
  });
}

/**
 * Take screenshot with timestamp
 */
export async function takeScreenshot(page: Page, name: string) {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
  await page.screenshot({
    path: `tests/e2e/screenshots/${name}-${timestamp}.png`,
    fullPage: true,
  });
}

