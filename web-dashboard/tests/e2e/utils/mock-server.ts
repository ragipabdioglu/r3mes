import { Page } from '@playwright/test';

/**
 * Mock backend API responses
 */
export async function setupMockServer(page: Page) {
  // Mock network stats
  await page.route('**/api/network/stats', (route) => {
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        total_miners: 150,
        total_validators: 25,
        network_hashrate: 1000.5,
        total_stake: 1000000.0,
        block_height: 12345,
      }),
    });
  });

  // Mock user info
  await page.route('**/api/user/info/*', (route) => {
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        wallet_address: 'remes1test123...',
        credits: 100.0,
        is_miner: true,
        last_mining_time: new Date().toISOString(),
      }),
    });
  });

  // Mock chat endpoint
  await page.route('**/api/chat', (route) => {
    const request = route.request();
    const postData = request.postDataJSON();
    
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        response: `Mock response to: ${postData?.message || ''}`,
        credits_used: 0.1,
        adapter_used: 'general',
      }),
    });
  });

  // Mock health check
  await page.route('**/api/health', (route) => {
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        status: 'healthy',
        timestamp: new Date().toISOString(),
      }),
    });
  });
}

/**
 * Disable mock server
 */
export async function disableMockServer(page: Page) {
  await page.unroute('**/api/**');
}

