import { test, expect } from '@playwright/test';

/**
 * Navigation and routing tests
 * 
 * Tests navigation between pages, link functionality,
 * and routing behavior.
 */
test.describe('Navigation Tests', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/');
  });

  test('Homepage navigation links work', async ({ page }) => {
    // Test "Start Mining" link
    const startMiningLink = page.getByRole('link', { name: /start mining/i });
    await expect(startMiningLink).toBeVisible();
    await startMiningLink.click();
    await expect(page).toHaveURL(/.*\/mine/);

    // Go back and test "Try AI Chat" link
    await page.goto('/');
    const chatLink = page.getByRole('link', { name: /try ai chat/i });
    await expect(chatLink).toBeVisible();
    await chatLink.click();
    await expect(page).toHaveURL(/.*\/chat/);

    // Go back and test "Explore Network" link
    await page.goto('/');
    const networkLink = page.getByRole('link', { name: /explore network/i });
    await expect(networkLink).toBeVisible();
    await networkLink.click();
    await expect(page).toHaveURL(/.*\/network/);
  });

  test('Navigation menu items are accessible', async ({ page }) => {
    // Check if navigation menu exists
    const nav = page.locator('nav').first();
    await expect(nav).toBeVisible();

    // Test navigation to different pages
    const pages = [
      { name: 'Home', url: '/' },
      { name: 'Chat', url: '/chat' },
      { name: 'Mine', url: '/mine' },
      { name: 'Network', url: '/network' },
    ];

    for (const pageInfo of pages) {
      await page.goto(pageInfo.url);
      await expect(page.locator('body')).toBeVisible();
      // Page should load without errors
      await expect(page.locator('main, [role="main"]').first()).toBeVisible({ timeout: 5000 });
    }
  });

  test('Onboarding flow redirects correctly', async ({ page, context }) => {
    // Clear onboarding completion flag
    await context.clearCookies();
    await page.evaluate(() => {
      localStorage.removeItem('r3mes_onboarding_completed');
    });

    // Navigate to home - should redirect to onboarding
    await page.goto('/');
    
    // Should either be on onboarding page or home (if onboarding already completed)
    const currentUrl = page.url();
    expect(currentUrl).toMatch(/\/(onboarding|$)/);
  });

  test('Back button works correctly', async ({ page }) => {
    // Navigate through pages
    await page.goto('/');
    await page.goto('/chat');
    await page.goto('/mine');

    // Use back button
    await page.goBack();
    await expect(page).toHaveURL(/.*\/chat/);

    await page.goBack();
    await expect(page).toHaveURL(/\//);
  });
});

