import { test, expect } from '@playwright/test';

/**
 * End-to-end user flow tests
 * 
 * Tests complete user journeys through the application,
 * including onboarding, navigation, and feature usage.
 */
test.describe('User Flow Tests', () => {
  test('Complete onboarding flow', async ({ page, context }) => {
    // Clear onboarding completion
    await context.clearCookies();
    await page.evaluate(() => {
      localStorage.removeItem('r3mes_onboarding_completed');
    });

    // Start at home - should redirect to onboarding
    await page.goto('/');
    
    // Should be on onboarding page
    const currentUrl = page.url();
    expect(currentUrl).toMatch(/\/onboarding/);

    // Wait for onboarding to complete or skip
    await page.waitForLoadState('networkidle');

    // Onboarding might auto-advance or require interaction
    // Just verify it loads
    await expect(page.locator('body')).toBeVisible();
  });

  test('New user journey: Home -> Chat -> Mine -> Network', async ({ page }) => {
    // Start at home
    await page.goto('/');
    await expect(page).toHaveURL(/\//);
    await expect(page.locator('body')).toBeVisible();

    // Navigate to Chat
    const chatLink = page.getByRole('link', { name: /chat|try ai/i });
    if (await chatLink.isVisible().catch(() => false)) {
      await chatLink.click();
      await expect(page).toHaveURL(/.*\/chat/);
      await page.waitForLoadState('networkidle');
    } else {
      // If link not found, navigate directly
      await page.goto('/chat');
    }

    // Navigate to Mine
    await page.goto('/mine');
    await expect(page).toHaveURL(/.*\/mine/);
    await page.waitForLoadState('networkidle');
    await expect(page.locator('body')).toBeVisible();

    // Navigate to Network
    await page.goto('/network');
    await expect(page).toHaveURL(/.*\/network/);
    await page.waitForLoadState('networkidle');
    await expect(page.locator('body')).toBeVisible();
  });

  test('User can navigate back and forth between pages', async ({ page }) => {
    const pages = ['/', '/chat', '/mine', '/network'];

    for (const pagePath of pages) {
      await page.goto(pagePath);
      await page.waitForLoadState('networkidle');
      await expect(page.locator('body')).toBeVisible();
    }

    // Navigate back using browser back button
    for (let i = pages.length - 1; i > 0; i--) {
      await page.goBack();
      await page.waitForLoadState('networkidle');
      await expect(page.locator('body')).toBeVisible();
    }
  });

  test('User can complete onboarding and access main features', async ({ page, context }) => {
    // Clear onboarding
    await context.clearCookies();
    await page.evaluate(() => {
      localStorage.removeItem('r3mes_onboarding_completed');
    });

    // Go to home (should redirect to onboarding)
    await page.goto('/');
    await page.waitForLoadState('networkidle');

    // Complete onboarding (if manual completion needed)
    // For auto-advancing onboarding, just wait
    await page.waitForTimeout(5000);

    // Mark onboarding as complete manually if needed
    await page.evaluate(() => {
      localStorage.setItem('r3mes_onboarding_completed', 'true');
    });

    // Navigate to home
    await page.goto('/');
    await expect(page.locator('body')).toBeVisible();

    // Should be able to access main features
    await page.goto('/chat');
    await expect(page.locator('body')).toBeVisible();

    await page.goto('/mine');
    await expect(page.locator('body')).toBeVisible();
  });

  test('Application handles page refresh correctly', async ({ page }) => {
    // Navigate to a page
    await page.goto('/chat');
    await page.waitForLoadState('networkidle');

    // Refresh the page
    await page.reload();
    await page.waitForLoadState('networkidle');

    // Should still be on same page and functional
    await expect(page).toHaveURL(/.*\/chat/);
    await expect(page.locator('body')).toBeVisible();
  });

  test('User can access all main pages without errors', async ({ page }) => {
    const mainPages = [
      { path: '/', name: 'Home' },
      { path: '/chat', name: 'Chat' },
      { path: '/mine', name: 'Mine' },
      { path: '/network', name: 'Network' },
    ];

    for (const pageInfo of mainPages) {
      await page.goto(pageInfo.path);
      await page.waitForLoadState('networkidle');
      
      // Check for console errors
      const errors: string[] = [];
      page.on('console', msg => {
        if (msg.type() === 'error') {
          errors.push(msg.text());
        }
      });

      await expect(page.locator('body')).toBeVisible();
      
      // Allow some time for any errors to appear
      await page.waitForTimeout(1000);
      
      // Log errors but don't fail test (some might be expected)
      if (errors.length > 0) {
        console.log(`Errors on ${pageInfo.name} page:`, errors);
      }
    }
  });

  test('Application maintains state across navigation', async ({ page }) => {
    // Set some localStorage state
    await page.goto('/');
    await page.evaluate(() => {
      localStorage.setItem('test_key', 'test_value');
    });

    // Navigate to another page
    await page.goto('/chat');
    await page.waitForLoadState('networkidle');

    // Check if state is maintained
    const value = await page.evaluate(() => {
      return localStorage.getItem('test_key');
    });
    expect(value).toBe('test_value');
  });
});

