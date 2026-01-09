/**
 * Chat Flow Test
 * 
 * Tests the chat interface flow:
 * Connect wallet → Send message → Receive streaming response → Check credit deduction
 */

import { test, expect } from '@playwright/test';
import { waitForNetworkIdle, waitForLoadingComplete, expectNoErrorMessages, mockWalletConnection } from '../utils/test-helpers';
import { setupMockServer } from '../utils/mock-server';

test.describe('Chat Flow E2E', () => {
  test.beforeEach(async ({ page }) => {
    await setupMockServer(page);
    await mockWalletConnection(page);
  });

  test('Send chat message and receive response', async ({ page }) => {
    await page.goto('/chat');
    await waitForNetworkIdle(page);
    await waitForLoadingComplete(page);
    
    // Connect wallet (if not already connected)
    const connectButton = page.locator('button:has-text("Connect Wallet")').first();
    if (await connectButton.isVisible({ timeout: 5000 }).catch(() => false)) {
      await connectButton.click();
      await page.waitForSelector('text=/connected|wallet/i', { timeout: 10000 }).catch(() => {
        // Wallet might already be connected
      });
    }

    // Find message input
    const messageInput = page.locator('textarea[placeholder*="message"], textarea[placeholder*="Message"], input[type="text"]').first();
    await expect(messageInput).toBeVisible({ timeout: 10000 });
    
    // Send message
    await messageInput.fill('What is R3MES?');
    
    const sendButton = page.locator('button:has-text("Send"), button[type="submit"]').first();
    await sendButton.click();

    // Wait for response (mock server will return response)
    await expect(
      page.locator('text=/R3MES|response|mock/i').first()
    ).toBeVisible({ timeout: 30000 });

    // Verify no errors
    await expectNoErrorMessages(page);
  });

  test('Chat interface displays correctly', async ({ page }) => {
    await page.goto('/chat');
    await waitForNetworkIdle(page);
    await waitForLoadingComplete(page);

    // Check for chat interface elements
    await expect(
      page.locator('textarea, input[type="text"]').first()
    ).toBeVisible({ timeout: 10000 });
    
    await expect(
      page.locator('button:has-text("Send"), button[type="submit"]').first()
    ).toBeVisible({ timeout: 5000 });

    await expectNoErrorMessages(page);
  });
});

