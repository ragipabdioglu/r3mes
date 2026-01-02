/**
 * Unit tests for logger utility
 */

import { describe, it, expect, beforeEach, afterEach } from '@jest/globals';

describe('Logger', () => {
  const originalEnv = process.env.NODE_ENV;
  let consoleLogSpy: jest.SpyInstance;
  let consoleErrorSpy: jest.SpyInstance;
  let consoleWarnSpy: jest.SpyInstance;
  let consoleInfoSpy: jest.SpyInstance;
  let consoleDebugSpy: jest.SpyInstance;

  beforeEach(() => {
    consoleLogSpy = jest.spyOn(console, 'log').mockImplementation();
    consoleErrorSpy = jest.spyOn(console, 'error').mockImplementation();
    consoleWarnSpy = jest.spyOn(console, 'warn').mockImplementation();
    consoleInfoSpy = jest.spyOn(console, 'info').mockImplementation();
    consoleDebugSpy = jest.spyOn(console, 'debug').mockImplementation();
  });

  afterEach(() => {
    process.env.NODE_ENV = originalEnv;
    jest.restoreAllMocks();
    // Clear module cache to allow re-import with new env
    jest.resetModules();
  });

  describe('Development mode', () => {
    it('should log messages in development', async () => {
      process.env.NODE_ENV = 'development';
      const { logger } = await import('@/lib/logger');
      
      logger.info('test log');
      expect(consoleLogSpy).toHaveBeenCalledWith('test log');
    });

    it('should log errors in development', async () => {
      process.env.NODE_ENV = 'development';
      const { logger } = await import('@/lib/logger');
      
      logger.error('test error');
      expect(consoleErrorSpy).toHaveBeenCalledWith('test error');
    });

    it('should log warnings in development', async () => {
      process.env.NODE_ENV = 'development';
      const { logger } = await import('@/lib/logger');
      
      logger.warn('test warning');
      expect(consoleWarnSpy).toHaveBeenCalledWith('test warning');
    });

    it('should log info in development', async () => {
      process.env.NODE_ENV = 'development';
      const { logger } = await import('@/lib/logger');
      
      logger.info('test info');
      expect(consoleInfoSpy).toHaveBeenCalledWith('test info');
    });

    it('should log debug in development', async () => {
      process.env.NODE_ENV = 'development';
      const { logger } = await import('@/lib/logger');
      
      logger.debug('test debug');
      expect(consoleDebugSpy).toHaveBeenCalledWith('test debug');
    });
  });

  describe('Production mode', () => {
    it('should not log messages in production', async () => {
      process.env.NODE_ENV = 'production';
      const { logger } = await import('@/lib/logger');
      
      logger.info('test log');
      expect(consoleLogSpy).not.toHaveBeenCalled();
    });

    it('should always log errors in production', async () => {
      process.env.NODE_ENV = 'production';
      const { logger } = await import('@/lib/logger');
      
      logger.error('test error');
      expect(consoleErrorSpy).toHaveBeenCalledWith('test error');
    });

    it('should not log warnings in production', async () => {
      process.env.NODE_ENV = 'production';
      const { logger } = await import('@/lib/logger');
      
      logger.warn('test warning');
      expect(consoleWarnSpy).not.toHaveBeenCalled();
    });

    it('should not log info in production', async () => {
      process.env.NODE_ENV = 'production';
      const { logger } = await import('@/lib/logger');
      
      logger.info('test info');
      expect(consoleInfoSpy).not.toHaveBeenCalled();
    });

    it('should not log debug in production', async () => {
      process.env.NODE_ENV = 'production';
      const { logger } = await import('@/lib/logger');
      
      logger.debug('test debug');
      expect(consoleDebugSpy).not.toHaveBeenCalled();
    });
  });
});

