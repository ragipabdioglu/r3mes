/**
 * **Feature: frontend-build-fix, Property 4: Build produces deployable bundle**
 * **Validates: Requirements 1.4**
 */

import { describe, it, expect } from '@jest/globals';
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

describe('Deployable Bundle Property Tests', () => {
  const projectRoot = path.resolve(__dirname, '../..');
  const buildDir = path.join(projectRoot, '.next');
  
  beforeAll(() => {
    // Ensure we have a fresh build for testing
    if (!fs.existsSync(buildDir)) {
      console.log('Building project for bundle tests...');
      execSync('npm run build', { 
        cwd: projectRoot,
        stdio: 'inherit',
        timeout: 120000
      });
    }
  });

  it('should produce a complete Next.js build output', () => {
    expect(fs.existsSync(buildDir)).toBe(true);
    
    // Check for essential build directories
    const staticDir = path.join(buildDir, 'static');
    const serverDir = path.join(buildDir, 'server');
    
    expect(fs.existsSync(staticDir)).toBe(true);
    expect(fs.existsSync(serverDir)).toBe(true);
    
    // Check for build manifest
    const buildManifest = path.join(buildDir, 'build-manifest.json');
    expect(fs.existsSync(buildManifest)).toBe(true);
    
    // Verify build manifest is valid JSON
    const manifestContent = fs.readFileSync(buildManifest, 'utf8');
    expect(() => JSON.parse(manifestContent)).not.toThrow();
  });

  it('should generate static pages correctly', () => {
    const serverAppDir = path.join(buildDir, 'server', 'app');
    expect(fs.existsSync(serverAppDir)).toBe(true);
    
    // Check for key pages (Next.js App Router uses .rsc files)
    const indexPage = path.join(serverAppDir, 'index.rsc');
    const appPage = path.join(serverAppDir, 'app.rsc');
    
    expect(fs.existsSync(indexPage)).toBe(true);
    expect(fs.existsSync(appPage)).toBe(true);
    
    // Verify pages are not empty
    const indexStats = fs.statSync(indexPage);
    const appStats = fs.statSync(appPage);
    
    expect(indexStats.size).toBeGreaterThan(0);
    expect(appStats.size).toBeGreaterThan(0);
  });

  it('should include all necessary static assets', () => {
    const staticDir = path.join(buildDir, 'static');
    
    // Check for chunks directory
    const chunksDir = path.join(staticDir, 'chunks');
    expect(fs.existsSync(chunksDir)).toBe(true);
    
    // Verify there are JavaScript chunks
    const chunkFiles = fs.readdirSync(chunksDir).filter(file => file.endsWith('.js'));
    expect(chunkFiles.length).toBeGreaterThan(0);
    
    // Check for CSS files if any
    const cssFiles = fs.readdirSync(staticDir, { recursive: true })
      .filter((file: any) => typeof file === 'string' && file.endsWith('.css'));
    
    // CSS files are optional but if they exist, they should be valid
    cssFiles.forEach((cssFile: string) => {
      const cssPath = path.join(staticDir, cssFile);
      const cssStats = fs.statSync(cssPath);
      expect(cssStats.size).toBeGreaterThan(0);
    });
  });

  it('should have proper standalone build configuration', () => {
    // Check for standalone build output
    const standaloneDir = path.join(buildDir, 'standalone');
    
    // Standalone might not exist if not configured, but if it does, validate it
    if (fs.existsSync(standaloneDir)) {
      const serverJs = path.join(standaloneDir, 'server.js');
      expect(fs.existsSync(serverJs)).toBe(true);
      
      const serverStats = fs.statSync(serverJs);
      expect(serverStats.size).toBeGreaterThan(0);
    }
    
    // Check Next.js config for standalone output
    const nextConfigPath = path.join(projectRoot, 'next.config.js');
    const configContent = fs.readFileSync(nextConfigPath, 'utf8');
    expect(configContent).toContain("output: 'standalone'");
  });

  it('should validate bundle size and performance', () => {
    const buildManifestPath = path.join(buildDir, 'build-manifest.json');
    const buildManifest = JSON.parse(fs.readFileSync(buildManifestPath, 'utf8'));
    
    // Check that we have reasonable bundle sizes
    const pages = buildManifest.pages || {};
    
    Object.keys(pages).forEach(page => {
      const pageAssets = pages[page];
      if (Array.isArray(pageAssets)) {
        pageAssets.forEach(asset => {
          if (asset.endsWith('.js')) {
            const assetPath = path.join(buildDir, 'static', asset);
            if (fs.existsSync(assetPath)) {
              const assetStats = fs.statSync(assetPath);
              // Reasonable size check - no single JS file should be over 5MB
              expect(assetStats.size).toBeLessThan(5 * 1024 * 1024);
            }
          }
        });
      }
    });
  });

  it('should include proper metadata and configuration files', () => {
    // Check for Next.js metadata
    const appManifest = path.join(buildDir, 'app-build-manifest.json');
    if (fs.existsSync(appManifest)) {
      const manifestContent = fs.readFileSync(appManifest, 'utf8');
      expect(() => JSON.parse(manifestContent)).not.toThrow();
    }
    
    // Check for routes manifest
    const routesManifest = path.join(buildDir, 'routes-manifest.json');
    if (fs.existsSync(routesManifest)) {
      const routesContent = fs.readFileSync(routesManifest, 'utf8');
      const routes = JSON.parse(routesContent);
      
      // Should have some routes defined
      expect(routes.pages || routes.dynamicRoutes || routes.staticRoutes).toBeDefined();
    }
  });

  it('should validate that critical pages are buildable', () => {
    // Test that key application pages exist in build (App Router uses .rsc files)
    const serverAppDir = path.join(buildDir, 'server', 'app');
    
    const criticalPages = [
      'index.rsc', // Root page
      'app.rsc', // Dashboard app page
      'network.rsc', // Network page
      'mine.rsc', // Mine page
    ];
    
    criticalPages.forEach(pagePath => {
      const fullPath = path.join(serverAppDir, pagePath);
      if (fs.existsSync(fullPath)) {
        const pageStats = fs.statSync(fullPath);
        expect(pageStats.size).toBeGreaterThan(0);
      }
    });
  });

  it('should validate build can be started', () => {
    // Test that the build can be started (dry run)
    const packageJsonPath = path.join(projectRoot, 'package.json');
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    
    // Verify start script exists
    expect(packageJson.scripts.start).toBeDefined();
    expect(packageJson.scripts.start).toBe('next start');
    
    // Verify Next.js is available
    const nextPath = path.join(projectRoot, 'node_modules', 'next');
    expect(fs.existsSync(nextPath)).toBe(true);
    
    // Check that server.js exists for standalone builds
    const standaloneServer = path.join(buildDir, 'standalone', 'server.js');
    if (fs.existsSync(standaloneServer)) {
      const serverContent = fs.readFileSync(standaloneServer, 'utf8');
      expect(serverContent).toContain('server');
    }
  });

  it('should validate Docker deployment readiness', () => {
    // Check for Dockerfile or docker-related files
    const dockerfilePath = path.join(projectRoot, 'Dockerfile');
    const dockerComposePath = path.join(projectRoot, 'docker-compose.yml');
    
    // Docker files are optional, but if they exist, validate basic structure
    if (fs.existsSync(dockerfilePath)) {
      const dockerfileContent = fs.readFileSync(dockerfilePath, 'utf8');
      expect(dockerfileContent).toContain('FROM');
    }
    
    // Check that build output is suitable for containerization
    const nextConfigPath = path.join(projectRoot, 'next.config.js');
    const configContent = fs.readFileSync(nextConfigPath, 'utf8');
    
    // Standalone output is ideal for Docker
    expect(configContent).toContain("output: 'standalone'");
    
    // Check that public assets are properly configured
    const publicDir = path.join(projectRoot, 'public');
    if (fs.existsSync(publicDir)) {
      const publicFiles = fs.readdirSync(publicDir);
      expect(publicFiles.length).toBeGreaterThan(0);
    }
  });
});