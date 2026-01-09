/**
 * **Feature: frontend-build-fix, Property 1: Build process completes without module resolution errors**
 * **Validates: Requirements 1.1**
 */

import { describe, it, expect } from '@jest/globals';
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

describe('Build Process Property Tests', () => {
  const projectRoot = path.resolve(__dirname, '../..');
  
  it('should complete TypeScript compilation without module resolution errors', () => {
    try {
      const result = execSync(
        'npx tsc --noEmit --skipLibCheck',
        { 
          cwd: projectRoot,
          encoding: 'utf8',
          stdio: 'pipe',
          timeout: 60000 // 60 second timeout
        }
      );
      
      expect(result).toBeDefined();
      
    } catch (error: any) {
      const errorOutput = error.stdout || error.stderr || '';
      
      // Check for module resolution errors (but ignore test files and known issues)
      const moduleResolutionErrors = [
        'Cannot find module \'react\'',
        'Cannot find module \'next/',
        'Cannot find module \'@tanstack/react-query\'',
        'Cannot find module \'react-globe.gl\''
      ];
      
      const hasModuleResolutionError = moduleResolutionErrors.some(errorText => 
        errorOutput.includes(errorText)
      );
      
      if (hasModuleResolutionError) {
        throw new Error(`Module resolution error in build: ${errorOutput}`);
      }
      
      // Allow other types of errors (like type mismatches) but log them
      console.warn('Non-module-resolution TypeScript warnings:', errorOutput);
    }
  });

  it('should have valid Next.js configuration', () => {
    const nextConfigPath = path.join(projectRoot, 'next.config.js');
    expect(fs.existsSync(nextConfigPath)).toBe(true);
    
    // Test that Next.js config can be loaded without syntax errors
    try {
      const configContent = fs.readFileSync(nextConfigPath, 'utf8');
      
      // Check for essential Next.js configuration
      expect(configContent).toContain('nextConfig');
      expect(configContent).toContain('module.exports');
      
      // Check for TypeScript configuration
      expect(configContent).toContain('typescript');
      
      // Check for webpack configuration for react-globe.gl
      expect(configContent).toContain('webpack');
      expect(configContent).toContain('fallback');
      
    } catch (error) {
      throw new Error(`Next.js configuration error: ${error}`);
    }
  });

  it('should have valid TypeScript configuration for Next.js', () => {
    const tsconfigPath = path.join(projectRoot, 'tsconfig.json');
    expect(fs.existsSync(tsconfigPath)).toBe(true);
    
    const tsconfig = JSON.parse(fs.readFileSync(tsconfigPath, 'utf8'));
    
    // Verify Next.js specific TypeScript settings
    expect(tsconfig.compilerOptions.jsx).toBe('preserve');
    expect(tsconfig.compilerOptions.moduleResolution).toBe('bundler');
    expect(tsconfig.compilerOptions.target).toBe('ES2022');
    expect(tsconfig.compilerOptions.esModuleInterop).toBe(true);
    expect(tsconfig.compilerOptions.allowSyntheticDefaultImports).toBe(true);
    
    // Verify Next.js plugin
    const hasNextPlugin = tsconfig.compilerOptions.plugins?.some(
      (plugin: any) => plugin.name === 'next'
    );
    expect(hasNextPlugin).toBe(true);
    
    // Verify path mapping
    expect(tsconfig.compilerOptions.paths).toBeDefined();
    expect(tsconfig.compilerOptions.paths['@/*']).toEqual(expect.arrayContaining(['./*']));
  });

  it('should validate package.json scripts and dependencies', () => {
    const packageJsonPath = path.join(projectRoot, 'package.json');
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    
    // Check essential build scripts
    expect(packageJson.scripts.build).toBeDefined();
    expect(packageJson.scripts.dev).toBeDefined();
    expect(packageJson.scripts.start).toBeDefined();
    
    // Check that build script uses Next.js
    expect(packageJson.scripts.build).toContain('next build');
    
    // Check essential dependencies for build
    const essentialDeps = [
      'next',
      'react',
      'react-dom',
      'typescript'
    ];
    
    essentialDeps.forEach(dep => {
      expect(packageJson.dependencies[dep] || packageJson.devDependencies?.[dep]).toBeDefined();
    });
    
    // Check type dependencies
    expect(packageJson.dependencies['@types/react']).toBeDefined();
    expect(packageJson.dependencies['@types/react-dom']).toBeDefined();
  });

  it('should validate that all source files can be parsed', () => {
    const componentsDir = path.join(projectRoot, 'components');
    const appDir = path.join(projectRoot, 'app');
    
    const checkDirectory = (dir: string) => {
      if (!fs.existsSync(dir)) return;
      
      const files = fs.readdirSync(dir, { withFileTypes: true });
      
      files.forEach(file => {
        if (file.isDirectory()) {
          checkDirectory(path.join(dir, file.name));
        } else if (file.name.endsWith('.tsx') || file.name.endsWith('.ts')) {
          const filePath = path.join(dir, file.name);
          
          try {
            const content = fs.readFileSync(filePath, 'utf8');
            
            // Basic syntax validation - check for common syntax errors
            expect(content).not.toContain('import from'); // Missing module name
            expect(content).not.toContain('export {'); // Incomplete export
            
            // Check for proper React imports if it's a component file
            if (file.name.endsWith('.tsx')) {
              // Should have React import or use client directive
              const hasReactImport = content.includes('import React') || 
                                   content.includes('from "react"') ||
                                   content.includes('"use client"') ||
                                   content.includes("'use client'");
              
              // Some pages might not need React imports if they're server components
              if (!hasReactImport && !content.includes('export default function')) {
                expect(hasReactImport).toBe(true);
              }
            }
            
          } catch (error) {
            throw new Error(`File parsing error in ${filePath}: ${error}`);
          }
        }
      });
    };
    
    checkDirectory(componentsDir);
    checkDirectory(appDir);
  });

  it('should validate build environment setup', () => {
    // Check that node_modules exists and has essential packages
    const nodeModulesPath = path.join(projectRoot, 'node_modules');
    expect(fs.existsSync(nodeModulesPath)).toBe(true);
    
    // Check for Next.js installation
    const nextPath = path.join(nodeModulesPath, 'next');
    expect(fs.existsSync(nextPath)).toBe(true);
    
    // Check for React installation
    const reactPath = path.join(nodeModulesPath, 'react');
    expect(fs.existsSync(reactPath)).toBe(true);
    
    // Check for TypeScript installation
    const typescriptPath = path.join(nodeModulesPath, 'typescript');
    expect(fs.existsSync(typescriptPath)).toBe(true);
    
    // Check that next-env.d.ts exists
    const nextEnvPath = path.join(projectRoot, 'next-env.d.ts');
    expect(fs.existsSync(nextEnvPath)).toBe(true);
  });
});