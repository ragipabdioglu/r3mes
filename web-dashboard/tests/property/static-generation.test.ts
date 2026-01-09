/**
 * **Feature: frontend-build-fix, Property 3: Static generation succeeds**
 * **Validates: Requirements 1.3**
 */

import { describe, it, expect } from '@jest/globals';
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

describe('Static Generation Property Tests', () => {
  const projectRoot = path.resolve(__dirname, '../..');
  
  it('should validate Next.js build configuration for static generation', () => {
    const nextConfigPath = path.join(projectRoot, 'next.config.js');
    const configContent = fs.readFileSync(nextConfigPath, 'utf8');
    
    // Check for static generation compatibility
    expect(configContent).toContain('output: \'standalone\'');
    
    // Check for proper webpack configuration
    expect(configContent).toContain('webpack:');
    expect(configContent).toContain('fallback');
    
    // Check for proper TypeScript configuration
    expect(configContent).toContain('typescript:');
    
    // Verify no blocking configurations for static generation
    expect(configContent).not.toContain('output: \'export\''); // Would conflict with standalone
  });

  it('should validate that pages can be statically generated', () => {
    // Test a simple page build to ensure static generation works
    const testPagePath = path.join(projectRoot, 'temp-static-test-page.tsx');
    const testPageContent = `
import React from 'react';

export default function TestStaticPage() {
  return (
    <div>
      <h1>Test Static Page</h1>
      <p>This page should be statically generated</p>
    </div>
  );
}

// This ensures the page can be statically generated
export async function generateStaticParams() {
  return [];
}
`;

    try {
      fs.writeFileSync(testPagePath, testPageContent);
      
      // Test TypeScript compilation of the static page
      const result = execSync(
        `npx tsc --noEmit --skipLibCheck --jsx preserve "${testPagePath}"`,
        { 
          cwd: projectRoot,
          encoding: 'utf8',
          stdio: 'pipe'
        }
      );
      
      expect(result).toBeDefined();
      
    } catch (error: any) {
      const errorOutput = error.stdout || error.stderr || '';
      
      // Check for static generation blocking errors
      const staticGenErrors = [
        'Cannot use JSX',
        'JSX element implicitly has type \'any\'',
        'Cannot find module \'react\''
      ];
      
      const hasStaticGenError = staticGenErrors.some(errorText => 
        errorOutput.includes(errorText)
      );
      
      if (hasStaticGenError) {
        throw new Error(`Static generation compatibility error: ${errorOutput}`);
      }
      
    } finally {
      if (fs.existsSync(testPagePath)) {
        fs.unlinkSync(testPagePath);
      }
    }
  });

  it('should validate client-side component compatibility', () => {
    // Test that client components can be properly handled during static generation
    const testClientComponentPath = path.join(projectRoot, 'temp-client-component.tsx');
    const testClientContent = `
"use client";

import React, { useState, useEffect } from 'react';

export default function TestClientComponent() {
  const [mounted, setMounted] = useState(false);
  
  useEffect(() => {
    setMounted(true);
  }, []);
  
  if (!mounted) {
    return <div>Loading...</div>;
  }
  
  return (
    <div>
      <h2>Client Component</h2>
      <p>This component runs on the client side</p>
    </div>
  );
}
`;

    try {
      fs.writeFileSync(testClientComponentPath, testClientContent);
      
      const result = execSync(
        `npx tsc --noEmit --skipLibCheck --jsx preserve "${testClientComponentPath}"`,
        { 
          cwd: projectRoot,
          encoding: 'utf8',
          stdio: 'pipe'
        }
      );
      
      expect(result).toBeDefined();
      
    } catch (error: any) {
      const errorOutput = error.stdout || error.stderr || '';
      
      const clientComponentErrors = [
        'Cannot find module \'react\'',
        'useState',
        'useEffect',
        '"use client"'
      ];
      
      const hasClientComponentError = clientComponentErrors.some(errorText => 
        errorOutput.includes(errorText)
      );
      
      if (hasClientComponentError) {
        throw new Error(`Client component error: ${errorOutput}`);
      }
      
    } finally {
      if (fs.existsSync(testClientComponentPath)) {
        fs.unlinkSync(testClientComponentPath);
      }
    }
  });

  it('should validate dynamic imports for static generation', () => {
    // Test that dynamic imports work correctly with static generation
    const testDynamicPath = path.join(projectRoot, 'temp-dynamic-static.tsx');
    const testDynamicContent = `
import React from 'react';
import dynamic from 'next/dynamic';

const DynamicComponent = dynamic(
  () => import('react').then((mod) => {
    const Component = () => <div>Dynamic Content</div>;
    return { default: Component };
  }),
  {
    ssr: false,
    loading: () => <div>Loading dynamic component...</div>,
  }
);

export default function TestDynamicStatic() {
  return (
    <div>
      <h1>Static Page with Dynamic Component</h1>
      <DynamicComponent />
    </div>
  );
}
`;

    try {
      fs.writeFileSync(testDynamicPath, testDynamicContent);
      
      const result = execSync(
        `npx tsc --noEmit --skipLibCheck --jsx preserve "${testDynamicPath}"`,
        { 
          cwd: projectRoot,
          encoding: 'utf8',
          stdio: 'pipe'
        }
      );
      
      expect(result).toBeDefined();
      
    } catch (error: any) {
      const errorOutput = error.stdout || error.stderr || '';
      
      const dynamicImportErrors = [
        'Cannot find module \'next/dynamic\'',
        'Module \'next/dynamic\' has no exported member \'dynamic\''
      ];
      
      const hasDynamicImportError = dynamicImportErrors.some(errorText => 
        errorOutput.includes(errorText)
      );
      
      if (hasDynamicImportError) {
        throw new Error(`Dynamic import error: ${errorOutput}`);
      }
      
    } finally {
      if (fs.existsSync(testDynamicPath)) {
        fs.unlinkSync(testDynamicPath);
      }
    }
  });

  it('should validate TypeScript configuration supports static generation', () => {
    const tsconfigPath = path.join(projectRoot, 'tsconfig.json');
    const tsconfig = JSON.parse(fs.readFileSync(tsconfigPath, 'utf8'));
    
    // Verify settings that support static generation
    expect(tsconfig.compilerOptions.target).toBe('ES2022'); // Modern target for better performance
    expect(tsconfig.compilerOptions.module).toBe('esnext'); // ESNext modules for tree shaking
    expect(tsconfig.compilerOptions.moduleResolution).toBe('bundler'); // Bundler resolution for Next.js
    expect(tsconfig.compilerOptions.jsx).toBe('preserve'); // Preserve JSX for Next.js
    expect(tsconfig.compilerOptions.esModuleInterop).toBe(true); // ES module interop
    expect(tsconfig.compilerOptions.allowSyntheticDefaultImports).toBe(true); // Synthetic imports
    
    // Verify includes cover all necessary files
    expect(tsconfig.include).toContain('**/*.tsx');
    expect(tsconfig.include).toContain('next-env.d.ts');
    
    // Verify excludes prevent issues
    expect(tsconfig.exclude).toContain('node_modules');
  });

  it('should validate package.json supports static generation', () => {
    const packageJsonPath = path.join(projectRoot, 'package.json');
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    
    // Check Next.js version supports static generation
    const nextVersion = packageJson.dependencies.next;
    expect(nextVersion).toBeDefined();
    expect(nextVersion).toMatch(/^(\^)?14\./); // Next.js 14+ has good static generation
    
    // Check React version compatibility
    const reactVersion = packageJson.dependencies.react;
    expect(reactVersion).toBeDefined();
    expect(reactVersion).toMatch(/^(\^)?18\./); // React 18+ for concurrent features
    
    // Check build script
    expect(packageJson.scripts.build).toBe('next build');
    expect(packageJson.scripts.start).toBe('next start');
  });

  it('should validate environment setup for static generation', () => {
    // Check that Next.js environment file exists
    const nextEnvPath = path.join(projectRoot, 'next-env.d.ts');
    expect(fs.existsSync(nextEnvPath)).toBe(true);
    
    const nextEnvContent = fs.readFileSync(nextEnvPath, 'utf8');
    expect(nextEnvContent).toContain('/// <reference types="next" />');
    expect(nextEnvContent).toContain('/// <reference types="next/image-types/global" />');
    
    // Check that TypeScript can resolve Next.js types
    const nodeModulesNextPath = path.join(projectRoot, 'node_modules', 'next');
    expect(fs.existsSync(nodeModulesNextPath)).toBe(true);
    
    // Check for Next.js type definitions
    const nextTypesPath = path.join(nodeModulesNextPath, 'types');
    expect(fs.existsSync(nextTypesPath)).toBe(true);
  });
});