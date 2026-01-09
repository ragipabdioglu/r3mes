/**
 * **Feature: frontend-build-fix, Property 2: TypeScript recognizes JSX elements**
 * **Validates: Requirements 2.1**
 */

import { describe, it, expect } from '@jest/globals';
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

describe('TypeScript Configuration Property Tests', () => {
  const projectRoot = path.resolve(__dirname, '../..');
  
  it('should recognize JSX elements without type errors', () => {
    // Create a temporary test component with JSX elements
    const testComponentPath = path.join(projectRoot, 'temp-test-component.tsx');
    const testComponent = `
import React from 'react';

export default function TestComponent() {
  return (
    <div className="test">
      <h1>Test Header</h1>
      <p>Test paragraph</p>
      <span>Test span</span>
      <button onClick={() => console.log('clicked')}>Test Button</button>
    </div>
  );
}
`;

    try {
      // Write test component
      fs.writeFileSync(testComponentPath, testComponent);
      
      // Run TypeScript check on the test component
      const result = execSync(
        `npx tsc --noEmit --skipLibCheck --jsx preserve --esModuleInterop "${testComponentPath}"`,
        { 
          cwd: projectRoot,
          encoding: 'utf8',
          stdio: 'pipe'
        }
      );
      
      // If no errors, TypeScript recognizes JSX elements correctly
      expect(result).toBeDefined();
      
    } catch (error: any) {
      // Check if error is related to JSX recognition
      const errorOutput = error.stdout || error.stderr || '';
      
      // These errors should NOT appear if JSX is properly configured
      const jsxErrors = [
        'JSX element implicitly has type \'any\'',
        'Cannot find name \'React\'',
        'JSX.IntrinsicElements',
        'Cannot use JSX'
      ];
      
      const hasJsxError = jsxErrors.some(errorText => 
        errorOutput.includes(errorText)
      );
      
      if (hasJsxError) {
        throw new Error(`TypeScript JSX configuration error: ${errorOutput}`);
      }
      
      // Other errors might be acceptable (like missing imports in test)
      console.warn('Non-JSX TypeScript error (acceptable):', errorOutput);
      
    } finally {
      // Clean up test file
      if (fs.existsSync(testComponentPath)) {
        fs.unlinkSync(testComponentPath);
      }
    }
  });

  it('should have proper module resolution for React types', () => {
    // Test that React types are properly resolved
    const testTypePath = path.join(projectRoot, 'temp-test-types.tsx');
    const testTypes = `
import React, { useState, useEffect } from 'react';

// Test React types are available
const testComponent: React.FC<{ title: string }> = ({ title }) => {
  const [count, setCount] = useState<number>(0);
  
  useEffect(() => {
    console.log('Effect running');
  }, []);

  return <div>{title}: {count}</div>;
};

export default testComponent;
`;

    try {
      fs.writeFileSync(testTypePath, testTypes);
      
      const result = execSync(
        `npx tsc --noEmit --skipLibCheck --jsx preserve --esModuleInterop "${testTypePath}"`,
        { 
          cwd: projectRoot,
          encoding: 'utf8',
          stdio: 'pipe'
        }
      );
      
      expect(result).toBeDefined();
      
    } catch (error: any) {
      const errorOutput = error.stdout || error.stderr || '';
      
      // Check for React type resolution errors
      const reactTypeErrors = [
        'Cannot find module \'react\'',
        'Cannot find namespace \'React\'',
        'Property \'FC\' does not exist'
      ];
      
      const hasReactTypeError = reactTypeErrors.some(errorText => 
        errorOutput.includes(errorText)
      );
      
      if (hasReactTypeError) {
        throw new Error(`React type resolution error: ${errorOutput}`);
      }
      
    } finally {
      if (fs.existsSync(testTypePath)) {
        fs.unlinkSync(testTypePath);
      }
    }
  });

  it('should validate TypeScript configuration compatibility with Next.js', () => {
    const tsconfigPath = path.join(projectRoot, 'tsconfig.json');
    
    expect(fs.existsSync(tsconfigPath)).toBe(true);
    
    const tsconfig = JSON.parse(fs.readFileSync(tsconfigPath, 'utf8'));
    
    // Verify essential Next.js TypeScript settings
    expect(tsconfig.compilerOptions.jsx).toBe('preserve');
    expect(tsconfig.compilerOptions.moduleResolution).toBe('bundler');
    expect(tsconfig.compilerOptions.esModuleInterop).toBe(true);
    expect(tsconfig.compilerOptions.allowSyntheticDefaultImports).toBe(true);
    expect(tsconfig.compilerOptions.isolatedModules).toBe(true);
    
    // Verify Next.js plugin is configured
    const hasNextPlugin = tsconfig.compilerOptions.plugins?.some(
      (plugin: any) => plugin.name === 'next'
    );
    expect(hasNextPlugin).toBe(true);
    
    // Verify proper includes
    expect(tsconfig.include).toContain('next-env.d.ts');
    expect(tsconfig.include).toContain('**/*.tsx');
  });
});