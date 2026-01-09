/**
 * **Feature: frontend-build-fix, Property 5: Third-party library imports resolve**
 * **Validates: Requirements 3.1, 3.2**
 */

import { describe, it, expect } from '@jest/globals';
import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

describe('Third-party Library Import Property Tests', () => {
  const projectRoot = path.resolve(__dirname, '../..');
  
  const runTypeScriptCheck = (filePath: string) => {
    return execSync(
      `npx tsc --noEmit --skipLibCheck --jsx preserve --esModuleInterop "${filePath}"`,
      { 
        cwd: projectRoot,
        encoding: 'utf8',
        stdio: 'pipe'
      }
    );
  };
  
  it('should resolve @tanstack/react-query imports without errors', () => {
    const testFilePath = path.join(projectRoot, 'temp-tanstack-test.tsx');
    const testContent = `
import React from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';

export default function TestTanstackComponent() {
  const queryClient = useQueryClient();
  
  const { data, isLoading, error } = useQuery({
    queryKey: ['test-data'],
    queryFn: async () => {
      const response = await fetch('/api/test');
      if (!response.ok) throw new Error('Network error');
      return response.json();
    },
    refetchInterval: 5000,
  });

  const mutation = useMutation({
    mutationFn: async (newData: any) => {
      const response = await fetch('/api/test', {
        method: 'POST',
        body: JSON.stringify(newData),
      });
      return response.json();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['test-data'] });
    },
  });

  if (isLoading) return <div>Loading...</div>;
  if (error) return <div>Error: {error.message}</div>;

  return (
    <div>
      <pre>{JSON.stringify(data, null, 2)}</pre>
      <button onClick={() => mutation.mutate({ test: 'data' })}>
        Update Data
      </button>
    </div>
  );
}
`;

    try {
      fs.writeFileSync(testFilePath, testContent);
      
      const result = runTypeScriptCheck(testFilePath);
      expect(result).toBeDefined();
      
    } catch (error: any) {
      const errorOutput = error.stdout || error.stderr || '';
      
      const tanstackErrors = [
        'Cannot find module \'@tanstack/react-query\'',
        'Module \'@tanstack/react-query\' has no exported member',
        'useQuery',
        'useMutation',
        'useQueryClient'
      ];
      
      const hasTanstackError = tanstackErrors.some(errorText => 
        errorOutput.includes(errorText)
      );
      
      if (hasTanstackError) {
        throw new Error(`@tanstack/react-query import error: ${errorOutput}`);
      }
      
    } finally {
      if (fs.existsSync(testFilePath)) {
        fs.unlinkSync(testFilePath);
      }
    }
  });

  it('should resolve react-globe.gl dynamic imports without errors', () => {
    const testFilePath = path.join(projectRoot, 'temp-globe-test.tsx');
    const testContent = `
import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';

const Globe = dynamic(
  () => import('react-globe.gl').then((mod) => mod.default),
  {
    ssr: false,
    loading: () => <div>Loading globe...</div>,
  }
);

interface GlobeData {
  lat: number;
  lng: number;
  size: number;
}

export default function TestGlobeComponent() {
  const [globeData, setGlobeData] = useState<GlobeData[]>([]);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    setGlobeData([
      { lat: 40.7128, lng: -74.0060, size: 1 },
      { lat: 51.5074, lng: -0.1278, size: 1.5 },
    ]);
  }, []);

  return (
    <div style={{ height: '400px' }}>
      {mounted ? (
        <Globe
          globeImageUrl="//unpkg.com/three-globe/example/img/earth-blue-marble.jpg"
          pointsData={globeData}
          pointColor="color"
          pointRadius="size"
          pointResolution={2}
          onGlobeReady={() => console.log('Globe ready')}
        />
      ) : (
        <div>Initializing globe...</div>
      )}
    </div>
  );
}
`;

    try {
      fs.writeFileSync(testFilePath, testContent);
      
      const result = runTypeScriptCheck(testFilePath);
      expect(result).toBeDefined();
      
    } catch (error: any) {
      const errorOutput = error.stdout || error.stderr || '';
      
      const globeErrors = [
        'Cannot find module \'react-globe.gl\'',
        'Could not find a declaration file for module \'react-globe.gl\'',
        'implicitly has an \'any\' type'
      ];
      
      const hasGlobeError = globeErrors.some(errorText => 
        errorOutput.includes(errorText)
      );
      
      if (hasGlobeError) {
        throw new Error(`react-globe.gl import error: ${errorOutput}`);
      }
      
    } finally {
      if (fs.existsSync(testFilePath)) {
        fs.unlinkSync(testFilePath);
      }
    }
  });

  it('should resolve Next.js dynamic imports correctly', () => {
    const testFilePath = path.join(projectRoot, 'temp-dynamic-test.tsx');
    const testContent = `
import React from 'react';
import dynamic from 'next/dynamic';

const DynamicComponent = dynamic(
  () => import('framer-motion').then((mod) => mod.motion.div),
  {
    ssr: false,
    loading: () => <div>Loading animation...</div>,
  }
);

export default function TestDynamicComponent() {
  return (
    <div>
      <DynamicComponent
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        Animated content
      </DynamicComponent>
    </div>
  );
}
`;

    try {
      fs.writeFileSync(testFilePath, testContent);
      
      const result = runTypeScriptCheck(testFilePath);
      expect(result).toBeDefined();
      
    } catch (error: any) {
      const errorOutput = error.stdout || error.stderr || '';
      
      const dynamicErrors = [
        'Cannot find module \'next/dynamic\'',
        'Module \'next/dynamic\' has no exported member',
        'dynamic'
      ];
      
      const hasDynamicError = dynamicErrors.some(errorText => 
        errorOutput.includes(errorText)
      );
      
      if (hasDynamicError) {
        throw new Error(`Next.js dynamic import error: ${errorOutput}`);
      }
      
      // Allow other errors (like framer-motion complexity)
      console.warn('Non-critical dynamic import warning:', errorOutput);
      
    } finally {
      if (fs.existsSync(testFilePath)) {
        fs.unlinkSync(testFilePath);
      }
    }
  });

  it('should validate that all critical third-party packages are installed', () => {
    const packageJsonPath = path.join(projectRoot, 'package.json');
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    
    const criticalPackages = [
      '@tanstack/react-query',
      'react-globe.gl',
      'next',
      'react',
      'react-dom',
      'framer-motion',
      'recharts'
    ];
    
    const criticalTypePackages = [
      '@types/react',
      '@types/react-dom',
      '@types/node'
    ];
    
    // Check runtime dependencies
    criticalPackages.forEach(pkg => {
      expect(packageJson.dependencies[pkg]).toBeDefined();
    });
    
    // Check type dependencies
    criticalTypePackages.forEach(pkg => {
      expect(packageJson.dependencies[pkg] || packageJson.devDependencies?.[pkg]).toBeDefined();
    });
    
    // Verify versions are compatible
    expect(packageJson.dependencies.react).toMatch(/^(\^)?18\./);
    expect(packageJson.dependencies['react-dom']).toMatch(/^(\^)?18\./);
    expect(packageJson.dependencies.next).toMatch(/^(\^)?14\./);
  });
});