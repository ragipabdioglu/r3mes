/**
 * Debug utilities for R3MES Frontend
 *
 * Provides frontend debug utilities, component render tracking, API call tracing, and state inspection UI.
 */

export interface DebugConfig {
  enabled: boolean;
  logLevel: 'TRACE' | 'DEBUG' | 'INFO' | 'WARN' | 'ERROR';
  components: string[];
}

export interface TraceEntry {
  traceId: string;
  component: string;
  operation: string;
  startTime: number;
  endTime?: number;
  durationMs?: number;
  fields?: Record<string, any>;
  error?: string;
}

/**
 * Load debug configuration from environment variables
 */
export function loadDebugConfig(): DebugConfig {
  // In browser, check localStorage or window.env
  const debugMode = 
    typeof window !== 'undefined' && 
    (localStorage.getItem('R3MES_DEBUG_MODE') === 'true' || 
     (window as any).__R3MES_DEBUG_MODE__ === 'true');
  
  const logLevel = 
    typeof window !== 'undefined' && localStorage.getItem('R3MES_DEBUG_LOG_LEVEL') as any ||
    'INFO';
  
  const componentsStr = 
    typeof window !== 'undefined' && localStorage.getItem('R3MES_DEBUG_COMPONENTS') ||
    '';
  
  const components = componentsStr 
    ? componentsStr.split(',').map(c => c.trim().toLowerCase())
    : ['*'];
  
  return {
    enabled: debugMode,
    logLevel: logLevel || 'INFO',
    components,
  };
}

/**
 * Check if debug is enabled for a component
 */
export function isComponentEnabled(config: DebugConfig, component: string): boolean {
  if (!config.enabled) {
    return false;
  }
  
  if (config.components.includes('*') || config.components.length === 0) {
    return true;
  }
  
  return config.components.includes(component.toLowerCase());
}

/**
 * Debug logger
 */
class DebugLogger {
  private config: DebugConfig;
  
  constructor() {
    this.config = loadDebugConfig();
  }
  
  private shouldLog(level: string): boolean {
    if (!this.config.enabled) {
      return false;
    }
    
    const levels = ['TRACE', 'DEBUG', 'INFO', 'WARN', 'ERROR'];
    const configLevel = levels.indexOf(this.config.logLevel);
    const logLevel = levels.indexOf(level);
    
    return logLevel >= configLevel;
  }
  
  trace(message: string, ...args: any[]): void {
    if (this.shouldLog('TRACE')) {
      console.trace(`[TRACE] ${message}`, ...args);
    }
  }
  
  debug(message: string, ...args: any[]): void {
    if (this.shouldLog('DEBUG')) {
      console.debug(`[DEBUG] ${message}`, ...args);
    }
  }
  
  info(message: string, ...args: any[]): void {
    if (this.shouldLog('INFO')) {
      console.info(`[INFO] ${message}`, ...args);
    }
  }
  
  warn(message: string, ...args: any[]): void {
    if (this.shouldLog('WARN')) {
      console.warn(`[WARN] ${message}`, ...args);
    }
  }
  
  error(message: string, ...args: any[]): void {
    if (this.shouldLog('ERROR')) {
      console.error(`[ERROR] ${message}`, ...args);
    }
  }
}

export const debugLogger = new DebugLogger();

/**
 * Component render tracker
 */
export function trackComponentRender(componentName: string, props?: Record<string, any>): void {
  const config = loadDebugConfig();
  if (!isComponentEnabled(config, 'frontend')) {
    return;
  }
  
  debugLogger.trace(`Component render: ${componentName}`, { props });
}

/**
 * API call tracer
 */
export function traceApiCall(
  method: string,
  url: string,
  startTime: number,
  endTime?: number,
  error?: Error
): TraceEntry {
  const config = loadDebugConfig();
  if (!isComponentEnabled(config, 'frontend')) {
    return {} as TraceEntry;
  }
  
  const traceId = `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  const durationMs = endTime ? endTime - startTime : undefined;
  
  const entry: TraceEntry = {
    traceId,
    component: 'frontend',
    operation: `api_${method.toLowerCase()}`,
    startTime,
    endTime,
    durationMs,
    fields: {
      method,
      url,
    },
    error: error?.message,
  };
  
  if (error) {
    debugLogger.error(`API call failed: ${method} ${url}`, { error, traceId });
  } else {
    debugLogger.trace(`API call: ${method} ${url}`, { durationMs, traceId });
  }
  
  return entry;
}

/**
 * State inspector
 */
export function inspectState(stateName: string, state: any): void {
  const config = loadDebugConfig();
  if (!isComponentEnabled(config, 'frontend')) {
    return;
  }
  
  debugLogger.debug(`State inspection: ${stateName}`, { state });
}

/**
 * Performance monitor
 */
export class PerformanceMonitor {
  private marks: Map<string, number> = new Map();
  
  mark(name: string): void {
    this.marks.set(name, performance.now());
  }
  
  measure(name: string, startMark: string, endMark?: string): number | null {
    const start = this.marks.get(startMark);
    if (!start) {
      return null;
    }
    
    const end = endMark ? this.marks.get(endMark) : performance.now();
    if (!end) {
      return null;
    }
    
    const duration = end - start;
    debugLogger.trace(`Performance: ${name}`, { duration, startMark, endMark });
    return duration;
  }
  
  clear(): void {
    this.marks.clear();
  }
}

export const performanceMonitor = new PerformanceMonitor();
