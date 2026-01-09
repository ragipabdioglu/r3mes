# Debug Mode Documentation

## Overview

R3MES provides a comprehensive debug mode system that enables detailed logging, performance profiling, state inspection, and distributed tracing across all components. Debug mode is designed for development, testing, and troubleshooting purposes.

## Quick Start

### Enable Debug Mode

```bash
# Set environment variables
export R3MES_DEBUG_MODE=true
export R3MES_DEBUG_LEVEL=verbose
export R3MES_DEBUG_COMPONENTS=blockchain,backend,miner

# Or use the startup script
source scripts/debug/start_debug_mode.sh

# Start components
remesd start  # Blockchain
python3 -m backend.app.main  # Backend
python3 miner_engine.py  # Miner
```

### Collect Debug Information

```bash
# Collect all debug logs, profiles, and traces
scripts/debug/collect_debug_info.sh

# Analyze log files
python3 scripts/debug/analyze_debug_logs.py ~/.r3mes/debug.log
```

## Configuration

### Debug Levels

- **minimal**: Only critical errors and performance metrics
- **standard**: Detailed logging, state inspection, performance profiling
- **verbose**: All system internals, trace logs, internal state dumps (default)

### Components

Specify which components should have debug enabled:

```bash
R3MES_DEBUG_COMPONENTS=blockchain,backend,miner,launcher,frontend
```

Or use `*` to enable all components.

### Feature Flags

- **R3MES_DEBUG_LOGGING**: Enable enhanced logging (default: true)
- **R3MES_DEBUG_PROFILING**: Enable performance profiling (default: true)
- **R3MES_DEBUG_STATE_INSPECTION**: Enable state inspection (default: true)
- **R3MES_DEBUG_TRACE**: Enable distributed tracing (default: true)

## Logging

### Log Levels

- **TRACE**: Most verbose, includes all internal operations
- **DEBUG**: Detailed debugging information
- **INFO**: General informational messages
- **WARN**: Warning messages
- **ERROR**: Error messages

### Log Formats

- **json**: Structured JSON format (default in debug mode)
- **text**: Human-readable text format

### Log Files

Debug logs are written to:
- Default: `~/.r3mes/debug.log`
- Configurable via: `R3MES_DEBUG_LOG_FILE`

Example log entry (JSON format):

```json
{
  "timestamp": "2024-01-01T12:00:00Z",
  "level": "DEBUG",
  "message": "Operation completed",
  "component": "blockchain",
  "block_height": 12345,
  "duration_ms": 42,
  "trace_id": "abc123"
}
```

## Performance Profiling

### Overview

Performance profiling tracks function execution times, memory usage, and system resources.

### Using the Profiler

#### Go (Blockchain)

```go
profiler := keeper.GetDebugProfiler()
if profiler != nil {
    end := profiler.StartTimer("my_operation")
    defer end()
    // ... your code ...
}
```

#### Python (Backend/Miner)

```python
from backend.app.performance_profiler import get_profiler

profiler = get_profiler()

# Profile a function
result = profiler.profile_function("my_function", lambda: my_function())

# Or use as context manager
with profiler.start_timer("my_operation"):
    # ... your code ...
    pass

# Export stats
profiler.export_stats("profile.json")
```

### Profile Output

Profiles are saved to: `~/.r3mes/profiles/` (configurable via `R3MES_DEBUG_PROFILE_OUTPUT`)

Profile file format:

```json
{
  "start_time": "2024-01-01T12:00:00Z",
  "end_time": "2024-01-01T12:01:00Z",
  "duration": 60.0,
  "profiles": {
    "my_function": {
      "function": "my_function",
      "call_count": 100,
      "total_duration": 5.2,
      "min_duration": 0.01,
      "max_duration": 0.5,
      "avg_duration": 0.052
    }
  },
  "memory_stats": {
    "current_mb": 128,
    "peak_mb": 256
  }
}
```

## State Inspection

### Overview

State inspection allows you to examine the internal state of the blockchain keeper and collections.

### Using State Inspector

#### Go (Blockchain)

```go
inspector := keeper.GetDebugStateInspector()
if inspector != nil {
    // Get state dump
    dump, err := inspector.GetStateDump(ctx)
    
    // Get collection stats
    stats, err := inspector.GetCollectionStats(ctx)
    
    // Get params
    params, err := inspector.GetParamsDump(ctx)
    
    // Get cache stats
    cacheStats, err := inspector.GetCacheStats(ctx)
}
```

### State Dump Format

```json
{
  "block_height": 12345,
  "block_time": "2024-01-01T12:00:00Z",
  "params": {
    "mining_difficulty": 1234.0
  },
  "collection_stats": [
    {
      "collection_name": "StoredGradients",
      "entry_count": 1000
    }
  ],
  "cache_stats": {
    "gradient_cache": {
      "enabled": true
    }
  }
}
```

## Distributed Tracing

### Overview

Distributed tracing tracks operations across multiple components using trace IDs.

### Trace IDs

Trace IDs are automatically generated and propagated across components. They allow you to correlate logs and operations across the system.

Example trace flow:

1. Request arrives at backend → Trace ID generated: `abc123`
2. Backend calls blockchain → Trace ID propagated: `abc123`
3. Blockchain processes request → Trace ID logged: `abc123`
4. All logs with trace ID `abc123` can be correlated

### Using Traces

Traces are automatically collected when `R3MES_DEBUG_TRACE=true`. Trace data is stored in memory and can be exported to files.

## Security Considerations

### Production Mode

**WARNING**: Debug mode should **NEVER** be enabled in production!

- Security validation will fail if `R3MES_DEBUG_MODE=true` in production
- Debug logs may contain sensitive information
- Performance profiling has overhead
- State inspection exposes internal state

### Sensitive Data

Debug logs automatically filter sensitive data:
- Private keys
- Passwords
- API keys
- Authentication tokens

However, always review logs before sharing them.

## Troubleshooting

### Debug Mode Not Working

1. Check environment variables:
   ```bash
   env | grep R3MES_DEBUG
   ```

2. Verify debug config is loaded:
   ```go
   // Go
   config := GetDebugConfig()
   fmt.Printf("Enabled: %v\n", config.Enabled)
   ```

   ```python
   # Python
   from backend.app.debug_config import get_debug_config
   config = get_debug_config()
   print(f"Enabled: {config.enabled}")
   ```

3. Check log file permissions:
   ```bash
   ls -la ~/.r3mes/debug.log
   ```

### Performance Issues

If debug mode causes performance issues:

1. Use `minimal` debug level:
   ```bash
   export R3MES_DEBUG_LEVEL=minimal
   ```

2. Disable specific features:
   ```bash
   export R3MES_DEBUG_PROFILING=false
   export R3MES_DEBUG_TRACE=false
   ```

3. Limit components:
   ```bash
   export R3MES_DEBUG_COMPONENTS=blockchain  # Only blockchain
   ```

### Log File Too Large

1. Use log rotation (automatic)
2. Reduce log level:
   ```bash
   export R3MES_DEBUG_LOG_LEVEL=INFO  # Less verbose
   ```
3. Clean up old logs:
   ```bash
   rm ~/.r3mes/debug.log
   ```

## Best Practices

1. **Use debug mode only in development/testing**
2. **Start with `standard` level, increase to `verbose` only when needed**
3. **Enable profiling only when investigating performance issues**
4. **Regularly clean up debug logs and profiles**
5. **Never commit debug logs to version control**
6. **Review logs before sharing them (check for sensitive data)**

## Examples

### Example 1: Debug a specific operation

```bash
# Enable debug mode
export R3MES_DEBUG_MODE=true
export R3MES_DEBUG_LEVEL=verbose
export R3MES_DEBUG_COMPONENTS=blockchain

# Run operation
remesd tx remes submit-gradient ...

# Check logs
tail -f ~/.r3mes/debug.log | grep "submit-gradient"
```

### Example 2: Profile performance

```python
from backend.app.performance_profiler import get_profiler

profiler = get_profiler()

# Profile inference
result = profiler.profile_function("inference", lambda: model.infer(data))

# Export profile
profiler.export_stats("inference_profile.json")

# Analyze
python3 scripts/debug/analyze_debug_logs.py inference_profile.json
```

### Example 3: Inspect state

```go
// Get state inspector
inspector := keeper.GetDebugStateInspector()
if inspector != nil {
    // Get full state dump
    dump, err := inspector.GetStateDump(ctx)
    
    // Convert to JSON for inspection
    jsonData, _ := json.MarshalIndent(dump, "", "  ")
    fmt.Println(string(jsonData))
}
```

## References

- Environment Variables: [16_environment_variables.md](./16_environment_variables.md)
- Security: [PRODUCTION_SECURITY.md](./PRODUCTION_SECURITY.md)
- Testing Guide: [11_testing_qa.md](./11_testing_qa.md)
