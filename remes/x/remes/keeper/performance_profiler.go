package keeper

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"
)

// PerformanceProfiler provides performance profiling capabilities
type PerformanceProfiler struct {
	config    *DebugConfig
	enabled   bool
	profiles  map[string]*ProfileEntry
	mu        sync.RWMutex
	startTime time.Time
}

// ProfileEntry represents a single profiling entry
type ProfileEntry struct {
	Function      string        `json:"function"`
	CallCount     int64         `json:"call_count"`
	TotalDuration time.Duration `json:"total_duration"`
	MinDuration   time.Duration `json:"min_duration"`
	MaxDuration   time.Duration `json:"max_duration"`
	AvgDuration   time.Duration `json:"avg_duration"`
	LastCallTime  time.Time     `json:"last_call_time"`
}

// ProfileStats represents overall profiling statistics
type ProfileStats struct {
	StartTime    time.Time                `json:"start_time"`
	EndTime      time.Time                `json:"end_time"`
	Duration     time.Duration            `json:"duration"`
	Profiles     map[string]*ProfileEntry `json:"profiles"`
	MemoryStats  runtime.MemStats         `json:"memory_stats"`
	GoroutineCount int                    `json:"goroutine_count"`
}

// NewPerformanceProfiler creates a new performance profiler
func NewPerformanceProfiler(config *DebugConfig) (*PerformanceProfiler, error) {
	profiler := &PerformanceProfiler{
		config:    config,
		enabled:   config != nil && config.Enabled && config.Profiling && config.IsBlockchainEnabled(),
		profiles:  make(map[string]*ProfileEntry),
		startTime: time.Now(),
	}

	return profiler, nil
}

// StartTimer starts a timer for a function/procedure
func (pp *PerformanceProfiler) StartTimer(functionName string) func() {
	if !pp.enabled {
		return func() {} // No-op if profiling is disabled
	}

	start := time.Now()

	return func() {
		duration := time.Since(start)
		pp.recordProfile(functionName, duration)
	}
}

// recordProfile records a profile entry
func (pp *PerformanceProfiler) recordProfile(functionName string, duration time.Duration) {
	pp.mu.Lock()
	defer pp.mu.Unlock()

	entry, exists := pp.profiles[functionName]
	if !exists {
		entry = &ProfileEntry{
			Function:     functionName,
			MinDuration:  duration,
			MaxDuration:  duration,
			LastCallTime: time.Now(),
		}
		pp.profiles[functionName] = entry
	}

	entry.CallCount++
	entry.TotalDuration += duration
	if duration < entry.MinDuration {
		entry.MinDuration = duration
	}
	if duration > entry.MaxDuration {
		entry.MaxDuration = duration
	}
	entry.AvgDuration = entry.TotalDuration / time.Duration(entry.CallCount)
	entry.LastCallTime = time.Now()
}

// GetStats returns current profiling statistics
func (pp *PerformanceProfiler) GetStats() *ProfileStats {
	pp.mu.RLock()
	defer pp.mu.RUnlock()

	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)

	// Create a copy of profiles map
	profilesCopy := make(map[string]*ProfileEntry)
	for k, v := range pp.profiles {
		entryCopy := *v
		profilesCopy[k] = &entryCopy
	}

	return &ProfileStats{
		StartTime:      pp.startTime,
		EndTime:        time.Now(),
		Duration:       time.Since(pp.startTime),
		Profiles:       profilesCopy,
		MemoryStats:    memStats,
		GoroutineCount: runtime.NumGoroutine(),
	}
}

// ExportStats exports profiling statistics to a file
func (pp *PerformanceProfiler) ExportStats(filename string) error {
	if !pp.enabled {
		return nil
	}

	stats := pp.GetStats()

	// Create directory if it doesn't exist
	dir := filepath.Dir(filename)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create profile directory: %w", err)
	}

	// Write stats as JSON
	data, err := json.MarshalIndent(stats, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal stats: %w", err)
	}

	if err := os.WriteFile(filename, data, 0644); err != nil {
		return fmt.Errorf("failed to write profile file: %w", err)
	}

	return nil
}

// Reset resets all profiling data
func (pp *PerformanceProfiler) Reset() {
	pp.mu.Lock()
	defer pp.mu.Unlock()

	pp.profiles = make(map[string]*ProfileEntry)
	pp.startTime = time.Now()
}

// ProfileFunction profiles a function execution
func (pp *PerformanceProfiler) ProfileFunction(functionName string, fn func() error) error {
	if !pp.enabled {
		return fn()
	}

	end := pp.StartTimer(functionName)
	defer end()
	return fn()
}

// GetMemoryStats returns current memory statistics
func (pp *PerformanceProfiler) GetMemoryStats() runtime.MemStats {
	var memStats runtime.MemStats
	runtime.ReadMemStats(&memStats)
	return memStats
}

// GetGoroutineCount returns the current number of goroutines
func (pp *PerformanceProfiler) GetGoroutineCount() int {
	return runtime.NumGoroutine()
}

// GetDebugProfiler returns the performance profiler for the keeper
func (k Keeper) GetDebugProfiler() *PerformanceProfiler {
	if k.debugConfig == nil || !k.debugConfig.Enabled || !k.debugConfig.Profiling || !k.debugConfig.IsBlockchainEnabled() {
		return nil
	}

	// Create profiler instance (cached would be better, but for simplicity we create on-demand)
	profiler, err := NewPerformanceProfiler(k.debugConfig)
	if err != nil {
		// If profiler creation fails, return nil (profiling is optional)
		return nil
	}

	return profiler
}
