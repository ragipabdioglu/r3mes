package keeper

import (
	"crypto/rand"
	"encoding/hex"
	"sync"
	"time"

	sdk "github.com/cosmos/cosmos-sdk/types"
)

// TraceEntry represents a single trace entry
type TraceEntry struct {
	TraceID      string                 `json:"trace_id"`
	Component    string                 `json:"component"`
	Operation    string                 `json:"operation"`
	StartTime    time.Time              `json:"start_time"`
	EndTime      time.Time              `json:"end_time,omitempty"`
	Duration     time.Duration          `json:"duration,omitempty"`
	BlockHeight  int64                  `json:"block_height"`
	Fields       map[string]interface{} `json:"fields,omitempty"`
	Error        string                 `json:"error,omitempty"`
}

// TraceCollector collects trace entries
type TraceCollector struct {
	config    *DebugConfig
	enabled   bool
	traces    map[string][]*TraceEntry
	mu        sync.RWMutex
	maxBuffer int
}

// NewTraceCollector creates a new trace collector
func NewTraceCollector(config *DebugConfig) *TraceCollector {
	return &TraceCollector{
		config:    config,
		enabled:   config != nil && config.Enabled && config.Trace && config.IsBlockchainEnabled(),
		traces:    make(map[string][]*TraceEntry),
		maxBuffer: config.TraceBufferSize,
	}
}

// GenerateTraceID generates a new trace ID
func GenerateTraceID() string {
	b := make([]byte, 16)
	rand.Read(b)
	return hex.EncodeToString(b)
}

// StartTrace starts a new trace
func (tc *TraceCollector) StartTrace(traceID, operation string, ctx sdk.Context) *TraceEntry {
	if !tc.enabled {
		return nil
	}

	entry := &TraceEntry{
		TraceID:     traceID,
		Component:   "blockchain",
		Operation:   operation,
		StartTime:   time.Now(),
		BlockHeight: ctx.BlockHeight(),
		Fields:      make(map[string]interface{}),
	}

	tc.mu.Lock()
	defer tc.mu.Unlock()

	tc.traces[traceID] = append(tc.traces[traceID], entry)

	// Trim buffer if needed
	if len(tc.traces[traceID]) > tc.maxBuffer {
		tc.traces[traceID] = tc.traces[traceID][len(tc.traces[traceID])-tc.maxBuffer:]
	}

	return entry
}

// EndTrace ends a trace entry
func (tc *TraceCollector) EndTrace(entry *TraceEntry, err error) {
	if !tc.enabled || entry == nil {
		return
	}

	entry.EndTime = time.Now()
	entry.Duration = entry.EndTime.Sub(entry.StartTime)
	if err != nil {
		entry.Error = err.Error()
	}
}

// GetTraces gets all traces for a trace ID
func (tc *TraceCollector) GetTraces(traceID string) []*TraceEntry {
	if !tc.enabled {
		return nil
	}

	tc.mu.RLock()
	defer tc.mu.RUnlock()

	traces, exists := tc.traces[traceID]
	if !exists {
		return nil
	}

	// Return a copy
	result := make([]*TraceEntry, len(traces))
	for i, trace := range traces {
		traceCopy := *trace
		result[i] = &traceCopy
	}

	return result
}

// GetAllTraceIDs gets all trace IDs
func (tc *TraceCollector) GetAllTraceIDs() []string {
	if !tc.enabled {
		return nil
	}

	tc.mu.RLock()
	defer tc.mu.RUnlock()

	traceIDs := make([]string, 0, len(tc.traces))
	for traceID := range tc.traces {
		traceIDs = append(traceIDs, traceID)
	}

	return traceIDs
}

// ClearTraces clears all traces
func (tc *TraceCollector) ClearTraces() {
	if !tc.enabled {
		return
	}

	tc.mu.Lock()
	defer tc.mu.Unlock()

	tc.traces = make(map[string][]*TraceEntry)
}

// GetTraceCollector returns the trace collector for the keeper
func (k Keeper) GetTraceCollector() *TraceCollector {
	if k.debugConfig == nil || !k.debugConfig.Enabled || !k.debugConfig.Trace || !k.debugConfig.IsBlockchainEnabled() {
		return nil
	}

	// For simplicity, create a new collector each time
	// In production, this should be cached
	return NewTraceCollector(k.debugConfig)
}
