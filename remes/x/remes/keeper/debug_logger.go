package keeper

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	sdk "github.com/cosmos/cosmos-sdk/types"
)

// DebugLogger provides enhanced logging capabilities for debug mode
type DebugLogger struct {
	config  *DebugConfig
	logFile *os.File
	enabled bool
}

// LogEntry represents a structured log entry
type LogEntry struct {
	Timestamp   time.Time              `json:"timestamp"`
	Level       string                 `json:"level"`
	Message     string                 `json:"message"`
	Component   string                 `json:"component,omitempty"`
	TraceID     string                 `json:"trace_id,omitempty"`
	RequestID   string                 `json:"request_id,omitempty"`
	BlockHeight int64                  `json:"block_height,omitempty"`
	Fields      map[string]interface{} `json:"fields,omitempty"`
	Duration    string                 `json:"duration,omitempty"`
	File        string                 `json:"file,omitempty"`
	Line        int                    `json:"line,omitempty"`
}

// NewDebugLogger creates a new debug logger instance
func NewDebugLogger(config *DebugConfig) (*DebugLogger, error) {
	logger := &DebugLogger{
		config:  config,
		enabled: config != nil && config.Enabled && config.Logging && config.IsBlockchainEnabled(),
	}

	if !logger.enabled {
		return logger, nil
	}

	// Setup log file if specified
	if config.LogFile != "" {
		// Create directory if it doesn't exist
		logDir := filepath.Dir(config.LogFile)
		if err := os.MkdirAll(logDir, 0755); err != nil {
			return nil, fmt.Errorf("failed to create log directory: %w", err)
		}

		// Open log file (append mode)
		logFile, err := os.OpenFile(config.LogFile, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
		if err != nil {
			return nil, fmt.Errorf("failed to open log file: %w", err)
		}

		logger.logFile = logFile
	}

	return logger, nil
}

// Close closes the log file if open
func (dl *DebugLogger) Close() error {
	if dl.logFile != nil {
		return dl.logFile.Close()
	}
	return nil
}

// shouldLog checks if a log level should be logged based on config
func (dl *DebugLogger) shouldLog(level string) bool {
	if !dl.enabled {
		return false
	}

	// Map log levels to priorities
	levelPriority := map[string]int{
		"TRACE": 0,
		"DEBUG": 1,
		"INFO":  2,
		"WARN":  3,
		"ERROR": 4,
	}

	configPriority := levelPriority[dl.config.LogLevel]
	logPriority := levelPriority[level]

	return logPriority >= configPriority
}

// log writes a log entry
func (dl *DebugLogger) log(entry LogEntry) {
	if !dl.shouldLog(entry.Level) {
		return
	}

	var output string
	if dl.config.LogFormat == "json" {
		// JSON format
		jsonData, err := json.Marshal(entry)
		if err != nil {
			// Fallback to text format if JSON marshaling fails
			output = fmt.Sprintf("[%s] %s: %s\n", entry.Timestamp.Format(time.RFC3339), entry.Level, entry.Message)
		} else {
			output = string(jsonData) + "\n"
		}
	} else {
		// Text format
		output = dl.formatTextLog(entry)
	}

	// Write to file if available
	if dl.logFile != nil {
		dl.logFile.WriteString(output)
		dl.logFile.Sync() // Ensure it's written immediately
	}

	// Also write to stdout for console output (if debug mode is verbose)
	if dl.config.Level == DebugLevelVerbose {
		fmt.Fprint(os.Stdout, output)
	}
}

// formatTextLog formats a log entry as text
func (dl *DebugLogger) formatTextLog(entry LogEntry) string {
	timestamp := entry.Timestamp.Format("2006-01-02 15:04:05.000")
	base := fmt.Sprintf("[%s] %-5s %s", timestamp, entry.Level, entry.Message)

	if entry.Component != "" {
		base += fmt.Sprintf(" [component=%s]", entry.Component)
	}
	if entry.TraceID != "" {
		base += fmt.Sprintf(" [trace_id=%s]", entry.TraceID)
	}
	if entry.RequestID != "" {
		base += fmt.Sprintf(" [request_id=%s]", entry.RequestID)
	}
	if entry.BlockHeight > 0 {
		base += fmt.Sprintf(" [block_height=%d]", entry.BlockHeight)
	}
	if entry.Duration != "" {
		base += fmt.Sprintf(" [duration=%s]", entry.Duration)
	}
	if len(entry.Fields) > 0 {
		for k, v := range entry.Fields {
			base += fmt.Sprintf(" [%s=%v]", k, v)
		}
	}
	if entry.File != "" {
		base += fmt.Sprintf(" [%s:%d]", entry.File, entry.Line)
	}

	return base + "\n"
}

// Trace logs a TRACE level message
func (dl *DebugLogger) Trace(ctx sdk.Context, message string, fields ...interface{}) {
	entry := LogEntry{
		Timestamp:   time.Now(),
		Level:       "TRACE",
		Message:     message,
		Component:   "blockchain",
		BlockHeight: ctx.BlockHeight(),
		Fields:      dl.parseFields(fields...),
	}
	dl.log(entry)
}

// Debug logs a DEBUG level message
func (dl *DebugLogger) Debug(ctx sdk.Context, message string, fields ...interface{}) {
	entry := LogEntry{
		Timestamp:   time.Now(),
		Level:       "DEBUG",
		Message:     message,
		Component:   "blockchain",
		BlockHeight: ctx.BlockHeight(),
		Fields:      dl.parseFields(fields...),
	}
	dl.log(entry)
}

// Info logs an INFO level message
func (dl *DebugLogger) Info(ctx sdk.Context, message string, fields ...interface{}) {
	entry := LogEntry{
		Timestamp:   time.Now(),
		Level:       "INFO",
		Message:     message,
		Component:   "blockchain",
		BlockHeight: ctx.BlockHeight(),
		Fields:      dl.parseFields(fields...),
	}
	dl.log(entry)
}

// Warn logs a WARN level message
func (dl *DebugLogger) Warn(ctx sdk.Context, message string, fields ...interface{}) {
	entry := LogEntry{
		Timestamp:   time.Now(),
		Level:       "WARN",
		Message:     message,
		Component:   "blockchain",
		BlockHeight: ctx.BlockHeight(),
		Fields:      dl.parseFields(fields...),
	}
	dl.log(entry)
}

// Error logs an ERROR level message
func (dl *DebugLogger) Error(ctx sdk.Context, message string, fields ...interface{}) {
	entry := LogEntry{
		Timestamp:   time.Now(),
		Level:       "ERROR",
		Message:     message,
		Component:   "blockchain",
		BlockHeight: ctx.BlockHeight(),
		Fields:      dl.parseFields(fields...),
	}
	dl.log(entry)
}

// parseFields parses key-value pairs from variadic arguments
// Supports: ("key1", value1, "key2", value2, ...)
func (dl *DebugLogger) parseFields(fields ...interface{}) map[string]interface{} {
	result := make(map[string]interface{})
	for i := 0; i < len(fields)-1; i += 2 {
		if key, ok := fields[i].(string); ok {
			result[key] = fields[i+1]
		}
	}
	return result
}

// WithTraceID adds a trace ID to a log entry
func (dl *DebugLogger) WithTraceID(ctx sdk.Context, traceID string) *DebugLoggerWithTrace {
	return &DebugLoggerWithTrace{
		logger:  dl,
		traceID: traceID,
	}
}

// WithRequestID adds a request ID to a log entry
func (dl *DebugLogger) WithRequestID(ctx sdk.Context, requestID string) *DebugLoggerWithRequest {
	return &DebugLoggerWithRequest{
		logger:    dl,
		requestID: requestID,
	}
}

// DebugLoggerWithTrace is a logger with a trace ID
type DebugLoggerWithTrace struct {
	logger  *DebugLogger
	traceID string
}

// Trace logs with trace ID
func (dlt *DebugLoggerWithTrace) Trace(ctx sdk.Context, message string, fields ...interface{}) {
	entry := LogEntry{
		Timestamp:   time.Now(),
		Level:       "TRACE",
		Message:     message,
		Component:   "blockchain",
		TraceID:     dlt.traceID,
		BlockHeight: ctx.BlockHeight(),
		Fields:      dlt.logger.parseFields(fields...),
	}
	dlt.logger.log(entry)
}

// Debug logs with trace ID
func (dlt *DebugLoggerWithTrace) Debug(ctx sdk.Context, message string, fields ...interface{}) {
	entry := LogEntry{
		Timestamp:   time.Now(),
		Level:       "DEBUG",
		Message:     message,
		Component:   "blockchain",
		TraceID:     dlt.traceID,
		BlockHeight: ctx.BlockHeight(),
		Fields:      dlt.logger.parseFields(fields...),
	}
	dlt.logger.log(entry)
}

// DebugLoggerWithRequest is a logger with a request ID
type DebugLoggerWithRequest struct {
	logger    *DebugLogger
	requestID string
}

// Trace logs with request ID
func (dlr *DebugLoggerWithRequest) Trace(ctx sdk.Context, message string, fields ...interface{}) {
	entry := LogEntry{
		Timestamp:   time.Now(),
		Level:       "TRACE",
		Message:     message,
		Component:   "blockchain",
		RequestID:   dlr.requestID,
		BlockHeight: ctx.BlockHeight(),
		Fields:      dlr.logger.parseFields(fields...),
	}
	dlr.logger.log(entry)
}

// TimeOperation times an operation and logs the duration
func (dl *DebugLogger) TimeOperation(ctx sdk.Context, operationName string, operation func() error) error {
	start := time.Now()
	err := operation()
	duration := time.Since(start)

	fields := map[string]interface{}{
		"operation": operationName,
		"duration_ms": duration.Milliseconds(),
	}
	if err != nil {
		fields["error"] = err.Error()
		dl.Error(ctx, fmt.Sprintf("Operation %s completed with error", operationName), dl.fieldsToArgs(fields)...)
	} else {
		dl.Debug(ctx, fmt.Sprintf("Operation %s completed", operationName), dl.fieldsToArgs(fields)...)
	}

	return err
}

// fieldsToArgs converts a map to key-value arguments
func (dl *DebugLogger) fieldsToArgs(fields map[string]interface{}) []interface{} {
	args := make([]interface{}, 0, len(fields)*2)
	for k, v := range fields {
		args = append(args, k, v)
	}
	return args
}

// GetDebugLogger returns the debug logger for the keeper
func (k Keeper) GetDebugLogger() *DebugLogger {
	if k.debugConfig == nil || !k.debugConfig.Enabled || !k.debugConfig.Logging || !k.debugConfig.IsBlockchainEnabled() {
		return nil
	}

	// Create logger instance (cached would be better, but for simplicity we create on-demand)
	logger, err := NewDebugLogger(k.debugConfig)
	if err != nil {
		// If logger creation fails, return nil (debug logging is optional)
		return nil
	}

	return logger
}
