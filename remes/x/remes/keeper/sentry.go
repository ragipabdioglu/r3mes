package keeper

import (
	"os"
	"time"

	"github.com/getsentry/sentry-go"
)

// InitSentry initializes Sentry for error tracking
func InitSentry() error {
	dsn := os.Getenv("SENTRY_DSN")
	if dsn == "" {
		// Sentry is optional, return nil if not configured
		return nil
	}

	err := sentry.Init(sentry.ClientOptions{
		Dsn:              dsn,
		Environment:      getEnvironment(),
		Release:          "remes@1.0.0",
		TracesSampleRate: 0.1,
		Debug:            os.Getenv("SENTRY_DEBUG") == "true",
	})
	if err != nil {
		return err
	}

	return nil
}

// CaptureException captures an exception and sends it to Sentry
func CaptureException(err error, tags ...map[string]string) {
	if err == nil {
		return
	}

	// Add tags if provided
	if len(tags) > 0 {
		sentry.WithScope(func(scope *sentry.Scope) {
			for key, value := range tags[0] {
				scope.SetTag(key, value)
			}
			sentry.CaptureException(err)
		})
		return
	}

	sentry.CaptureException(err)
}

// CaptureMessage captures a message and sends it to Sentry
func CaptureMessage(message string) {
	sentry.CaptureMessage(message)
}

// Flush flushes any buffered events to Sentry
func FlushSentry() {
	sentry.Flush(2 * time.Second)
}

// getEnvironment returns the current environment
func getEnvironment() string {
	env := os.Getenv("REMES_ENV")
	if env == "" {
		env = "development"
	}
	return env
}
