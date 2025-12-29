package keeper

import (
	"os"
	"time"

	"github.com/getsentry/sentry-go"
)

// InitSentry initializes Sentry for error tracking
func InitSentry() error {
	sentryDSN := os.Getenv("SENTRY_DSN")
	if sentryDSN == "" {
		// Sentry is optional - don't fail if DSN is not set
		return nil
	}

	environment := os.Getenv("R3MES_ENV")
	if environment == "" {
		environment = "development"
	}

	tracesSampleRate := 0.1
	if environment == "development" {
		tracesSampleRate = 1.0
	}

	err := sentry.Init(sentry.ClientOptions{
		Dsn:              sentryDSN,
		Environment:      environment,
		TracesSampleRate: tracesSampleRate,
		Release:          os.Getenv("R3MES_VERSION"),
		ServerName:       os.Getenv("HOSTNAME"),
		BeforeSend: func(event *sentry.Event, hint *sentry.EventHint) *sentry.Event {
			// Filter out sensitive data
			if event.Request != nil {
				// Remove sensitive headers
				if event.Request.Headers != nil {
					delete(event.Request.Headers, "Authorization")
					delete(event.Request.Headers, "Cookie")
					delete(event.Request.Headers, "X-Api-Key")
				}
			}
			return event
		},
	})

	if err != nil {
		return err
	}

	// Flush buffered events before the program terminates
	defer sentry.Flush(2 * time.Second)

	return nil
}

// CaptureException captures an exception to Sentry
func CaptureException(err error, tags map[string]string) {
	if err == nil {
		return
	}

	sentry.WithScope(func(scope *sentry.Scope) {
		for k, v := range tags {
			scope.SetTag(k, v)
		}
		sentry.CaptureException(err)
	})
}

// CaptureMessage captures a message to Sentry
func CaptureMessage(message string, level sentry.Level, tags map[string]string) {
	sentry.WithScope(func(scope *sentry.Scope) {
		for k, v := range tags {
			scope.SetTag(k, v)
		}
		sentry.CaptureMessage(message)
	})
}

