package keeper

import (
	"fmt"
	"net/http"
	"os"
	"runtime/debug"
	"strings"
)

// PanicRecoveryMiddleware recovers from panics in HTTP handlers
// and returns a proper error response instead of crashing the server
func PanicRecoveryMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if err := recover(); err != nil {
				// Log the panic with stack trace
				stack := debug.Stack()
				
				// Fallback: log to stderr
				fmt.Fprintf(os.Stderr, "Panic recovered in HTTP handler: %v\nPath: %s\nMethod: %s\nStack:\n%s\n",
					err, r.URL.Path, r.Method, string(stack))

				// Return 500 Internal Server Error
				w.WriteHeader(http.StatusInternalServerError)
				w.Header().Set("Content-Type", "application/json")
				
				// In production, don't expose internal error details
				isProduction := os.Getenv("R3MES_ENV") == "production"
				if isProduction {
					fmt.Fprintf(w, `{"error":"INTERNAL_SERVER_ERROR","message":"An internal error occurred"}`)
				} else {
					// In development, show error details (sanitized)
					// Escape stack trace for JSON
					stackStr := string(stack)
					stackStr = strings.ReplaceAll(stackStr, "\n", "\\n")
					stackStr = strings.ReplaceAll(stackStr, "\"", "\\\"")
					fmt.Fprintf(w, `{"error":"INTERNAL_SERVER_ERROR","message":"%v","stack":"%s"}`, err, stackStr)
				}
			}
		}()

		next.ServeHTTP(w, r)
	})
}
