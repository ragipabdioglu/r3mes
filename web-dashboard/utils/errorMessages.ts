/**
 * User-friendly error message utilities
 * 
 * Converts technical error messages to user-friendly ones
 */

/**
 * Convert technical error to user-friendly message
 * 
 * @param error - Error object or message string
 * @returns User-friendly error message
 */
export function getUserFriendlyError(error: Error | string | unknown): string {
  let errorMessage: string;
  
  if (error instanceof Error) {
    errorMessage = error.message;
  } else if (typeof error === "string") {
    errorMessage = error;
  } else {
    errorMessage = "An unexpected error occurred";
  }

  // Network errors
  if (
    errorMessage.includes("fetch") ||
    errorMessage.includes("network") ||
    errorMessage.includes("ECONNREFUSED") ||
    errorMessage.includes("Failed to fetch")
  ) {
    return "Unable to connect to server. Please restart the application or check your internet connection.";
  }

  // Timeout errors
  if (errorMessage.includes("timeout") || errorMessage.includes("TIMEOUT")) {
    return "Request timed out. Please try again.";
  }

  // Authentication errors
  if (
    errorMessage.includes("401") ||
    errorMessage.includes("Unauthorized") ||
    errorMessage.includes("Invalid or expired API key")
  ) {
    return "Your session has expired. Please log in again.";
  }

  // Payment/credit errors
  if (
    errorMessage.includes("402") ||
    errorMessage.includes("Insufficient credits")
  ) {
    return "Insufficient credits. Please earn credits by mining.";
  }

  // Not found errors
  if (
    errorMessage.includes("404") ||
    errorMessage.includes("Not Found") ||
    errorMessage.includes("not found")
  ) {
    return "The requested resource was not found. Please refresh the page.";
  }

  // Server errors
  if (
    errorMessage.includes("500") ||
    errorMessage.includes("Internal Server Error")
  ) {
    return "A server error occurred. Please try again later.";
  }

  // Database errors
  if (errorMessage.includes("database") || errorMessage.includes("SQL")) {
    return "Database error. Please restart the application.";
  }

  // Generic fallback
  return errorMessage || "An error occurred. Please try again.";
}

/**
 * Get error title based on error type
 * 
 * @param error - Error object or message string
 * @returns Error title
 */
export function getErrorTitle(error: Error | string | unknown): string {
  let errorMessage: string;
  
  if (error instanceof Error) {
    errorMessage = error.message;
  } else if (typeof error === "string") {
    errorMessage = error;
  } else {
    return "Error";
  }

  if (errorMessage.includes("network") || errorMessage.includes("fetch")) {
    return "Connection Error";
  }

  if (errorMessage.includes("401") || errorMessage.includes("Unauthorized")) {
    return "Authorization Error";
  }

  if (errorMessage.includes("402") || errorMessage.includes("credits")) {
    return "Insufficient Credits";
  }

  if (errorMessage.includes("404")) {
    return "Not Found";
  }

  if (errorMessage.includes("500")) {
    return "Server Error";
  }

  return "Error";
}

