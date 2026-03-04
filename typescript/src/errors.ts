/**
 * Custom error classes for @langchain/kserve.
 *
 * Error hierarchy:
 *   KServeError (base)
 *   ├── KServeConnectionError   – network/DNS failures, unreachable host
 *   ├── KServeAuthenticationError – 401/403 responses
 *   ├── KServeModelNotFoundError  – 404, model not loaded
 *   ├── KServeInferenceError      – 4xx/5xx during inference
 *   └── KServeTimeoutError        – request timeout exceeded
 */

/**
 * Base error class for all KServe-related errors.
 */
export class KServeError extends Error {
  /** HTTP status code, if available */
  public readonly statusCode?: number;

  constructor(message: string, statusCode?: number) {
    super(message);
    this.name = "KServeError";
    this.statusCode = statusCode;
    // Maintain proper prototype chain in transpiled environments
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Thrown when the client cannot establish a connection to the KServe endpoint.
 * Typical causes: wrong URL, network policy, service not running.
 */
export class KServeConnectionError extends KServeError {
  constructor(message: string, statusCode?: number) {
    super(message, statusCode);
    this.name = "KServeConnectionError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Thrown when the server returns a 401 or 403 response.
 * Check your apiKey / tokenProvider configuration.
 */
export class KServeAuthenticationError extends KServeError {
  constructor(message: string, statusCode?: number) {
    super(message, statusCode);
    this.name = "KServeAuthenticationError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Thrown when the server returns 404 or the model name is not found.
 * Verify the modelName and that the InferenceService is in Ready state.
 */
export class KServeModelNotFoundError extends KServeError {
  constructor(message: string, statusCode?: number) {
    super(message, statusCode);
    this.name = "KServeModelNotFoundError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Thrown when the inference request itself fails (e.g., 422 bad input, 500 runtime error).
 */
export class KServeInferenceError extends KServeError {
  constructor(message: string, statusCode?: number) {
    super(message, statusCode);
    this.name = "KServeInferenceError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Thrown when the request exceeds the configured timeout.
 * Consider increasing the `timeout` option for cold-start scenarios.
 */
export class KServeTimeoutError extends KServeError {
  constructor(message: string) {
    super(message);
    this.name = "KServeTimeoutError";
    Object.setPrototypeOf(this, new.target.prototype);
  }
}

/**
 * Map an HTTP status code to the appropriate KServeError subclass.
 *
 * @param statusCode - HTTP response status code
 * @param message    - Error message to include
 */
export function mapHttpErrorToKServeError(
  statusCode: number,
  message: string
): KServeError {
  if (statusCode === 401 || statusCode === 403) {
    return new KServeAuthenticationError(message, statusCode);
  }
  if (statusCode === 404) {
    return new KServeModelNotFoundError(message, statusCode);
  }
  return new KServeInferenceError(message, statusCode);
}
