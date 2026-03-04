import { describe, it, expect } from "vitest";
import {
  KServeError,
  KServeConnectionError,
  KServeAuthenticationError,
  KServeModelNotFoundError,
  KServeInferenceError,
  KServeTimeoutError,
  mapHttpErrorToKServeError,
} from "../../src/errors.js";

describe("KServeError hierarchy", () => {
  it("KServeError is an Error", () => {
    const err = new KServeError("base error");
    expect(err).toBeInstanceOf(Error);
    expect(err).toBeInstanceOf(KServeError);
    expect(err.name).toBe("KServeError");
    expect(err.message).toBe("base error");
    expect(err.statusCode).toBeUndefined();
  });

  it("KServeError stores statusCode", () => {
    const err = new KServeError("error", 500);
    expect(err.statusCode).toBe(500);
  });

  it("KServeConnectionError extends KServeError", () => {
    const err = new KServeConnectionError("connection failed");
    expect(err).toBeInstanceOf(KServeError);
    expect(err).toBeInstanceOf(KServeConnectionError);
    expect(err.name).toBe("KServeConnectionError");
  });

  it("KServeAuthenticationError extends KServeError", () => {
    const err = new KServeAuthenticationError("unauthorized", 401);
    expect(err).toBeInstanceOf(KServeError);
    expect(err).toBeInstanceOf(KServeAuthenticationError);
    expect(err.name).toBe("KServeAuthenticationError");
    expect(err.statusCode).toBe(401);
  });

  it("KServeModelNotFoundError extends KServeError", () => {
    const err = new KServeModelNotFoundError("not found", 404);
    expect(err).toBeInstanceOf(KServeError);
    expect(err).toBeInstanceOf(KServeModelNotFoundError);
    expect(err.name).toBe("KServeModelNotFoundError");
    expect(err.statusCode).toBe(404);
  });

  it("KServeInferenceError extends KServeError", () => {
    const err = new KServeInferenceError("inference failed", 422);
    expect(err).toBeInstanceOf(KServeError);
    expect(err).toBeInstanceOf(KServeInferenceError);
    expect(err.name).toBe("KServeInferenceError");
  });

  it("KServeTimeoutError extends KServeError", () => {
    const err = new KServeTimeoutError("timed out");
    expect(err).toBeInstanceOf(KServeError);
    expect(err).toBeInstanceOf(KServeTimeoutError);
    expect(err.name).toBe("KServeTimeoutError");
  });
});

describe("mapHttpErrorToKServeError", () => {
  it("maps 401 to KServeAuthenticationError", () => {
    const err = mapHttpErrorToKServeError(401, "Unauthorized");
    expect(err).toBeInstanceOf(KServeAuthenticationError);
    expect(err.statusCode).toBe(401);
  });

  it("maps 403 to KServeAuthenticationError", () => {
    const err = mapHttpErrorToKServeError(403, "Forbidden");
    expect(err).toBeInstanceOf(KServeAuthenticationError);
    expect(err.statusCode).toBe(403);
  });

  it("maps 404 to KServeModelNotFoundError", () => {
    const err = mapHttpErrorToKServeError(404, "Not found");
    expect(err).toBeInstanceOf(KServeModelNotFoundError);
    expect(err.statusCode).toBe(404);
  });

  it("maps 500 to KServeInferenceError", () => {
    const err = mapHttpErrorToKServeError(500, "Internal server error");
    expect(err).toBeInstanceOf(KServeInferenceError);
    expect(err.statusCode).toBe(500);
  });

  it("maps 422 to KServeInferenceError", () => {
    const err = mapHttpErrorToKServeError(422, "Unprocessable entity");
    expect(err).toBeInstanceOf(KServeInferenceError);
    expect(err.statusCode).toBe(422);
  });

  it("includes the message in the error", () => {
    const err = mapHttpErrorToKServeError(500, "Something went wrong");
    expect(err.message).toBe("Something went wrong");
  });
});
