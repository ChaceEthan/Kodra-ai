import { test } from "node:test";
import assert from "node:assert/strict";

// Run via `node --import tsx/esm --test ...` (see package.json test:server)
const { resolvePort, parseAllowedOrigins } = await import("../server-utils.ts");

test("resolvePort uses the fallback when unset", () => {
  assert.equal(resolvePort(undefined, 3000), 3000);
});

test("resolvePort parses a custom APP_PORT", () => {
  assert.equal(resolvePort("4321", 3000), 4321);
});

test("resolvePort parses a custom backend port", () => {
  assert.equal(resolvePort("9001", 8000), 9001);
});

test("resolvePort falls back on garbage input", () => {
  assert.equal(resolvePort("not-a-port", 3000), 3000);
  assert.equal(resolvePort("-5", 3000), 3000);
  assert.equal(resolvePort("0", 3000), 3000);
});

test("parseAllowedOrigins uses the local-dev default when unset", () => {
  assert.deepEqual(parseAllowedOrigins(undefined), ["http://localhost:3000"]);
  assert.deepEqual(parseAllowedOrigins(""), ["http://localhost:3000"]);
});

test("parseAllowedOrigins supports a single origin", () => {
  assert.deepEqual(parseAllowedOrigins("http://localhost:3000"), ["http://localhost:3000"]);
});

test("parseAllowedOrigins supports comma-separated origins with whitespace", () => {
  assert.deepEqual(
    parseAllowedOrigins("http://localhost:3000, http://example.com ,http://foo.com"),
    ["http://localhost:3000", "http://example.com", "http://foo.com"]
  );
});

test("parseAllowedOrigins does not default to a bare wildcard", () => {
  const result = parseAllowedOrigins(undefined);
  assert.ok(!result.includes("*"));
});
