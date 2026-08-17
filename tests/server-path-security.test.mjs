import { test } from "node:test";
import assert from "node:assert/strict";
import path from "node:path";

// Run via `node --import tsx/esm --test ...` (see package.json test:server)
// so this TypeScript import resolves without a separate build step.
const { resolveWithinRoot } = await import("../server-utils.ts");

const ROOT = path.resolve("kodra-core");

test("allows a normal nested path", () => {
  const result = resolveWithinRoot(ROOT, "model/gpt_model.py");
  assert.equal(result, path.join(ROOT, "model", "gpt_model.py"));
});

test("allows the root itself", () => {
  const result = resolveWithinRoot(ROOT, ".");
  assert.equal(result, ROOT);
});

test("rejects simple ../ traversal", () => {
  const result = resolveWithinRoot(ROOT, "../package.json");
  assert.equal(result, null);
});

test("rejects deep ../ traversal to an absolute-looking target", () => {
  const result = resolveWithinRoot(ROOT, "../../../../etc/passwd");
  assert.equal(result, null);
});

test("rejects an absolute path outside the root", () => {
  const result = resolveWithinRoot(ROOT, path.resolve("package.json"));
  assert.equal(result, null);
});

test("rejects a sibling directory that string-prefixes the root name", () => {
  // e.g. root ".../kodra-core" vs sibling ".../kodra-core-secrets/x.txt"
  const siblingAbsolute = ROOT + "-secrets" + path.sep + "x.txt";
  const result = resolveWithinRoot(ROOT, siblingAbsolute);
  assert.equal(result, null);
});

test("rejects encoded traversal that still resolves outside after decoding", () => {
  const decoded = decodeURIComponent("..%2F..%2Fpackage.json");
  const result = resolveWithinRoot(ROOT, decoded);
  assert.equal(result, null);
});
