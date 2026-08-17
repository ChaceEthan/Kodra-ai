import path from "path";

/**
 * Resolves `relativePath` against `root` and verifies the canonical result is
 * actually inside `root` - not merely string-prefixed by it.
 *
 * A naive `fullPath.startsWith(root)` check is bypassable in two ways:
 *   1. `../` traversal that `path.resolve` would normally contain, but only
 *      if you check the *resolved* path (never the raw input string).
 *   2. A sibling directory that happens to share `root`'s name as a string
 *      prefix, e.g. root "/work/kodra-core" vs sibling
 *      "/work/kodra-core-secrets" - the latter *starts with* the former as
 *      a string despite being a completely different, unrelated directory.
 *
 * This function resolves both root and target to absolute, normalized paths
 * and requires the target to equal the root or be nested under it via an
 * actual path separator, which closes both bypasses.
 */
export function resolveWithinRoot(root: string, relativePath: string): string | null {
  const resolvedRoot = path.resolve(root);
  const full = path.resolve(resolvedRoot, relativePath);
  if (full !== resolvedRoot && !full.startsWith(resolvedRoot + path.sep)) {
    return null;
  }
  return full;
}

/** Parses an integer port from an env var string, falling back to a safe
 * default when unset or unparsable (never throws, never returns NaN). */
export function resolvePort(raw: string | undefined, fallback: number): number {
  if (!raw) return fallback;
  const parsed = parseInt(raw, 10);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

/** Parses a comma-separated KODRA_ALLOWED_ORIGINS value into a trimmed,
 * non-empty list of origins, falling back to the local-dev default. */
export function parseAllowedOrigins(raw: string | undefined, fallback: string[] = ["http://localhost:3000"]): string[] {
  if (!raw || !raw.trim()) return fallback;
  const origins = raw.split(",").map((o) => o.trim()).filter(Boolean);
  return origins.length > 0 ? origins : fallback;
}
