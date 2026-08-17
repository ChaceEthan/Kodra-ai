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
