"""
Scalable source-code corpus pipeline for Kodra AI Agent (Phase 2+).

This module prepares LOCAL, explicitly-approved/licensed source trees for
training. It intentionally does NOT scrape the internet or clone arbitrary
repositories — callers are responsible for only pointing `build_manifest` at
directories they have the rights to train on.

Pipeline stages: recursive discovery -> binary/size filtering -> secret
scrubbing -> exact-duplicate removal -> quality filtering -> deterministic
train/val/test split -> manifest with provenance, license, and token stats.
"""
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional

# --- Supported languages -----------------------------------------------
EXTENSION_LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java",
    ".c": "c", ".cpp": "cpp", ".h": "c", ".hpp": "cpp", ".cs": "csharp",
    ".go": "go", ".rs": "rust", ".php": "php", ".rb": "ruby",
    ".swift": "swift", ".kt": "kotlin", ".kts": "kotlin", ".sql": "sql",
    ".sh": "shell", ".ps1": "powershell", ".html": "html", ".css": "css",
    ".scss": "scss", ".json": "json", ".yaml": "yaml", ".yml": "yaml",
    ".xml": "xml", ".md": "markdown", ".rst": "rst", ".toml": "toml",
}

# Directories that are never source-of-truth training data: build output,
# vendored/third-party code, VCS internals, dependency caches.
EXCLUDED_DIR_NAMES = {
    "node_modules", "dist", "build", "out", "target", "__pycache__",
    ".git", ".hg", ".svn", "vendor", "venv", ".venv", "env",
    ".next", ".nuxt", "coverage", ".pytest_cache", ".mypy_cache",
    "checkpoints", ".idea", ".vscode", "site-packages",
}

DEFAULT_MAX_FILE_SIZE_BYTES = 1_000_000  # 1MB per-file cap


# --- Secret / credential filtering --------------------------------------
_SECRET_PATTERNS = [
    re.compile(r"AKIA[0-9A-Z]{16}"),                       # AWS access key
    re.compile(r"-----BEGIN (RSA|OPENSSH|EC|DSA) PRIVATE KEY-----"),
    re.compile(r"sk-[A-Za-z0-9]{20,}"),                    # generic secret-key-style token
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),             # GitHub tokens
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),           # Slack tokens
    re.compile(r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"),
    re.compile(r"[a-zA-Z0-9_\-]{2,}:[a-zA-Z0-9_\-]{6,}@[\w.-]+"),  # user:pass@host URL
]


def contains_secret(text: str) -> bool:
    """Heuristic secret/credential detector. Not exhaustive - a best-effort
    filter to keep obvious credentials out of training data, not a security
    scanner."""
    return any(p.search(text) for p in _SECRET_PATTERNS)


# --- PII filtering hook (placeholder) -----------------------------------
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")


def contains_likely_pii(text: str) -> bool:
    """Minimal placeholder PII hook (currently only flags email addresses).
    Real PII scrubbing (names, phone numbers, addresses) needs a proper NLP
    pipeline and is intentionally out of scope for Phase 2 — this hook
    exists so callers have a place to plug that in later."""
    return bool(_EMAIL_RE.search(text))


# --- Quality filters ------------------------------------------------------
def filter_too_short(text: str, min_chars: int = 20) -> bool:
    """Returns True if the file should be KEPT."""
    return len(text.strip()) >= min_chars


def filter_not_mostly_binary_garbage(text: str) -> bool:
    printable = sum(1 for c in text if c.isprintable() or c in "\n\t\r")
    return len(text) == 0 or (printable / len(text)) > 0.85


# --- Metadata-contamination filtering --------------------------------------
# A record is "contaminated" when dataset/record bookkeeping (a field like
# `prompt`, `completion`, `target`, `category`, or `metadata` from an
# instruction-tuning or manifest record) gets accidentally serialized into
# what is supposed to be plain training text - e.g. a bug that writes
# `str(example_dict)` or a JSON blob into a .py file that then gets read
# back in as source code. The key is required to be quoted (`"target":` /
# `'target':`) because that is what JSON/dict serialization always produces.
# This intentionally does NOT flag a bare `target:` with no quotes, since
# that is ordinary, valid Python (e.g. a variable named `target` followed by
# the colon of an `if`/`elif` block, as in binary search) and must never be
# rejected as if it were leaked metadata.
_METADATA_KEY_PATTERN = re.compile(
    r"""["'](prompt|completion|target|category|metadata)["']\s*:"""
)


def contains_serialized_metadata(text: str) -> bool:
    """True if `text` contains a quoted dataset-record key (e.g. `"target":`)
    indicating accidental metadata serialization rather than legitimate code
    or prose."""
    return bool(_METADATA_KEY_PATTERN.search(text))


def contains_replacement_character(text: str) -> bool:
    """True if `text` contains the Unicode replacement character (U+FFFD),
    which signals upstream encoding corruption and must never enter the
    training corpus."""
    return "�" in text


def filter_no_serialized_metadata(text: str) -> bool:
    return not contains_serialized_metadata(text)


def filter_no_replacement_character(text: str) -> bool:
    return not contains_replacement_character(text)


# --- Minified / generated-file filtering -----------------------------------
# Minified or machine-generated code (bundler output, compiled protobufs,
# lockfiles-as-code, etc.) teaches a code model the wrong style and is not
# something a human wrote by hand, so it is excluded from the LM corpus even
# though it may be syntactically valid.
_GENERATED_FILE_MARKERS = (
    "do not edit", "auto-generated", "autogenerated", "@generated",
    "this file was automatically generated", "code generated by",
    "generated by the protocol buffer compiler",
)


def looks_minified_or_generated(text: str) -> bool:
    """True if `text` looks like minified or machine-generated code: a
    generated-file banner in the first few lines, or very long lines with
    little whitespace (the hallmark of minification)."""
    lines = text.splitlines()
    head = "\n".join(lines[:5]).lower()
    if any(marker in head for marker in _GENERATED_FILE_MARKERS):
        return True
    if lines:
        longest_line = max(len(l) for l in lines)
        avg_line_len = len(text) / len(lines)
        if longest_line > 1000 and avg_line_len > 300:
            return True
    return False


def filter_not_minified_or_generated(text: str) -> bool:
    return not looks_minified_or_generated(text)


DEFAULT_QUALITY_FILTERS: List[Callable[[str], bool]] = [
    filter_too_short,
    filter_not_mostly_binary_garbage,
    filter_no_serialized_metadata,
    filter_no_replacement_character,
    filter_not_minified_or_generated,
]


# --- Generated dependency-lockfile exclusion --------------------------------
# Package-manager lockfiles are 100% machine-generated, extremely repetitive,
# and not representative of how a human writes code - unlike other JSON/YAML
# config files (package.json, jsconfig.json, tsconfig.json, vercel.json, ...),
# which remain fully eligible for training. This is deliberately a narrow,
# exact-basename match (case-insensitive) rather than a content heuristic, so
# it can never accidentally catch a legitimate, hand-authored config file
# that merely happens to be JSON/YAML or to mention "lock" somewhere in it.
GENERATED_LOCKFILE_BASENAMES = frozenset({
    "package-lock.json", "npm-shrinkwrap.json", "yarn.lock",
    "pnpm-lock.yaml", "composer.lock", "cargo.lock", "gemfile.lock",
    "poetry.lock", "pipfile.lock",
})


def is_generated_lockfile(path: str) -> bool:
    """True if `path`'s basename is an exact, well-known package-manager
    lockfile name. Never a substring or extension-based match."""
    return os.path.basename(path).lower() in GENERATED_LOCKFILE_BASENAMES


# --- File records -----------------------------------------------------
@dataclass
class SourceFileRecord:
    relative_path: str
    language: str
    size_bytes: int
    sha256: str
    char_count: int
    token_estimate: int  # ~= char_count for byte-level tokenizers; documented approximation
    split: str = "train"
    license: str = "unknown"
    source: str = "local"


@dataclass
class DatasetManifest:
    created_at: float
    root: str
    seed: int
    license: str
    source: str
    num_files: int
    num_duplicates_removed: int
    num_filtered_out: int
    num_secrets_redacted: int
    num_encoding_rejected: int
    num_lockfiles_rejected: int
    total_chars: int
    total_token_estimate: int
    language_counts: Dict[str, int]
    split_counts: Dict[str, int]
    # Breakdown of num_filtered_out by the specific quality-filter function
    # name that rejected each file (e.g. "filter_no_serialized_metadata"),
    # so a report can explain WHY files were rejected, not just how many.
    filtered_reasons: Dict[str, int]
    files: List[Dict]


# --- Discovery -----------------------------------------------------------
def is_binary(path: str, sniff_bytes: int = 8000) -> bool:
    try:
        with open(path, "rb") as f:
            chunk = f.read(sniff_bytes)
        return b"\x00" in chunk
    except OSError:
        return True


def discover_source_files(root: str, max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES) -> List[str]:
    """Recursively discover candidate source files under `root`, skipping
    excluded/vendor/build directories, binaries, and oversized files."""
    matches: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIR_NAMES and not d.startswith(".")]
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in EXTENSION_LANGUAGE_MAP:
                continue
            full_path = os.path.join(dirpath, fname)
            try:
                size = os.path.getsize(full_path)
            except OSError:
                continue
            if size == 0 or size > max_file_size_bytes:
                continue
            if is_binary(full_path):
                continue
            matches.append(full_path)
    return sorted(matches)  # sorted for deterministic ordering given a fixed seed


def _deterministic_split(relative_path: str, seed: int, val_ratio: float, test_ratio: float) -> str:
    """Assigns train/val/test deterministically from a hash of (seed, path),
    so re-running the pipeline on the same corpus always yields the same
    split without needing to persist assignment state."""
    digest = hashlib.sha256(f"{seed}:{relative_path}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF  # in [0, 1)
    if bucket < test_ratio:
        return "test"
    if bucket < test_ratio + val_ratio:
        return "val"
    return "train"


def build_manifest(
    root: str,
    seed: int = 42,
    val_ratio: float = 0.05,
    test_ratio: float = 0.05,
    max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
    license: str = "unknown",
    source: str = "local",
    quality_filters: Optional[List[Callable[[str], bool]]] = None,
    redact_secrets: bool = True,
) -> DatasetManifest:
    """Build a deterministic dataset manifest for a locally-approved source
    tree. Does not write any files itself; see `write_manifest`."""
    quality_filters = quality_filters if quality_filters is not None else DEFAULT_QUALITY_FILTERS

    seen_hashes = set()
    records: List[SourceFileRecord] = []
    num_duplicates = 0
    num_filtered = 0
    num_secrets = 0
    num_encoding_rejected = 0
    num_lockfiles_rejected = 0
    filtered_reasons: Dict[str, int] = {}
    language_counts: Dict[str, int] = {}
    split_counts: Dict[str, int] = {"train": 0, "val": 0, "test": 0}
    total_chars = 0

    for path in discover_source_files(root, max_file_size_bytes):
        if is_generated_lockfile(path):
            num_lockfiles_rejected += 1
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        except (UnicodeDecodeError, OSError):
            # Strict UTF-8 decode failure - not valid UTF-8 text at all.
            num_encoding_rejected += 1
            continue

        sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if sha256 in seen_hashes:
            num_duplicates += 1
            continue

        if redact_secrets and contains_secret(text):
            num_secrets += 1
            continue

        rejecting_filter = next((f for f in quality_filters if not f(text)), None)
        if rejecting_filter is not None:
            num_filtered += 1
            filtered_reasons[rejecting_filter.__name__] = filtered_reasons.get(rejecting_filter.__name__, 0) + 1
            continue

        seen_hashes.add(sha256)
        rel_path = os.path.relpath(path, root).replace(os.sep, "/")
        ext = os.path.splitext(path)[1].lower()
        language = EXTENSION_LANGUAGE_MAP.get(ext, "unknown")
        split = _deterministic_split(rel_path, seed, val_ratio, test_ratio)

        record = SourceFileRecord(
            relative_path=rel_path,
            language=language,
            size_bytes=len(text.encode("utf-8")),
            sha256=sha256,
            char_count=len(text),
            token_estimate=len(text),  # byte-level approximation; refine per-tokenizer later
            split=split,
            license=license,
            source=source,
        )
        records.append(record)
        language_counts[language] = language_counts.get(language, 0) + 1
        split_counts[split] += 1
        total_chars += record.char_count

    return DatasetManifest(
        created_at=time.time(),
        root=os.path.abspath(root),
        seed=seed,
        license=license,
        source=source,
        num_files=len(records),
        num_duplicates_removed=num_duplicates,
        num_filtered_out=num_filtered,
        num_secrets_redacted=num_secrets,
        num_encoding_rejected=num_encoding_rejected,
        num_lockfiles_rejected=num_lockfiles_rejected,
        total_chars=total_chars,
        total_token_estimate=total_chars,
        language_counts=language_counts,
        split_counts=split_counts,
        filtered_reasons=filtered_reasons,
        files=[asdict(r) for r in records],
    )


def write_manifest(manifest: DatasetManifest, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)


def build_training_text(manifest: DatasetManifest, split: str = "train") -> str:
    """Concatenates the raw file contents (never manifest bookkeeping fields
    like license/source/sha256) for one split into a single training string,
    reading each file fresh from disk by its recorded relative_path. This is
    the only sanctioned way to turn a manifest into LM training text, so
    manifest metadata can never accidentally leak into what the model
    actually trains on."""
    parts = []
    for record in manifest.files:
        if record["split"] != split:
            continue
        full_path = os.path.join(manifest.root, record["relative_path"])
        with open(full_path, "r", encoding="utf-8") as f:
            parts.append(f.read())
    return "\n\n".join(parts)


# --- Near-duplicate detection (documented strategy, not exact matching) --
# Exact duplicates are removed above via SHA-256 content hashing. True
# near-duplicate detection (e.g. MinHash/SimHash over shingles, or embedding
# similarity) requires a much larger reference corpus to be worth the
# compute cost and is deferred to Phase 2 dataset scale-up. The hook below
# documents where that would plug in.
def near_duplicate_hook(_record: SourceFileRecord, _text: str) -> bool:
    """Placeholder: return True if a file should be treated as a near-dup of
    something already kept. Always returns False until a real near-dup
    strategy (MinHash/SimHash) is implemented for larger corpora."""
    return False
