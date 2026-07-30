//! Containment rules for cache paths built from untrusted values.
//!
//! Two sources feed the cache layout: the model reference a user types,
//! and values the registry returns (commit hash, blob ETag, manifest
//! filenames). Neither is trusted — the endpoint is configurable through
//! `MODEL_ENDPOINT`/`HF_ENDPOINT`, so its responses are remote input that
//! must never steer a write, or a delete, outside the hub directory.
//!
//! The rule here is containment, not a list of forbidden characters. A
//! blocklist only rejects the escapes someone thought to enumerate;
//! asking "does the finished path still land where it should?" rejects
//! every escape, including forms nobody anticipated.

use std::path::{Component, Path, PathBuf};

use crate::error::PacaError;

/// Resolves `.` and `..` without touching the filesystem.
/// [`std::fs::canonicalize`] is not usable here: these are paths we are
/// about to create, so they do not exist yet.
fn resolve(path: &Path) -> PathBuf {
    let mut resolved = PathBuf::new();

    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                resolved.pop();
            }
            component => resolved.push(component),
        }
    }

    resolved
}

/// Joins `name` under `parent` as exactly one new level.
///
/// For values that stand for a single directory entry: a model directory,
/// a blob hash, a commit hash. Resolution must be a no-op — if resolving
/// `.`/`..` changes the path at all, `name` carried structure it should
/// not have, whatever that structure was made of.
pub(crate) fn join_child(parent: &Path, name: &str) -> Result<PathBuf, PacaError> {
    let parent = resolve(parent);
    let candidate = parent.join(name);

    if resolve(&candidate) == candidate && candidate.parent() == Some(parent.as_path()) {
        return Ok(candidate);
    }

    Err(PacaError::UnsafePath(name.to_string()))
}

/// Joins a relative `path` under `root`, which it must land strictly
/// inside.
///
/// Unlike [`join_child`] this permits nesting, because manifest filenames
/// legitimately look like `BF16/model-00001-of-00030.gguf`.
pub(crate) fn join_within(root: &Path, path: &str) -> Result<PathBuf, PacaError> {
    let root = resolve(root);
    let candidate = root.join(path);

    if resolve(&candidate) == candidate && candidate != root && candidate.starts_with(&root) {
        return Ok(candidate);
    }

    Err(PacaError::UnsafePath(path.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> PathBuf {
        PathBuf::from("/hub")
    }

    #[test]
    fn join_child_accepts_a_single_level() {
        assert_eq!(
            join_child(&root(), "models--unsloth--GLM-4.7-Flash-GGUF").unwrap(),
            PathBuf::from("/hub/models--unsloth--GLM-4.7-Flash-GGUF")
        );
        assert_eq!(
            join_child(&root(), "abc123def456").unwrap(),
            PathBuf::from("/hub/abc123def456")
        );
    }

    #[test]
    fn join_child_rejects_climbing_out() {
        assert!(join_child(&root(), "..").is_err());
        assert!(join_child(&root(), "../evil").is_err());
        assert!(join_child(&root(), "../../../../tmp/evil").is_err());
    }

    #[test]
    fn join_child_rejects_landing_back_on_the_parent() {
        assert!(join_child(&root(), ".").is_err());
        assert!(join_child(&root(), "").is_err());
        assert!(join_child(&root(), "models--owner--../..").is_err());
    }

    #[test]
    fn join_child_rejects_a_sibling_reached_by_traversal() {
        // Lands inside /hub, but not where the name says it does.
        assert!(join_child(&root(), "models--owner--../victim").is_err());
    }

    #[test]
    fn join_child_accepts_a_name_that_merely_ends_in_dots() {
        // Only a component that is exactly ".." traverses; this is one
        // literal directory name, so it stays a child of the root.
        assert_eq!(
            join_child(&root(), "models--owner--..").unwrap(),
            PathBuf::from("/hub/models--owner--..")
        );
    }

    #[test]
    fn join_child_rejects_extra_levels() {
        assert!(join_child(&root(), "a/b").is_err());
        assert!(join_child(&root(), "BF16/model.gguf").is_err());
    }

    #[test]
    fn join_child_rejects_absolute_paths() {
        assert!(join_child(&root(), "/etc/passwd").is_err());
    }

    #[test]
    fn join_within_accepts_nesting() {
        assert_eq!(
            join_within(&root(), "model-Q4.gguf").unwrap(),
            PathBuf::from("/hub/model-Q4.gguf")
        );
        assert_eq!(
            join_within(&root(), "BF16/model-BF16-00001-of-00002.gguf").unwrap(),
            PathBuf::from("/hub/BF16/model-BF16-00001-of-00002.gguf")
        );
    }

    #[test]
    fn join_within_rejects_escaping() {
        assert!(join_within(&root(), "..").is_err());
        assert!(join_within(&root(), "../../evil.gguf").is_err());
        assert!(join_within(&root(), "BF16/../../evil.gguf").is_err());
        assert!(join_within(&root(), "/etc/passwd").is_err());
        assert!(join_within(&root(), "").is_err());
    }

    #[test]
    fn join_within_rejects_traversal_that_stays_inside() {
        // Lands at /hub/b, but only after climbing; a well-formed manifest
        // filename never needs this, and allowing it would desync the
        // `../` depth that the snapshot symlink target is built from.
        assert!(join_within(&root(), "a/../b.gguf").is_err());
    }
}
