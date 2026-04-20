use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use crate::cache::{HubLayout, ModelPaths, derive_tag, is_gguf};
use crate::error::{ModelRefError, PacaError};
use crate::model::ModelRef;

/// What to remove from the cache.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RemoveTarget {
    /// Remove the entire model repository regardless of tag
    Repo { owner: String, model: String },
    /// Remove a single tag
    Tag(ModelRef),
}

impl FromStr for RemoveTarget {
    type Err = ModelRefError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.contains(':') {
            return Ok(Self::Tag(s.parse()?));
        }

        let (owner, model) = s.split_once('/').ok_or(ModelRefError::MissingOwner)?;
        if owner.is_empty() || model.is_empty() {
            return Err(ModelRefError::MissingOwner);
        }

        Ok(Self::Repo {
            owner: owner.to_string(),
            model: model.to_string(),
        })
    }
}

#[derive(Debug)]
pub struct RemoveResult {
    pub removed_files: Vec<PathBuf>,
}

pub fn remove_model(target: &str, hub_dir: Option<PathBuf>) -> Result<RemoveResult, PacaError> {
    let target: RemoveTarget = target.parse()?;
    let hub = HubLayout::open(hub_dir)?;

    match target {
        RemoveTarget::Repo { owner, model } => remove_repo(hub.root(), &owner, &model),
        RemoveTarget::Tag(model_ref) => remove_tag(&hub, &model_ref),
    }
}

fn remove_repo(hub_dir: &Path, owner: &str, model: &str) -> Result<RemoveResult, PacaError> {
    let model_dir = hub_dir.join(format!("models--{owner}--{model}"));
    if !model_dir.is_dir() {
        return Err(PacaError::ModelNotInstalled(format!("{owner}/{model}")));
    }

    let mut removed_files = Vec::new();
    remove_path_collect(&model_dir, &mut removed_files)?;
    Ok(RemoveResult { removed_files })
}

fn remove_tag(hub: &HubLayout, model_ref: &ModelRef) -> Result<RemoveResult, PacaError> {
    let paths = hub.model(model_ref);
    let model_dir = paths.dir();

    if !model_dir.is_dir() {
        return Err(PacaError::ModelNotInstalled(model_ref.to_string()));
    }

    let commit = paths
        .read_ref()
        .map(|c| c.trim().to_string())
        .ok_or_else(|| PacaError::ModelNotInstalled(model_ref.to_string()))?;

    let snapshot_dir = paths.snapshot(&commit);
    if !snapshot_dir.is_dir() {
        return Err(PacaError::ModelNotInstalled(model_ref.to_string()));
    }

    let tag_entries = find_tag_entries(&snapshot_dir, &model_ref.model, &model_ref.tag)?;
    if tag_entries.is_empty() {
        return Err(PacaError::ModelNotInstalled(model_ref.to_string()));
    }

    let mut removed_files = Vec::new();
    for entry in &tag_entries {
        remove_path_collect(entry, &mut removed_files)?;
    }

    prune_orphaned_blobs(&paths, &snapshot_dir, &mut removed_files)?;

    if !snapshot_contains_gguf(&snapshot_dir)? {
        remove_path_collect(&model_dir, &mut removed_files)?;
    }

    Ok(RemoveResult { removed_files })
}

fn prune_orphaned_blobs(
    paths: &ModelPaths<'_>,
    snapshot_dir: &Path,
    removed_files: &mut Vec<PathBuf>,
) -> Result<(), PacaError> {
    let mut referenced_blobs: HashSet<String> = HashSet::new();
    collect_referenced_blob_hashes(snapshot_dir, &mut referenced_blobs)?;

    let blobs = paths.blobs();
    if !blobs.is_dir() {
        return Ok(());
    }

    for blob_entry in fs::read_dir(&blobs).map_err(PacaError::CacheDir)? {
        let blob_entry = blob_entry.map_err(PacaError::CacheDir)?;
        let name = blob_entry.file_name().to_string_lossy().into_owned();
        if !referenced_blobs.contains(&name) {
            fs::remove_file(blob_entry.path()).map_err(PacaError::FileDelete)?;
            removed_files.push(blob_entry.path());
        }
    }
    Ok(())
}

/// Returns the paths that should be deleted to remove `tag` from a snapshot.
///
/// For a flat-layout tag (gguf files directly in the snapshot root), each
/// matching file is returned individually. For a shard-layout tag (gguf
/// files nested under a `{tag}/` subdirectory), the subdirectory itself is
/// returned once so callers can remove the entire shard tree with a single
/// call.
fn find_tag_entries(
    snapshot_dir: &Path,
    model: &str,
    tag: &str,
) -> Result<Vec<PathBuf>, PacaError> {
    let mut entries: Vec<PathBuf> = Vec::new();
    collect_tag_entries_recursive(snapshot_dir, snapshot_dir, model, tag, &mut entries)?;
    Ok(entries)
}

fn collect_tag_entries_recursive(
    base: &Path,
    dir: &Path,
    model: &str,
    tag: &str,
    entries: &mut Vec<PathBuf>,
) -> Result<(), PacaError> {
    for entry in fs::read_dir(dir).map_err(PacaError::CacheDir)? {
        let entry = entry.map_err(PacaError::CacheDir)?;
        let path = entry.path();

        if path.is_dir() {
            collect_tag_entries_recursive(base, &path, model, tag, entries)?;
            continue;
        }

        if !is_gguf(&path) {
            continue;
        }

        let name = entry.file_name().to_string_lossy().into_owned();
        let relative = path
            .strip_prefix(base)
            .unwrap_or(&path)
            .to_string_lossy()
            .into_owned();

        if let Some((subdir, _)) = relative.split_once('/') {
            if subdir == tag {
                let subdir_path = base.join(subdir);
                if !entries.contains(&subdir_path) {
                    entries.push(subdir_path);
                }
            }
        } else if derive_tag(&name, model).as_deref() == Some(tag) {
            entries.push(path);
        }
    }
    Ok(())
}

fn collect_referenced_blob_hashes(
    dir: &Path,
    referenced: &mut HashSet<String>,
) -> Result<(), PacaError> {
    for entry in fs::read_dir(dir).map_err(PacaError::CacheDir)? {
        let entry = entry.map_err(PacaError::CacheDir)?;
        let path = entry.path();

        if path.is_dir() {
            collect_referenced_blob_hashes(&path, referenced)?;
            continue;
        }

        let Ok(target) = fs::read_link(&path) else {
            continue;
        };

        if let Some((_, hash)) = target.to_string_lossy().rsplit_once("blobs/") {
            referenced.insert(hash.to_string());
        }
    }
    Ok(())
}

fn snapshot_contains_gguf(dir: &Path) -> Result<bool, PacaError> {
    for entry in fs::read_dir(dir).map_err(PacaError::CacheDir)? {
        let entry = entry.map_err(PacaError::CacheDir)?;
        let path = entry.path();

        if path.is_dir() {
            if snapshot_contains_gguf(&path)? {
                return Ok(true);
            }
        } else if is_gguf(&path) {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Recursively removes `path`, recording each *file* deleted (not directories)
/// into `removed`. Symlinks are treated as files.
fn remove_path_collect(path: &Path, removed: &mut Vec<PathBuf>) -> Result<(), PacaError> {
    let is_symlink = path
        .symlink_metadata()
        .is_ok_and(|m| m.file_type().is_symlink());

    if path.is_dir() && !is_symlink {
        for entry in fs::read_dir(path).map_err(PacaError::CacheDir)? {
            let entry = entry.map_err(PacaError::CacheDir)?;
            remove_path_collect(&entry.path(), removed)?;
        }
        fs::remove_dir(path).map_err(PacaError::FileDelete)?;
    } else {
        fs::remove_file(path).map_err(PacaError::FileDelete)?;
        removed.push(path.to_path_buf());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{setup_model_dir, write_blob, write_ref, write_snapshot_symlink};

    #[test]
    fn parses_target_with_tag_as_tag_variant() {
        let target: RemoveTarget = "owner/model-GGUF:Q4".parse().unwrap();
        let RemoveTarget::Tag(model_ref) = target else {
            panic!("expected Tag variant");
        };
        assert_eq!(model_ref.owner, "owner");
        assert_eq!(model_ref.model, "model-GGUF");
        assert_eq!(model_ref.tag, "Q4");
    }

    #[test]
    fn parses_target_without_tag_as_repo_variant() {
        let target: RemoveTarget = "owner/model-GGUF".parse().unwrap();
        assert_eq!(
            target,
            RemoveTarget::Repo {
                owner: "owner".to_string(),
                model: "model-GGUF".to_string(),
            }
        );
    }

    #[test]
    fn parses_target_without_owner_errors() {
        let result: Result<RemoveTarget, _> = "model-only".parse();
        assert!(matches!(result, Err(ModelRefError::MissingOwner)));
    }

    #[test]
    fn parses_target_with_empty_owner_errors() {
        let result: Result<RemoveTarget, _> = "/model".parse();
        assert!(matches!(result, Err(ModelRefError::MissingOwner)));
    }

    #[test]
    fn remove_model_errors_when_repo_not_installed() {
        let dir = tempfile::tempdir().unwrap();
        let result = remove_model("owner/missing-GGUF", Some(dir.path().to_path_buf()));
        assert!(matches!(result, Err(PacaError::ModelNotInstalled(_))));
    }

    #[test]
    fn remove_model_errors_when_tag_not_installed_for_existing_repo() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = setup_model_dir(dir.path(), "owner", "model-GGUF");
        write_blob(&model_dir, "hash_q4");
        write_ref(&model_dir, "commit1");
        write_snapshot_symlink(&model_dir, "commit1", "model-Q4.gguf", "hash_q4");

        let result = remove_model(
            "owner/model-GGUF:NONEXISTENT",
            Some(dir.path().to_path_buf()),
        );
        assert!(matches!(result, Err(PacaError::ModelNotInstalled(_))));
    }

    #[test]
    fn remove_tag_removes_single_file_symlink_and_orphaned_blob() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = setup_model_dir(dir.path(), "owner", "model-GGUF");
        write_blob(&model_dir, "hash_q4");
        write_ref(&model_dir, "commit1");
        write_snapshot_symlink(&model_dir, "commit1", "model-Q4.gguf", "hash_q4");

        let result = remove_model("owner/model-GGUF:Q4", Some(dir.path().to_path_buf())).unwrap();

        let symlink_path = model_dir.join("snapshots/commit1/model-Q4.gguf");
        let blob_path = model_dir.join("blobs/hash_q4");
        assert!(!model_dir.exists());
        assert!(result.removed_files.contains(&symlink_path));
        assert!(result.removed_files.contains(&blob_path));
    }

    #[test]
    fn remove_tag_preserves_other_tags_in_same_repo() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = setup_model_dir(dir.path(), "owner", "model-GGUF");
        write_blob(&model_dir, "hash_q4");
        write_blob(&model_dir, "hash_q8");
        write_ref(&model_dir, "commit1");
        write_snapshot_symlink(&model_dir, "commit1", "model-Q4.gguf", "hash_q4");
        write_snapshot_symlink(&model_dir, "commit1", "model-Q8.gguf", "hash_q8");

        let result = remove_model("owner/model-GGUF:Q4", Some(dir.path().to_path_buf())).unwrap();

        assert!(!model_dir.join("snapshots/commit1/model-Q4.gguf").exists());
        assert!(!model_dir.join("blobs/hash_q4").exists());
        assert!(model_dir.join("snapshots/commit1/model-Q8.gguf").exists());
        assert!(model_dir.join("blobs/hash_q8").exists());
        assert!(model_dir.exists());

        assert!(
            result
                .removed_files
                .contains(&model_dir.join("snapshots/commit1/model-Q4.gguf"))
        );
        assert!(
            result
                .removed_files
                .contains(&model_dir.join("blobs/hash_q4"))
        );
        assert!(
            !result
                .removed_files
                .contains(&model_dir.join("snapshots/commit1/model-Q8.gguf"))
        );
    }

    #[test]
    fn remove_tag_removes_subdir_sharded_tag_and_its_blobs() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = setup_model_dir(dir.path(), "owner", "model-GGUF");
        write_blob(&model_dir, "hash_q4");
        write_blob(&model_dir, "hash_bf16_1");
        write_blob(&model_dir, "hash_bf16_2");
        write_ref(&model_dir, "commit1");
        write_snapshot_symlink(&model_dir, "commit1", "model-Q4.gguf", "hash_q4");
        write_snapshot_symlink(
            &model_dir,
            "commit1",
            "BF16/model-BF16-00001-of-00002.gguf",
            "hash_bf16_1",
        );
        write_snapshot_symlink(
            &model_dir,
            "commit1",
            "BF16/model-BF16-00002-of-00002.gguf",
            "hash_bf16_2",
        );

        let result = remove_model("owner/model-GGUF:BF16", Some(dir.path().to_path_buf())).unwrap();

        assert!(!model_dir.join("snapshots/commit1/BF16").exists());
        assert!(!model_dir.join("blobs/hash_bf16_1").exists());
        assert!(!model_dir.join("blobs/hash_bf16_2").exists());
        assert!(model_dir.join("snapshots/commit1/model-Q4.gguf").exists());
        assert!(model_dir.join("blobs/hash_q4").exists());

        assert!(
            result
                .removed_files
                .contains(&model_dir.join("blobs/hash_bf16_1"))
        );
        assert!(
            result
                .removed_files
                .contains(&model_dir.join("blobs/hash_bf16_2"))
        );
    }

    #[test]
    fn remove_repo_removes_entire_model_dir_and_all_blobs() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = setup_model_dir(dir.path(), "owner", "model-GGUF");
        write_blob(&model_dir, "hash_q4");
        write_blob(&model_dir, "hash_q8");
        write_ref(&model_dir, "commit1");
        write_snapshot_symlink(&model_dir, "commit1", "model-Q4.gguf", "hash_q4");
        write_snapshot_symlink(&model_dir, "commit1", "model-Q8.gguf", "hash_q8");

        let result = remove_model("owner/model-GGUF", Some(dir.path().to_path_buf())).unwrap();

        assert!(!model_dir.exists());
        assert!(
            result
                .removed_files
                .contains(&model_dir.join("blobs/hash_q4"))
        );
        assert!(
            result
                .removed_files
                .contains(&model_dir.join("blobs/hash_q8"))
        );
        assert!(
            result
                .removed_files
                .contains(&model_dir.join("snapshots/commit1/model-Q4.gguf"))
        );
        assert!(
            result
                .removed_files
                .contains(&model_dir.join("snapshots/commit1/model-Q8.gguf"))
        );
    }

    #[test]
    fn remove_repo_leaves_sibling_repos_untouched() {
        let dir = tempfile::tempdir().unwrap();
        let target_dir = setup_model_dir(dir.path(), "owner", "target-GGUF");
        write_blob(&target_dir, "target_blob");
        write_ref(&target_dir, "commit1");
        write_snapshot_symlink(&target_dir, "commit1", "target-Q4.gguf", "target_blob");

        let sibling_dir = setup_model_dir(dir.path(), "owner", "sibling-GGUF");
        write_blob(&sibling_dir, "sibling_blob");
        write_ref(&sibling_dir, "commit1");
        write_snapshot_symlink(&sibling_dir, "commit1", "sibling-Q4.gguf", "sibling_blob");

        remove_model("owner/target-GGUF", Some(dir.path().to_path_buf())).unwrap();

        assert!(!target_dir.exists());
        assert!(sibling_dir.join("blobs/sibling_blob").exists());
        assert!(
            sibling_dir
                .join("snapshots/commit1/sibling-Q4.gguf")
                .exists()
        );
    }
}
