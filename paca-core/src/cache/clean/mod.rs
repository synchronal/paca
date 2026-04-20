use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::cache::get_hub_dir;
use crate::error::PacaError;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum CleanReason {
    BrokenSymlink,
    OrphanedBlob,
    OrphanedSnapshot,
    PartialBlob,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RemovedFile {
    pub path: PathBuf,
    pub reason: CleanReason,
}

#[derive(Debug)]
pub struct CleanResult {
    pub removed_files: Vec<RemovedFile>,
}

pub fn clean_cache(hub_dir: Option<PathBuf>) -> Result<CleanResult, PacaError> {
    let hub_dir = match hub_dir {
        Some(dir) => dir,
        None => get_hub_dir()?,
    };

    if !hub_dir.exists() {
        return Ok(CleanResult {
            removed_files: Vec::new(),
        });
    }

    let mut removed_files = Vec::new();

    for entry in fs::read_dir(&hub_dir)? {
        let entry = entry?;
        let dir_name = entry.file_name().to_string_lossy().to_string();

        if !dir_name.starts_with("models--") || !entry.path().is_dir() {
            continue;
        }

        let model_dir = entry.path();
        clean_model_dir(&model_dir, &mut removed_files)?;
    }

    removed_files.sort_by(|a, b| a.path.cmp(&b.path));

    Ok(CleanResult { removed_files })
}

fn clean_model_dir(
    model_dir: &Path,
    removed_files: &mut Vec<RemovedFile>,
) -> Result<(), PacaError> {
    let refs_dir = model_dir.join("refs");
    let snapshots_dir = model_dir.join("snapshots");
    let blobs_dir = model_dir.join("blobs");

    // Read the current commit from refs/main
    let current_commit = refs_dir
        .join("main")
        .exists()
        .then(|| fs::read_to_string(refs_dir.join("main")).ok())
        .flatten();

    // Remove snapshots not referenced by refs/main
    if snapshots_dir.is_dir() {
        for snapshot_entry in fs::read_dir(&snapshots_dir)? {
            let snapshot_entry = snapshot_entry?;
            let snapshot_name = snapshot_entry.file_name().to_string_lossy().to_string();

            if current_commit.as_deref() != Some(&snapshot_name) {
                remove_dir_recursive(
                    &snapshot_entry.path(),
                    CleanReason::OrphanedSnapshot,
                    removed_files,
                )?;
            }
        }
    }

    // Walk remaining snapshots to collect referenced blob hashes and remove broken symlinks
    let mut referenced_blobs: HashSet<String> = HashSet::new();

    if snapshots_dir.is_dir() {
        collect_blob_refs_recursive(&snapshots_dir, &mut referenced_blobs, removed_files)?;
    }

    // Remove orphaned blobs and stray .partial files from aborted downloads
    if blobs_dir.is_dir() {
        for blob_entry in fs::read_dir(&blobs_dir)? {
            let blob_entry = blob_entry?;
            let blob_name = blob_entry.file_name().to_string_lossy().to_string();

            let reason = if blob_name.ends_with(".partial") {
                Some(CleanReason::PartialBlob)
            } else if !referenced_blobs.contains(&blob_name) {
                Some(CleanReason::OrphanedBlob)
            } else {
                None
            };

            if let Some(reason) = reason {
                fs::remove_file(blob_entry.path()).map_err(PacaError::FileDelete)?;
                removed_files.push(RemovedFile {
                    path: blob_entry.path(),
                    reason,
                });
            }
        }
    }

    Ok(())
}

fn collect_blob_refs_recursive(
    dir: &Path,
    referenced_blobs: &mut HashSet<String>,
    removed_files: &mut Vec<RemovedFile>,
) -> Result<(), PacaError> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            collect_blob_refs_recursive(&path, referenced_blobs, removed_files)?;
        } else if path
            .symlink_metadata()
            .is_ok_and(|m| m.file_type().is_symlink())
        {
            if let Ok(target) = fs::read_link(&path) {
                let target_str = target.to_string_lossy();
                if let Some(hash) = target_str.rsplit_once("blobs/").map(|(_, h)| h.to_string()) {
                    // Check if the symlink target actually exists
                    if path.exists() {
                        referenced_blobs.insert(hash);
                    } else {
                        fs::remove_file(&path).map_err(PacaError::FileDelete)?;
                        removed_files.push(RemovedFile {
                            path,
                            reason: CleanReason::BrokenSymlink,
                        });
                    }
                }
            } else {
                fs::remove_file(&path).map_err(PacaError::FileDelete)?;
                removed_files.push(RemovedFile {
                    path,
                    reason: CleanReason::BrokenSymlink,
                });
            }
        }
    }
    Ok(())
}

fn remove_dir_recursive(
    dir: &Path,
    reason: CleanReason,
    removed_files: &mut Vec<RemovedFile>,
) -> Result<(), PacaError> {
    if !dir.exists() && dir.symlink_metadata().is_err() {
        return Ok(());
    }

    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            let is_symlink = path
                .symlink_metadata()
                .is_ok_and(|m| m.file_type().is_symlink());

            if path.is_dir() && !is_symlink {
                remove_dir_recursive(&path, reason.clone(), removed_files)?;
            } else {
                fs::remove_file(&path).map_err(PacaError::FileDelete)?;
                removed_files.push(RemovedFile {
                    path,
                    reason: reason.clone(),
                });
            }
        }
        fs::remove_dir(dir).map_err(PacaError::FileDelete)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs;
    use std::os::unix::fs::symlink;

    fn setup_model_dir(hub: &std::path::Path, owner: &str, model: &str) -> PathBuf {
        let model_dir = hub.join(format!("models--{owner}--{model}"));
        fs::create_dir_all(model_dir.join("blobs")).unwrap();
        fs::create_dir_all(model_dir.join("refs")).unwrap();
        fs::create_dir_all(model_dir.join("snapshots")).unwrap();
        model_dir
    }

    fn write_blob(model_dir: &std::path::Path, hash: &str) {
        fs::write(model_dir.join("blobs").join(hash), b"fake blob data").unwrap();
    }

    fn write_ref(model_dir: &std::path::Path, commit: &str) {
        fs::write(model_dir.join("refs").join("main"), commit).unwrap();
    }

    fn write_snapshot_symlink(
        model_dir: &std::path::Path,
        commit: &str,
        filename: &str,
        blob_hash: &str,
    ) {
        let snapshot_dir = model_dir.join("snapshots").join(commit);
        let symlink_path = snapshot_dir.join(filename);
        if let Some(parent) = symlink_path.parent() {
            fs::create_dir_all(parent).unwrap();
        }
        let depth = filename.matches('/').count() + 2;
        let target = format!("{}blobs/{}", "../".repeat(depth), blob_hash);
        symlink(&target, &symlink_path).unwrap();
    }

    #[test]
    fn clean_cache_empty_dir_removes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();
        assert!(result.removed_files.is_empty());
    }

    #[test]
    fn clean_cache_nonexistent_dir_removes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let nonexistent = dir.path().join("does-not-exist");
        let result = clean_cache(Some(nonexistent)).unwrap();
        assert!(result.removed_files.is_empty());
    }

    #[test]
    fn clean_cache_complete_model_removes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = setup_model_dir(dir.path(), "owner", "model-GGUF");

        write_blob(&model_dir, "abc123");
        write_ref(&model_dir, "commit1");
        write_snapshot_symlink(&model_dir, "commit1", "model-Q4.gguf", "abc123");

        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();
        assert!(result.removed_files.is_empty());
    }

    #[test]
    fn clean_cache_complete_sharded_model_removes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = setup_model_dir(dir.path(), "owner", "model-GGUF");

        write_blob(&model_dir, "hash1");
        write_blob(&model_dir, "hash2");
        write_ref(&model_dir, "commit1");
        write_snapshot_symlink(
            &model_dir,
            "commit1",
            "BF16/model-BF16-00001-of-00002.gguf",
            "hash1",
        );
        write_snapshot_symlink(
            &model_dir,
            "commit1",
            "BF16/model-BF16-00002-of-00002.gguf",
            "hash2",
        );
        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();
        assert!(result.removed_files.is_empty());
    }

    #[test]
    fn clean_cache_removes_orphaned_snapshot() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = setup_model_dir(dir.path(), "owner", "model-GGUF");

        write_blob(&model_dir, "abc123");
        write_ref(&model_dir, "commit2");
        // Current snapshot
        write_snapshot_symlink(&model_dir, "commit2", "model-Q4.gguf", "abc123");
        // Old snapshot
        write_blob(&model_dir, "oldhash");
        write_snapshot_symlink(&model_dir, "commit1", "model-Q4.gguf", "oldhash");

        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();

        // Old snapshot's symlink should be removed
        let orphaned_snapshot_reasons: Vec<_> = result
            .removed_files
            .iter()
            .filter(|r| r.reason == CleanReason::OrphanedSnapshot)
            .collect();
        assert_eq!(orphaned_snapshot_reasons.len(), 1);

        // oldhash blob should be orphaned too
        let orphaned_blob_reasons: Vec<_> = result
            .removed_files
            .iter()
            .filter(|r| r.reason == CleanReason::OrphanedBlob)
            .collect();
        assert_eq!(orphaned_blob_reasons.len(), 1);

        // Current snapshot should still exist
        assert!(model_dir.join("snapshots/commit2/model-Q4.gguf").exists());
        assert!(model_dir.join("blobs/abc123").exists());
    }

    #[test]
    fn clean_cache_removes_broken_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = setup_model_dir(dir.path(), "owner", "model-GGUF");

        // Don't write the actual blob — symlink will be broken
        write_ref(&model_dir, "commit1");
        write_snapshot_symlink(&model_dir, "commit1", "model-Q4.gguf", "nonexistent_hash");

        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();

        let broken_reasons: Vec<_> = result
            .removed_files
            .iter()
            .filter(|r| r.reason == CleanReason::BrokenSymlink)
            .collect();
        assert_eq!(broken_reasons.len(), 1);
    }

    #[test]
    fn clean_cache_removes_partial_blob() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = setup_model_dir(dir.path(), "owner", "model-GGUF");

        write_blob(&model_dir, "used_hash");
        write_ref(&model_dir, "commit1");
        write_snapshot_symlink(&model_dir, "commit1", "model-Q4.gguf", "used_hash");

        // Stray .partial from an aborted download
        fs::write(
            model_dir.join("blobs").join("aborted_hash.partial"),
            b"half a download",
        )
        .unwrap();

        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();

        let partial_reasons: Vec<_> = result
            .removed_files
            .iter()
            .filter(|r| r.reason == CleanReason::PartialBlob)
            .collect();
        assert_eq!(partial_reasons.len(), 1);
        assert!(partial_reasons[0].path.ends_with("aborted_hash.partial"));

        // Final blob and symlink untouched
        assert!(model_dir.join("blobs/used_hash").exists());
        assert!(!model_dir.join("blobs/aborted_hash.partial").exists());
    }

    #[test]
    fn clean_cache_does_not_treat_partial_as_orphaned_blob() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = setup_model_dir(dir.path(), "owner", "model-GGUF");

        write_ref(&model_dir, "commit1");
        fs::write(model_dir.join("blobs").join("abc.partial"), b"x").unwrap();

        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();

        // Should be classified as PartialBlob, not OrphanedBlob
        assert_eq!(result.removed_files.len(), 1);
        assert_eq!(result.removed_files[0].reason, CleanReason::PartialBlob);
    }

    #[test]
    fn clean_cache_removes_orphaned_blob() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = setup_model_dir(dir.path(), "owner", "model-GGUF");

        write_blob(&model_dir, "used_hash");
        write_blob(&model_dir, "orphaned_hash");
        write_ref(&model_dir, "commit1");
        write_snapshot_symlink(&model_dir, "commit1", "model-Q4.gguf", "used_hash");

        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();

        let orphaned_reasons: Vec<_> = result
            .removed_files
            .iter()
            .filter(|r| r.reason == CleanReason::OrphanedBlob)
            .collect();
        assert_eq!(orphaned_reasons.len(), 1);
        assert!(orphaned_reasons[0].path.ends_with("orphaned_hash"));

        // Used blob should still exist
        assert!(model_dir.join("blobs/used_hash").exists());
    }

    #[test]
    fn clean_cache_mixed_scenario() {
        let dir = tempfile::tempdir().unwrap();

        // Complete model — should be kept
        let good_dir = setup_model_dir(dir.path(), "owner", "good-GGUF");
        write_blob(&good_dir, "good_hash");
        write_ref(&good_dir, "commit1");
        write_snapshot_symlink(&good_dir, "commit1", "model-Q4.gguf", "good_hash");

        // Model with orphaned blob — blob should be removed
        let blob_dir = setup_model_dir(dir.path(), "owner", "blob-GGUF");
        write_blob(&blob_dir, "used");
        write_blob(&blob_dir, "orphaned");
        write_ref(&blob_dir, "commit1");
        write_snapshot_symlink(&blob_dir, "commit1", "model.gguf", "used");

        let result = clean_cache(Some(dir.path().to_path_buf())).unwrap();

        // Good model untouched
        assert!(good_dir.join("blobs/good_hash").exists());
        assert!(good_dir.join("snapshots/commit1/model-Q4.gguf").exists());

        // Orphaned blob removed
        assert!(!blob_dir.join("blobs/orphaned").exists());
        assert!(blob_dir.join("blobs/used").exists());

        assert_eq!(result.removed_files.len(), 1);
        assert_eq!(result.removed_files[0].reason, CleanReason::OrphanedBlob);
    }
}
