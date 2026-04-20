//! Shared fixtures for tests across the `cache` submodules.
//!
//! The `snapshots`/`blobs`/`refs` tree that HuggingFace's cache builds is
//! annoying to construct by hand, so the helpers here produce a
//! well-formed tree with the symlink structure our code expects.

use std::fs;
use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

pub(crate) fn setup_model_dir(hub: &Path, owner: &str, model: &str) -> PathBuf {
    let model_dir = hub.join(format!("models--{owner}--{model}"));
    fs::create_dir_all(model_dir.join("blobs")).unwrap();
    fs::create_dir_all(model_dir.join("refs")).unwrap();
    fs::create_dir_all(model_dir.join("snapshots")).unwrap();
    model_dir
}

pub(crate) fn write_blob(model_dir: &Path, hash: &str) {
    fs::write(model_dir.join("blobs").join(hash), b"fake blob data").unwrap();
}

pub(crate) fn write_ref(model_dir: &Path, commit: &str) {
    fs::write(model_dir.join("refs").join("main"), commit).unwrap();
}

pub(crate) fn write_snapshot_symlink(
    model_dir: &Path,
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
    let target = format!("{}blobs/{blob_hash}", "../".repeat(depth));
    symlink(&target, &symlink_path).unwrap();
}
