# Changelog

## v0.3.0

- Fix `clean` deleting the current snapshot and its blobs when `refs/main` ends in whitespace.
- Fix `rm <tag>` deleting blobs still referenced by other snapshots.
- Fix `list` and `outdated` failing outright when a snapshot holds a broken symlink.
- Check disk space against only the files that still need downloading.
- Keep completed chunks on disk when a parallel download fails size verification.
- Reject model references and registry values that resolve outside the cache directory.
- Report mid-download network failures as HTTP errors rather than manifest errors.
- Trim whitespace from `HF_TOKEN`, and error rather than panic when it cannot be used.
- Print an error to stderr when `outdated` cannot reach a repository.

## v0.2.0

- Add file size summaries to `list` output.

## v0.1.2

- Save off multiple partial files during downloads.
- Fix byte tracking during downloads with retries.

## v0.1.1

- Fix per-crate READMEs.

## v0.1.0

- Initial release.
